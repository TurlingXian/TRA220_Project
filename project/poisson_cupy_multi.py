import cupy as cp
import numpy as np
import time
import config



def enable_p2p_if_possible(dev0=0, dev1=1):
    can01 = int(cp.cuda.runtime.deviceCanAccessPeer(dev0, dev1))
    can10 = int(cp.cuda.runtime.deviceCanAccessPeer(dev1, dev0))
    print(f"canAccessPeer {dev0}->{dev1}: {can01}, {dev1}->{dev0}: {can10}")

    if not (can01 and can10):
        print("⚠️ GPU P2P not available (deviceCanAccessPeer=0), fallback to CPU copy")
        return False

    def _enable(a, b):
        try:
            with cp.cuda.Device(a):
                cp.cuda.runtime.deviceEnablePeerAccess(b)
            return True
        except cp.cuda.runtime.CUDARuntimeError as e:
            msg = str(e).lower()
            # 关键：把 already enabled 当成成功
            if "peeraccessalreadyenabled" in msg or "already enabled" in msg:
                return True
            raise

    try:
        _enable(dev0, dev1)
        _enable(dev1, dev0)
        print("✅ GPU P2P enabled")
        return True
    except Exception as e:
        print(f"⚠️ EnablePeerAccess failed (real): {repr(e)}")
        return False


def solve_cupy_2gpu(nx=config.NX, ny=config.NY,
                   max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    dx2 = dx * dx
    dy2 = dy * dy
    div_term = 1.0 / (2.0 * (dx2 + dy2))

    # --- domain split (y direction) ---
    mid = ny // 2

    # GPU 0 handles [0 : mid+1]
    # GPU 1 handles [mid-1 : ny]
    ny0 = mid + 1
    ny1 = ny - mid + 1

    # --- allocate explicitly on each GPU ---
    with cp.cuda.Device(0):
        p0 = cp.zeros((ny0, nx), dtype=cp.float32)
        pd0 = cp.zeros_like(p0)
        b0 = cp.zeros_like(p0)

    with cp.cuda.Device(1):
        p1 = cp.zeros((ny1, nx), dtype=cp.float32)
        pd1 = cp.zeros_like(p1)
        b1 = cp.zeros_like(p1)

    # --- source terms (write on the correct device) ---
    iy_a = int(ny / 4)
    ix_a = int(nx / 4)
    if iy_a < mid:
        with cp.cuda.Device(0):
            b0[iy_a, ix_a] = 100
    else:
        with cp.cuda.Device(1):
            b1[iy_a - mid + 1, ix_a] = 100

    iy_b = int(3 * ny / 4)
    ix_b = int(3 * nx / 4)
    if iy_b < mid:
        with cp.cuda.Device(0):
            b0[iy_b, ix_b] = -100
    else:
        with cp.cuda.Device(1):
            b1[iy_b - mid + 1, ix_b] = -100

    # --- try P2P ---
    p2p = enable_p2p_if_possible(0, 1)

    # --- warmup / init ---
    with cp.cuda.Device(0):
        p0[1:-1, 1:-1] = 0.0
    with cp.cuda.Device(1):
        p1[1:-1, 1:-1] = 0.0
    cp.cuda.Device(0).synchronize()
    cp.cuda.Device(1).synchronize()

    start_time = time.time()
    final_it = max_iter

    for it in range(max_iter):
        # snapshot
        with cp.cuda.Device(0):
            pd0[:] = p0
        with cp.cuda.Device(1):
            pd1[:] = p1

        # --- GPU 0 compute (excluding bottom halo) ---
        with cp.cuda.Device(0):
            p0[1:-1, 1:-1] = (
                ((pd0[1:-1, 2:] + pd0[1:-1, :-2]) * dy2 +
                 (pd0[2:, 1:-1] + pd0[:-2, 1:-1]) * dx2 -
                 b0[1:-1, 1:-1] * dx2 * dy2) * div_term
            )

        # --- GPU 1 compute (excluding top halo) ---
        with cp.cuda.Device(1):
            p1[1:-1, 1:-1] = (
                ((pd1[1:-1, 2:] + pd1[1:-1, :-2]) * dy2 +
                 (pd1[2:, 1:-1] + pd1[:-2, 1:-1]) * dx2 -
                 b1[1:-1, 1:-1] * dx2 * dy2) * div_term
            )

        # --- halo exchange ---
        if p2p:
            # GPU0 -> GPU1 : p0[-2, :] to p1[0, :]
            with cp.cuda.Device(1):
                cp.copyto(p1[0, :], p0[-2, :])

            # GPU1 -> GPU0 : p1[1, :] to p0[-1, :]
            with cp.cuda.Device(0):
                cp.copyto(p0[-1, :], p1[1, :])
        else:
            # Host staging fallback (slow, but correct)
            with cp.cuda.Device(0):
                tmp0 = cp.asnumpy(p0[-2, :])
            with cp.cuda.Device(1):
                tmp1 = cp.asnumpy(p1[1, :])

            with cp.cuda.Device(1):
                p1[0, :] = cp.asarray(tmp0)
            with cp.cuda.Device(0):
                p0[-1, :] = cp.asarray(tmp1)

        # --- boundary conditions (apply on each GPU) ---
        with cp.cuda.Device(0):
            p0[0, :] = 0.0
            p0[:, 0] = 0.0
            p0[:, -1] = 0.0

        with cp.cuda.Device(1):
            p1[-1, :] = 0.0
            p1[:, 0] = 0.0
            p1[:, -1] = 0.0

        # --- convergence check ---
        if (not config.BENCHMARK_MODE) and (it % config.CHECK_INTERVAL == 0):
            with cp.cuda.Device(0):
                diff0 = cp.max(cp.abs(p0 - pd0)).item()
            with cp.cuda.Device(1):
                diff1 = cp.max(cp.abs(p1 - pd1)).item()

            diff = max(diff0, diff1)
            if diff < tol:
                final_it = it
                break

    cp.cuda.Device(0).synchronize()
    cp.cuda.Device(1).synchronize()
    total_time = time.time() - start_time

    return final_it, total_time
