import numpy as np
import time
import config
import os
import numba
from numba import njit, prange


@njit
def poisson_step_serial(p, pd, b, dx2, dy2, div_term, nx, ny):
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            p[y, x] = (((pd[y, x + 1] + pd[y, x - 1]) * dy2 +
                        (pd[y + 1, x] + pd[y - 1, x]) * dx2 -
                        b[y, x] * dx2 * dy2) * div_term)


@njit(parallel=True)
def poisson_step_parallel(p, pd, b, dx2, dy2, div_term, nx, ny):
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            p[y, x] = (((pd[y, x + 1] + pd[y, x - 1]) * dy2 +
                        (pd[y + 1, x] + pd[y - 1, x]) * dx2 -
                        b[y, x] * dx2 * dy2) * div_term)


def solve_cpu_auto(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    """
    Auto solver:
      - if numba.get_num_threads() <= 1: run serial kernel
      - else: run parallel kernel
    """
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    p = np.zeros((ny, nx), dtype=np.float64)
    pd = np.zeros((ny, nx), dtype=np.float64)
    b = np.zeros((ny, nx), dtype=np.float64)

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    b[int(ny / 4), int(nx / 4)] = 100.0
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100.0

    dx2 = dx * dx
    dy2 = dy * dy
    div_term = 1.0 / (2.0 * (dx2 + dy2))

    threads = numba.get_num_threads()

    start_time = time.time()
    final_it = max_iter

    for it in range(max_iter):
        pd[:] = p

        if threads <= 1:
            poisson_step_serial(p, pd, b, dx2, dy2, div_term, nx, ny)
        else:
            poisson_step_parallel(p, pd, b, dx2, dy2, div_term, nx, ny)

        # boundary
        p[0, :] = 0.0
        p[-1, :] = 0.0
        p[:, 0] = 0.0
        p[:, -1] = 0.0

        # convergence check
        if (not config.BENCHMARK_MODE) and (it % config.CHECK_INTERVAL == 0):
            final_error = np.abs(p - pd).max()
            if final_error < tol:
                final_it = it
                break

    return x, y, p, final_it, time.time() - start_time


def configure_numba_threads_from_env(default_threads: int = 1) -> int:
    """
    Ensures numba uses the thread count from NUMBA_NUM_THREADS if set.
    Call this once at program start.
    """
    n = int(os.environ.get("NUMBA_NUM_THREADS", str(default_threads)))
    numba.set_num_threads(n)
    return numba.get_num_threads()
