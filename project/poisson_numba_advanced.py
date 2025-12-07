from numba import cuda, float32
import numpy as np
import math
import time
import config

TILE_SIZE = 16

@cuda.jit
def poisson_shared_kernel(p_out, p_in, b, dx2, dy2, div_term, nx, ny):
    s_p = cuda.shared.array((TILE_SIZE + 2, TILE_SIZE + 2), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x, y = cuda.grid(2)
    sx, sy = tx + 1, ty + 1
    
    if x < nx and y < ny:
        s_p[sy, sx] = p_in[y, x]
    else:
        s_p[sy, sx] = 0.0

    if tx == 0 and x > 0: s_p[sy, 0] = p_in[y, x - 1]
    if tx == TILE_SIZE - 1 and x < nx - 1: s_p[sy, sx + 1] = p_in[y, x + 1]
    if ty == 0 and y > 0: s_p[0, sx] = p_in[y - 1, x]
    if ty == TILE_SIZE - 1 and y < ny - 1: s_p[sy + 1, sx] = p_in[y + 1, x]
        
    cuda.syncthreads()
    
    if x > 0 and x < nx - 1 and y > 0 and y < ny - 1:
        val = (((s_p[sy, sx+1] + s_p[sy, sx-1]) * dy2 +
                (s_p[sy+1, sx] + s_p[sy-1, sx]) * dx2 -
                b[y, x] * dx2 * dy2) * div_term)
        p_out[y, x] = val

def solve_numba_shared(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER):
    if not cuda.is_available():
        return None, None, None, None, None

    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    
    p_host = np.zeros((ny, nx), dtype=np.float32)
    b_host = np.zeros((ny, nx), dtype=np.float32)
    b_host[int(ny / 4), int(nx / 4)] = 100
    b_host[int(3 * ny / 4), int(3 * nx / 4)] = -100

    d_p_in = cuda.to_device(p_host)
    d_p_out = cuda.to_device(p_host)
    d_b = cuda.to_device(b_host)

    dx2, dy2 = dx**2, dy**2
    div_term = 1.0 / (2 * (dx2 + dy2))

    threads_per_block = (TILE_SIZE, TILE_SIZE)
    blocks_per_grid = (math.ceil(nx / TILE_SIZE), math.ceil(ny / TILE_SIZE))

    start_time = time.time()
    final_it = 0

    for it in range(max_iter):
        poisson_shared_kernel[blocks_per_grid, threads_per_block](
            d_p_out, d_p_in, d_b, dx2, dy2, div_term, nx, ny
        )
        cuda.synchronize()
        
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            p_curr = d_p_out.copy_to_host()
            p_prev = d_p_in.copy_to_host()
            diff = np.max(np.abs(p_curr - p_prev))
            if diff < config.TOLERANCE:
                final_it = it
                print(f"   ✅ Numba Shared Converged at {it}, err={diff:.2e}")
                d_p_in = d_p_out
                break
        
        d_p_in, d_p_out = d_p_out, d_p_in

    else:
        final_it = max_iter

    cuda.synchronize()
    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), d_p_in.copy_to_host(), final_it, time.time() - start_time