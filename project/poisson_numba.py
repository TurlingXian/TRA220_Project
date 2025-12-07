from numba import cuda
import numpy as np
import math
import time
import config

@cuda.jit
def poisson_kernel(p_out, p_in, b, dx2, dy2, div_term, nx, ny):
    x, y = cuda.grid(2)
    if x > 0 and x < nx - 1 and y > 0 and y < ny - 1:
        p_out[y, x] = (((p_in[y, x+1] + p_in[y, x-1]) * dy2 +
                        (p_in[y+1, x] + p_in[y-1, x]) * dx2 -
                        b[y, x] * dx2 * dy2) * div_term)

def solve_numba(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER):
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

    threads_per_block = (16, 16)
    blocks_per_grid = (math.ceil(nx / 16), math.ceil(ny / 16))

    start_time = time.time()
    final_it = 0

    for it in range(max_iter):
        poisson_kernel[blocks_per_grid, threads_per_block](
            d_p_out, d_p_in, d_b, dx2, dy2, div_term, nx, ny
        )
        cuda.synchronize()
        
        # 折中优化：每 CHECK_INTERVAL 步拷贝回 CPU 检查一次
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            p_curr = d_p_out.copy_to_host()
            p_prev = d_p_in.copy_to_host()
            diff = np.max(np.abs(p_curr - p_prev))
            if diff < config.TOLERANCE:
                final_it = it
                print(f"   ✅ Numba Converged at {it}, err={diff:.2e}")
                d_p_in = d_p_out
                break
        
        d_p_in, d_p_out = d_p_out, d_p_in

    else:
        final_it = max_iter

    cuda.synchronize()
    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), d_p_in.copy_to_host(), final_it, time.time() - start_time