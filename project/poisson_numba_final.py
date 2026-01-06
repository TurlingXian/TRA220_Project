from numba import cuda, float32
import numpy as np
import math
import time
import config

# Tile and Shared Memory sizes (16x16 threads + 1px halo on all sides)
TILE_X = 16
TILE_Y = 16

@cuda.jit
def poisson_shared_kernel(p_out, p_in, b, dx2, dy2, div_term, nx, ny):
    # Fixed shape for A100 shared memory (16+2 x 16+2)
    s_p = cuda.shared.array(shape=(18, 18), dtype=float32)
    
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x, y = cuda.grid(2)
    
    # Shared memory local index (offset by 1 for halo)
    sx, sy = tx + 1, ty + 1
    
    # 1. Load core data
    if x < nx and y < ny:
        s_p[sy, sx] = p_in[y, x]
    else:
        s_p[sy, sx] = 0.0

    # 2. Load Halo (Boundary data)
    if tx == 0 and x > 0:           s_p[sy, 0] = p_in[y, x - 1]
    if tx == TILE_X-1 and x < nx-1: s_p[sy, sx+1] = p_in[y, x + 1]
    if ty == 0 and y > 0:           s_p[0, sx] = p_in[y - 1, x]
    if ty == TILE_Y-1 and y < ny-1: s_p[sy+1, sx] = p_in[y + 1, x]
        
    cuda.syncthreads()
    
    # 3. Compute Jacobi
    if 0 < x < nx - 1 and 0 < y < ny - 1:
        val = (((s_p[sy, sx+1] + s_p[sy, sx-1]) * dy2 +
                (s_p[sy+1, sx] + s_p[sy-1, sx]) * dx2 -
                b[y, x] * dx2 * dy2) * div_term)
        p_out[y, x] = val

def solve_numba_shared(nx, ny, max_iter, tol):
    # Setup physics and grid
    # --- 修改这里：手动计算 dx 和 dy ---
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    
    # 显式转换为 float32 以匹配 GPU 精度
    dx = float32((xmax - xmin) / (nx - 1))
    dy = float32((ymax - ymin) / (ny - 1))
    dx2, dy2 = dx**2, dy**2
    div_term = float32(1.0 / (2 * (dx2 + dy2)))
    # -----------------------------------
    
    # Initialize host data
    p_host = np.zeros((ny, nx), dtype=np.float32)
    b_host = np.zeros((ny, nx), dtype=np.float32)
    b_host[int(ny / 4), int(nx / 4)] = 100
    b_host[int(3 * ny / 4), int(3 * nx / 4)] = -100

    # Device allocation
    d_p_in = cuda.to_device(p_host)
    d_p_out = cuda.to_device(p_host)
    d_b = cuda.to_device(b_host)

    threads = (TILE_X, TILE_Y)
    blocks = (math.ceil(nx / TILE_X), math.ceil(ny / TILE_Y))
    
    start_time = time.time()
    final_it = 0
    
    # Kernel Fusion: Increase check_interval to eliminate T_gap
    check_interval = 100 
    
    for it in range(0, max_iter, check_interval):
        for _ in range(check_interval):
            poisson_shared_kernel[blocks, threads](d_p_out, d_p_in, d_b, dx2, dy2, div_term, nx, ny)
            d_p_in, d_p_out = d_p_out, d_p_in # Pointer swap
        
        # Synchronization only happens here, reducing scheduling overhead
        cuda.synchronize()
        final_it = it + check_interval
        
        # Convergence Check (Copy to host for error calculation)
        p_curr = d_p_in.copy_to_host()
        p_prev = d_p_out.copy_to_host()
        if np.max(np.abs(p_curr - p_prev)) < tol:
            break

    total_duration = time.time() - start_time
    p_final = d_p_in.copy_to_host()

    # --- THIS IS THE MISSING PART ---
    # Return 5 values to match: _, _, _, iters, duration
    return (np.linspace(config.X_MIN, config.X_MAX, nx), 
            np.linspace(config.Y_MIN, config.Y_MAX, ny), 
            p_final, 
            final_it, 
            total_duration)