from numba import cuda, float32
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

def solve_numba(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    if not cuda.is_available():
        return None, None, None, None, None

    # --- 1. 准备 CPU 数据 ---
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = float32((xmax - xmin) / (nx - 1))
    dy = float32((ymax - ymin) / (ny - 1))
    
    p_host = np.zeros((ny, nx), dtype=np.float32)
    b_host = np.zeros((ny, nx), dtype=np.float32)
    b_host[int(ny / 4), int(nx / 4)] = 100
    b_host[int(3 * ny / 4), int(3 * nx / 4)] = -100

    dx2, dy2 = dx**2, dy**2
    div_term = float32(1.0 / (2 * (dx2 + dy2)))

    threads_per_block = (16, 16)
    blocks_per_grid = (int(math.ceil(nx / 16)), int(math.ceil(ny / 16)))
    
    start_time = time.time()
    final_it = 0

    # --- 2. 关键修改：用 Context Manager 包裹所有 GPU 操作 ---
    # 这样可以保证 d_p_in 分配时的上下文和 kernel 运行时的上下文是同一个
    try:
        with cuda.gpus[0]:
            # 显存分配
            d_p_in = cuda.to_device(p_host)
            d_p_out = cuda.to_device(p_host)
            d_b = cuda.to_device(b_host)
            
            # 迭代循环
            for it in range(max_iter):
                poisson_kernel[blocks_per_grid, threads_per_block](
                    d_p_out, d_p_in, d_b, dx2, dy2, div_term, nx, ny
                )
                # 同步
                cuda.synchronize()
                
                # 收敛检查
                if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
                    p_curr = d_p_out.copy_to_host()
                    p_prev = d_p_in.copy_to_host()
                    diff = np.max(np.abs(p_curr - p_prev))
                    
                    if diff < tol:
                        final_it = it
                        # 确保最后的结果在 d_p_in 里
                        d_p_in.copy_to_device(d_p_out) 
                        break
                
                # 交换指针 (仅在 GPU 内部进行)
                d_p_in, d_p_out = d_p_out, d_p_in
            else:
                final_it = max_iter
            
            # 取回结果
            p_final = d_p_in.copy_to_host()
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return None, None, None, None, None

    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), p_final, final_it, time.time() - start_time