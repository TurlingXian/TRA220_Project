import cupy as cp
import numpy as np
import time
import config

def solve_cupy(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = np.float32((xmax - xmin) / (nx - 1))
    dy = np.float32((ymax - ymin) / (ny - 1))

    # 1. 显式使用 float32
    p = cp.zeros((ny, nx), dtype=cp.float32)
    pd = cp.zeros((ny, nx), dtype=cp.float32)
    b = cp.zeros((ny, nx), dtype=cp.float32)

    # 源项位置必须与 CPU 版严格对齐
    b[int(ny / 4), int(nx / 4)] = 100.0
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100.0

    dx2 = dx**2
    dy2 = dy**2
    div_term = cp.float32(1.0 / (2 * (dx2 + dy2)))
    
    # 2. 预热 (Warmup)
    p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
                      (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
                      b[1:-1, 1:-1] * dx2 * dy2) * div_term)
    cp.cuda.Device().synchronize() 

    start_time = time.time()
    final_it = 0

    for it in range(max_iter):
        pd[:] = p[:] # GPU 内部拷贝
        
        p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
                          (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
                          b[1:-1, 1:-1] * dx2 * dy2) * div_term)

        p[0, :], p[-1, :], p[:, 0], p[:, -1] = 0.0, 0.0, 0.0, 0.0
        
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            diff = cp.max(cp.abs(p - pd))
            if diff < tol:
                final_it = it
                break
    else:
        final_it = max_iter

    cp.cuda.Device().synchronize() # 计时结束前必须同步
    total_time = time.time() - start_time
    
    return None, None, cp.asnumpy(p), final_it, total_time