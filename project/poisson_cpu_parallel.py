import numpy as np
import time
import config
from numba import jit, prange

@jit(nopython=True, parallel=True)
def poisson_step_parallel(p, pd, b, dx2, dy2, div_term, nx, ny):
    # prange 自动将外层循环分配给不同的 CPU 核心
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            p[y, x] = (((pd[y, x+1] + pd[y, x-1]) * dy2 +
                        (pd[y+1, x] + pd[y-1, x]) * dx2 -
                        b[y, x] * dx2 * dy2) * div_term)
    return p

# 修改点：增加 tol 参数
def solve_cpu_parallel(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    p = np.zeros((ny, nx))
    pd = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    b[int(ny / 4), int(nx / 4)] = 100
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100

    dx2 = dx**2
    dy2 = dy**2
    div_term = 1.0 / (2 * (dx2 + dy2))

    # 预热编译 (Numba 首次运行需要编译)
    poisson_step_parallel(p, pd, b, dx2, dy2, div_term, nx, ny)

    start_time = time.time()
    final_it = 0

    for it in range(max_iter):
        # 高效的原地复制
        pd[:] = p[:]
        
        # 并行计算
        poisson_step_parallel(p, pd, b, dx2, dy2, div_term, nx, ny)
        
        # 边界条件 (在 CPU 上串行执行，但这部分开销相对于内部的 O(N^2) 计算很小)
        p[0, :] = 0; p[-1, :] = 0
        p[:, 0] = 0; p[:, -1] = 0

        # 收敛检查
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            final_error = np.abs(p - pd).max()
            if final_error < tol: # 使用传入的 tol
                final_it = it
                # print(f"   ✅ CPU Parallel Converged at {it}")
                break
    else:
        final_it = max_iter

    return x, y, p, final_it, time.time() - start_time