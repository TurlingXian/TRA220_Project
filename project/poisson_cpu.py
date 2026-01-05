import numpy as np
import time
import config

# 修改点：增加了 tol=config.TOLERANCE 参数
def solve_cpu(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
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
    
    start_time = time.time()
    final_it = 0
    final_error = 0.0

    # print(f"CPU Baseline: Running {max_iter} steps (tol={tol:.1e})...")

    for it in range(max_iter):
        pd[:] = p[:]
        p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
                          (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
                          b[1:-1, 1:-1] * dx2 * dy2) * div_term)

        p[0, :] = 0; p[-1, :] = 0
        p[:, 0] = 0; p[:, -1] = 0
        
        # 收敛检查逻辑
        # 修改点：使用传入的 tol，而不是 config.TOLERANCE
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            final_error = np.abs(p - pd).max()
            if final_error < tol:
                final_it = it
                # print(f"   ✅ CPU Converged at {it}, err={final_error:.2e}")
                break
    else:
        final_it = max_iter

    return x, y, p, final_it, time.time() - start_time