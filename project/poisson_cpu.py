import numpy as np
import time
import config

def solve_cpu(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = np.float32((xmax - xmin) / (nx - 1))
    dy = np.float32((ymax - ymin) / (ny - 1))

    # 统一使用 float32
    p = np.zeros((ny, nx), dtype=np.float32)
    pd = np.zeros((ny, nx), dtype=np.float32)
    b = np.zeros((ny, nx), dtype=np.float32)
    
    x = np.linspace(xmin, xmax, nx, dtype=np.float32)
    y = np.linspace(ymin, ymax, ny, dtype=np.float32)

    b[int(ny / 4), int(nx / 4)] = 100.0
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100.0

    dx2 = dx**2
    dy2 = dy**2
    div_term = np.float32(1.0 / (2 * (dx2 + dy2)))
    
    start_time = time.time()
    final_it = 0

    for it in range(max_iter):
        pd[:] = p[:]
        # 5点模板计算
        p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
                          (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
                          b[1:-1, 1:-1] * dx2 * dy2) * div_term)

        p[0, :] = 0; p[-1, :] = 0
        p[:, 0] = 0; p[:, -1] = 0
        
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            final_error = np.abs(p - pd).max()
            if final_error < tol:
                final_it = it
                break
    else:
        final_it = max_iter

    return x, y, p, final_it, time.time() - start_time