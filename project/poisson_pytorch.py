import torch
import numpy as np
import time
import config

def solve_pytorch(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    device = torch.device('cuda')
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = np.float32((xmax - xmin) / (nx - 1))
    dy = np.float32((ymax - ymin) / (ny - 1))

    # 使用 float32
    p = torch.zeros((ny, nx), device=device, dtype=torch.float32)
    p_old = torch.zeros((ny, nx), device=device, dtype=torch.float32)
    b = torch.zeros((ny, nx), device=device, dtype=torch.float32)

    b[int(ny / 4), int(nx / 4)] = 100.0
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100.0

    dx2 = dx**2
    dy2 = dy**2
    div_term = torch.tensor(1.0 / (2 * (dx2 + dy2)), device=device, dtype=torch.float32)
    
    # 预热
    torch.cuda.synchronize()

    start_time = time.time()
    final_it = 0
    
    for it in range(max_iter):
        p_old.copy_(p) # 原地拷贝，避免重新分配内存
        
        p[1:-1, 1:-1] = (((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy2 +
                          (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx2 -
                          b[1:-1, 1:-1] * dx2 * dy2) * div_term)
        
        p[0, :], p[-1, :], p[:, 0], p[:, -1] = 0.0, 0.0, 0.0, 0.0
        
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            diff = torch.max(torch.abs(p - p_old)).item()
            if diff < tol:
                final_it = it
                break
    else:
        final_it = max_iter

    torch.cuda.synchronize()
    return None, None, p.cpu().numpy(), final_it, time.time() - start_time