import torch
import numpy as np
import time
import config

# 修改点：增加 tol 参数，默认值为 config.TOLERANCE
def solve_pytorch(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
        
    device = torch.device('cuda')
    
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    p = torch.zeros((ny, nx), device=device, dtype=torch.float32)
    b = torch.zeros((ny, nx), device=device, dtype=torch.float32)

    b[int(ny / 4), int(nx / 4)] = 100
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100

    dx2 = dx**2
    dy2 = dy**2
    div_term = 1.0 / (2 * (dx2 + dy2))
    
    start_time = time.time()
    final_it = 0
    
    # 预热 GPU (可选，但推荐在 benchmark 脚本里做)
    # torch.cuda.synchronize() 

    for it in range(max_iter):
        p_old = p.clone()
        
        p[1:-1, 1:-1] = (((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy2 +
                          (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx2 -
                          b[1:-1, 1:-1] * dx2 * dy2) * div_term)
        
        p[0, :] = 0; p[-1, :] = 0
        p[:, 0] = 0; p[:, -1] = 0
        
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            # 关键优化：只传标量 .item()
            diff = torch.max(torch.abs(p - p_old)).item()
            
            # 修改点：使用传入的动态 tol 进行判断
            if diff < tol:
                final_it = it
                # print(f"   ✅ PyTorch Converged at {it}, err={diff:.2e}")
                break
    else:
        final_it = max_iter

    torch.cuda.synchronize()
    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), p.cpu().numpy(), final_it, time.time() - start_time