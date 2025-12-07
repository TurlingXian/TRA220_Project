import cupy as cp
import numpy as np
import time
import config

def solve_cupy(nx=config.NX, ny=config.NY, max_iter=config.MAX_ITER, tol=config.TOLERANCE):
    """
    GPU Implementation using CuPy (Drop-in replacement for NumPy)
    """
    # 1. 准备参数 (这些是在 CPU 上的标量)
    xmin, xmax = config.X_MIN, config.X_MAX
    ymin, ymax = config.Y_MIN, config.Y_MAX
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # 2. 初始化 (直接在 GPU 显存中创建数组)
    # cp.zeros 对应 np.zeros，但数据直接分配在 GPU 上
    p = cp.zeros((ny, nx), dtype=cp.float32)
    pd = cp.zeros((ny, nx), dtype=cp.float32)
    b = cp.zeros((ny, nx), dtype=cp.float32)

    # 设置源项
    b[int(ny / 4), int(nx / 4)] = 100
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100

    # 预计算常数
    dx2 = dx**2
    dy2 = dy**2
    div_term = 1.0 / (2 * (dx2 + dy2))
    
    # 3. 预热 (Warmup)
    # CuPy 在第一次运行时会编译 Kernel，为了计时准确，建议先跑一步
    p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
                      (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
                      b[1:-1, 1:-1] * dx2 * dy2) * div_term)
    cp.cuda.Device().synchronize() # 确保预热完成

    print(f"CuPy: Starting simulation on GPU...")
    start_time = time.time()
    final_it = 0

    # 4. 主循环
    for it in range(max_iter):
        pd = p.copy() # 在 GPU 上进行显存拷贝 (非常快)
        
        # 核心计算：语法和 NumPy 一模一样！
        p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
                          (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
                          b[1:-1, 1:-1] * dx2 * dy2) * div_term)

        # 边界条件
        p[0, :] = 0; p[-1, :] = 0
        p[:, 0] = 0; p[:, -1] = 0
        
        # 收敛检查
        if not config.BENCHMARK_MODE and it % config.CHECK_INTERVAL == 0:
            # cp.max 和 cp.abs 都是 GPU 操作
            # 当我们做 < tol 判断时，CuPy 会自动同步并将布尔值传回 CPU
            diff = cp.max(cp.abs(p - pd))
            if diff < tol:
                final_it = it
                # print(f"   ✅ CuPy Converged at {it}, err={diff:.2e}")
                break
    else:
        final_it = max_iter

    # 5. 计时结束前的同步
    # GPU 是异步运行的，必须等待所有指令完成才能停止计时
    cp.cuda.Device().synchronize()
    total_time = time.time() - start_time

    # 6. 将结果从 GPU 取回 CPU (用于画图或验证)
    p_cpu = cp.asnumpy(p)
    
    return np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), p_cpu, final_it, total_time