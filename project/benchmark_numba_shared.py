import numpy as np
import time
import math
from numba import cuda
import sys

# 导入刚才保存的 shared memory 内核
from poisson_numba_shared import poisson_shared_kernel, TILE_SIZE

def run_benchmark():
    # 测试的网格大小列表
    grid_sizes = [50, 100, 200, 400, 800, 1000, 2000]
    
    print("=======================================================================")
    print(f"   Numba (Shared Memory) Benchmark")
    print(f"   Tile Size: {TILE_SIZE}x{TILE_SIZE} | Shared Mem: {(TILE_SIZE+2)**2*4/1024:.2f} KB/block")
    print("=======================================================================")
    print(f"{'Grid':<10} | {'Tol Used':<10} | {'Steps':<10} | {'Time (s)':<10} | {'Status':<12}")
    print("-" * 68)

    for n in grid_sizes:
        nx, ny = n, n
        
        # 1. 动态容差设置
        base_tol = 1e-7
        tolerance = base_tol / ((n / 50.0) ** 2)

        # 2. 准备数据
        # 注意：使用 float32 配合 GPU
        p_host = np.zeros((ny, nx), dtype=np.float32)
        b_host = np.zeros((ny, nx), dtype=np.float32)
        
        # 设置源项
        b_host[int(ny / 4), int(nx / 4)] = 100
        b_host[int(3 * ny / 4), int(3 * nx / 4)] = -100

        # 3. 拷贝到 GPU
        d_p_in = cuda.to_device(p_host)
        d_p_out = cuda.to_device(p_host)
        d_b = cuda.to_device(b_host)

        # 4. 计算参数
        xmin, xmax = 0.0, 2.0
        ymin, ymax = 0.0, 1.0
        dx = (xmax - xmin) / (nx - 1)
        dy = (ymax - ymin) / (ny - 1)
        dx2, dy2 = dx**2, dy**2
        div_term = 1.0 / (2 * (dx2 + dy2))

        # 5. 设置 Grid/Block 维度
        # 这是 Shared Memory 版本的关键：Block 大小必须匹配 TILE_SIZE
        threads_per_block = (TILE_SIZE, TILE_SIZE)
        blocks_per_grid = (math.ceil(nx / TILE_SIZE), math.ceil(ny / TILE_SIZE))

        # 6. 预热 (Warmup) - 强制 JIT 编译
        poisson_shared_kernel[blocks_per_grid, threads_per_block](
            d_p_out, d_p_in, d_b, dx2, dy2, div_term, nx, ny
        )
        cuda.synchronize()

        # 7. 开始计时循环
        start_time = time.time()
        max_iter = 200000
        converged = False
        steps = 0
        
        # 为了减少 Host-Device 通信开销，每 1000 步检查一次
        check_interval = 1000

        for i in range(0, max_iter, check_interval):
            # 运行 check_interval 次
            for _ in range(check_interval):
                poisson_shared_kernel[blocks_per_grid, threads_per_block](
                    d_p_out, d_p_in, d_b, dx2, dy2, div_term, nx, ny
                )
                # 交换指针
                d_p_in, d_p_out = d_p_out, d_p_in
            
            cuda.synchronize()
            steps += check_interval

            # 检查收敛
            p_curr = d_p_in.copy_to_host()
            p_prev = d_p_out.copy_to_host() # 注意因为刚才交换了，out 其实是旧的
            diff = np.max(np.abs(p_curr - p_prev))
            
            if diff < tolerance:
                converged = True
                break

        total_time = time.time() - start_time
        status = "Converged" if converged else "Max Iter"
        
        print(f"{f'{n}x{n}':<10} | {tolerance:.1e}  | {steps:<10} | {total_time:<10.4f} | {status:<12}")

if __name__ == "__main__":
    if not cuda.is_available():
        print("Error: CUDA not available.")
    else:
        run_benchmark()