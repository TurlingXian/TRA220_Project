# benchmark_gpu.py
import config
import time
import numpy as np
from poisson_cupy import solve_cupy
from poisson_pytorch import solve_pytorch

def run_gpu_comparison():
    # 强制固定步数模式，用于测吞吐量
    config.BENCHMARK_MODE = True
    FIXED_ITER = 1000 
    
    # 规模必须与 CPU 实验对齐，才能画全架构对比图
    sizes = [128, 256, 512, 1024, 2048, 4096]

    print(f"==========================================================")
    print(f" GPU Comparison: CuPy vs PyTorch (Fixed {FIXED_ITER} iters)")
    print(f"==========================================================")
    print(f"{'Grid':^10} | {'CuPy (s)':^10} | {'PyTorch (s)':^10} | {'Diff (x)':^10}")
    print("-" * 55)

    for size in sizes:
        # CuPy 运行
        _, _, _, _, time_cupy = solve_cupy(nx=size, ny=size, max_iter=FIXED_ITER)
        # PyTorch 运行
        _, _, _, _, time_pt = solve_pytorch(nx=size, ny=size, max_iter=FIXED_ITER)
        
        diff = time_cupy / time_pt if time_pt > 0 else 0
        print(f"{size}x{size:<5} | {time_cupy:^10.4f} | {time_pt:^10.4f} | {diff:^10.2f}")

if __name__ == "__main__":
    run_gpu_comparison()