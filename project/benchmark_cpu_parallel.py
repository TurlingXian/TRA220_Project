import os
import time
import config
import numba
import numpy as np
from poisson_cpu_parallel import solve_cpu_auto, configure_numba_threads_from_env

def run_scaling_benchmark():
    config.BENCHMARK_MODE = True # 统一固定步数
    FIXED_ITER = 1000
    
    threads = configure_numba_threads_from_env(default_threads=1)
    
    # 预热编译 (Warm-up)
    solve_cpu_auto(nx=128, ny=128, max_iter=10)

    sizes = [128, 256, 512, 1024, 2048]

    print("==========================================================")
    print(f" CPU Parallel Benchmark (Threads: {threads}, float32)")
    print("==========================================================")
    print(f"{'Grid':^10} | {'Time (s)':^10} | {'Speedup':^10} | {'Max Val':^10}")
    print("-" * 55)

    for size in sizes:
        _, _, p, iters, duration = solve_cpu_auto(
            nx=size, ny=size, max_iter=FIXED_ITER
        )
        print(f"{size}x{size:<5} | {duration:^10.4f} | {'N/A':^10} | {np.max(p):^10.4f}")

if __name__ == "__main__":
    run_scaling_benchmark()