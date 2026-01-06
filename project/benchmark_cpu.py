import config
import time
import numpy as np
from poisson_cpu import solve_cpu

def run_benchmark():
    # --- 配置：性能模式 (用于加速比图表) ---
    config.BENCHMARK_MODE = True
    FIXED_ITER = 1000 
    
    # --- 配置：收敛模式 (用于验证，规模设小一点) ---
    # config.BENCHMARK_MODE = False
    # config.MAX_ITER = 100000 
    
    BASE_TOL = 1e-7
    BASE_SIZE = 50
    sizes = [128, 256, 512, 1024, 2048]

    print(f"==========================================================")
    print(f" CPU Benchmark (Fixed {FIXED_ITER} iterations, float32)")
    print(f"==========================================================")
    print(f"{'Grid':^10} | {'Time (s)':^10} | {'Time/Iter (ms)':^15} | {'Max Val':^10}")
    print("-" * 55)

    for size in sizes:
        # 动态容差仅在 BENCHMARK_MODE=False 时生效
        scaling_factor = (size / BASE_SIZE) ** 2
        dynamic_tol = BASE_TOL / scaling_factor
        
        _, _, p, iters, duration = solve_cpu(
            nx=size, ny=size, max_iter=FIXED_ITER, tol=dynamic_tol
        )
        
        avg_ms = (duration / iters) * 1000
        print(f"{size}x{size:<5} | {duration:^10.4f} | {avg_ms:^15.4f} | {np.max(p):^10.4f}")

if __name__ == "__main__":
    run_benchmark()