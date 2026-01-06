import time
from poisson_numba_final import solve_numba_shared
import config

def run_extreme_benchmark():
    # 锁定 A100 实验参数
    sizes = [1024, 2048, 4096]
    print(f"{'Size':<10} | {'Method':<15} | {'Time (s)':<10} | {'GUPS':<10}")
    print("-" * 50)

    for n in sizes:
        start = time.time()
        # 调用优化后的共享内存版
        _, _, _, iters, duration = solve_numba_shared(n, n, 1000, 1e-9)
        
        gups = (n**2 * 1000) / (duration * 1e9)
        print(f"{n}x{n:<5} | Numba-Shared   | {duration:<10.4f} | {gups:<10.2f}")

if __name__ == "__main__":
    run_extreme_benchmark()