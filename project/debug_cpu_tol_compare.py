import time
import config
from poisson_cpu_parallel import solve_cpu_parallel

def run_one(nx, ny, tol, label):
    print("=" * 80)
    print(f"[RUN] {label}  nx={nx}, ny={ny}, tol={tol:.2e}")
    config.BENCHMARK_MODE = False          # 确保真的在做收敛检查
    config.MAX_ITER = 200000               # 足够大
    config.CHECK_INTERVAL = 50             # 每 50 步检测一次

    t0 = time.time()
    x, y, p, iters, total_time = solve_cpu_parallel(
        nx=nx, ny=ny, max_iter=config.MAX_ITER, tol=tol
    )
    t1 = time.time()

    avg_step_time = total_time / max(iters, 1)
    print(f"  Steps taken : {iters}")
    print(f"  Total time  : {total_time:.4f} s (wall: {t1 - t0:.4f} s)")
    print(f"  Avg per step: {avg_step_time:.6e} s")
    print(f"  Field stats : min={p.min():.3e}, max={p.max():.3e}, mean={p.mean():.3e}")
    print("=" * 80)
    return iters, total_time, avg_step_time

if __name__ == "__main__":
    # 按你之前 GPU/CPU 的公式，用动态 tol
    BASE_TOL = 1e-7
    BASE_SIZE = 50

    size_1 = 1000
    size_2 = 2000

    tol_1 = BASE_TOL / (size_1 / BASE_SIZE) ** 2
    tol_2 = BASE_TOL / (size_2 / BASE_SIZE) ** 2

    print("Dynamic tolerances:")
    print(f"  size={size_1}: tol={tol_1:.2e}")
    print(f"  size={size_2}: tol={tol_2:.2e}")

    # 在同一个进程里依次跑 1000 和 2000（保证环境一致）
    run_one(size_1, size_1, tol_1, "CPU-1000")
    run_one(size_2, size_2, tol_2, "CPU-2000")
