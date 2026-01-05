import os
import time
import config
import numba
from poisson_cpu_parallel import solve_cpu_auto, configure_numba_threads_from_env


def run_scaling_benchmark():
    # 1) 固定为收敛模式（与你现在一致）
    config.BENCHMARK_MODE = False
    config.MAX_ITER = 200000

    # 2) 设置线程数（从环境变量 NUMBA_NUM_THREADS 读）
    threads = configure_numba_threads_from_env(default_threads=1)

    # 3) 你的 dynamic tol 逻辑
    BASE_TOL = 1e-7
    BASE_SIZE = 50
    # sizes = [50, 100, 200, 400, 800, 1000, 2000]
    sizes = [50, 100, 200, 400, 800, 1000]

    # 4) 预热编译（避免把 numba 首次编译时间算进 benchmark）
    #    Numba 的编译按“类型签名”走，数组 shape 不影响签名，所以用小网格即可。
    _old_bm = config.BENCHMARK_MODE
    _old_ci = config.CHECK_INTERVAL
    config.BENCHMARK_MODE = True
    config.CHECK_INTERVAL = 1
    solve_cpu_auto(nx=50, ny=50, max_iter=1, tol=1.0)
    config.BENCHMARK_MODE = _old_bm
    config.CHECK_INTERVAL = _old_ci

    print("=======================================================================")
    print(" CPU SCALING BENCHMARK (Auto: serial@1 thread, parallel@>1 threads)")
    print(f" Threads Active (numba.get_num_threads): {numba.get_num_threads()}")
    print(f" NUMBA_NUM_THREADS env: {os.environ.get('NUMBA_NUM_THREADS', 'unset')}")
    print(f" Base Tol: {BASE_TOL} (at {BASE_SIZE}x{BASE_SIZE})")
    print(" Dynamic Tol: Tol = BASE_TOL / (size/BASE_SIZE)^2")
    print("=======================================================================")
    print(f"{'Grid':<10} | {'Tol Used':<10} | {'Steps':<10} | {'Time (s)':<10} | {'Status':<12}")
    print("-" * 68)

    for size in sizes:
        scaling_factor = (size / BASE_SIZE) ** 2
        dynamic_tol = BASE_TOL / scaling_factor

        try:
            _, _, _, iters, duration = solve_cpu_auto(
                nx=size,
                ny=size,
                max_iter=config.MAX_ITER,
                tol=dynamic_tol
            )

            status = "Converged" if iters < config.MAX_ITER else "Max Reached"
            print(f"{size}x{size:<5} | {dynamic_tol:.1e} | {iters:<10} | {duration:<10.4f} | {status:<12}")

        except Exception as e:
            print(f"{size}x{size:<5} | ERROR      | {str(e)}")

    print("-" * 68)
    print("Done.")


if __name__ == "__main__":
    run_scaling_benchmark()
