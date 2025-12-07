import config
import time
import os
import numpy as np
# 请确认你的并行求解器文件名是 poisson_cpu_parallel 还是 poisson_cpu_advanced
# 根据你之前的描述，应该是 poisson_cpu_parallel，如果文件名改了请自行调整
from poisson_cpu_parallel import solve_cpu_parallel 

def run_parallel_benchmark():
    # 强制关闭 Benchmark 模式，使用 Dynamic Tolerance 进行公平的物理收敛测试
    config.BENCHMARK_MODE = False
    
    # 增加最大迭代次数，保证收敛
    config.MAX_ITER = 200000 
    
    # 更严的基准容差 (与单核版保持完全一致)
    BASE_TOL = 1e-7  
    BASE_SIZE = 50

    # 我们只测试大网格，小网格多核没有意义
    # 这里保持和你单核版一样的列表，方便对比
    sizes = [50, 100, 200, 400, 800, 1000, 2000]

    # 获取当前环境变量中的线程数
    num_threads = os.environ.get('NUMBA_NUM_THREADS', 'Default')

    print(f"=======================================================================")
    print(f"   CPU PARALLEL Benchmark (Numba Multicore)")
    print(f"   Threads Active: {num_threads}")
    print(f"   Base Tol: {BASE_TOL} (at 50x50) | Logic: Tol_new = Tol_base / (Size/50)^2")
    print(f"=======================================================================")
    
    # --- 修改点 1: 表头增加了 'Tol Used' 列，格式与单核版完全对齐 ---
    print(f"{'Grid':^10} | {'Tol Used':^10} | {'Steps':^10} | {'Time (s)':^10} | {'Status':^12}")
    print("-" * 68)

    for size in sizes:
        # 动态容差计算
        scaling_factor = (size / BASE_SIZE) ** 2
        dynamic_tol = BASE_TOL / scaling_factor
        
        try:
            # 调用并行求解器
            # --- 关键修改 2: 务必传入 tol=dynamic_tol ---
            # 如果不传这个，它就会用默认的 1e-5，导致大网格假收敛
            _, _, _, iters, duration = solve_cpu_parallel(
                nx=size, 
                ny=size, 
                max_iter=config.MAX_ITER,
                tol=dynamic_tol 
            )
            
            status = "Converged" if iters < config.MAX_ITER else "Max Reached"
            
            # --- 修改点 3: 打印行增加了 dynamic_tol 的显示 ---
            print(f"{size}x{size:<5} | {dynamic_tol:.1e}  | {iters:^10} | {duration:^10.4f} | {status:^12}")

        except TypeError as e:
            print(f"{size}x{size:<5} |   ERROR    | 函数参数错误: {e}")
            print("提示: 请检查 poisson_cpu_parallel.py 中的 solve_cpu_parallel 是否定义了 'tol' 参数")
            break
        except Exception as e:
             print(f"{size}x{size:<5} |   ERROR    | {str(e)}")

    print("-" * 68)

if __name__ == "__main__":
    run_parallel_benchmark()