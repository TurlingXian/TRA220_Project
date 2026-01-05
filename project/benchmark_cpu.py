import config
import time
from poisson_cpu import solve_cpu

def run_benchmark():
    # 强制关闭 Benchmark 模式，开启收敛检查
    config.BENCHMARK_MODE = False
    
    # 设置一个足够大的最大步数，保证能跑到真正的收敛
    config.MAX_ITER = 200000 
    
    # 基础容差 (以 50x50 为基准)
    BASE_TOL = 1e-7
    BASE_SIZE = 50

    # sizes = [50, 100, 200, 400, 800, 1000, 2000]
    sizes = [50, 100, 200, 400, 800, 1000]

    print(f"=======================================================================")
    print(f"   CPU Benchmark (Dynamic Tolerance Mode)")
    print(f"   Base Tol: {BASE_TOL} (at 50x50) | Logic: Tol_new = Tol_base / (Size/50)^2")
    print(f"=======================================================================")
    print(f"{'Grid':^10} | {'Tol Used':^10} | {'Steps':^10} | {'Time (s)':^10} | {'Status':^12}")
    print("-" * 68)

    for size in sizes:
        # --- 核心修改：动态计算容差 ---
        # 网格每大一倍，容差严 4 倍，抵消 dx^2 带来的数值减小
        scaling_factor = (size / BASE_SIZE) ** 2
        dynamic_tol = BASE_TOL / scaling_factor
        
        try:
            # 传入动态计算的 tol
            _, _, _, iters, duration = solve_cpu(
                nx=size, 
                ny=size, 
                max_iter=config.MAX_ITER,
                tol=dynamic_tol
            )
            
            status = "Converged" if iters < config.MAX_ITER else "Max Reached"
            
            # 打印结果
            print(f"{size}x{size:<5} | {dynamic_tol:.1e}  | {iters:^10} | {duration:^10.4f} | {status:^12}")

        except KeyboardInterrupt:
            print(f"\nStopped by user.")
            break
        except Exception as e:
             print(f"{size}x{size:<5} |   ERROR    | {str(e)}")

    print("-" * 68)

if __name__ == "__main__":
    run_benchmark()