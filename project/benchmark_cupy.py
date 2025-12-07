import config
import time
# 只导入 CuPy 求解器
from poisson_cupy import solve_cupy

def run_benchmark():
    # --- 1. 强制对齐实验环境 ---
    # 强制关闭 Benchmark 模式，开启收敛检查
    config.BENCHMARK_MODE = False
    
    # 设置一个足够大的最大步数，保证能跑到真正的收敛
    config.MAX_ITER = 200000 
    
    # --- 2. 物理参数对齐 (完全一致) ---
    # 基础容差 (以 50x50 为基准)
    BASE_TOL = 1e-7
    BASE_SIZE = 50

    # 尺寸列表 (完全一致)
    sizes = [50, 100, 200, 400, 800, 1000, 2000]

    print(f"=======================================================================")
    print(f"   CuPy Benchmark (Dynamic Tolerance Mode)")
    print(f"   Base Tol: {BASE_TOL} (at 50x50) | Logic: Tol_new = Tol_base / (Size/50)^2")
    print(f"=======================================================================")
    # 表头格式完全对齐 CPU 版
    print(f"{'Grid':^10} | {'Tol Used':^10} | {'Steps':^10} | {'Time (s)':^10} | {'Status':^12}")
    print("-" * 68)

    # 预热 (Warmup)
    # CuPy 第一次运行需要编译 Kernel，为了不影响 50x50 的计时，我们在循环外先跑一次
    try:
        solve_cupy(nx=50, ny=50, max_iter=5, tol=1.0)
    except Exception:
        pass

    for size in sizes:
        # --- 核心逻辑：动态计算容差 ---
        # 网格每大一倍，容差严 4 倍，抵消 dx^2 带来的数值减小
        scaling_factor = (size / BASE_SIZE) ** 2
        dynamic_tol = BASE_TOL / scaling_factor
        
        try:
            # 传入动态计算的 tol
            _, _, _, iters, duration = solve_cupy(
                nx=size, 
                ny=size, 
                max_iter=config.MAX_ITER,
                tol=dynamic_tol
            )
            
            status = "Converged" if iters < config.MAX_ITER else "Max Reached"
            
            # 打印结果 (格式与 CPU 版一模一样)
            print(f"{size}x{size:<5} | {dynamic_tol:.1e}  | {iters:^10} | {duration:^10.4f} | {status:^12}")

        except Exception as e:
             print(f"{size}x{size:<5} |   ERROR    | {str(e)}")

    print("-" * 68)

if __name__ == "__main__":
    run_benchmark()