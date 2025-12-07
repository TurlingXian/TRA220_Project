import config
import time
# 导入更新后的 PyTorch 求解器
from poisson_pytorch import solve_pytorch

def run_torch_benchmark():
    # --- 1. 强制对齐实验环境 ---
    config.BENCHMARK_MODE = False
    config.MAX_ITER = 200000 
    
    # --- 2. 物理参数对齐 (必须和 CPU/CuPy 完全一致) ---
    BASE_TOL = 1e-7
    BASE_SIZE = 50

    sizes = [50, 100, 200, 400, 800, 1000, 2000]

    print(f"=======================================================================")
    print(f"   PyTorch Benchmark (Dynamic Tolerance Mode)")
    print(f"   Base Tol: {BASE_TOL} (at 50x50) | Logic: Tol_new = Tol_base / (Size/50)^2")
    print(f"=======================================================================")
    print(f"{'Grid':^10} | {'Tol Used':^10} | {'Steps':^10} | {'Time (s)':^10} | {'Status':^12}")
    print("-" * 68)

    for size in sizes:
        # 动态容差计算
        scaling_factor = (size / BASE_SIZE) ** 2
        dynamic_tol = BASE_TOL / scaling_factor
        
        try:
            # 预热 (Warmup): PyTorch 也有启动开销
            if size == sizes[0]:
                solve_pytorch(nx=50, ny=50, max_iter=10, tol=1.0)

            # 正式运行
            _, _, _, iters, duration = solve_pytorch(
                nx=size, 
                ny=size, 
                max_iter=config.MAX_ITER,
                tol=dynamic_tol # <--- 传入动态容差
            )
            
            status = "Converged" if iters < config.MAX_ITER else "Max Reached"
            
            print(f"{size}x{size:<5} | {dynamic_tol:.1e}  | {iters:^10} | {duration:^10.4f} | {status:^12}")

        except Exception as e:
             print(f"{size}x{size:<5} |   ERROR    | {str(e)}")

    print("-" * 68)

if __name__ == "__main__":
    run_torch_benchmark()