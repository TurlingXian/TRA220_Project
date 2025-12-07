import config
from poisson_cpu import solve_cpu
from poisson_cpu_parallel import solve_cpu_parallel
from poisson_pytorch import solve_pytorch
from poisson_numba import solve_numba
from poisson_numba_shared import solve_numba_shared
from visualize import save_plot
import numpy as np

def print_result(name, t, it, fps, base_t=None):
    speedup = ""
    if base_t:
        speedup = f"| Speedup: {base_t/t:.2f}x"
    print(f"{name:<20} | Time: {t:.4f}s | Steps: {it:<5} | FPS: {fps:<8.1f} {speedup}")

def main():
    print(f"==========================================================")
    print(f"   TRA220: GPU Accelerated Poisson Solver")
    print(f"   Grid: {config.NX}x{config.NY} | Max Steps: {config.MAX_ITER}")
    print(f"   Mode: {'BENCHMARK (Fixed Steps)' if config.BENCHMARK_MODE else 'PHYSICS (Convergence Check)'}")
    print(f"==========================================================\n")

    # 1. CPU Baseline
    # 如果网格太大，CPU 太慢，可以选择性跳过
    if config.NX > 1000:
        print("⚠️ Grid > 1000, skipping Single Core CPU to save time...")
        t_cpu = None
        p_ref = None
    else:
        x, y, p_ref, it_cpu, t_cpu = solve_cpu()
        print_result("CPU Single Core", t_cpu, it_cpu, it_cpu/t_cpu)
        save_plot(x, y, p_ref, "result_cpu.png")

    # 2. CPU Parallel
    try:
        x, y, p_cpu_p, it_cpu_p, t_cpu_p = solve_cpu_parallel()
        base_t = t_cpu if t_cpu else t_cpu_p # 如果跳过单核，就用多核当基准
        print_result("CPU Parallel", t_cpu_p, it_cpu_p, it_cpu_p/t_cpu_p, t_cpu)
    except Exception as e:
        print(f"CPU Parallel Failed: {e}")

    print("-" * 60)

    # 3. PyTorch
    try:
        x, y, p_torch, it_torch, t_torch = solve_pytorch()
        print_result("PyTorch (GPU)", t_torch, it_torch, it_torch/t_torch, base_t)
        save_plot(x, y, p_torch, "result_pytorch.png")
    except Exception as e:
        print(f"PyTorch Failed: {e}")

    # 4. Numba Basic
    try:
        x, y, p_numba, it_numba, t_numba = solve_numba()
        if p_numba is None:
            print("Numba Basic skipped (No GPU).")
        else:
            print_result("Numba Basic", t_numba, it_numba, it_numba/t_numba, base_t)
            save_plot(x, y, p_numba, "result_numba.png")
    except Exception as e:
        print(f"Numba Basic Failed: {e}")

    # 5. Numba Shared (Optimized)
    try:
        x, y, p_shared, it_shared, t_shared = solve_numba_shared()
        if p_shared is None:
            print("Numba Shared skipped (No GPU).")
        else:
            print_result("Numba Shared", t_shared, it_shared, it_shared/t_shared, base_t)
            
            # 显示优化提升
            if 't_numba' in locals() and t_numba:
                print(f"   >>> Optimization Gain (Shared vs Basic): {t_numba/t_shared:.2f}x")
            
            save_plot(x, y, p_shared, "result_numba_shared.png")
    except Exception as e:
        print(f"Numba Shared Failed: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()