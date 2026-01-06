# profile_target.py
import config
from poisson_numba_final import solve_numba_shared

def run_profile():
    # 模拟 4.2 节中观察到的基准规模
    n = 2048 
    max_iter = 20000 # 跑足够长的时间以捕获稳定的周期
    tolerance = 1e-12 # 故意设低，确保它跑满循环
    
    print(f"Starting Profile Target: {n}x{n} grid...")
    # 调用你最新的共享内存+内核融合版本
    solve_numba_shared(n, n, max_iter, tolerance)
    print("Profile Target Finished.")

if __name__ == "__main__":
    run_profile()