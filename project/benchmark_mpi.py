import os
from mpi4py import MPI

# 1. 物理隔离
localid = os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(localid)
os.environ["NUMBA_CUDA_USE_NVIDIA_BINDING"] = "1"
os.environ["NUMBA_CUDA_REDUCE_MEMORY_USAGE"] = "0"

from poisson_numba_mpi import solve_2gpu_mpi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_weak_scaling():
    # --- 弱扩展性设置 ---
    # 每个 GPU 负责 2048x2048 的局部网格
    nx = 2048
    base_ny = 2048
    
    # 总规模随节点数线性增长
    total_ny = base_ny * size 
    
    iters = 5000
    check_interval = 200

    if rank == 0:
        print(f"=== Weak Scaling Test ({size}-GPU) ===")
        print(f"Local Load per GPU : {nx} x {base_ny}")
        print(f"Total Grid Size    : {nx} x {total_ny}")
        print(f"Total Updates      : {(nx * total_ny * iters) / 1e9:.2f} G-Points")
        print("----------------------------------------------", flush=True)

    # 执行计算
    t = solve_2gpu_mpi(nx, total_ny, max_iter=iters, check_interval=check_interval)

    if rank == 0:
        # 计算总吞吐量
        gups = (nx * total_ny * iters) / (t * 1e9)
        print(f"Execution Time : {t:.4f} s")
        print(f"Throughput     : {gups:.2f} GUPS")
        
        # 理想情况：GUPS 应该随 size 线性增长
        if size > 1:
            # 假设之前单卡纯净版是 62 GUPS
            efficiency = (gups / (62.05 * size)) * 100
            print(f"Weak Scaling Efficiency: {efficiency:.1f}%")
        print("==============================================", flush=True)

if __name__ == "__main__":
    run_weak_scaling()