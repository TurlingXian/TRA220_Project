import os
import numpy as np

# 1) 先拿 local id（单节点双 task -> 0/1）
localid = os.environ.get("SLURM_LOCALID")
if localid is None:
    localid = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")

# 2) 在 import numba 前隔离 GPU（因为你用的是 --gpus-per-node）
os.environ["CUDA_VISIBLE_DEVICES"] = str(localid)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(
    f"[rank {rank}] SLURM_LOCALID={localid} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"NUMBA_CUDA_USE_NVIDIA_BINDING={os.environ.get('NUMBA_CUDA_USE_NVIDIA_BINDING')}",
    flush=True
)

try:
    from numba import cuda

    # 隔离后只可见 1 张卡，因此选 0
    cuda.select_device(0)

    # 强制创建 context
    ctx = cuda.current_context()
    dev = cuda.get_current_device()
    print(f"[rank {rank}] context OK. device name={dev.name}, visible_id={dev.id}", flush=True)

    # 分配+拷贝
    h = np.array([123.45 + rank], dtype=np.float32)
    d = cuda.to_device(h)
    out = d.copy_to_host()[0]

    print(f"[rank {rank}] SUCCESS: value={out}", flush=True)

except Exception as e:
    print(f"[rank {rank}] FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()

comm.Barrier()
if rank == 0:
    print("[rank 0] All ranks reached MPI barrier.", flush=True)
