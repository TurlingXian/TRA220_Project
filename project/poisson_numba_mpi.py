import numpy as np
from mpi4py import MPI

def solve_2gpu_mpi(nx: int, ny: int, max_iter: int = 1000, check_interval: int = 100) -> float:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 允许 size 为 1 或 2
    if size > 2:
        raise RuntimeError(f"This solver supports 1 or 2 ranks, got {size}")

    if ny % size != 0:
        raise RuntimeError(f"ny={ny} must be divisible by size={size}")

    from numba import cuda, float32
    cuda.select_device(0)
    cuda.current_context()

    dx2 = (1.0 / (nx - 1)) ** 2
    dy2 = (1.0 / (ny - 1)) ** 2
    div_term = 1.0 / (2.0 * (dx2 + dy2))
    local_ny = ny // size

    p = cuda.to_device(np.zeros((local_ny + 2, nx), dtype=np.float32))
    pd = cuda.to_device(np.zeros((local_ny + 2, nx), dtype=np.float32))
    b = cuda.to_device(np.zeros((local_ny + 2, nx), dtype=np.float32))

    @cuda.jit
    def solve_kernel(p, pd, b, div_term, dx2, dy2, local_ny, nx, rank, size):
        s_p = cuda.shared.array(shape=(18, 18), dtype=float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        bx, by = cuda.blockIdx.x, cuda.blockIdx.y
        ix, iy = bx * 16 + tx, by * 16 + ty

        if ix < nx and iy < local_ny + 2:
            s_p[ty + 1, tx + 1] = pd[iy, ix]
            if tx == 0 and ix > 0:           s_p[ty + 1, 0] = pd[iy, ix - 1]
            if tx == 15 and ix < nx - 1:     s_p[ty + 1, 17] = pd[iy, ix + 1]
            if ty == 0 and iy > 0:           s_p[0, tx + 1] = pd[iy - 1, ix]
            if ty == 15 and iy < local_ny+1: s_p[17, tx + 1] = pd[iy + 1, ix]
        cuda.syncthreads()

        if 0 < ix < nx - 1 and 0 < iy <= local_ny:
            # 物理边界保护：单卡时 rank=0, size=1，if 逻辑依然成立
            if not ((rank == 0 and iy == 1) or (rank == size - 1 and iy == local_ny)):
                p[iy, ix] = ((s_p[ty+1, tx+2] + s_p[ty+1, tx]) * dy2 + 
                            (s_p[ty+2, tx+1] + s_p[ty, tx+1]) * dx2 - 
                            b[iy, ix] * dx2 * dy2) * div_term

    threads = (16, 16)
    blocks = (int(np.ceil(nx / 16)), int(np.ceil((local_ny + 2) / 16)))

    host_send = cuda.pinned_array(shape=(nx,), dtype=np.float32)
    host_recv = cuda.pinned_array(shape=(nx,), dtype=np.float32)

    # Warm-up (对齐 4.4 节的纯净测试环境)
    comm.Barrier()
    solve_kernel[blocks, threads](p, pd, b, div_term, dx2, dy2, local_ny, nx, rank, size)
    cuda.synchronize()
    comm.Barrier()

    start_time = MPI.Wtime()

    for it in range(0, max_iter, check_interval):
        for _ in range(check_interval):
            pd, p = p, pd
            solve_kernel[blocks, threads](p, pd, b, div_term, dx2, dy2, local_ny, nx, rank, size)

        cuda.synchronize()

        # 核心修改：如果是单卡运行，跳过 MPI 通信
        if size > 1:
            if rank == 0:
                p[local_ny, :].copy_to_host(host_send)
                comm.Sendrecv(sendbuf=host_send, dest=1, sendtag=11, recvbuf=host_recv, source=1, recvtag=22)
                p[local_ny + 1, :].copy_to_device(host_recv)
            else:
                p[1, :].copy_to_host(host_send)
                comm.Sendrecv(sendbuf=host_send, dest=0, sendtag=22, recvbuf=host_recv, source=0, recvtag=11)
                p[0, :].copy_to_device(host_recv)

    comm.Barrier()
    return MPI.Wtime() - start_time