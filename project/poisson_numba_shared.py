from numba import cuda, float32
import numba      # 如果别处用到 numba.float32 之类可以保留
import numpy as np

TILE_SIZE = 16
SHARED_Y = TILE_SIZE + 2
SHARED_X = TILE_SIZE + 2

@cuda.jit
def poisson_shared_kernel(p_out, p_in, b, dx2, dy2, div_term, nx, ny):
    # ✅ 使用模块级常量 + 关键字参数
    s_p = cuda.shared.array(shape=(SHARED_Y, SHARED_X), dtype=float32)
    
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    x, y = cuda.grid(2)
    
    sx, sy = tx + 1, ty + 1
    
    if x < nx and y < ny:
        s_p[sy, sx] = p_in[y, x]
    else:
        s_p[sy, sx] = 0.0

    if tx == 0 and x > 0:
        s_p[sy, 0] = p_in[y, x - 1]
    if tx == TILE_SIZE - 1 and x < nx - 1:
        s_p[sy, sx + 1] = p_in[y, x + 1]
    if ty == 0 and y > 0:
        s_p[0, sx] = p_in[y - 1, x]
    if ty == TILE_SIZE - 1 and y < ny - 1:
        s_p[sy + 1, sx] = p_in[y + 1, x]
        
    cuda.syncthreads()
    
    if x > 0 and x < nx - 1 and y > 0 and y < ny - 1:
        val = (((s_p[sy, sx+1] + s_p[sy, sx-1]) * dy2 +
                (s_p[sy+1, sx] + s_p[sy-1, sx]) * dx2 -
                b[y, x] * dx2 * dy2) * div_term)
        p_out[y, x] = val
