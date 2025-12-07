import time
import numpy as np
import matplotlib.pyplot as plt
from visualize_2d import save_plot_2d


def solve_poisson_fixed(nx, ny, n_steps,
                        xmin=0.0, xmax=2.0,
                        ymin=0.0, ymax=1.0):
    """
    最简单的 2D Poisson Jacobi 迭代版本：
    - 固定步数 n_steps
    - Dirichlet 边界 p=0
    - 源项 b 在两个点为 ±100
    """
    # 网格步长
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # 场变量
    p = np.zeros((ny, nx))
    pd = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    # 坐标（只用于画图）
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    # 源项（和你之前保持一致）
    b[int(ny / 4), int(nx / 4)] = 100
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100

    # 预计算常数
    dx2 = dx ** 2
    dy2 = dy ** 2
    div_term = 1.0 / (2.0 * (dx2 + dy2))

    print(f"\n=== Run {n_steps} iterations on {nx}x{ny} grid ===")
    start_time = time.time()
    last_diff = None

    for it in range(n_steps):
        pd = p.copy()

        # Jacobi 更新（五点差分）
        p[1:-1, 1:-1] = (
            ((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2 +
             (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2 -
             b[1:-1, 1:-1] * dx2 * dy2) * div_term
        )

        # 边界条件 p = 0
        p[0, :] = 0.0
        p[-1, :] = 0.0
        p[:, 0] = 0.0
        p[:, -1] = 0.0

        # 记录最后一步的变化量（也可以改成每步都记录做曲线）
        last_diff = np.abs(p - pd).max()

    total_time = time.time() - start_time
    print(f"Finished {n_steps} steps in {total_time:.4f} s")
    print(f"   Max |p_new - p_old| at last step: {last_diff:.2e}")

    return x, y, p, total_time, last_diff

def main():
    # ======= 在这里改参数做实验 =======
    NX = 50      # 网格点数（x 方向）
    NY = 50      # 网格点数（y 方向）
    steps_list = [10, 50, 100]  # 固定迭代步数
    XMIN, XMAX = 0.0, 2.0
    YMIN, YMAX = 0.0, 1.0
    # =================================

    print("==========================================================")
    print("   Basic Poisson Solver (NO config.py)")
    print(f"   Grid: {NX}x{NY}")
    print("==========================================================")

    for n_steps in steps_list:
        x, y, p, t, diff = solve_poisson_fixed(
            NX, NY, n_steps,
            xmin=XMIN, xmax=XMAX,
            ymin=YMIN, ymax=YMAX
        )

        filename = f"poisson_{NX}x{NY}_{n_steps}steps.png"
        title = f"Poisson solution ({NX}x{NY}), {n_steps} steps"
        save_plot_2d(x, y, p, filename, title=title)

        print("-" * 50)


if __name__ == "__main__":
    main()
