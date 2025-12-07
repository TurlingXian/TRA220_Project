import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import config

def save_plot(x, y, p, filename="result.png"):
    """
    绘制 3D 表面图。包含智能检查，防止大网格卡死。
    """
    if not config.ENABLE_PLOTTING:
        # print(f"   [Plotting skipped by config]")
        return

    # 安全检查：如果网格太大，强制不画图
    if len(x) * len(y) > 500 * 500:
        print(f"   [Plotting skipped: Grid too large for Matplotlib]")
        return

    print(f"   Saving plot to {filename}...")
    try:
        fig = plt.figure(figsize=(11, 7), dpi=100)
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                               linewidth=0, antialiased=False)
        ax.view_init(30, 225)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.savefig(filename)
        plt.close()
    except Exception as e:
        print(f"   [Plotting failed: {e}]")