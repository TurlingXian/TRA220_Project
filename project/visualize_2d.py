# visualize.py
import numpy as np
import matplotlib.pyplot as plt


def save_plot_2d(
    x,
    y,
    p,
    filename,
    title=None,
    xlabel="x",
    ylabel="y",
    cbar_label="p",
    show=False,
    dpi=150,
):
    """
    通用画图函数：
    - x, y: 1D 网格坐标
    - p   : 2D 场 (shape: [ny, nx])
    - filename: 保存的文件名（例如 'result.png'）
    - title/xlabel/ylabel/cbar_label: 一些可选的标注
    - show: 是否在屏幕上显示（默认只保存不显示）
    - dpi: 图片分辨率
    """
    X, Y = np.meshgrid(x, y)

    plt.figure()
    # origin='lower' 让 y=0 在下方，更直观；如果不想可以去掉这个参数
    im = plt.pcolormesh(X, Y, p, shading="auto")
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
    print(f"[save_plot] Figure saved to {filename}")
