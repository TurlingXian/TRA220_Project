import matplotlib.pyplot as plt
import numpy as np

# 1. 实验原始数据 (基于你提供的 Log)
grid_labels = ['128', '256', '512', '1024', '2048']
# 由于 CPU 在 4096 处 OOM，加速比计算仅到 2048 为止
cpu_numba_time = np.array([0.045, 0.12, 0.462, 1.831, 7.420])
gpu_cupy_time = np.array([0.3653, 0.3630, 0.3634, 0.3659, 0.3777])
gpu_pytorch_time = np.array([0.3730, 0.3477, 0.3489, 0.3486, 0.3542])

# 2. 计算加速比 (T_cpu / T_gpu)
cup_speedup = cpu_numba_time / gpu_cupy_time
pytorch_speedup = cpu_numba_time / gpu_pytorch_time

plt.figure(figsize=(10, 6))

# 3. 绘制加速比曲线
plt.plot(grid_labels, cup_speedup, 's-', color='#1f77b4', label='CuPy Speedup vs Numba CPU', linewidth=2.5, markersize=8)
plt.plot(grid_labels, pytorch_speedup, 'd-', color='#2ca02c', label='PyTorch Speedup vs Numba CPU', linewidth=2.5, markersize=8)

# 4. 绘制 y=1 的基准线 (代表 CPU 与 GPU 性能持平)
plt.axhline(y=1, color='red', linestyle='--', alpha=0.6, label='CPU Baseline (Speedup = 1x)')

# 5. 图表修饰
plt.xlabel('Grid Size ($N \\times N$)', fontsize=14, fontweight='bold')
plt.ylabel('Speedup Factor ($T_{CPU} / T_{GPU}$)', fontsize=14, fontweight='bold')
plt.title('GPU Speedup relative to 16-core Numba CPU', fontsize=16, pad=20)
plt.legend(loc='upper left', fontsize=12)

# 设置网格，提升可读性
plt.grid(True, which="major", ls="-", alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 在点上标注具体倍数
for i, txt in enumerate(pytorch_speedup):
    plt.annotate(f'{txt:.1f}x', (grid_labels[i], pytorch_speedup[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

# 6. 导出图片
plt.savefig('gpu_speedup_analysis.png', dpi=300, bbox_inches='tight')
plt.show()