import matplotlib.pyplot as plt
import numpy as np

# 实验数据
grid_labels = ['128', '256', '512', '1024', '2048', '4096']
x = np.arange(len(grid_labels))

# 填入实验数据
# CPU 在 4096 处设为 None
cpu_numba_time = [0.045, 0.12, 0.462, 1.831, 7.420, None]
gpu_cupy_time = [0.3653, 0.3630, 0.3634, 0.3659, 0.3777, 1.2830]
gpu_pytorch_time = [0.3730, 0.3477, 0.3489, 0.3486, 0.3542, 1.2667]

plt.figure(figsize=(10, 6))

# 绘制曲线
# CPU 线只绘制有数据的前5个点
plt.plot(grid_labels[:5], cpu_numba_time[:5], 'o--', color='#d62728', label='Numba CPU (16-core)', linewidth=2, markersize=8)
# GPU 线绘制所有点
plt.plot(grid_labels, gpu_cupy_time, 's-', color='#1f77b4', label='CuPy GPU (A100)', linewidth=2, markersize=8)
plt.plot(grid_labels, gpu_pytorch_time, 'd-', color='#2ca02c', label='PyTorch GPU (A100)', linewidth=2, markersize=8)

# --- 已移除标注箭头 ---
# plt.annotate(...) 代码块被删除

# 格式设置
plt.yscale('log') # 保持对数坐标
plt.xlabel('Grid Size ($N \\times N$)', fontsize=14, fontweight='bold')
plt.ylabel('Execution Time (s) [Log Scale]', fontsize=14, fontweight='bold')
plt.title('Performance Scaling and Latency Plateau (1000 Iterations)', fontsize=16)
plt.legend(loc='upper left', fontsize=12)

# 设置网格线，使其更易读
plt.grid(True, which="major", ls="-", color='gray', alpha=0.4)
plt.grid(True, which="minor", ls="--", color='gray', alpha=0.2)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 导出高质量图片
plt.savefig('gpu_performance_scaling_clean.png', dpi=300, bbox_inches='tight')
plt.show()