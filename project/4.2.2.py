import matplotlib.pyplot as plt
import numpy as np

# 实验数据
grid_labels = ['128', '256', '512', '1024', '2048', '4096']
x = np.arange(len(grid_labels))

# 填入你之前的实验数据
# CPU 在 4096 处设为 None
cpu_numba_time = [0.045, 0.12, 0.462, 1.831, 7.420, None] 
gpu_cupy_time = [0.3653, 0.3630, 0.3634, 0.3659, 0.3777, 1.2830]
gpu_pytorch_time = [0.3730, 0.3477, 0.3489, 0.3486, 0.3542, 1.2667]

# 计算 GUPS = (N*N * iterations) / (time * 1e9)
def calc_gups(n, time):
    if time is None: return 0
    return (int(n)**2 * 1000) / (time * 1e9)

grids = ['512', '1024', '2048', '4096']
cpu_gups = [calc_gups(g, cpu_numba_time[i+2]) for i, g in enumerate(grids)]
gpu_gups = [calc_gups(g, gpu_pytorch_time[i+2]) for i, g in enumerate(grids)]

x = np.arange(len(grids))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, cpu_gups, width, label='Numba CPU', color='#ff9896')
rects2 = ax.bar(x + width/2, gpu_gups, width, label='PyTorch GPU', color='#1f77b4')

# 标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Throughput (GUPS)', fontsize=12)
ax.set_title('Throughput Growth: Breaking the Memory Wall', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(grids)
ax.legend()

# 标注 CPU 缺失
# ax.text(3-width/2, 0.5, 'N/A (OOM)', ha='center', color='red', fontweight='bold')

plt.savefig('gups_comparison.png', dpi=300, bbox_inches='tight')
plt.show()