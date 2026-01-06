import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 ---
# 规模标签
labels = ['1024^2', '2048^2', '4096^2']

# PyTorch (Section 4.2) 的 GUPS 数据
pytorch_gups = [11.13, 11.82, 13.24] 

# Numba-Shared (Section 4.3) 的 GUPS 数据
numba_shared_gups = [2.42, 15.89, 15.48] 

x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱状条宽度

# --- 2. 绘图设置 ---
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, pytorch_gups, width, label='PyTorch (High-level)', color='green', alpha=0.7)
rects2 = ax.bar(x + width/2, numba_shared_gups, width, label='Numba-Shared (Optimized)', color='red')

# --- 3. 装饰与标注 ---
ax.set_ylabel('Giga-Updates Per Second (GUPS)', fontsize=12)
ax.set_title('Throughput Performance: Breaking the Framework Limit', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 自动标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点纵向偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# --- 4. 保存图片 ---
plt.tight_layout()
plt.savefig('gups_comparison_final.png', dpi=300)
plt.show()

print("GUPS 柱状图已生成：gups_comparison_final.png")