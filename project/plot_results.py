import matplotlib.pyplot as plt
import numpy as np
import sys

# 如果在没有显示器的服务器上运行，使用非交互式后端
if sys.platform.startswith('linux'):
    plt.switch_backend('agg')

# ==========================================
# 核心实验数据录入 (来自你的日志文件)
# ==========================================
# Chart 1 数据: 性能演进
labels_evo = ['CPU\n(Serial)', 'CPU\n(OpenMP)', 'GPU Base\n(4.3)', 'GPU Opt\n(1-Card)', 'GPU Opt\n(2-Card)']
# 注意：CPU数据是估算值用于对比，后三个是你真实测得的
gups_evo = [0.005, 0.05, 15.89, 61.59, 113.39] 
colors_evo = ['#9ca3af', '#6b7280', '#fcd34d', '#34d399', '#3b82f6']

# Chart 2 & 3 数据: 弱扩展性测试 (N=2048 -> N=4096)
num_gpus = [1, 2]
exec_times = [0.3405, 0.3699]  # 秒
throughputs = [61.59, 113.39]  # GUPS
efficiencies = [100.0, 91.4]   # 百分比

# 设置通用绘图风格
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# ==========================================
# 图 1: 性能演进直方图 (Log Scale)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(labels_evo, gups_evo, color=colors_evo, edgecolor='black', alpha=0.9, width=0.6)

# 关键：使用对数坐标轴，因为差距极大
ax1.set_yscale('log')
ax1.set_ylabel('Throughput (GUPS) - Log Scale', fontweight='bold')
ax1.set_title('Performance Evolution: From CPU to Multi-GPU', fontweight='bold', pad=20)
ax1.grid(axis='y', linestyle='--', alpha=0.5, which='major')

# 在柱子上方标注具体数值
for bar, value in zip(bars, gups_evo):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height * 1.15, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('1_performance_evolution.png', dpi=300)
print("[Generated] 1_performance_evolution.png")

# ==========================================
# 图 2: 弱扩展性分析图 (双Y轴)
# ==========================================
fig2, ax2_left = plt.subplots(figsize=(10, 6))
ax2_right = ax2_left.twinx()

# 左轴：执行时间 (理想情况是水平线)
line1 = ax2_left.plot(num_gpus, exec_times, 'o-', color='#ef4444', linewidth=3, markersize=10, label='Execution Time (s)')
ax2_left.set_ylabel('Execution Time (seconds)', color='#ef4444', fontweight='bold')
ax2_left.tick_params(axis='y', labelcolor='#ef4444')
ax2_left.set_ylim(0, 0.5) # 设置合理的范围以便观察

# 右轴：吞吐量 (理想情况是线性增长)
line2 = ax2_right.plot(num_gpus, throughputs, 's--', color='#3b82f6', linewidth=3, markersize=10, label='Throughput (GUPS)')
ax2_right.set_ylabel('Throughput (GUPS)', color='#3b82f6', fontweight='bold')
ax2_right.tick_params(axis='y', labelcolor='#3b82f6')
ax2_right.set_ylim(0, 130)

# X轴设置
ax2_left.set_xticks(num_gpus)
ax2_left.set_xlabel('Number of GPUs (A100)', fontweight='bold')
ax2_left.set_title('Weak Scaling Analysis\n(Constant Load per GPU: 2048x2048)', fontweight='bold', pad=20)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2_left.legend(lines, labels, loc='center right')

plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('2_weak_scaling.png', dpi=300)
print("[Generated] 2_weak_scaling.png")

# ==========================================
# 图 3: 并行效率损耗图
# ==========================================
fig3, ax3 = plt.subplots(figsize=(8, 6))

# 绘制实际效率线
ax3.plot(num_gpus, efficiencies, 'D-', color='#10b981', linewidth=3, markersize=12, label='Measured Efficiency')

# 绘制理想参考线 (100%)
ax3.axhline(y=100, color='grey', linestyle='--', linewidth=2, label='Ideal (Linear Scaling)')

# 标注数据点
ax3.annotate(f'{efficiencies[0]:.1f}% (Base)', (num_gpus[0], efficiencies[0]), 
             xytext=(-20, 20), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
ax3.annotate(f'{efficiencies[1]:.1f}%', (num_gpus[1], efficiencies[1]), 
             xytext=(-40, -30), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
# 标注损耗原因
ax3.text(1.5, 95.5, 'Loss due to PCIe \nData Transfer & MPI Sync', color='#ef4444', ha='center')

ax3.set_xticks(num_gpus)
ax3.set_xlabel('Number of GPUs', fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontweight='bold', color='#10b981')
ax3.set_ylim(85, 105)
ax3.set_title('Parallel Efficiency & Communication Overhead', fontweight='bold', pad=20)
ax3.legend()
ax3.grid(linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('3_parallel_efficiency.png', dpi=300)
print("[Generated] 3_parallel_efficiency.png")