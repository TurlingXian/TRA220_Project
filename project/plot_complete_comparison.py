import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 (严格对齐你的实验日志) ---
# 网格规模 (N x N)
sizes_standard = np.array([128, 256, 512, 1024, 2048, 4096])
# Numba Basic 的特殊采样点
sizes_basic = np.array([100, 200, 400, 800, 1000, 2000])

# 执行时间 (1000次迭代)
# CPU (16-core Numba)
t_cpu = np.array([0.045, 0.120, 0.462, 1.831, 7.420, np.nan]) 

# PyTorch (Section 4.2)
t_pytorch = np.array([0.3730, 0.3477, 0.3489, 0.3486, 0.3542, 1.2667])

# Numba-Basic (受限于频繁同步与Python循环)
t_numba_basic = np.array([0.4512, 1.3236, 3.5222, 7.7829, 9.7226, 29.0137])

# Numba-Shared (4.3极致优化版)
t_numba_opt = np.array([np.nan, np.nan, np.nan, 0.4332, 0.2640, 1.0837])

# --- 2. 绘图：执行耗时对比 (Log-Log) ---
plt.figure(figsize=(12, 7))
plt.loglog(sizes_standard, t_cpu, 'o--', label='Numba CPU (16-core)', color='gray', alpha=0.8)
plt.loglog(sizes_standard, t_pytorch, 's-', label='PyTorch (High-level GPU)', color='green')
plt.loglog(sizes_basic, t_numba_basic, 'x-', label='Numba-Basic (Naive GPU)', color='orange')
plt.loglog(sizes_standard, t_numba_opt, 'D-', label='Numba-Shared (Fused+Shared)', color='red', linewidth=2.5)

plt.title('Performance Evolution: Breaking the Latency Plateau', fontsize=16, fontweight='bold')
plt.xlabel('Grid Size (N)', fontsize=13)
plt.ylabel('Time for 1000 iterations (s)', fontsize=13)
plt.xticks(sizes_standard, labels=sizes_standard)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(fontsize=11)

# 标注 4.3 节的关键突破
plt.annotate('Optimization Victory!', xy=(2048, 0.264), xytext=(400, 0.1),
             arrowprops=dict(facecolor='red', shrink=0.05, width=2), fontsize=12, color='red', fontweight='bold')
plt.savefig('complete_time_scaling.png', dpi=300)

# --- 3. 绘图：加速比对比 (Speedup) ---
plt.figure(figsize=(12, 7))
# 以 CPU 2048x2048 的 7.42s 为基准
speedup_pt = 7.420 / 0.3542 # PyTorch at 2048
speedup_opt = 7.420 / 0.2640 # Numba-Shared at 2048
speedup_basic = 7.420 / 29.0137 # Numba-Basic at 2000

methods = ['Numba-Basic', 'PyTorch', 'Numba-Shared']
speedups = [speedup_basic, speedup_pt, speedup_opt]
colors = ['orange', 'green', 'red']

bars = plt.bar(methods, speedups, color=colors, alpha=0.8)
plt.axhline(y=1, color='black', linestyle='--', label='CPU Baseline (1.0x)')

# 添加数值标注
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Speedup Factor vs. 16-core CPU (at ~2048^2 Grid)', fontsize=16)
plt.ylabel('Speedup Factor ($T_{CPU} / T_{GPU}$)', fontsize=13)
plt.ylim(0, 35)
plt.legend()
plt.savefig('complete_speedup_bar.png', dpi=300)

plt.show()