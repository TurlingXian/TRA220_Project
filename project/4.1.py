import matplotlib.pyplot as plt
import numpy as np

# --- 填入你的实验数据 ---
sizes = np.array([128, 256, 512, 1024, 2048])
time_np = np.array([0.0380, 0.1255, 0.4837, 3.1655, 13.4633])
# 注意：这里我暂且用 24.0s 作为 4096 的预估值，如果你重测了请修改
time_parallel = np.array([0.0362, 0.1180, 0.4229, 1.6201, 5.7754])

# 计算加速比
speedup = time_np / time_parallel

# 设置全局字体风格（可选，增加学术感）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True

# --- 图 1: 执行时间 (Log-Log Plot) ---
plt.figure(figsize=(5, 4)) # 适合单栏宽度
plt.loglog(sizes, time_np, 'o-', label='NumPy (Single Core)', linewidth=2, markersize=7)
plt.loglog(sizes, time_parallel, 's--', label='Numba (16 Threads)', linewidth=2, markersize=7)

plt.xlabel('Grid Size ($N \\times N$)', fontsize=11)
plt.ylabel('Execution Time (s)', fontsize=11)
plt.title('CPU Execution Time Scaling (1000 Iterations)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('cpu_time_loglog.png', dpi=300)
print("Saved: cpu_time_loglog.png")

# --- 图 2: 加速比 (Speedup Factor) ---
plt.figure(figsize=(5, 4)) # 适合单栏宽度
plt.plot(sizes, speedup, 'r-D', linewidth=2, markersize=7, label='Numba Speedup')

# 绘制 y=1 的参考线
plt.axhline(y=1.0, color='black', linestyle=':', alpha=0.6)

plt.xlabel('Grid Size ($N \\times N$)', fontsize=11)
plt.ylabel('Speedup Factor ($x$)', fontsize=11)
plt.title('Parallel Speedup relative to NumPy Baseline', fontsize=12)
plt.xticks(sizes) # 确保横坐标刻度对应实验规模
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('cpu_speedup.png', dpi=300)
print("Saved: cpu_speedup.png")