# config.py

# --- 模式选择 ---
# 建议设为 True。
# 论文 4.1-4.3 节主要对比“计算吞吐量”，即在相同步数下谁跑得快。
BENCHMARK_MODE = True 

# --- 网格设置 ---
# 这里的 NX, NY 是默认值。
# 在执行脚本（如 benchmark_cpu.py）时，代码会循环遍历不同规模，覆盖掉这里的 50。
NX = 512 
NY = 512

# --- 迭代设置 ---
# 性能测试模式下，建议统一设为 1000。
# 1000 步足以消除计时的随机误差，同时保证 CPU 不会跑太久。
MAX_ITER = 1000 

# --- 物理参数 ---
X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = 0.0, 1.0

# --- 收敛判定 ---
# 虽然 BENCHMARK_MODE=True 时不生效，但建议留着用于最后的物理验证。
TOLERANCE = 1e-6 
CHECK_INTERVAL = 100 

# --- 绘图开关 ---
ENABLE_PLOTTING = False