# config.py

# --- 模式选择 ---
# True:  纯性能测试模式 (推荐)。强制跑满 MAX_ITER，不检查收敛。
#        这是展示 GPU 吞吐量 (Speedup) 的最佳方式。
# False: 物理模拟模式。当误差小于 TOLERANCE 时提前停止。
BENCHMARK_MODE = True

# --- 网格设置 ---
# 推荐: 1000 或 2000 以展示 GPU 优势
NX = 50
NY = 50

# --- 迭代设置 ---
# 如果是 Benchmark 模式，建议设为 2000~5000 步
# 如果是 Physics 模式，可能需要更多步数
MAX_ITER = 2000

# --- 物理参数 ---
X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = 0.0, 1.0

# --- 收敛判定 ---
# 仅在 BENCHMARK_MODE = False 时生效
# 建议设得很小，或者采用 relative tolerance，这里用绝对误差演示
TOLERANCE = 1e-6 
CHECK_INTERVAL = 100 # 每隔多少步检查一次误差 (减少同步开销)

# --- 绘图开关 ---
# 网格大于 500x500 时建议 False，否则 Matplotlib 会卡死
ENABLE_PLOTTING = False