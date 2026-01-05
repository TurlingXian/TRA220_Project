import matplotlib.pyplot as plt
import os

# === 1. 数据配置 ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# A. Log 文件 (确保这些文件在 logs 目录下)
LOG_FILES = {
    'CPU (32 Cores)': 'cpu_test_7723800_0.log', 
    'CPU (64 Cores)': 'cpu_test_7723800_1.log',
    'CuPy':           'cupy_7723807.log',
    'PyTorch':        'torch_7723809.log',
    'Numba Basic':    'numba_7724473.log',       # <--- 确认这里有 Numba Basic
    'Numba Shared':   'numba_shared_7724481.log'
}

# B. 手动录入的数据 (单核 CPU)
MANUAL_DATA = {
    'CPU (Single Core)': {
        50: 0.0703,
        100: 0.6452,
        200: 4.1267,
        400: 42.6447,
        800: 831.4497,
        1000: 1561.5273,
        2000: 9069.6256
    }
}

# === 2. 样式配置 ===
STYLES = {
    'CPU (Single Core)': {'color': 'black',  'marker': 'x', 'linestyle': '--', 'linewidth': 1.0}, 
    'CPU (32 Cores)':    {'color': 'gray',   'marker': 'o', 'linestyle': '--', 'linewidth': 1.5},
    'CPU (64 Cores)':    {'color': 'purple', 'marker': 'v', 'linestyle': '-.', 'linewidth': 1.5}, 
    'CuPy':              {'color': 'blue',   'marker': 's', 'linestyle': '-'},
    'PyTorch':           {'color': 'orange', 'marker': '^', 'linestyle': '-'},
    # Numba Basic 用绿色菱形虚线表示
    'Numba Basic':       {'color': 'green',  'marker': 'd', 'linestyle': ':',  'linewidth': 2.0},
    'Numba Shared':      {'color': 'red',    'marker': '*', 'linestyle': '-',  'linewidth': 2.5, 'markersize': 10} 
}

# === 3. 解析函数 ===
def parse_log(filename):
    filepath = os.path.join(BASE_DIR, filename)
    data = {}
    if not os.path.exists(filepath):
        print(f"⚠️ 跳过: 找不到文件 {filename}")
        return data
    
    with open(filepath, 'r') as f:
        for line in f:
            if "x" in line and "Converged" in line:
                try:
                    parts = line.split('|')
                    grid_size = int(parts[0].strip().split('x')[0])
                    time_val = float(parts[3].strip())
                    data[grid_size] = time_val
                except (ValueError, IndexError):
                    continue
    return dict(sorted(data.items()))

# === 4. 整合数据 ===
all_data = {}

for label, filename in LOG_FILES.items():
    parsed = parse_log(filename)
    if parsed:
        all_data[label] = parsed

for label, data in MANUAL_DATA.items():
    all_data[label] = data

# === 5. 绘图核心函数 ===
def plot_chart(dataset_names, title, filename_suffix, y_log=False, max_x_limit=None):
    plt.figure(figsize=(12, 8), dpi=150)
    save_path = os.path.join(BASE_DIR, f"result_{filename_suffix}.png")
    
    temp_grids = set()

    for name in dataset_names:
        # 检查数据是否存在
        if name not in all_data or not all_data[name]:
            print(f"⚠️ 警告: 数据集 [{name}] 为空或未找到，将不会绘制在图表中。")
            continue
        
        raw_x = list(all_data[name].keys())
        raw_y = list(all_data[name].values())
        
        final_x = []
        final_y = []
        
        for x_val, y_val in zip(raw_x, raw_y):
            if max_x_limit is not None and x_val > max_x_limit:
                continue
            final_x.append(x_val)
            final_y.append(y_val)
            temp_grids.add(x_val)

        if not final_x: continue

        style = STYLES.get(name, {})
        plt.plot(final_x, final_y, label=name, **style)

    if not temp_grids:
        print(f"❌ 错误: 没有可绘制的数据，无法生成图表 {filename_suffix}。")
        plt.close()
        return

    sorted_grids = sorted(list(temp_grids))

    plt.xlabel('Grid Size ($N \\times N$)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(sorted_grids, rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    if y_log:
        plt.yscale('log')
        plt.ylabel('Time (seconds) - Log Scale', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 图表已生成: {save_path}")

# === 6. 生成图表 (确保列表里有 Numba Basic) ===

# 图 1: 完整全景 (Log Scale)
plot_chart(
    ['CPU (Single Core)', 'CPU (32 Cores)', 'CPU (64 Cores)', 'Numba Basic', 'CuPy', 'PyTorch', 'Numba Shared'], 
    'Full Overview: CPU vs GPU (Log Scale)', 
    'full_overview', 
    y_log=True
)

# 图 2: CPU 扩展性 (线性坐标, 限1000)
plot_chart(
    ['CPU (Single Core)', 'CPU (32 Cores)', 'CPU (64 Cores)'], 
    'CPU Scaling Analysis (Up to 1000x1000)', 
    'cpu_scaling', 
    y_log=False,
    max_x_limit=1000
)

# 图 3: GPU 细节 (线性坐标)
plot_chart(
    ['Numba Basic', 'CuPy', 'PyTorch', 'Numba Shared'], 
    'GPU Implementations Comparison', 
    'gpu_details', 
    y_log=False
)