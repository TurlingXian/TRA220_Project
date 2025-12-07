#!/bin/bash
#SBATCH --job-name=cupy_bench
#SBATCH --account=C3SE2025-2-17       # 使用你提供的项目ID
#SBATCH --time=00:15:00               # CuPy 跑得非常快，15分钟绰绰有余
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1           # <---【关键】申请 1 块 GPU
#SBATCH --partition=vera              # Vera 分区 (部分节点含 GPU)
#SBATCH --output=logs/cupy_%j.log     # 日志保存到 logs 文件夹
#SBATCH --error=logs/cupy_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODELIST"

# 1. 打印 GPU 信息 (确认我们拿到了显卡)
echo "------------------------------------------------"
echo "Checking GPU status..."
nvidia-smi
echo "------------------------------------------------"

# 2. 加载环境 (保持和 CPU 脚本一致)
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a

# 2. 【关键新增】加载 CUDA 模块！
# CuPy 需要 libnvrtc.so，这个文件在这个模块里
# 我们尝试加载 CUDA 12 (因为你装的是 cupy-cuda12x)
module load CUDA/12.1.1
# 如果上面这行报错说找不到，试一下: module load CUDA

# 3. 激活虚拟环境
echo "Activating virtual environment..."
source $HOME/venvs/poisson_venv/bin/activate

# 4. 确认 Python 和 CuPy 安装
which python
# 简单检查一下 cupy 是否能导入，防止跑了一半才报错
python -c "import cupy; print(f'CuPy imported successfully! Device: {cupy.cuda.Device()}')"

echo "------------------------------------------------"
echo "Starting CuPy Benchmark..."

# 5. 运行 Python 脚本
# 记得加上 -u 防止日志丢失
python -u benchmark_cupy.py

echo "Done."