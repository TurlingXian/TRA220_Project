#!/bin/bash
#SBATCH --job-name=cupy_2gpu
#SBATCH --account=C3SE2025-2-17
#SBATCH --partition=vera
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # 保持 32 (16核 x 2卡)
#SBATCH --gpus-per-node=A100:2  # 申请双卡
#SBATCH --time=00:30:00
#SBATCH --output=logs/gpu/cupy_2gpu_%j.log
#SBATCH --error=logs/gpu/cupy_2gpu_%j.err

set -euo pipefail

# 1. 清理并加载正确的模块 (跟你那个成功的 cuda 脚本保持一致)
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a
module load CUDA/12.1.1

# 2. 激活 poisson_venv (我们知道这个环境本身是没问题的)
source $HOME/venvs/poisson_venv/bin/activate

# 3. [关键步骤] 自动检查并安装 CuPy
# 如果环境里已经有 cupy，这步会瞬间完成；如果没有，它会自动安装
echo "--- Installing/Checking CuPy ---"
pip install cupy-cuda12x

mkdir -p logs/gpu

echo "=== GPU info ==="
nvidia-smi

echo "=== Python info ==="
which python
python --version

# 4. 运行双卡 Benchmark
echo "--- Starting Benchmark ---"
python -u benchmark_cupy_multi.py