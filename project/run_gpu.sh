#!/bin/bash
#SBATCH --job-name=gpu_fix
#SBATCH --account=C3SE2025-2-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # 必须匹配 A100 节点的 16 核要求
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:30:00
#SBATCH --partition=vera
#SBATCH --output=logs/gpu/profile_%j.log
#SBATCH --error=logs/gpu/profile_%j.err

set -euo pipefail

# 1. 环境清理与加载
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a
module load CUDA/12.1.1 

# 2. 激活虚拟环境
source $HOME/venvs/poisson_venv/bin/activate

# --- 重要：自检与修复 torch ---
echo "Checking for torch and cupy..."
if ! python -c "import torch; import cupy; print('Dependencies found!')" &> /dev/null; then
    echo "Dependencies missing in venv. Installing..."
    pip install torch cupy-cuda12x  # 确保 venv 内有这些库
fi

# 3. 运行 Profiling
# 注意：将 %j 替换为 $SLURM_JOB_ID
echo "Starting GPU Benchmark with Nsight Systems..."

srun nsys profile \
    -t cuda,osrt,nvtx \
    -o "logs/gpu/report_$SLURM_JOB_ID" \
    -f true \
    python -u benchmark_gpu.py

echo "Done. Report saved to logs/gpu/report_$SLURM_JOB_ID.nsys-rep"