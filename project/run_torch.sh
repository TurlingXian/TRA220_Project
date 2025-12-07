#!/bin/bash
#SBATCH --job-name=torch_bench
#SBATCH --account=C3SE2025-2-17       # 你的项目ID
#SBATCH --time=00:15:00
#SBATCH --nodes=1

# --- 关键：锁定 A100 显卡 ---
#SBATCH --gpus-per-node=A100:1

#SBATCH --partition=vera
#SBATCH --output=logs/torch_%j.log
#SBATCH --error=logs/torch_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODELIST"

# 1. 确认显卡
echo "------------------------------------------------"
echo "GPU Allocated:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "------------------------------------------------"

# 2. 加载环境 (保持一致)
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a

# 3. 激活虚拟环境
echo "Activating virtual environment..."
source $HOME/venvs/poisson_venv/bin/activate

# 4. 运行脚本 (-u 防止日志丢失)
echo "Starting PyTorch Benchmark..."
python -u benchmark_torch.py

echo "Done."