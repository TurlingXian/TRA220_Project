#!/bin/bash
#SBATCH --job-name=numba_shared
#SBATCH --account=C3SE2025-2-17
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --partition=vera
#SBATCH --output=logs/numba_shared_%j.log
#SBATCH --error=logs/numba_shared_%j.err

echo "Job ID: $SLURM_JOB_ID"

# 1. 加载模块
module purge
module load GCC/12.3.0
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# 2. 激活 GPU 环境
source $HOME/project/gpu_env/bin/activate

# 3. 关键配置：强制使用 NVIDIA Bindings
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
unset CUDA_HOME
unset NUMBA_CUDA_DIR

# 4. 运行 Shared Memory Benchmark
echo "Starting Numba Shared Memory Benchmark..."
python -u benchmark_numba_shared.py

echo "Done."