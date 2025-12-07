#!/bin/bash
#SBATCH --job-name=numba_bench
#SBATCH --account=C3SE2025-2-17
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --partition=vera
#SBATCH --output=logs/numba_%j.log
#SBATCH --error=logs/numba_%j.err

echo "Job ID: $SLURM_JOB_ID"

# 1. 模块加载：使用和刚才 Smoke Test 成功时完全一样的配置
module purge
module load GCC/12.3.0
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# 2. 激活环境：指向我们刚才修好的 gpu_env
# (假设 gpu_env 在你的 project 目录下，建议用绝对路径更稳妥)
source $HOME/project/gpu_env/bin/activate

# 3. 核心配置：复制刚才成功的“必胜配置”
# 不要再用旧脚本里的 NUMBA_CUDA_DRIVER=/usr/lib64... 那些了，容易冲突
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
unset CUDA_HOME
unset NUMBA_CUDA_DIR

# 4. 运行你的 Benchmark
echo "Starting Numba Benchmark..."
python -u benchmark_numba.py

echo "Done."