#!/bin/bash
#SBATCH --job-name=cpu_np
#SBATCH --account=C3SE2025-2-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=vera
#SBATCH --constraint=ZEN4
#SBATCH --output=logs/cpu/baseline_%j.log

# 1. 环境准备
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a
source $HOME/venvs/poisson_venv/bin/activate

# 2. 核心设置：强制单线程，确保是“纯正”的单核基准
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# 3. 创建日志目录
mkdir -p logs/cpu

# 4. 运行
echo "Starting NumPy CPU Baseline..."
srun python -u benchmark_cpu.py