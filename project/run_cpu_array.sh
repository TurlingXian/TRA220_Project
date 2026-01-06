#!/bin/bash
#SBATCH --job-name=cpu_par
#SBATCH --account=C3SE2025-2-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=vera
#SBATCH --constraint=ZEN4
#SBATCH --output=logs/cpu/parallel_%j.log

# 1. 环境准备
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a
source $HOME/venvs/poisson_venv/bin/activate

# 2. 线程设置：利用申请的 16 个核心
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# 3. 运行
echo "Starting Numba Parallel (16 Threads)..."
srun --cpu-bind=cores python -u benchmark_cpu_parallel.py