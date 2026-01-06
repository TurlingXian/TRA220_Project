#!/bin/bash
#SBATCH --job-name=numba_weak_scaling
#SBATCH --account=C3SE2025-2-17
#SBATCH --partition=vera
#SBATCH --nodes=1
#SBATCH --ntasks=1             # 分别测试 1 和 2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=A100:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu/weak_scaling_%j.log

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.1.1
source "$HOME/venvs/poisson_venv/bin/activate"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "Starting Weak Scaling Experiment..."
srun --cpu-bind=cores python -u benchmark_mpi.py