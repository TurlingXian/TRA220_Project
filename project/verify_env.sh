#!/bin/bash
#SBATCH --job-name=verify_mpi_cuda
#SBATCH --account=C3SE2025-2-17
#SBATCH --partition=vera
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=A100:2
#SBATCH --time=00:05:00
#SBATCH --output=verify_%j.log
#SBATCH --error=verify_%j.err

set -euo pipefail

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.1.1
source "$HOME/venvs/poisson_venv/bin/activate"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 关键：启用 NVIDIA binding
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export NUMBA_CUDA_REDUCE_MEMORY_USAGE=0

echo "=== node GPU list ==="
nvidia-smi -L
echo

echo "=== per-task slurm env ==="
srun bash -c 'echo "task=$SLURM_PROCID localid=$SLURM_LOCALID CVD(before)=$CUDA_VISIBLE_DEVICES"'
echo

echo "=== run verify ==="
srun python -u verify_mpi_cuda.py
