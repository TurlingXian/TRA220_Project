#!/bin/bash
#SBATCH --job-name=numba_mpi_2gpu
#SBATCH --account=C3SE2025-2-17
#SBATCH --partition=vera
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=A100:2 
#SBATCH --time=00:15:00
#SBATCH --output=logs/gpu/numba_mpi_%j.log
#SBATCH --error=logs/gpu/numba_mpi_%j.err

module purge
module load GCC/12.3.0 OpenMPI/4.1.5 CUDA/12.1.1
source $HOME/venvs/poisson_venv/bin/activate

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# 禁用显存池防止冲突
export NUMBA_CUDA_REDUCE_MEMORY_USAGE=0

mkdir -p logs/gpu

echo "Starting 2-GPU task with srun (N=4096)..."
# srun 会自动设置 SLURM_LOCALID，这是我们脚本依赖的核心变量
srun python benchmark_mpi.py