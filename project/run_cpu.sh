#!/bin/bash
#SBATCH --job-name=cpu_baseline
#SBATCH --account=C3SE2025-2-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=vera
#SBATCH --constraint=ZEN4
#SBATCH --output=logs/cpu/baseline_%j.log
#SBATCH --error=logs/cpu/baseline_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"

# 环境
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a
source $HOME/venvs/poisson_venv/bin/activate

# 强制所有潜在并行库只用 1 线程（保证“真单核”基线）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS"

# 建议用 srun，确保绑定与资源一致
srun --cpu-bind=cores python -u benchmark_cpu.py
