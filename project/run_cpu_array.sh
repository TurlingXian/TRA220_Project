#!/bin/bash
#SBATCH --job-name=cpu_scaling_auto
#SBATCH --account=C3SE2025-2-17
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=vera
#SBATCH --constraint=ZEN4
#SBATCH --output=logs/cpu/auto_scale_%a_%j.log
#SBATCH --error=logs/cpu/auto_scale_%a_%j.err

set -euo pipefail

CORES_LIST=(1 2 4 8 16 32 64)
CURRENT_CORES=${CORES_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running scaling test for: $CURRENT_CORES threads"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Node(s): $SLURM_JOB_NODELIST"

module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a
source $HOME/venvs/poisson_venv/bin/activate

# 强制线程数一致（Numba + OpenMP）
export NUMBA_NUM_THREADS=$CURRENT_CORES
export OMP_NUM_THREADS=$CURRENT_CORES

# 绑定/放置策略（降低漂移）
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export NUMBA_THREADING_LAYER=omp

# 避免别的库偷偷开线程影响结果（可选但推荐）
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs/cpu

srun --cpu-bind=cores python -u benchmark_cpu_parallel.py
