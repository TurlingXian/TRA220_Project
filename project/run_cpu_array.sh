#!/bin/bash
#SBATCH --job-name=cpu_scaling
#SBATCH --account=C3SE2025-2-17
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --partition=vera
#SBATCH --constraint=ZEN4             # <---【关键修改1】保证两个任务都在同款 CPU 上跑
#SBATCH --output=logs/cpu_test_%A_%a.log
#SBATCH --error=logs/cpu_test_%A_%a.err

echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $SLURMD_NODELIST"

# 加载环境
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0 SciPy-bundle/2024.05-gfbf-2024a

# 激活虚拟环境
echo "Activating virtual environment..."
source $HOME/venvs/poisson_venv/bin/activate

# 确认 Python 路径
which python

# 根据 Array ID 决定核数
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    CORES=32
    echo "Configuring for 32 Cores..."
else
    CORES=64
    echo "Configuring for 64 Cores..."
fi

# 设置环境变量
export NUMBA_NUM_THREADS=$CORES
export OMP_NUM_THREADS=$CORES

# 运行 Python 脚本
# 你的 benchmark_cpu_advanced.py 会读取上面的 NUMBA_NUM_THREADS 变量
python -u benchmark_cpu_parallel.py   # <---【关键修改2】加上 -u 防止日志丢失

echo "Done with $CORES cores test."