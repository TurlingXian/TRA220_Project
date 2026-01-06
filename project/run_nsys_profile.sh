#!/bin/bash
#SBATCH --job-name=nsys_profile
#SBATCH --account=C3SE2025-2-17
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --partition=vera

# 1. 环境准备 (务必包含这些环境变量)
module purge
module load GCC/12.3.0 CUDA/12.1.1 Python/3.11.3-GCCcore-12.3.0
source $HOME/project/gpu_env/bin/activate
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

# 2. 核心剖析命令
# 增加 --trace=cuda 以确保强制抓取 GPU 数据
# 使用绝对路径以防万一
nsys profile \
    --trace=cuda,osrt \
    --output=reports/poisson_final_trace_v2 \
    --force-overwrite=true \
    python profile_target.py