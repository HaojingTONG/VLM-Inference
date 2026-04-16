#!/bin/bash
#SBATCH --job-name=vlm-token-compress
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

module purge
module load cuda/12.1

# ======== TODO: 根据你的实际情况修改以下两项 ========
# 1. conda 路径: 运行 `which conda` 确认
# 2. 环境名: 你创建环境时用的名字
source ~/.bashrc
conda activate vlm-compress
# ==================================================

# 切换到项目目录 (TODO: 改成你的 NetID)
cd /scratch/<your-netid>/Project

python scripts/run_benchmark.py \
    --config configs/default.yaml \
    --output results/
