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
module load python/3.10

source ~/.bashrc
conda activate vlm-compress  # or your env name

cd $SLURM_SUBMIT_DIR

python scripts/run_benchmark.py \
    --config configs/default.yaml \
    --output results/
