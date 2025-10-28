#!/bin/bash
#SBATCH --job-name=post3r_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=/$WRKDIR/.huggingface_cache
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions

source activate post3r


DATA_DIR="/scratch/work/liz23/POST3R/post3r/data/ytvis2021_resized"
OUTPUT_DIR="/scratch/work/liz23/POST3R/logs"

srun python scripts/train.py "configs/train/ytvis2021.yaml" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR} \
        --wandb

