#!/bin/bash
#SBATCH --job-name=post3r_train
#SBATCH --account=project_462001066
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================
# POST3R Training on CSC LUMI with ROCm
# ============================================
# 

SINGULARITY_IMAGE="/scratch/project_462001066/post3r-lumi_latest.sif"
DATA_DIR="/scratch/project_462001066/POST3R/data/ytvis2021_resized"
OUTPUT_DIR="/scratch/project_462001066/POST3R/output"

singularity exec \
    --rocm \
    --bind "/scratch/project_462001066" \
    --bind "/project/project_462001066/POST3R:/workspace" \
    "${SINGULARITY_IMAGE}" \
    python scripts/train.py "configs/train/ytvis2021.yaml" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR}

