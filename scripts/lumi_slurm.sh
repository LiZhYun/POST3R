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

module load LUMI/24.03  partition/G
module load OpenGL/24.03-cpeGNU-24.03
module load cray-mpich
module load cray-libfabric
module load rocm
module load cray-python
source /scratch/project_462001066/post3r/bin/activate

DATA_DIR="/scratch/project_462001066/POST3R/data/ytvis2021_resized"
OUTPUT_DIR="/scratch/project_462001066/POST3R/output"

srun python scripts/train.py "configs/train/ytvis2021.yaml" \
        --data-dir ${DATA_DIR} \
        --log-dir ${OUTPUT_DIR}

