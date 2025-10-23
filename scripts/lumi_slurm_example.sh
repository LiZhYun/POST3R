#!/bin/bash
#SBATCH --job-name=post3r_train
#SBATCH --account=project_<YOUR_PROJECT_ID>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================
# POST3R Training on CSC LUMI with ROCm
# ============================================
# 
# Before running:
# 1. Replace <YOUR_PROJECT_ID> with your LUMI project ID
# 2. Update paths for data and singularity image
# 3. Create logs directory: mkdir -p logs
# 4. Submit with: sbatch lumi_slurm_example.sh
#

set -e

# Environment setup for LUMI
module purge
module load LUMI/23.09
module load partition/G
module load rocm/5.7.1
module load singularity-bindings

# Paths (modify these for your setup)
SINGULARITY_IMAGE="/path/to/post3r_rocm.sif"
DATA_DIR="/scratch/project_<YOUR_PROJECT_ID>/post3r/data"
OUTPUT_DIR="/scratch/project_<YOUR_PROJECT_ID>/post3r/outputs"
CHECKPOINT_PATH="/scratch/project_<YOUR_PROJECT_ID>/post3r/checkpoints/cut3r_512_dpt_4_64.pth"

# Create output directory
mkdir -p "${OUTPUT_DIR}/logs"

# Verify GPU availability
echo "============================================"
echo "Job Information"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo ""
rocm-smi
echo ""

# Set up Weights & Biases (optional)
# export WANDB_API_KEY="your_api_key_here"
# Or use offline mode
export WANDB_MODE=offline

# Training configuration
CONFIG="configs/train/ytvis2021.yaml"
BATCH_SIZE=4
NUM_WORKERS=8
NUM_GPUS=4

echo "============================================"
echo "Starting POST3R Training"
echo "============================================"
echo "Config: ${CONFIG}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE} (per GPU)"
echo "Workers: ${NUM_WORKERS}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# Run training inside Singularity container
singularity exec \
    --rocm \
    --bind "${DATA_DIR}:/workspace/post3r/data" \
    --bind "${OUTPUT_DIR}:/workspace/outputs" \
    --bind "${CHECKPOINT_PATH}:/workspace/submodules/ttt3r/src/cut3r_512_dpt_4_64.pth" \
    "${SINGULARITY_IMAGE}" \
    python /workspace/scripts/train.py "${CONFIG}" \
        --data-dir /workspace/post3r/data/ytvis2021_resized \
        --log-dir /workspace/outputs \
        dataset.batch_size=${BATCH_SIZE} \
        dataset.num_workers=${NUM_WORKERS} \
        trainer.devices=${NUM_GPUS} \
        trainer.strategy=ddp \
        trainer.precision=32

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Optional: Sync wandb runs if using offline mode
# singularity exec "${SINGULARITY_IMAGE}" wandb sync "${OUTPUT_DIR}/*/logs/wandb/offline-*"
