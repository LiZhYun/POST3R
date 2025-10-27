#!/bin/bash
#SBATCH --job-name=post3r_train
#SBATCH --account=project_462001066
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=400:30:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================
# POST3R Training on CSC LUMI with ROCm
# ============================================
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
# srun --account=project_462001066 --partition=small-g --gpus-per-node=1  --time=00:30:00 --nodes=1 --pty bash
# singularity shell --rocm -B /scratch/project_462001066,/project/project_462001066,/project/project_462001066/POST3R:/workspace  /scratch/project_462001066/post3r-lumi_latest.sif
# python scripts/train.py configs/train/ytvis2021.yaml --data-dir /scratch/project_462001066/POST3R/data/ytvis2021_resized --log-dir /scratch/project_462001066/POST3R/output
# Run training inside Singularity container

# module load LUMI/24.03  partition/G
# module load OpenGL/24.03-cpeGNU-24.03
# module load cray-mpich
# module load cray-libfabric
# module load rocm
# module load cray-python
# source /scratch/project_462001066/post3r/bin/activate

# export SIF=/scratch/project_462001066/lumi-pytorch-rocm-6.0.3-python-3.12-pytorch-v2.3.1.sif
# singularity shell --rocm -B /opt/cray/libfabric/1.15.2.0/lib64/,/opt/cray/pe/mpich/8.1.29/gtl/lib/,/opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/lib,/usr/lib64:/usr/lib64,/usr/lib:/usr/lib,/scratch/project_462001066,/project/project_462001066,/project/project_462001066/POST3R:/workspace $SIF
# singularity exec $SIF bash -c '$WITH_CONDA && pip list'
# module use /appl/local/containers/ai-modules
# module load singularity-AI-bindings
# $WITH_CONDA
# source /scratch/project_462001066/post3r/bin/activate
# export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:/opt/cray/pe/mpich/8.1.29/gtl/lib/:/opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/lib:$LD_LIBRARY_PATH
# ldconfig -p | grep libmpi_cray


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
