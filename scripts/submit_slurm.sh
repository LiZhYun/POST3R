#!/bin/bash
#SBATCH --job-name=post3r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Load modules (adjust for your cluster)
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
source ~/.bashrc
conda activate post3r

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Set paths (MODIFY THESE)
DATA_DIR="/path/to/ytvis2021_resized"
OUTPUT_DIR="/path/to/outputs"

# Weights & Biases (optional - uncomment to enable)
# export WANDB_API_KEY="your_api_key_here"
WANDB_PROJECT="post3r"
WANDB_ENTITY="your-team-name"

# Create log directory
mkdir -p logs

# Training command
# Add --wandb flag to enable W&B logging
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir $DATA_DIR \
    --log-dir $OUTPUT_DIR \
    --wandb \
    --wandb-project $WANDB_PROJECT \
    --wandb-entity $WANDB_ENTITY \
    trainer.devices=4 \
    trainer.strategy=ddp

# Alternative: without W&B
# python scripts/train.py configs/train/ytvis2021.yaml \
#     --data-dir $DATA_DIR \
#     --log-dir $OUTPUT_DIR \
#     trainer.devices=4 \
#     trainer.strategy=ddp

# Print end time
echo "End time: $(date)"
