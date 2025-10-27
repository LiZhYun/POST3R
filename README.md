# POST3R: Persistent Object Slots for Temporal 3D Representation from Video

Object-centric learning with 3D geometric reasoning for temporally consistent video understanding.

## Overview

POST3R combines Slot Attention with TTT3R's 3D backbone to learn object-centric representations that maintain consistent 3D geometry and identity over time. Each slot reconstructs a 3D point cloud instead of 2D features, enabling robust tracking and scene understanding

---

## Setup Instructions

Choose the setup guide that matches your system:
- **[Setup for NVIDIA GPUs (Local/Cluster)](#setup-for-nvidia-gpus)** - For local machines or clusters with NVIDIA GPUs
- **[Setup for CSC LUMI (AMD GPUs)](#setup-for-csc-lumi-amd-gpus)** - For LUMI supercomputer with AMD GPUs

---

## Setup for NVIDIA GPUs

This guide is for local machines or clusters with NVIDIA GPUs using conda.

### Step 1: Clone Repository

```bash
git clone --recursive git@github.com:LiZhYun/POST3R.git
cd POST3R
```

### Step 2: Create Conda Environment

**Option A: Using environment.yml (Recommended)**
```bash
conda env create -f environment.yml
conda activate post3r
```

**Option B: Manual Setup**
```bash
# Create environment
conda create -n post3r python=3.11 cmake=3.14.0
conda activate post3r

# Install PyTorch with CUDA support (adjust pytorch-cuda version for your system)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Fix for PyTorch dataloader issue (see https://github.com/pytorch/pytorch/issues/99625)
conda install 'llvm-openmp<16'
```

### Step 3: Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

> **⚠️ Important**: Ensure your CUDA toolkit version matches PyTorch. Check with `nvcc --version` and `python -c "import torch; print(torch.version.cuda)"`

### Step 4: Setup Weights & Biases (Optional)

```bash
wandb login
# Follow the prompt to enter your API key from https://wandb.ai/authorize
```

### Step 5: Compile CUDA Kernels

```bash
cd submodules/ttt3r/src/croco/models/curope/
python setup.py build_ext --inplace
cd -
```

### Step 6: Download TTT3R Checkpoint

**Required checkpoint**: `cut3r_512_dpt_4_64.pth` (~2.8GB)

```bash
cd submodules/ttt3r/src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd -
```

**Verify checkpoint**:
```bash
ls -lh submodules/ttt3r/src/cut3r_512_dpt_4_64.pth
# Should be ~2.8GB
```

> **Alternative**: Download manually from [Google Drive](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link) and place in `submodules/ttt3r/src/`

### Step 7: Prepare Dataset

See [Dataset Preparation](#dataset-preparation) section below.

---

## Setup for CSC LUMI (AMD GPUs)

This guide is specifically for the LUMI supercomputer with AMD GPUs using ROCm 6.0.

### Step 1: Clone Repository

```bash
git clone --recursive git@github.com:LiZhYun/POST3R.git
cd POST3R
```

### Step 2: Load Python Module

```bash
module load cray-python
```

### Step 3: Create Virtual Environment

```bash
# Create venv in your home directory
python -m venv ~/post3r

# Activate the environment
source ~/post3r/bin/activate
```

### Step 4: Install PyTorch with ROCm 6.0

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify GPU Support

```bash
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

> **💡 Tip**: Use `rocm-smi` to check GPU status on LUMI compute nodes.

### Step 7: Setup Weights & Biases (Optional)

```bash
wandb login
# Or set API key as environment variable
export WANDB_API_KEY="your_api_key_here"
```

### Step 8: Fix RoPE2D for ROCm Compatibility

```bash
# Follow https://github.com/CUT3R/CUT3R/issues/26 to replace RoPE2D code
```

### Step 9: Download TTT3R Checkpoint

**Required checkpoint**: `cut3r_512_dpt_4_64.pth` (~2.8GB)

```bash
cd submodules/ttt3r/src
# Download on your local machine and transfer to LUMI, or use wget if accessible
# gdown may not work on LUMI compute nodes
cd -
```

### Step 10: Prepare Dataset

See [Dataset Preparation](#dataset-preparation) section below.

> **💾 Storage**: Use `/scratch/project_<your_project>/` for datasets and outputs on LUMI for better I/O performance.

---

## Dataset Preparation

**YTVIS2021** (recommended for training):

1. **Download raw data**:
```bash
mkdir -p post3r/data/ytvis2021_raw
cd post3r/data/ytvis2021_raw
# Download from: https://codalab.lisn.upsaclay.fr/competitions/7680
# You need: train.zip, valid.zip, train.json, valid.json
```

2. **Extract**:
```bash
unzip train.zip
unzip valid.zip
```

3. **Preprocess to WebDataset format**:
```bash
cd ..
python save_ytvis2021.py \
    --input ytvis2021_raw \
    --output ytvis2021_resized \
    --size 518 \
    --chunk-size 1000
```

This creates `ytvis2021_resized/` with `.tar` files for efficient loading.

**Expected structure**:
```
post3r/data/
└── ytvis2021_resized/
    ├── ytvis-train-000000.tar
    ├── ytvis-train-000001.tar
    ├── ...
    └── ytvis-valid-000000.tar
```

---

## Training

### Basic Training Command

```bash
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir post3r/data/ytvis2021_resized \
    --log-dir outputs
```

**With Weights & Biases logging**:
```bash
# First login (one time setup)
wandb login

# Train with W&B
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir post3r/data/ytvis2021_resized \
    --log-dir outputs \
    --wandb \
    --wandb-project post3r \
    --wandb-entity your-team-name
```

### Multi-GPU Training (Recommended)

```bash
# 4 GPUs with DDP
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir post3r/data/ytvis2021_resized \
    --log-dir outputs \
    trainer.devices=4 \
    trainer.strategy=ddp

# 8 GPUs
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir post3r/data/ytvis2021_resized \
    --log-dir outputs \
    trainer.devices=8 \
    trainer.strategy=ddp
```

### NVIDIA Cluster with SLURM

Create `scripts/submit_train_nvidia.sh`:
```bash
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

# Activate environment
source ~/.bashrc
conda activate post3r

# Training
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir /path/to/ytvis2021_resized \
    --log-dir /path/to/outputs \
    trainer.devices=4 \
    trainer.strategy=ddp
```

Submit with: `sbatch scripts/submit_train_nvidia.sh`

### LUMI SLURM

Create `scripts/submit_train_lumi.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=post3r-train
#SBATCH --account=project_<your_project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create logs directory
mkdir -p logs

# Load required modules
module load cray-python

# Activate virtual environment
source ~/post3r/bin/activate

# Set environment variables for ROCm
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# Training
python scripts/train.py configs/train/ytvis2021.yaml \
    --data-dir /scratch/project_<your_project>/post3r/data/ytvis2021_resized \
    --log-dir /scratch/project_<your_project>/post3r/outputs \
    trainer.devices=8 \
    trainer.strategy=ddp

# Clean up
rm -rf ${MIOPEN_USER_DB_PATH}
```

Submit with: `sbatch scripts/submit_train_lumi.sh`

> **📖 More LUMI Examples**: See [LUMI_SETUP.md](LUMI_SETUP.md) for multi-node training and additional SLURM configurations.

### Key Configuration Options

**Model hyperparameters** (in config or via CLI):
```yaml
model:
  num_slots: 11              # Number of object slots (11 for YTVIS)
  slot_dim: 128              # Slot feature dimension
  num_iterations: 3          # Slot attention iterations
  decoder_resolution: [518, 518]  # Decoder output size
  learning_rate: 4e-4        # Base learning rate
  weight_decay: 0.0          # L2 regularization
  visualize: true            # Enable visualizations
  visualize_every_n_steps: 1000  # Visualization frequency
```

**Dataset** (adjust for your hardware):
```yaml
dataset:
  batch_size: 4              # Per-GPU batch size (reduce if OOM)
  num_workers: 4             # Dataloader workers per GPU
  chunk_size: 6              # Frames per video clip
  train_resampling: true     # Resample for distributed training
```

**Trainer**:
```yaml
trainer:
  max_steps: 100000          # Total training steps
  val_check_interval: 5000   # Validation frequency
  log_every_n_steps: 100     # Logging frequency
  gradient_clip_val: 1.0     # Gradient clipping
  precision: 32              # Use 16 for mixed precision
```

**Override from CLI**:
```bash
python scripts/train.py configs/train/ytvis2021.yaml \
    model.learning_rate=1e-3 \
    dataset.batch_size=2 \
    trainer.precision=16
```

### Resume Training

```bash
# Resume from experiment directory (finds last.ckpt automatically)
python scripts/train.py configs/train/ytvis2021.yaml \
    --continue outputs/post3r/20251022_143000

# Resume from specific checkpoint
python scripts/train.py configs/train/ytvis2021.yaml \
    --continue outputs/post3r/20251022_143000/results/checkpoints/last.ckpt
```

### Output Directory Structure

```
outputs/
└── post3r/                    # experiment_name from config
    └── 20251022_143000/       # timestamp
        ├── config.yaml        # Saved configuration
        ├── logs/              # Training logs
        │   ├── tensorboard/   # TensorBoard events
        │   └── metrics/       # CSV metrics
        ├── results/           # Training results
        │   └── checkpoints/   # Model checkpoints
        │       ├── last.ckpt
        │       └── epoch=XX-step=YYYYYY.ckpt
        └── visualizations/    # Visualization outputs
```

### Monitor Training

**TensorBoard**:
```bash
tensorboard --logdir=outputs/post3r/20251022_143000/logs/tensorboard
```

**Weights & Biases**:
- View at: https://wandb.ai/your-entity/post3r
- Automatic logging of metrics, system stats, and hyperparameters
- Compare multiple runs easily

**Check metrics (CSV)**:
```bash
# View loss curves
cat outputs/post3r/20251022_143000/logs/metrics/metrics.csv | column -t -s,

# Watch live
watch -n 5 'tail -n 20 outputs/post3r/20251022_143000/logs/metrics/metrics.csv'
```

**GPU usage**:
```bash
# For NVIDIA GPUs
watch -n 1 nvidia-smi

# For AMD GPUs (LUMI)
watch -n 1 rocm-smi
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)
```bash
# Reduce batch size
python scripts/train.py configs/train/ytvis2021.yaml dataset.batch_size=2

# Use gradient accumulation
python scripts/train.py configs/train/ytvis2021.yaml 
    dataset.batch_size=2 
    trainer.accumulate_grad_batches=2

# Use mixed precision
python scripts/train.py configs/train/ytvis2021.yaml trainer.precision=16

# Reduce model size
python scripts/train.py configs/train/ytvis2021.yaml 
    model.num_slots=8 
    model.slot_dim=64
```

### Dataloader Hanging/Slow
```bash
# Reduce workers
python scripts/train.py configs/train/ytvis2021.yaml dataset.num_workers=2

# Install llvm-openmp fix
conda install 'llvm-openmp<16'
```

### TTT3R Checkpoint Not Found
```bash
# Verify path
ls -lh submodules/ttt3r/src/cut3r_512_dpt_4_64.pth

# Check size (should be ~2.8GB)
du -h submodules/ttt3r/src/cut3r_512_dpt_4_64.pth

# Re-download if needed
cd submodules/ttt3r/src
rm cut3r_512_dpt_4_64.pth
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
```

### RoPE CUDA/ROCm Kernel Errors

**For NVIDIA GPUs:**
```bash
# Check CUDA version match
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Recompile kernels
cd submodules/ttt3r/src/croco/models/curope/
python setup.py clean --all
python setup.py build_ext --inplace
```

**For AMD GPUs (LUMI):**
```bash
# Check ROCm version
python -c "import torch; print(torch.version.hip)"

# Recompile kernels
cd submodules/ttt3r/src/croco/models/curope/
python setup.py clean --all
python setup.py build_ext --inplace
```

### Weights & Biases Issues
```bash
# Not logged in
wandb login

# Set API key via environment variable (for clusters)
export WANDB_API_KEY="your_api_key_here"

# Run in offline mode (sync later)
export WANDB_MODE=offline
python scripts/train.py configs/train/ytvis2021.yaml --wandb ...

# Sync offline runs later
wandb sync outputs/post3r/*/logs/wandb/offline-*

# Disable W&B if issues persist
# Simply don't use --wandb flag
```

### Training Not Starting
```bash
# Test with dry run
python scripts/train.py configs/train/ytvis2021.yaml --dry -v

# Check data loading
python -c "
from post3r.training import data_module as dm
from omegaconf import OmegaConf
config = OmegaConf.load('configs/train/ytvis2021.yaml')
datamodule = dm.build(config.dataset, data_dir='post3r/data/ytvis2021_resized')
datamodule.setup('fit')
print('Train batches:', len(datamodule.train_dataloader()))
print('Val batches:', len(datamodule.val_dataloader()))
"
```

---

## Quick Reference

### Minimal Working Example

**For NVIDIA GPUs:**
```bash
# 1. Setup
conda env create -f environment.yml
conda activate post3r
conda install 'llvm-openmp<16'
wandb login  # Optional: for W&B logging
cd submodules/ttt3r/src/croco/models/curope/
python setup.py build_ext --inplace
cd -
```

**For AMD GPUs (ROCm 6.0) - LUMI:**
```bash
# 1. Setup
module load cray-python
python -m venv ~/post3r
source ~/post3r/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
wandb login  # Optional: for W&B logging

# 2. Fix RoPE2D for ROCm (see https://github.com/CUT3R/CUT3R/issues/26)

# 3. Compile kernels
cd submodules/ttt3r/src/croco/models/curope/
python setup.py build_ext --inplace
cd -
```

**Continue with (both NVIDIA and AMD):**
```bash

# 2. Download checkpoint
cd submodules/ttt3r/src
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd -

# 3. Prepare data (assuming you have YTVIS2021)
python post3r/data/save_ytvis2021.py 
    --input post3r/data/ytvis2021_raw 
    --output post3r/data/ytvis2021_resized

# 4. Train
python scripts/train.py configs/train/ytvis2021.yaml 
    --data-dir post3r/data/ytvis2021_resized
```

### File Checklist Before Training

**Both NVIDIA and LUMI:**
- ✅ `submodules/ttt3r/src/cut3r_512_dpt_4_64.pth` exists (~2.8GB)
- ✅ `post3r/data/ytvis2021_resized/*.tar` files exist
- ✅ `submodules/ttt3r/src/croco/models/curope/*.so` compiled

**NVIDIA only:**
- ✅ `conda list | grep llvm-openmp` shows version < 16

### GPU Memory Requirements
| Batch Size | Slots | GPU Memory | Recommended GPU |
|------------|-------|------------|-----------------|
| 4          | 11    | ~24GB      | RTX 3090, A5000 |
| 2          | 11    | ~14GB      | RTX 3080        |
| 1          | 11    | ~8GB       | RTX 3070        |
| 4          | 8     | ~18GB      | RTX 3080 Ti     |

---

## License & Acknowledgements

**License**: MIT (see [LICENSE](LICENSE))

**Built upon**:
- [TTT3R](https://github.com/Inception3D/TTT3R) - 3D reconstruction backbone
- [SlotContrast](https://github.com/martius-lab/slotcontrast) - Object-centric learning reference

**Note**: Submodules have their own licenses. Check their repositories for details.

## Development

### Code Quality
```bash
# Format code
black post3r/ scripts/
isort post3r/ scripts/

# Run tests
pytest tests/

# Type checking
mypy post3r/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Submit a pull request

## Citation

```bibtex
@article{post3r2025,
    title={POST3R: Persistent Object Slots for Temporal 3D Representation from Video},
    author={Your Name},
    year={2025}
}
```

## Acknowledgements

This project builds upon excellent prior work:

- **[TTT3R](https://github.com/Inception3D/TTT3R)**: 3D reconstruction backbone with test-time training
- **[SlotContrast](https://github.com/martius-lab/slotcontrast)**: Object-centric learning with temporal consistency
- **[CUT3R](https://github.com/CUT3R/CUT3R)**: Multi-view 3D reconstruction
- **[DUSt3R](https://github.com/naver/dust3r)**: Dense reconstruction from uncalibrated images

We thank the authors for releasing their code and models!

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

Note: Submodules (TTT3R, SlotContrast) have their own licenses. Please refer to their respective repositories.
