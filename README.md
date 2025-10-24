# POST3R: Persistent Object Slots for Temporal 3D Representation from Video

Object-centric learning with 3D geometric reasoning for temporally consistent video understanding.

## Overview

POST3R combines Slot Attention with TTT3R's 3D backbone to learn object-centric representations that maintain consistent 3D geometry and identity over time. Each slot reconstructs a 3D point cloud instead of 2D features, enabling robust tracking and scene understanding

---

## Setup for Cluster Training

This guide provides everything needed to run training on a computing cluster.

### Step 1: Environment Setup

**Clone repository**:
```bash
git clone --recursive git@github.com:LiZhYun/POST3R.git
cd POST3R
```

**Create conda environment**:
```bash
conda create -n post3r python=3.11 cmake=3.14.0
conda activate post3r
```

**Install PyTorch with GPU support**:
```bash
# For NVIDIA GPUs (adjust pytorch-cuda version for your system)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# For AMD GPUs (ROCm): Use Docker instead (see below)
```

**Install other dependencies**:
```bash
pip install -r requirements.txt
```

**Fix for PyTorch dataloader issue**:
```bash
# See https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
```

**Install evaluation tools**:
```bash
pip install evo
pip install open3d
```

**Verify GPU support**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

> **⚠️ Important**: Always install PyTorch via conda with the appropriate CUDA version before installing other dependencies. This ensures GPU support is properly configured.

#### Alternative: Using environment.yml (All-in-one)

```bash
conda env create -f environment.yml
conda activate post3r
```

#### For AMD GPUs (ROCm) - Use Docker

```bash
# Pull the ROCm container with PyTorch 2.5.1
docker pull rocm/pytorch:rocm7.0.2_ubuntu22.04_py3.10_pytorch_release_2.5.1

# Build POST3R container
docker build -t post3r-rocm .

# Run container
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size 8G \
  -v $(pwd):/workspace post3r-rocm
```

**Setup Weights & Biases (optional)**:
```bash
wandb login
# Follow the prompt to enter your API key from https://wandb.ai/authorize
```

**Compile CUDA kernels** (required for TTT3R):
```bash
cd submodules/ttt3r/src/croco/models/curope/
python setup.py build_ext --inplace
cd -
```

> **Note**: Ensure your CUDA toolkit version matches PyTorch. Check with `nvcc --version` and `python -c "import torch; print(torch.version.cuda)"`

### Step 2: Download TTT3R Checkpoint

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

### Step 3: Prepare Dataset

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

### Cluster-Specific: SLURM

Create `scripts/submit_train.sh`:
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

Submit with: `sbatch scripts/submit_train.sh`

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
watch -n 1 nvidia-smi
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

### RoPE CUDA Kernel Errors
```bash
# Check CUDA version match
nvcc --version
python -c "import torch; print(torch.version.cuda)"

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
```bash
# 1. Setup
conda env create -f environment.yml
conda activate post3r
conda install 'llvm-openmp<16'
wandb login  # Optional: for W&B logging
cd submodules/ttt3r/src/croco/models/curope/
python setup.py build_ext --inplace
cd -

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
- ✅ `submodules/ttt3r/src/cut3r_512_dpt_4_64.pth` exists (~2.8GB)
- ✅ `post3r/data/ytvis2021_resized/*.tar` files exist
- ✅ `submodules/ttt3r/src/croco/models/curope/*.so` compiled
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
