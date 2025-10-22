#!/bin/bash
# Setup script for POST3R environment
# This script creates a conda environment and compiles necessary CUDA kernels

set -e  # Exit on error

echo "========================================="
echo "POST3R Environment Setup"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found: $(nvcc --version | grep release)"
else
    echo "⚠️  Warning: nvcc not found. CUDA may not be properly configured."
    echo "   RoPE kernel compilation may fail."
fi
echo ""

# Create conda environment
echo "Creating conda environment 'post3r'..."
echo "This may take several minutes..."
echo ""

if conda env list | grep -q "^post3r "; then
    echo "⚠️  Environment 'post3r' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n post3r -y
    else
        echo "Skipping environment creation."
        echo "To activate existing environment: conda activate post3r"
        exit 0
    fi
fi

conda env create -f environment.yml

echo ""
echo "✓ Conda environment created successfully"
echo ""

# Activate environment for remaining steps
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate post3r

# Compile RoPE CUDA kernels
echo ""
echo "========================================="
echo "Compiling RoPE CUDA Kernels"
echo "========================================="
echo ""

if [ ! -d "submodules/ttt3r" ]; then
    echo "❌ Error: TTT3R submodule not found"
    echo "Please initialize submodules first:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

cd submodules/ttt3r/src/croco/models/curope/

echo "Compiling curope extension..."
if python setup.py build_ext --inplace; then
    echo "✓ RoPE CUDA kernels compiled successfully"
else
    echo "❌ Error: RoPE kernel compilation failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Ensure CUDA toolkit is installed: conda install cuda-toolkit -c nvidia"
    echo "2. Check CUDA version matches PyTorch: python -c 'import torch; print(torch.version.cuda)'"
    echo "3. Verify nvcc is in PATH: which nvcc"
    echo ""
    echo "You can continue, but TTT3R may run slower without compiled kernels."
fi

cd ../../../../..

# Install POST3R package in development mode
echo ""
echo "========================================="
echo "Installing POST3R Package"
echo "========================================="
echo ""

echo "Installing POST3R in development mode..."
if pip install -e .; then
    echo "✓ POST3R package installed successfully"
else
    echo "❌ Error: POST3R package installation failed"
    exit 1
fi

# Check for TTT3R checkpoint
echo ""
echo "========================================="
echo "Checking TTT3R Checkpoint"
echo "========================================="
echo ""

CHECKPOINT_PATH="submodules/ttt3r/src/cut3r_512_dpt_4_64.pth"
if [ -f "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
    echo "✓ TTT3R checkpoint found ($CHECKPOINT_SIZE)"
else
    echo "⚠️  TTT3R checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "To download the checkpoint:"
    echo "  cd submodules/ttt3r/src"
    echo "  pip install gdown"
    echo "  gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"
    echo "  cd ../../.."
    echo ""
    echo "Or download manually from:"
    echo "  https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link"
fi

# Summary
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To get started:"
echo "  1. Activate the environment:"
echo "     conda activate post3r"
echo ""
echo "  2. Download TTT3R checkpoint (if not done):"
echo "     cd submodules/ttt3r/src && gdown --fuzzy <URL> && cd ../../.."
echo ""
echo "  3. Prepare your dataset:"
echo "     python post3r/data/save_ytvis2021.py --input data/ytvis2021_raw --output data/ytvis2021_resized"
echo ""
echo "  4. Start training:"
echo "     python scripts/train.py configs/train/ytvis2021.yaml --data-dir post3r/data/ytvis2021_resized"
echo ""
echo "For more information, see README.md"
echo ""
