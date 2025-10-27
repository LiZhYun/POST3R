# Dockerfile for POST3R on AMD GPUs (CSC LUMI)
# Using ROCm 7.0 with Ubuntu 24.04, Python 3.12, PyTorch 2.5.1
FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.11_pytorch_release_2.5.1

# Set up working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace/

# Note: PyTorch 2.5.1 with ROCm support is already included in the base image
# Verify what's installed
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    (python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')" || echo "TorchVision not pre-installed")

# Install Python dependencies
# We set PIP_NO_BUILD_ISOLATION and PIP_DISABLE_PIP_VERSION_CHECK to avoid issues
# The key is to use --no-build-isolation which allows pip to see already installed packages
ENV PIP_NO_BUILD_ISOLATION=1

RUN pip install --no-cache-dir --no-build-isolation \
    numpy==1.26.4 \
    einops>=0.6.0 \
    omegaconf>=2.3.0 \
    hydra-core>=1.3.0 \
    pytorch-lightning>=2.0.0 \
    tensorboard>=2.13.0 \
    wandb>=0.15.0 \
    accelerate>=0.20.0 \
    transformers \
    roma \
    huggingface-hub>=0.22 \
    trimesh \
    h5py \
    matplotlib>=3.7.0 \
    opencv-python>=4.8.0 \
    pillow==10.3.0 \
    viser>=0.1.0 \
    gradio \
    "pyglet<2" \
    tqdm>=4.65.0 \
    pyyaml>=6.0.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    torchmetrics>=1.0.0 \
    lpips>=0.1.4 \
    evo \
    open3d \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8 \
    av>=10.0.0 \
    moviepy>=1.0.3 \
    webdataset>=0.2.35 \
    timm>=0.9.7 \
    requests>=2.31.0 \
    pycocotools>=2.0.6

ENV PIP_NO_BUILD_ISOLATION=0

# Install POST3R in development mode
RUN pip install --no-cache-dir -e .

# Verify PyTorch installation (should still be ROCm version from base image)
RUN python -c "import torch; print(f'Final PyTorch version: {torch.__version__}'); print(f'CUDA/ROCm available: {torch.cuda.is_available()}')" && \
    (python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')" || echo "TorchVision not available")

# Try to build CuRoPE kernels (may fail on ROCm, will fall back to CPU)
RUN cd submodules/ttt3r/src/croco/models/curope && \
    python setup.py build_ext --inplace || \
    echo "Warning: CuRoPE compilation failed, using CPU fallback"

# Set environment variables
ENV PYTHONPATH=/workspace
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ROCm/MIOpen configuration to avoid database errors
# Create a writable cache directory for MIOpen
ENV MIOPEN_USER_DB_PATH=/tmp/miopen-cache
ENV MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache
RUN mkdir -p /tmp/miopen-cache && chmod 777 /tmp/miopen-cache

# Disable MIOpen cache warnings
ENV MIOPEN_DISABLE_CACHE=0
ENV MIOPEN_DEBUG_DISABLE_FIND_DB=0

# Create data and output directories
RUN mkdir -p /workspace/outputs /workspace/data

# Default shell
CMD ["/bin/bash"]
