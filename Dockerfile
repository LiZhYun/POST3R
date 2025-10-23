# Dockerfile for POST3R on AMD GPUs (CSC LUMI)
FROM rocm/pytorch:latest

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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install POST3R in development mode
RUN pip install --no-cache-dir -e .

# Try to build CuRoPE kernels (may fail on ROCm, will fall back to CPU)
RUN cd submodules/ttt3r/src/croco/models/curope && \
    python setup.py build_ext --inplace || \
    echo "Warning: CuRoPE compilation failed, using CPU fallback"

# Set environment variables
ENV PYTHONPATH=/workspace
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create data and output directories
RUN mkdir -p /workspace/outputs /workspace/data

# Default shell
CMD ["/bin/bash"]
