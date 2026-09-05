#!/bin/bash
set -e

echo "=== [1/6] Kiểm tra phần cứng & NVIDIA Driver ==="
nvidia-smi
nvcc --version || true
python3 --version

echo "=== [2/6] Cài đặt System packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    ca-certificates

echo "=== [3/6] Cài đặt Python Dependencies cơ bản ==="
pip install --upgrade pip setuptools wheel packaging ninja
pip install trimesh pyvista xformers huggingface_hub einops diffusers transformers accelerate imageio imageio-ffmpeg scipy rembg onnxruntime
pip install spconv-cu120 || true

echo "=== [4/6] Cài đặt NATTEN (Prebuilt wheel) ==="
pip install natten==0.17.1+torch240cu124 -f https://shi-labs.com/natten/wheels/ || pip install natten -f https://shi-labs.com/natten/wheels/

echo "=== [5/6] Cài đặt Flash Attention ==="
pip install flash-attn --no-build-isolation || true

echo "=== [6/6] Cài đặt nvdiffrast ==="
pip install git+https://github.com/NVlabs/nvdiffrast.git || true

mkdir -p /workspace/models
mkdir -p /workspace/samples
mkdir -p /workspace/wheels_backup

echo "=== Setup hoàn tất! ==="
