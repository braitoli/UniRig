#!/bin/bash
# ==============================================================================
# AUTOMATED SETUP SCRIPT FOR TRELLIS 2 & PIXAL3D BENCHMARK
# Target Environments: Vast.ai, RunPod, LambdaLabs, Cloud GPU Instances
# OS: Ubuntu 22.04+ (x86_64), PyTorch 2.4/2.6 with CUDA 12.4+
# ==============================================================================

set -e

echo "=================================================================="
echo "🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG TOÀN DIỆN CHO 3D STATUE STUDIO POC"
echo "=================================================================="

# 1. Kích hoạt Virtual Environment (nếu có)
if [ -d "/venv/main" ]; then
    echo "[+] Đang kích hoạt virtualenv /venv/main..."
    source /venv/main/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[+] Đang sử dụng active virtualenv: $VIRTUAL_ENV"
fi

# 2. Cài đặt các công cụ hệ thống cần thiết (apt)
echo "[+] Cài đặt build tools và thư viện đồ họa hệ thống..."
apt-get update -qq && apt-get install -y -qq \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    tmux \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev

# 3. Cài đặt các thư viện Python chuẩn
echo "[+] Cài đặt các thư viện xử lý 3D, Computer Vision và AI..."
pip install --upgrade pip setuptools wheel ninja
pip install \
    trimesh \
    pyvista \
    xatlas \
    vtk \
    scipy \
    rembg \
    diffusers \
    transformers \
    accelerate \
    huggingface_hub \
    pymeshlab \
    open3d \
    easydict \
    plyfile \
    zstandard \
    imageio \
    imageio-ffmpeg \
    pillow

# 4. Cài đặt nvdiffrast (bắt buộc cờ --no-build-isolation)
echo "[+] Biên dịch và cài đặt nvdiffrast..."
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

# 5. Cài đặt và build o_voxel, cumesh, flex_gemm
echo "[+] Biên dịch o_voxel và submodules CuMesh, FlexGEMM..."
WORKSPACE_DIR="/workspace"
if [ ! -d "$WORKSPACE_DIR/trellis2" ]; then
    echo "[!] Không tìm thấy $WORKSPACE_DIR/trellis2. Đang kiểm tra repo hiện tại..."
    if [ -d "./trellis2" ]; then
        WORKSPACE_DIR="."
    fi
fi

if [ -d "$WORKSPACE_DIR/trellis2/o-voxel" ]; then
    cd "$WORKSPACE_DIR/trellis2/o-voxel"
    pip install -e . --no-build-isolation
    cd -
fi

# 6. Cài đặt flash-attn
echo "[+] Cài đặt FlashAttention-2..."
pip install flash-attn --no-build-isolation || pip install flash-attn

# 7. Kiểm tra trạng thái các extensions
echo "=================================================================="
echo "🔍 KIỂM TRA TOÀN BỘ EXTENSIONS ĐÃ CÀI ĐẶT THÀNH CÔNG:"
python3 -c '
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

for mod in ["nvdiffrast", "o_voxel", "cumesh", "flex_gemm", "flash_attn", "trimesh", "pyvista"]:
    try:
        __import__(mod)
        print(f"  ✅ {mod:<15}: OK")
    except Exception as e:
        print(f"  ❌ {mod:<15}: FAILED ({e})")
'
echo "=================================================================="
echo "🎉 HOÀN TẤT CÀI ĐẶT MÔI TRƯỜNG!"
