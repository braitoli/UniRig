#!/bin/bash
# Run UniRig skeleton prediction on a single .glb or .obj file.
# Usage: bash run_skeleton.sh <input.glb> [output_dir]
#
# This wrapper uses a trimesh-based preprocessor instead of the original
# bpy-based extractor (bpy is not available for aarch64 Linux). It also
# disables the FBX export in the writer (the .obj skeleton is still produced).
set -e

PYTHON=/home/braitoli/miniconda/envs/unirig312/bin/python
# Disable user site-packages so a stray CPU-only torch in ~/.local doesn't
# shadow the CUDA torch that lives inside the env (which would break
# torchvision::nms dispatch at import time).
export PYTHONNOUSERSITE=1
INPUT="$1"
OUTPUT_DIR="${2:-results_skeleton}"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input.glb|.obj> [output_dir]"
    echo "Example (run from project root): $0 examples/giraffe.glb results/giraffe"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "ERROR: input file '$INPUT' not found"
    exit 1
fi

# run.py / get_files use the literal input string to compute the npz path,
# so we must pass the same string to the preprocessor — use a path relative
# to the project root, not an absolute one.
REL_INPUT="${INPUT#./}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Resolve to project-relative if an absolute path was given
case "$REL_INPUT" in
    /*) REL_INPUT="$(realpath --relative-to="$PROJECT_ROOT" "$REL_INPUT")" ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "Input:  $REL_INPUT"
echo "Output: $OUTPUT_DIR"

# Step 1: trimesh-based preprocessor (replaces bpy-based extract.sh)
echo ""
echo "=== Step 1/2: preprocessing (trimesh) ==="
rm -rf tmp_skel
$PYTHON preprocess_trimesh.py \
    --input "$REL_INPUT" \
    --output_dir tmp_skel \
    --faces_target_count 50000

# Step 2: skeleton inference
echo ""
echo "=== Step 2/2: skeleton inference ==="
$PYTHON run.py \
    --task=configs/task/quick_inference_skeleton_articulationxl_ar_256_nofbx.yaml \
    --input="$REL_INPUT" \
    --output_dir="$OUTPUT_DIR" \
    --npz_dir=tmp_skel \
    --seed=12345

echo ""
echo "=== Done ==="
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
