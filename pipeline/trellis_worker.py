import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image

# Ensure trellis env bin is in PATH for ninja/nvcc
os.environ['PATH'] = f"/home/braitoli/miniconda/envs/trellis/bin:{os.environ.get('PATH', '')}"
os.environ['CUMM_CUDA_ARCH_LIST'] = '8.6+PTX'
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'

# Add trellis-playground to path
sys.path.insert(0, '/home/braitoli/workspace/namnh/code/poc/trellis-playground')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()

    print(f"[TrellisWorker] Loading TRELLIS-image-large on GPU...")
    t0 = time.time()
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
    pipeline.cuda()
    print(f"[TrellisWorker] Pipeline loaded in {time.time() - t0:.2f}s")

    img = Image.open(args.image_path).convert('RGBA')
    print(f"[TrellisWorker] Input image size: {img.size}")

    print(f"[TrellisWorker] Generating 3D representation via neural diffusion...")
    t1 = time.time()
    outputs = pipeline.run(
        img,
        seed=args.seed,
        sparse_structure_sampler_params={'steps': args.steps, 'cfg_strength': 7.5},
        slat_sampler_params={'steps': args.steps, 'cfg_strength': 3.0},
    )
    print(f"[TrellisWorker] Neural inference completed in {time.time() - t1:.2f}s")

    print(f"[TrellisWorker] Post-processing to GLB mesh & textures...")
    t2 = time.time()
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    
    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_file))
    print(f"[TrellisWorker] Exported {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MB) in {time.time() - t2:.2f}s")

if __name__ == '__main__':
    main()
