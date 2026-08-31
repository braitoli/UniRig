import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Essential environment settings
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['ATTN_BACKEND'] = 'flash_attn'
os.environ['PATH'] = f"/home/braitoli/miniconda/envs/trellis/bin:{os.environ.get('PATH', '')}"

# Add trellis2 repository to sys.path
sys.path.insert(0, '/home/braitoli/workspace/namnh/code/poc/trellis2')

def preprocess_image_for_trellis(input_img: Image.Image) -> Image.Image:
    """Preprocess input image to centered foreground RGBA with bounding box crop."""
    has_alpha = False
    if input_img.mode == 'RGBA':
        alpha = np.array(input_img)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
            
    max_size = max(input_img.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input_img = input_img.resize((int(input_img.width * scale), int(input_img.height * scale)), Image.Resampling.LANCZOS)
        
    if has_alpha:
        output = input_img
    else:
        try:
            from rembg import remove
            output = remove(input_img)
        except Exception:
            # Fallback background removal using corner color estimation
            rgb = np.array(input_img.convert('RGB'))
            bg = rgb[0, 0].astype(np.float32)
            diff = np.linalg.norm(rgb.astype(np.float32) - bg, axis=-1)
            mask = (diff > 30).astype(np.uint8) * 255
            output = Image.fromarray(np.dstack([rgb, mask]))

    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) > 0:
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.05)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)
        
    output = np.array(output).astype(np.float32) / 255.0
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def main():
    parser = argparse.ArgumentParser(description="Run Microsoft TRELLIS.2-4B Image-to-3D Inference")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input 2D image")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output .glb file")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--resolution', type=str, default="1024", choices=["512", "1024", "1536"])
    parser.add_argument('--decimation_target', type=int, default=300000)
    parser.add_argument('--texture_size', type=int, default=4096)
    parser.add_argument('--tex_slat_steps', type=int, default=30,
                        help="texture SLAT sampler steps; the worker's own default is 30")
    parser.add_argument('--sparse_structure_steps', type=int, default=12)
    parser.add_argument('--shape_slat_steps', type=int, default=12)
    args = parser.parse_args()

    print(f"[TRELLIS.2] Loading Trellis2ImageTo3DPipeline (microsoft/TRELLIS.2-4B)...")
    t0 = time.time()
    
    import torch
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print(f"[TRELLIS.2] Pipeline loaded in {time.time() - t0:.2f}s")

    raw_img = Image.open(args.image_path)
    img = preprocess_image_for_trellis(raw_img)
    print(f"[TRELLIS.2] Preprocessed input image: {args.image_path} -> {img.size}")

    t1 = time.time()
    print(f"[TRELLIS.2] Running 4B neural flow-matching transformer diffusion...")
    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
    }[args.resolution]
    with torch.inference_mode():
        outputs, _ = pipeline.run(
            img,
            seed=args.seed,
            preprocess_image=False,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={
                "steps": args.sparse_structure_steps,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            shape_slat_sampler_params={
                "steps": args.shape_slat_steps,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            tex_slat_sampler_params={
                "steps": args.tex_slat_steps,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            return_latent=True,
        )
    print(f"[TRELLIS.2] Diffusion finished in {time.time() - t1:.2f}s")


    mesh = outputs[0] if isinstance(outputs, list) else outputs
    if hasattr(mesh, 'simplify'):
        mesh.simplify(16777216)

    print(f"[TRELLIS.2] Generating PBR textured mesh via o_voxel...")
    t2 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   pipeline.pbr_attr_layout if hasattr(pipeline, 'pbr_attr_layout') else mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   args.decimation_target,
        texture_size        =   args.texture_size,
        remesh              =   True,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_path), extension_webp=True)

    print(f"[TRELLIS.2] SUCCESS! Exported GLB to {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB) in {time.time() - t2:.2f}s")

if __name__ == '__main__':
    main()
