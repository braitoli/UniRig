"""
Standalone Pixal3D Multi-View (MV) inference script for UniRig Pipeline.
Takes a views directory containing transforms.json and 4 canonical views,
runs the multi-view flow-matching diffusion cascade, and exports a clean PBR GLB mesh.
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Initialize CUDA context early
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Environment configuration
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
os.environ['PATH'] = f"/home/braitoli/miniconda/envs/trellis/bin:{os.environ.get('PATH', '')}"

PIXAL3D_ROOT = '/home/braitoli/workspace/namnh/code/poc/Pixal3D'
if PIXAL3D_ROOT not in sys.path:
    sys.path.insert(0, PIXAL3D_ROOT)

os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(PIXAL3D_ROOT, 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '0'

from pixal3d.pipelines import Pixal3DMVImageTo3DPipeline
import o_voxel

MODEL_PATH = "TencentARC/Pixal3D"
CONFIG_FILE = "pipeline_mv.json"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
        "multiview_fusion": "average",
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
        "multiview_fusion": "average",
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
        "multiview_fusion": "average",
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
        "multiview_fusion": "average",
    },
}


def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjMultiViewFeatureExtractor,
    )
    model = DinoV3ProjMultiViewFeatureExtractor(**config)
    model.eval()
    return model


def init_mv_pipeline(model_path=MODEL_PATH, config_file=CONFIG_FILE, device="cuda", low_vram=False):
    print(f"[Pixal3D-MV] Loading MV pipeline from {model_path} ({config_file})...")
    pipeline = Pixal3DMVImageTo3DPipeline.from_pretrained(model_path, config_file)

    print("[Pixal3D-MV] Building DinoV3ProjMultiViewFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    cond_attrs = ['image_cond_model_ss', 'image_cond_model_shape_512',
                  'image_cond_model_shape_1024', 'image_cond_model_tex_1024']

    if low_vram:
        for attr in cond_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pixal3D-MV] Low-VRAM mode enabled.")
    else:
        pipeline.low_vram = False
        pipeline.cuda()
        for attr in cond_attrs:
            getattr(pipeline, attr).cuda()
        print("[Pixal3D-MV] Pre-loading NAF upsampler...")
        for attr in cond_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pixal3D-MV] Standard mode (models on GPU).")

    return pipeline


def make_rembg(pipeline):
    def rembg(image: Image.Image) -> Image.Image:
        if getattr(pipeline, 'low_vram', False):
            pipeline.rembg_model.to(pipeline.device)
        output = pipeline.rembg_model(image.convert('RGB'))
        if getattr(pipeline, 'low_vram', False):
            pipeline.rembg_model.cpu()
        return output
    return rembg


def load_rgba(path: str, rembg=None):
    image = Image.open(path)
    alpha = np.array(image.getchannel(3)) if image.mode == 'RGBA' else None
    if alpha is not None and not np.all(alpha == 255):
        return image.convert('RGBA'), False
    if rembg is None:
        raise ValueError(f"{path} has no alpha channel and no matting model was given")
    return rembg(image).convert('RGBA'), True


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)


def load_views(views_dir: str, num_views: int = None, rembg=None, image_sizes=(512, 1024)) -> dict:
    import json
    with open(os.path.join(views_dir, 'transforms.json')) as f:
        meta = json.load(f)
    frames = meta['frames']
    if num_views is not None:
        if num_views > len(frames):
            raise ValueError(f"--num_views {num_views} > {len(frames)} views in {views_dir}")
        frames = frames[:num_views]

    def camera_angle_x_of(frame):
        for src in (frame, meta):
            if 'camera_angle_x' in src:
                return float(src['camera_angle_x'])
        raise KeyError(f"'camera_angle_x' missing for {frame.get('file_path')}")

    transform_matrix = torch.tensor([fr['transform_matrix'] for fr in frames],
                                    dtype=torch.float32)[None]              # [1, V, 4, 4]
    camera_angle_x = torch.tensor([camera_angle_x_of(fr) for fr in frames],
                                  dtype=torch.float32)[None]                # [1, V]
    camera_distance = torch.norm(transform_matrix[:, :, :3, 3], dim=-1)      # [1, V]

    paths = [os.path.join(views_dir, fr['file_path']) for fr in frames]
    loaded = [load_rgba(p, rembg) for p in paths]
    rgba = [im for im, _ in loaded]
    matted = sum(was_matted for _, was_matted in loaded)
    if matted:
        print(f"[Pixal3D-MV] Rembg matted {matted}/{len(paths)} view(s) that had no alpha channel")

    images = {
        size: torch.stack([to_cond_tensor(im, size) for im in rgba], dim=0)[None]
        for size in image_sizes
    }

    view_names = [fr.get('name', os.path.splitext(fr['file_path'])[0]) for fr in frames]
    mesh_scale = float(meta.get('mesh_scale', 1.0))
    print(f"[Pixal3D-MV] Loaded V={len(frames)} views ({', '.join(view_names)}) from {views_dir}")
    print(f"[Pixal3D-MV] Camera: fov={math.degrees(float(camera_angle_x[0, 0])):.2f}deg, "
          f"distance={float(camera_distance[0, 0]):.4f}, mesh_scale={mesh_scale:.4f}")

    return {
        'images': images,
        'camera_angle_x': camera_angle_x,
        'camera_distance': camera_distance,
        'transform_matrix': transform_matrix,
        'mesh_scale': mesh_scale,
        'view_names': view_names,
    }


def check_main_view(views: dict, atol: float = 1e-4):
    """
    Warn when frame 0 is not the canonical front view.

    Every view is mapped through calc_mat_i = F @ inv(C_0) @ C_i, so the main view is
    always snapped onto F. If C_0 is not itself a front view the whole rig ends up
    rotated relative to the object, and the mesh comes out in a different frame than
    the denoisers were trained for -- which shows up as a model facing the wrong way.
    """
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGridMV
    F = ProjGridMV(grid_resolution=2, image_resolution=64).front_view_transform_matrix.clone()
    F[1, 3] = -views['camera_distance'][0, 0]
    err = float((views['transform_matrix'][0, 0] - F).abs().max())
    if err > atol:
        print(f"[Pixal3D-MV] WARNING: main view (frame 0) is not the canonical front view "
              f"(max deviation {err:.3e}). The result will be posed in that view's frame.")
    else:
        print(f"[Pixal3D-MV] Main view == canonical front view (max err {err:.1e})")


def main():
    parser = argparse.ArgumentParser(description="Pixal3D Multi-View Inference")
    parser.add_argument("--views_dir", type=str, required=True, help="Directory with transforms.json and view images")
    parser.add_argument("--output_path", type=str, required=True, help="Destination .glb file")
    parser.add_argument("--num_views", type=int, default=4, help="Number of views (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    # Defaults match inference_mv.py in the Pixal3D repo, which is what the released
    # results were produced with; the playground overrides them per detail preset.
    parser.add_argument("--resolution", type=int, default=1536, help="Resolution (1024 or 1536)")
    parser.add_argument("--ss_sampling_steps", type=int, default=12)
    parser.add_argument("--shape_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_slat_steps", type=int, default=12)
    parser.add_argument("--decimation_target", type=int, default=1000000)
    parser.add_argument("--texture_size", type=int, default=4096)
    parser.add_argument("--low_vram", action="store_true", help="Enable low-VRAM mode")

    args = parser.parse_args()
    t0 = time.time()

    pipeline = init_mv_pipeline(low_vram=args.low_vram)
    views = load_views(args.views_dir, num_views=args.num_views, rembg=make_rembg(pipeline))
    check_main_view(views)

    print(f"[Pixal3D-MV] Running flow matching multi-view diffusion (resolution={args.resolution})...")
    torch.manual_seed(args.seed)

    ss_sampler_override = {
        "steps": args.ss_sampling_steps, "guidance_strength": 7.5,
        "guidance_rescale": 0.7, "rescale_t": 5.0,
    }
    shape_sampler_override = {
        "steps": args.shape_slat_sampling_steps, "guidance_strength": 7.5,
        "guidance_rescale": 0.5, "rescale_t": 3.0,
    }
    tex_sampler_override = {
        "steps": args.tex_slat_steps, "guidance_strength": 1.0,
        "guidance_rescale": 0.0, "rescale_t": 3.0,
    }

    pipeline_type = f"{args.resolution}_cascade"
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run_mv(
        views,
        seed=args.seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=49152,
    )

    mesh = mesh_list[0]
    print(f"[Pixal3D-MV] Decoding textured mesh (target {args.decimation_target} faces, tex {args.texture_size})...")

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target, texture_size=args.texture_size,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )

    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    out_p = Path(args.output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_p), extension_webp=True)

    size_mb = out_p.stat().st_size / (1024 * 1024)
    print(f"[Pixal3D-MV] SUCCESS! Saved multi-view GLB to {out_p} ({size_mb:.2f} MB) in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
