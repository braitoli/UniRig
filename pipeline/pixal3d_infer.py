import os
import sys
import math
import time
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Essential environment settings
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
os.environ['PATH'] = f"/home/braitoli/miniconda/envs/trellis/bin:{os.environ.get('PATH', '')}"

PIXAL3D_ROOT = '/home/braitoli/workspace/namnh/code/poc/Pixal3D'
if PIXAL3D_ROOT not in sys.path:
    sys.path.insert(0, PIXAL3D_ROOT)

os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(PIXAL3D_ROOT, 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '0'

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
DEFAULT_MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model

def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())

def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}

def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

def init_pipeline(model_path=DEFAULT_MODEL_PATH, device="cuda", low_vram=False):
    print(f"[Pixal3D] Loading pipeline from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[Pixal3D] Building DinoV3 feature extractors...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if low_vram:
        print("[Pixal3D] Pre-downloading NAF upsampler weights (CPU mode)...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pixal3D] Low-VRAM mode enabled.")
    else:
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pixal3D] Standard mode (all models loaded on GPU).")

    return pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Tencent ARC Pixal3D Image-to-3D Inference")
    parser.add_argument('--image_path', type=str, required=True, help="Input 2D image path")
    parser.add_argument('--output_path', type=str, required=True, help="Output .glb path")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--resolution', type=str, default="1024", choices=["512", "1024", "1536"])
    parser.add_argument('--low_vram', action="store_true", help="Enable low-VRAM mode")
    parser.add_argument('--fov', type=float, default=-1.0, help="Manual camera FOV in rad (-1 for MoGe-2 auto)")
    parser.add_argument('--ss_sampling_steps', type=int, default=12)
    parser.add_argument('--shape_slat_sampling_steps', type=int, default=12)
    parser.add_argument('--tex_slat_steps', type=int, default=12)
    parser.add_argument('--decimation_target', type=int, default=300000)
    parser.add_argument('--texture_size', type=int, default=4096)
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    t0 = time.time()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Initialize Pixal3D pipeline
    pipeline = init_pipeline(args.model_path, low_vram=args.low_vram)

    # 2. Preprocess input image
    print(f"[Pixal3D] Preprocessing input image: {args.image_path}")
    raw_img = Image.open(args.image_path)
    img_preprocessed = pipeline.preprocess_image(raw_img)

    # 3. Camera parameters.
    # MoGe-2 must see the *preprocessed* image, not the original one: the FOV it returns
    # is derived from fx_normalized * width, so running it on the uncropped frame yields
    # the camera of a picture the denoisers never see, and the shape is then solved under
    # a perspective that does not match its conditioning. Both inference.py and the HF
    # Space estimate the camera after preprocessing, for this reason.
    camera_params = None
    if args.fov > 0:
        camera_angle_x = float(args.fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        dist_x = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0, 511]),
            mesh_scale=1.0, image_resolution=512
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': dist_x, 'mesh_scale': 1.0}
        print(f"[Pixal3D] Manual FOV: {math.degrees(camera_angle_x):.2f}° ({camera_angle_x:.4f} rad), distance={dist_x:.4f}")
    else:
        tmp_dir = Path(args.output_path).parent
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"_tmp_preprocessed_{int(time.time() * 1000)}.png"
        img_preprocessed.save(str(tmp_path))
        try:
            print("[Pixal3D] Estimating camera parameters via MoGe-2...")
            moge_model = load_moge_model(device="cuda")
            camera_params = get_camera_params_wild_moge(
                str(tmp_path), moge_model, device="cuda",
                mesh_scale=1.0, extend_pixel=0, image_resolution=512
            )
            print(f"[Pixal3D] Estimated camera: angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
            moge_model.cpu()
            del moge_model
        except Exception as e:
            print(f"[Pixal3D] Warning: MoGe camera estimation failed ({e}), falling back to default FOV 49.1°")
            camera_angle_x = 0.8575560450553894
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            dist_x = distance_from_fov(
                camera_angle_x, grid_point,
                torch.tensor([0, 511]),
                mesh_scale=1.0, image_resolution=512
            )["distance_from_x"]
            camera_params = {'camera_angle_x': camera_angle_x, 'distance': dist_x, 'mesh_scale': 1.0}
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    torch.manual_seed(args.seed)

    ss_sampler = {
        "steps": args.ss_sampling_steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    shape_sampler = {
        "steps": args.shape_slat_sampling_steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    tex_sampler = {
        "steps": args.tex_slat_steps,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }

    pipeline_type = f"{args.resolution}_cascade" if args.resolution in ["1024", "1536"] else "1024_cascade"
    print(f"[Pixal3D] Running neural flow-matching transformer diffusion ({pipeline_type})...")
    t1 = time.time()

    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        img_preprocessed,
        camera_params=camera_params,
        seed=args.seed,
        sparse_structure_sampler_params=ss_sampler,
        shape_slat_sampler_params=shape_sampler,
        tex_slat_sampler_params=tex_sampler,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=49152,
    )
    print(f"[Pixal3D] Diffusion completed in {time.time() - t1:.2f}s")

    mesh = mesh_list[0]
    print(f"[Pixal3D] Generating PBR textured mesh via o_voxel (target {args.decimation_target} faces, tex {args.texture_size})...")
    t2 = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    # Coordinate transform: camera space to 3D mesh space
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    out_file = Path(args.output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_file), extension_webp=True)

    print(f"[Pixal3D] SUCCESS! Saved GLB to {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MB) in {time.time() - t2:.2f}s (total: {time.time() - t0:.2f}s)")

if __name__ == '__main__':
    main()
