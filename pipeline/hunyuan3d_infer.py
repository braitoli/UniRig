"""Hunyuan3D-2.1 image-to-3D inference (shape + PBR paint), run as a subprocess.

Mirrors the official demo.py / gradio_app.py flow from the Hunyuan3D-2.1 repo:
shape via Hunyuan3DDiTFlowMatchingPipeline, then PBR texture via Hunyuan3DPaintPipeline.
Both stages need the repo on sys.path and its relative ckpt/cfg paths resolved from
the repo root, so we chdir there and keep only absolute paths for user-supplied I/O.
"""
import os
import sys
import time
import argparse
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

HUNYUAN_ROOT = Path(os.environ.get(
    'HUNYUAN3D_ROOT', '/home/braitoli/workspace/namnh/code/poc/Hunyuan3D-2.1'))

sys.path.insert(0, str(HUNYUAN_ROOT / 'hy3dshape'))
sys.path.insert(0, str(HUNYUAN_ROOT / 'hy3dpaint'))
sys.path.insert(0, str(HUNYUAN_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run Tencent Hunyuan3D-2.1 Image-to-3D Inference")
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--octree_resolution', type=int, default=256)
    parser.add_argument('--max_num_view', type=int, default=9, help="6..9; measured: 9 gives +16% sharpness on the back")
    parser.add_argument('--paint_resolution', type=int, default=768, choices=[512, 768])
    parser.add_argument('--skip_paint', action='store_true', help="shape only, no PBR texture")
    args = parser.parse_args()

    image_path = str(Path(args.image_path).resolve())
    out_path = Path(args.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Relative ckpt/cfg paths inside the repo only resolve from its root.
    os.chdir(str(HUNYUAN_ROOT))

    import torch
    from PIL import Image
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        from torchvision_fix import apply_fix
        apply_fix()
    except Exception as e:
        print(f"[Hunyuan3D] torchvision fix skipped: {e}")

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    print("[Hunyuan3D] Loading shape pipeline (tencent/Hunyuan3D-2.1)...", flush=True)
    t0 = time.time()
    shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
    print(f"[Hunyuan3D] Shape pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    image = Image.open(image_path).convert("RGBA")
    # A fully opaque image still has a background to strip.
    import numpy as np
    if np.all(np.array(image)[:, :, 3] == 255):
        print("[Hunyuan3D] Removing background...", flush=True)
        image = BackgroundRemover()(image.convert("RGB"))

    print(f"[Hunyuan3D] Generating shape (steps={args.steps}, "
          f"octree={args.octree_resolution})...", flush=True)
    t1 = time.time()
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(int(args.seed))
    mesh = shape_pipe(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        generator=generator,
    )[0]
    print(f"[Hunyuan3D] Shape done in {time.time() - t1:.1f}s "
          f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)", flush=True)

    if args.skip_paint:
        mesh.export(str(out_path))
        print(f"[Hunyuan3D] SUCCESS (shape only) -> {out_path} "
              f"({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return

    # Paint needs a mesh file on disk as input.
    shape_glb = out_path.parent / f"{out_path.stem}_shape.glb"
    mesh.export(str(shape_glb))

    # Free the shape model before loading the paint model.
    del shape_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    print(f"[Hunyuan3D] Loading PBR paint pipeline "
          f"(views={args.max_num_view}, res={args.paint_resolution})...", flush=True)
    t2 = time.time()
    conf = Hunyuan3DPaintConfig(args.max_num_view, args.paint_resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint_pipe = Hunyuan3DPaintPipeline(conf)
    print(f"[Hunyuan3D] Paint pipeline loaded in {time.time() - t2:.1f}s", flush=True)

    print("[Hunyuan3D] Painting PBR texture...", flush=True)
    t3 = time.time()
    # The paint pipeline always writes Wavefront OBJ + MTL + separate PBR jpgs,
    # whatever extension is asked for, so give it an .obj name and convert after.
    obj_path = out_path.parent / f"{out_path.stem}_painted.obj"
    painted = paint_pipe(
        mesh_path=str(shape_glb),
        image_path=image_path,
        output_mesh_path=str(obj_path),
    )
    print(f"[Hunyuan3D] Paint done in {time.time() - t3:.1f}s", flush=True)

    produced = Path(painted) if painted else obj_path
    if not produced.exists():
        raise RuntimeError(f"Paint pipeline produced no mesh at {produced}")

    print("[Hunyuan3D] Packing OBJ + PBR maps into GLB...", flush=True)
    convert_painted_obj_to_glb(produced, out_path)

    print(f"[Hunyuan3D] SUCCESS! Exported textured GLB to {out_path} "
          f"({out_path.stat().st_size / 1024 / 1024:.2f} MB) in {time.time() - t0:.1f}s total")


def convert_painted_obj_to_glb(obj_path: Path, out_path: Path):
    """Repack the paint pipeline's OBJ + map_Kd/map_Pm/map_Pr into a PBR GLB.

    glTF carries roughness in the green channel and metallic in blue of a single
    metallicRoughness texture, so the two grayscale maps have to be merged.
    """
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial

    mesh = trimesh.load(str(obj_path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    stem = obj_path.stem
    base_img = None
    for cand in (obj_path.parent / f"{stem}.jpg", obj_path.parent / f"{stem}.png"):
        if cand.exists():
            base_img = Image.open(cand).convert("RGB")
            break
    if base_img is None and isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        base_img = getattr(mesh.visual.material, "image", None)
    if base_img is None:
        raise RuntimeError(f"No base color map found next to {obj_path}")

    metal_p = obj_path.parent / f"{stem}_metallic.jpg"
    rough_p = obj_path.parent / f"{stem}_roughness.jpg"
    mr_img = None
    if metal_p.exists() and rough_p.exists():
        metal = Image.open(metal_p).convert("L")
        rough = Image.open(rough_p).convert("L")
        if rough.size != metal.size:
            rough = rough.resize(metal.size, Image.LANCZOS)
        mr_img = Image.merge("RGB", (Image.new("L", metal.size, 0), rough, metal))

    uv = mesh.visual.uv if isinstance(mesh.visual, trimesh.visual.TextureVisuals) else None
    if uv is None:
        raise RuntimeError("Painted mesh has no UVs")

    material = PBRMaterial(
        baseColorTexture=base_img,
        metallicRoughnessTexture=mr_img,
        metallicFactor=1.0 if mr_img is not None else 0.0,
        roughnessFactor=1.0,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.export(str(out_path), extension_webp=True)


if __name__ == '__main__':
    main()
