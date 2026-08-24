import os
import sys
import time
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from PIL import Image
import numpy as np

# Essential GPU Environment Settings
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['ATTN_BACKEND'] = 'flash_attn'
os.environ['PATH'] = f"/home/braitoli/miniconda/envs/trellis/bin:{os.environ.get('PATH', '')}"

# Add trellis2 repository to sys.path
TRELLIS_ROOT = Path('/home/braitoli/workspace/namnh/code/poc/trellis2')
if TRELLIS_ROOT.exists() and str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

import torch
# PyTorch TensorFloat-32 & Fast CUDA Acceleration
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

app = FastAPI(title="TRELLIS.2-4B Persistent GPU Worker Service")

# Global pipeline instance held resident in GPU VRAM
global_pipeline: Optional[Trellis2ImageTo3DPipeline] = None
is_pipeline_loading = False

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

def load_global_pipeline():
    global global_pipeline, is_pipeline_loading
    if global_pipeline is not None:
        return global_pipeline
    is_pipeline_loading = True
    print("[TRELLIS-Worker] Pre-loading Trellis2ImageTo3DPipeline (microsoft/TRELLIS.2-4B) into GPU VRAM...")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    global_pipeline = pipe
    is_pipeline_loading = False
    print(f"[TRELLIS-Worker] Model successfully resident in GPU VRAM ({time.time() - t0:.2f}s). Ready for fast instant inference!")
    return global_pipeline

@app.on_event("startup")
async def on_startup():
    import threading
    t = threading.Thread(target=load_global_pipeline, daemon=True)
    t.start()


@app.get("/health")
async def health_check():
    return {
        "status": "online" if global_pipeline is not None else ("loading" if is_pipeline_loading else "initializing"),
        "model": "microsoft/TRELLIS.2-4B",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0
    }

class GenerateRequest(BaseModel):
    image_path: str
    output_path: str
    seed: int = 42
    resolution: str = "1024"
    decimation_target: int = 300000
    texture_size: int = 2048

@app.post("/generate")
async def generate_3d(req: GenerateRequest):
    pipeline = load_global_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="TRELLIS pipeline is still loading into GPU VRAM")

    img_p = Path(req.image_path)
    if not img_p.exists():
        raise HTTPException(status_code=404, detail=f"Input image not found: {req.image_path}")

    t_start = time.time()
    raw_img = Image.open(str(img_p))
    img = preprocess_image_for_trellis(raw_img)

    t_diff = time.time()
    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
    }.get(req.resolution, "1024_cascade")

    with torch.inference_mode():
        outputs, _ = pipeline.run(
            img,
            seed=req.seed,
            preprocess_image=False,
            pipeline_type=pipeline_type,
            sparse_structure_sampler_params={
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            },
            shape_slat_sampler_params={
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            },
            tex_slat_sampler_params={
                "steps": 12,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            return_latent=True,
        )

    diff_time = round(time.time() - t_diff, 2)
    mesh = outputs[0] if isinstance(outputs, list) else outputs
    if hasattr(mesh, 'simplify'):
        mesh.simplify(16777216)

    t_post = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   pipeline.pbr_attr_layout if hasattr(pipeline, 'pbr_attr_layout') else mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   req.decimation_target,
        texture_size        =   req.texture_size,
        remesh              =   True,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   False
    )

    out_p = Path(req.output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_p), extension_webp=True)

    total_time = round(time.time() - t_start, 2)
    post_time = round(time.time() - t_post, 2)
    print(f"[TRELLIS-Worker] Done in {total_time}s (Diffusion: {diff_time}s, Mesh: {post_time}s) -> {out_p.stat().st_size / 1024 / 1024:.2f} MB")

    return {
        "status": "success",
        "output_path": str(out_p),
        "total_time_sec": total_time,
        "diffusion_time_sec": diff_time,
        "postprocess_time_sec": post_time,
        "size_bytes": out_p.stat().st_size
    }

if __name__ == '__main__':
    port = int(os.environ.get("TRELLIS_PORT", "7865"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
