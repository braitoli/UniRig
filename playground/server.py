import os
import sys
import re
import time
import socket
import shutil
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import uvicorn
import trimesh
import zipfile
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure root dir in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.unirig_pipeline import UniRigPipeline
from pipeline.statue_pipeline import Statue3DPipeline
from pipeline.statue_optimizer import STATUE_PALETTE
from playground import database
from playground.automation_service import StatueAutomationService, test_webhook_endpoint

app = FastAPI(title="UniRig 3D Rigging & Statue Painting Playground")

# Enable CORS for LAN access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path(__file__).resolve().parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/examples", StaticFiles(directory=str(ROOT_DIR / "examples")), name="examples")

PAINTING_DIST_DIR = ROOT_DIR / "3DPainting" / "dist"
if PAINTING_DIST_DIR.exists():
    app.mount("/3dpainting", StaticFiles(directory=str(PAINTING_DIST_DIR), html=True), name="3dpainting")

database.init_db()
pipeline_runner = UniRigPipeline(root_dir=str(ROOT_DIR))
statue_runner = Statue3DPipeline(root_dir=str(ROOT_DIR))
automation_service = StatueAutomationService(server_port=7860)


def ensure_trellis_worker_background():
    trellis_python = "/home/braitoli/miniconda/envs/trellis/bin/python"
    worker_script = ROOT_DIR / "pipeline" / "trellis_worker_service.py"
    if Path(trellis_python).exists() and worker_script.exists():
        try:
            import requests
            r = requests.get("http://127.0.0.1:7865/health", timeout=0.8)
            if r.status_code == 200:
                print("⚡ Persistent TRELLIS GPU worker is already online on port 7865.")
                return
        except Exception:
            pass
        import subprocess
        print("⚡ Auto-starting persistent TRELLIS GPU worker on port 7865...")
        subprocess.Popen([trellis_python, str(worker_script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

@app.on_event("startup")
async def server_startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, ensure_trellis_worker_background)
    cfg = database.get_automation_config()
    if cfg.get("enabled", False):
        automation_service.start()


def get_lan_ips() -> List[str]:
    """Find all accessible IPv4 LAN addresses."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass

    import subprocess
    try:
        out = subprocess.check_output(["ip", "-4", "addr"], text=True)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("inet ") and not line.startswith("inet 127."):
                ip = line.split()[1].split("/")[0]
                if not ip.startswith("172."): # Skip docker bridge networks
                    ips.add(ip)
    except Exception:
        pass
        
    return sorted(list(ips))

def run_job_background(job_id: str):
    """Executes the 5 pipeline stages (Stage 0 Image-to-3D if 2D image -> Stage 1..4) in background and updates DB state."""
    job = database.get_job(job_id)
    if not job:
        return
        
    t_start = time.time()
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = job["input_file_path"]
    metadata = job.get("metadata", {})
    
    ext = Path(input_file).suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    current_3d_input = input_file
    
    try:
        if ext in image_exts:
            # Stage 0: 2D Image to 3D Generation (TRELLIS.2-4B / Tencent Hunyuan3D-2.1)
            generator_choice = metadata.get("generator", "trellis")
            gen_folder = "stage0_hunyuan3d" if "hunyuan" in generator_choice.lower() else "stage0_trellis"
            
            def on_stage0_progress(pct: int, step_name: str, step_idx: int = 1, total_steps: int = 5):
                metadata["progress"] = {
                    "pct": pct,
                    "step_name": step_name,
                    "step_idx": step_idx,
                    "total_steps": total_steps
                }
                database.update_job(job_id, status="processing_image_to_3d", stage=0, metadata=metadata)

            on_stage0_progress(10, "Bắt đầu tiến trình tạo 3D từ ảnh 2D...", 1, 5)
            stage0_res = pipeline_runner.generate_3d_from_image(
                image_path=input_file,
                output_dir=str(job_dir / gen_folder),
                generator_type=generator_choice,
                mesh_detail=metadata.get("mesh_detail"),
                texture_detail=metadata.get("texture_detail"),
                progress_callback=on_stage0_progress
            )
            current_3d_input = stage0_res["output_glb_path"]
            metadata["stage0"] = {
                "generated_glb": current_3d_input,
                "generation_time_sec": stage0_res["generation_time_sec"],
                "model_used": stage0_res["model_used"],
                "generator_type": generator_choice,
                "mesh_detail": stage0_res.get("mesh_detail"),
                "texture_detail": stage0_res.get("texture_detail")
            }
            metadata["progress"] = {
                "pct": 100,
                "step_name": "Đã tạo xong Model 3D từ Ảnh 2D!",
                "step_idx": 5,
                "total_steps": 5
            }
            
            # Check if mode is 3d_only (only generate 3D GLB without auto-rigging)
            job_mode = metadata.get("mode", "3d_only")
            if job_mode == "3d_only":
                t_end = time.time()
                total_duration = round(t_end - t_start, 2)
                database.update_job(
                    job_id,
                    status="completed_3d_only",
                    stage=0,
                    duration_sec=total_duration,
                    metadata=metadata
                )
                return

        # Stage 1: Preprocess
        metadata["progress"] = {
            "pct": 20,
            "step_name": "Giai đoạn 1/4: Chuẩn hóa Mesh, tính toán Normals bề mặt...",
            "step_idx": 1,
            "total_steps": 4
        }
        database.update_job(job_id, status="processing_prep", stage=1, metadata=metadata)
        prep_res = pipeline_runner.preprocess_mesh(
            input_path=current_3d_input,
            output_dir=str(job_dir / "stage1_prep")
        )
        metadata["prep"] = {
            "num_vertices": prep_res["num_vertices"],
            "num_faces": prep_res["num_faces"],
            "norm_glb": str(prep_res["norm_glb_path"])
        }
        database.update_job(
            job_id,
            num_vertices=prep_res["num_vertices"],
            num_faces=prep_res["num_faces"],
            metadata=metadata
        )
        
        # Stage 2: Skeleton Prediction
        metadata["progress"] = {
            "pct": 45,
            "step_name": "Giai đoạn 2/4: Dự đoán hệ khớp xương UniRig AR Transformer...",
            "step_idx": 2,
            "total_steps": 4
        }
        database.update_job(job_id, status="processing_skeleton", stage=2, metadata=metadata)
        skel_res = pipeline_runner.predict_skeleton(
            input_mesh_path=current_3d_input,
            npz_dir=str(job_dir / "stage1_prep"),
            output_dir=str(job_dir / "stage2_skel")
        )
        metadata["skel"] = {
            "num_bones": skel_res["num_bones"],
            "tree": skel_res["tree"],
            "names": skel_res["names"],
            "inference_time_sec": skel_res["inference_time_sec"]
        }
        database.update_job(
            job_id,
            num_bones=skel_res["num_bones"],
            metadata=metadata
        )
        
        # Stage 3: Skinning Weights & Heatmaps
        metadata["progress"] = {
            "pct": 75,
            "step_name": "Giai đoạn 3/4: Tính toán ma trận trọng số gán da Laplacian Diffusion...",
            "step_idx": 3,
            "total_steps": 4
        }
        database.update_job(job_id, status="processing_skin", stage=3, metadata=metadata)
        skin_res = pipeline_runner.predict_skin(
            vertices=prep_res["vertices"],
            faces=prep_res["faces"],
            joints=skel_res["joints"],
            parents=skel_res["parents"],
            names=skel_res["names"],
            output_dir=str(job_dir / "stage3_skin"),
            use_neural=True,
            input_mesh_path=current_3d_input,
            skel_stage_dir=skel_res["skel_npz_path"]
        )
        metadata["skin"] = {
            "bone_stats": skin_res["bone_stats"],
            "calc_time_sec": skin_res["calc_time_sec"]
        }
        database.update_job(job_id, metadata=metadata)
        
        # Stage 4: Rigged GLB & Animations
        metadata["progress"] = {
            "pct": 92,
            "step_name": "Giai đoạn 4/4: Tạo chuyển động Mocap Retargeting & xuất file Rigged GLB...",
            "step_idx": 4,
            "total_steps": 4
        }
        database.update_job(job_id, status="processing_rig", stage=4, metadata=metadata)
        rigged_glb_path = str(job_dir / f"{prep_res['stem']}_rigged_animated.glb")
        rig_res = pipeline_runner.export_rigged_and_animated(
            vertices=prep_res["vertices"],
            faces=prep_res["faces"],
            joints=skel_res["joints"],
            parents=skel_res["parents"],
            skin_weights=skin_res["weights"],
            normals=prep_res["normals"],
            names=skel_res["names"],
            colors=prep_res.get("colors"),
            uvs=prep_res.get("uvs"),
            base_color_texture=prep_res.get("base_color_texture"),
            metallic_roughness_texture=prep_res.get("metallic_roughness_texture"),
            output_glb_path=rigged_glb_path
        )

        
        t_end = time.time()
        total_duration = round(t_end - t_start, 2)
        
        metadata["rig"] = {
            "glb_path": rigged_glb_path,
            "glb_size_bytes": rig_res["glb_size_bytes"],
            "animations": rig_res["animations"],
            "blendshapes": rig_res.get("blendshapes", []),
            "export_time_sec": rig_res["export_time_sec"],
            "total_duration_sec": total_duration
        }
        metadata["progress"] = {
            "pct": 100,
            "step_name": "Hoàn thành toàn bộ Pipeline!",
            "step_idx": 4,
            "total_steps": 4
        }
        
        database.update_job(
            job_id,
            status="completed",
            stage=4,
            duration_sec=total_duration,
            metadata=metadata
        )

        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error in job {job_id}:\n{tb}")
        database.update_job(
            job_id,
            status="failed",
            error_message=str(e),
            metadata={"traceback": tb}
        )

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(
        content=index_file.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )


@app.get("/api/info")
async def get_system_info():
    lan_ips = get_lan_ips()
    port = 7860
    urls = [f"http://{ip}:{port}" for ip in lan_ips] + [f"http://localhost:{port}"]
    return {
        "server_status": "online",
        "lan_ips": lan_ips,
        "urls": urls,
        "primary_url": urls[0] if urls else f"http://localhost:{port}",
        "port": port
    }

@app.get("/api/examples")
async def list_examples():
    examples_dir = ROOT_DIR / "examples"
    items = []
    if examples_dir.exists():
        for f in sorted(examples_dir.glob("*.glb")):
            items.append({
                "id": f.stem,
                "name": f.name,
                "path": str(f.relative_to(ROOT_DIR)),
                "size_kb": round(f.stat().st_size / 1024, 1)
            })
    return items

@app.get("/api/jobs")
async def get_all_jobs():
    return JSONResponse(
        content=database.list_jobs(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )


@app.post("/api/jobs/preset")
async def create_preset_job(
    preset_name: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    examples_dir = ROOT_DIR / "examples"
    preset_file = examples_dir / f"{preset_name}.glb"
    if not preset_file.exists():
        preset_file = examples_dir / preset_name
    if not preset_file.exists():
        raise HTTPException(status_code=404, detail=f"Preset {preset_name} not found")
        
    job_id = f"job_{int(time.time())}_{preset_name}"
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    target_input = job_dir / preset_file.name
    shutil.copyfile(str(preset_file), str(target_input))
    
    job = database.create_job(
        job_id=job_id,
        title=f"Preset: {preset_file.stem}",
        input_filename=preset_file.name,
        input_file_path=str(target_input)
    )
    
    background_tasks.add_task(run_job_background, job_id)
    return job

@app.post("/api/jobs/upload")
async def upload_custom_job(
    file: UploadFile = File(...),
    mode: str = Form("3d_only"),
    generator: str = Form("trellis"),
    mesh_detail: str = Form("high"),
    texture_detail: str = Form("high"),
    background_tasks: BackgroundTasks = None
):
    stem = Path(file.filename).stem
    clean_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', stem)
    job_id = f"job_{int(time.time())}_{clean_stem}"
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    clean_suffix = Path(file.filename).suffix.lower()
    clean_filename = f"{clean_stem}{clean_suffix}"
    target_input = job_dir / clean_filename
    with open(target_input, "wb") as f:
        content = await file.read()
        f.write(content)
        
    job = database.create_job(
        job_id=job_id,
        title=f"Custom: {file.filename}",
        input_filename=clean_filename,
        input_file_path=str(target_input),
        metadata={"mode": mode, "generator": generator,
                  "mesh_detail": mesh_detail, "texture_detail": texture_detail,
                  "original_filename": file.filename}
    )
    
    background_tasks.add_task(run_job_background, job_id)
    return job

@app.post("/api/jobs/{job_id}/continue_rigging")
async def continue_rigging_endpoint(
    job_id: str,
    background_tasks: BackgroundTasks = None
):
    """Triggers Stage 1..4 (Rig & Motion) for a job that completed Stage 0 (3D Generation) or has a 3D model."""
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    metadata = job.get("metadata", {})
    stage0_meta = metadata.get("stage0", {})
    current_3d = stage0_meta.get("generated_glb")
    job_dir = STORAGE_DIR / job_id
    
    # Flexible fallbacks to find current 3D mesh
    if not current_3d or not Path(current_3d).exists():
        matches0 = list(job_dir.glob("stage0_*/*_generated_3d.glb"))
        if matches0 and matches0[0].exists():
            current_3d = str(matches0[0])
        else:
            matches1 = list(job_dir.glob("stage1_prep/*_input.glb"))
            if matches1 and matches1[0].exists():
                current_3d = str(matches1[0])
            else:
                input_p = Path(job["input_file_path"])
                if input_p.exists() and input_p.suffix.lower() in [".glb", ".gltf", ".obj"]:
                    current_3d = str(input_p)
                else:
                    raise HTTPException(status_code=400, detail="Không tìm thấy 3D model để tiếp tục Rigging")
        
    database.update_job(job_id, status="processing_prep", stage=1)
    
    # Define worker to run stage 1..4 in background
    def run_continue_worker():
        t_start = time.time()
        job_dir = STORAGE_DIR / job_id
        try:
            # Stage 1
            metadata["progress"] = {
                "pct": 20,
                "step_name": "Giai đoạn 1/4: Chuẩn hóa Mesh, tính toán Normals bề mặt...",
                "step_idx": 1,
                "total_steps": 4
            }
            database.update_job(job_id, status="processing_prep", stage=1, metadata=metadata)
            prep_res = pipeline_runner.preprocess_mesh(
                input_path=current_3d,
                output_dir=str(job_dir / "stage1_prep")
            )
            metadata["prep"] = {
                "num_vertices": prep_res["num_vertices"],
                "num_faces": prep_res["num_faces"],
                "norm_glb": str(prep_res["norm_glb_path"])
            }
            database.update_job(job_id, num_vertices=prep_res["num_vertices"], num_faces=prep_res["num_faces"], metadata=metadata)

            # Stage 2
            metadata["progress"] = {
                "pct": 45,
                "step_name": "Giai đoạn 2/4: Dự đoán hệ khớp xương UniRig AR Transformer...",
                "step_idx": 2,
                "total_steps": 4
            }
            database.update_job(job_id, status="processing_skeleton", stage=2, metadata=metadata)
            skel_res = pipeline_runner.predict_skeleton(
                input_mesh_path=current_3d,
                npz_dir=str(job_dir / "stage1_prep"),
                output_dir=str(job_dir / "stage2_skel")
            )
            metadata["skel"] = {
                "num_bones": skel_res["num_bones"],
                "tree": skel_res["tree"],
                "names": skel_res["names"],
                "inference_time_sec": skel_res["inference_time_sec"]
            }
            database.update_job(job_id, num_bones=skel_res["num_bones"], metadata=metadata)

            # Stage 3
            metadata["progress"] = {
                "pct": 75,
                "step_name": "Giai đoạn 3/4: Tính toán ma trận trọng số gán da Laplacian Diffusion...",
                "step_idx": 3,
                "total_steps": 4
            }
            database.update_job(job_id, status="processing_skin", stage=3, metadata=metadata)
            skin_res = pipeline_runner.predict_skin(
                vertices=prep_res["vertices"],
                faces=prep_res["faces"],
                joints=skel_res["joints"],
                parents=skel_res["parents"],
                names=skel_res["names"],
                output_dir=str(job_dir / "stage3_skin"),
                use_neural=True,
                input_mesh_path=current_3d,
                skel_stage_dir=skel_res["skel_npz_path"]
            )
            metadata["skin"] = {
                "bone_stats": skin_res["bone_stats"],
                "calc_time_sec": skin_res["calc_time_sec"]
            }
            database.update_job(job_id, metadata=metadata)

            # Stage 4
            metadata["progress"] = {
                "pct": 92,
                "step_name": "Giai đoạn 4/4: Tạo chuyển động Mocap Retargeting & xuất file Rigged GLB...",
                "step_idx": 4,
                "total_steps": 4
            }
            database.update_job(job_id, status="processing_rig", stage=4, metadata=metadata)
            rigged_glb_path = str(job_dir / f"{prep_res['stem']}_rigged_animated.glb")
            rig_res = pipeline_runner.export_rigged_and_animated(
                vertices=prep_res["vertices"],
                faces=prep_res["faces"],
                joints=skel_res["joints"],
                parents=skel_res["parents"],
                skin_weights=skin_res["weights"],
                normals=prep_res["normals"],
                names=skel_res["names"],
                colors=prep_res.get("colors"),
                uvs=prep_res.get("uvs"),
                base_color_texture=prep_res.get("base_color_texture"),
                metallic_roughness_texture=prep_res.get("metallic_roughness_texture"),
                output_glb_path=rigged_glb_path
            )

            t_end = time.time()
            total_duration = round(t_end - t_start, 2)
            metadata["rig"] = {
                "glb_path": rigged_glb_path,
                "glb_size_bytes": rig_res["glb_size_bytes"],
                "animations": rig_res["animations"],
                "blendshapes": rig_res.get("blendshapes", []),
                "export_time_sec": rig_res["export_time_sec"],
                "total_duration_sec": total_duration
            }
            metadata["progress"] = {
                "pct": 100,
                "step_name": "Hoàn thành toàn bộ Pipeline!",
                "step_idx": 4,
                "total_steps": 4
            }
            database.update_job(job_id, status="completed", stage=4, duration_sec=total_duration, metadata=metadata)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error continuing rigging for job {job_id}:\n{tb}")
            database.update_job(job_id, status="failed", error_message=str(e), metadata={"traceback": tb})

    background_tasks.add_task(run_continue_worker)
    return {"status": "started", "job_id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(
        content=job,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )


@app.post("/api/jobs/{job_id}/reanimate")
async def reanimate_job_endpoint(
    job_id: str,
    use_neural_pan: bool = True,
    bvh_file_path: Optional[str] = None
):
    """
    Fast re-generation of Stage 4 animations (takes < 0.2s for kinematic, ~1-2s for Neural PAN)
    without re-running Neural mesh & skinning models.

    Defaults to the mocap path, matching what the full pipeline runs. Defaulting to the
    procedural one meant re-animating a job silently downgraded its motion -- and skipped the
    Nod/HeadShake clips, which only the mocap path produces.
    """
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_dir = STORAGE_DIR / job_id
    prep_npz = job_dir / "stage1_prep/raw_data.npz"
    skin_npz = job_dir / "stage3_skin/skin_weights.npz"
    
    if not prep_npz.exists() or not skin_npz.exists():
        raise HTTPException(status_code=400, detail="Missing cached prep/skin stage files for fast re-animation")
        
    try:
        t0 = time.time()
        prep_data = np.load(str(prep_npz), allow_pickle=True)
        vertices = prep_data["vertices"]
        faces = prep_data["faces"]
        normals = prep_data.get("vertex_normals", None)
        colors = prep_data.get("colors", None)
        
        skin_data = np.load(str(skin_npz), allow_pickle=True)
        skin_weights = skin_data["weights"]
        names = skin_data["names"].tolist()
        
        # Load joints and parents from skin_npz, stage2_skel, or DB metadata
        if "joints" in skin_data and len(skin_data["joints"]) > 0:
            joints = skin_data["joints"]
            parents = [None if (p is None or p < 0) else int(p) for p in skin_data["parents"]]
        else:
            skel_matches = list((job_dir / "stage2_skel").glob("**/predict_skeleton.npz"))
            if skel_matches:
                skel_data = np.load(str(skel_matches[0]), allow_pickle=True)
                joints = skel_data["joints"]
                parents = [None if (p is None or p < 0) else int(p) for p in skel_data["parents"]]
        if len(joints) == 0:
            raise HTTPException(status_code=400, detail="Job này được tạo từ phiên bản cũ chưa lưu cache Skeleton. Vui lòng nhấn 'Chạy Full Pipeline' một lần để lưu cache.")

        # Unnormalize joints if still in raw [-1, 1] model space
        b_min = vertices.min(axis=0)
        b_max = vertices.max(axis=0)
        center = (b_max + b_min) / 2.0
        scale = np.max(b_max - b_min) / 2.0
        j_span = joints.max(axis=0) - joints.min(axis=0)
        if np.max(j_span) <= 2.5 and joints.min(axis=0)[1] < -0.2 and b_min[1] >= -0.1:
            joints = joints * scale + center
            
        # Overwrite the GLB the pipeline actually produced, rather than deriving a name from
        # the uploaded file. Those two disagree whenever stage 0 generated the mesh: the
        # upload is "test15.png" but the pipeline works on "test15_generated_3d", so this
        # endpoint used to write a second, differently-named GLB beside the original and the
        # viewer went on serving the stale one.
        stem = Path(job["input_filename"]).stem
        rigged_glb_path = (job.get("metadata", {}).get("rig", {}).get("glb_path")
                           or str(job_dir / f"{stem}_rigged_animated.glb"))

        # raw_data.npz cannot hold images, so recover UV + PBR atlas from the cached stage1 GLB;
        # without this a re-animation would silently drop the texture.
        uvs = None
        base_color_texture = None
        metallic_roughness_texture = None
        prep_glb = next((job_dir / "stage1_prep").glob("*_input.glb"), None)
        if prep_glb is not None:
            try:
                import trimesh
                pm = trimesh.load(str(prep_glb), force="mesh", process=False)
                if isinstance(pm, trimesh.Scene):
                    pm = pm.dump(concatenate=True)
                if (isinstance(pm.visual, trimesh.visual.TextureVisuals)
                        and pm.visual.uv is not None
                        and len(pm.visual.uv) == len(vertices)):
                    uvs = np.asarray(pm.visual.uv, dtype=np.float32)
                    base_color_texture = getattr(pm.visual.material, "baseColorTexture", None)
                    metallic_roughness_texture = getattr(
                        pm.visual.material, "metallicRoughnessTexture", None)
            except Exception as e:
                print(f"[server] Could not recover texture from {prep_glb}: {e}")

        rig_res = pipeline_runner.export_rigged_and_animated(
            vertices=vertices,
            faces=faces,
            joints=joints,
            parents=parents,
            skin_weights=skin_weights,
            normals=normals,
            names=names,
            colors=colors,
            uvs=uvs,
            base_color_texture=base_color_texture,
            metallic_roughness_texture=metallic_roughness_texture,
            output_glb_path=rigged_glb_path,
            use_pan_retargeting=True,
            use_neural_pan=use_neural_pan,
            bvh_file_path=bvh_file_path
        )
        t1 = time.time()
        
        metadata = job.get("metadata", {})
        metadata["rig"] = {
            "glb_path": rigged_glb_path,
            "glb_size_bytes": rig_res["glb_size_bytes"],
            "animations": rig_res["animations"],
            "blendshapes": rig_res.get("blendshapes", []),
            "export_time_sec": round(t1 - t0, 3)
        }
        database.update_job(job_id, status="completed", metadata=metadata)
        
        return {
            "status": "success",
            "job_id": job_id,
            "reanimation_time_sec": round(t1 - t0, 3),
            "glb_size_bytes": rig_res["glb_size_bytes"]
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error reanimating job {job_id}:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Failed to reanimate: {e}")

@app.delete("/api/jobs/{job_id}")
async def delete_job_endpoint(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    database.delete_job(job_id)
    job_dir = STORAGE_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(str(job_dir), ignore_errors=True)
    return {"status": "deleted", "job_id": job_id}

@app.api_route("/api/jobs/{job_id}/files/generated_3d_glb", methods=["GET", "HEAD"])
async def get_generated_3d_glb(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = STORAGE_DIR / job_id
    matches = list(job_dir.glob("stage0_*/*_generated_3d.glb"))
    if matches and matches[0].exists():
        return FileResponse(
            str(matches[0]),
            media_type="model/gltf-binary",
            filename=f"{Path(job['input_filename']).stem}_3d.glb"
        )
    matches1 = list(job_dir.glob("stage1_prep/*_input.glb"))
    if matches1 and matches1[0].exists():
        return FileResponse(
            str(matches1[0]),
            media_type="model/gltf-binary",
            filename=f"{Path(job['input_filename']).stem}_3d.glb"
        )
    input_p = Path(job["input_file_path"])
    if input_p.exists() and input_p.suffix.lower() in [".glb", ".gltf"]:
        return FileResponse(
            str(input_p),
            media_type="model/gltf-binary",
            filename=f"{Path(job['input_filename']).stem}.glb"
        )
    raise HTTPException(status_code=404, detail="Generated 3D GLB file not found")

@app.api_route("/api/jobs/{job_id}/files/input_image", methods=["GET", "HEAD"])
async def get_input_image(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    input_p = Path(job["input_file_path"])
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if input_p.exists() and input_p.suffix.lower() in image_exts:
        media_type = f"image/{input_p.suffix.lower().removeprefix('.')}"
        if media_type == "image/jpg":
            media_type = "image/jpeg"
        return FileResponse(str(input_p), media_type=media_type)
        
    raise HTTPException(status_code=404, detail="Original 2D image file not found")

@app.api_route("/api/jobs/{job_id}/files/input_model", methods=["GET", "HEAD"])
async def get_input_model(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = STORAGE_DIR / job_id

    # Prioritize Stage 0 AI Generated 3D GLB (with high-res PBR textures from TRELLIS)
    stage0_matches = list(job_dir.glob("stage0_*/*_generated_3d.glb"))
    if stage0_matches and stage0_matches[0].exists():
        return FileResponse(str(stage0_matches[0]), media_type="model/gltf-binary")

    # Look for normalized glb in stage1_prep
    norm_matches = list(job_dir.glob("stage1_prep/*_input.glb"))
    if norm_matches and norm_matches[0].exists():
        return FileResponse(str(norm_matches[0]), media_type="model/gltf-binary")

    # Fallback to original input
    input_p = Path(job["input_file_path"])
    if input_p.exists():
        return FileResponse(str(input_p), media_type="model/gltf-binary" if input_p.suffix.lower() == ".glb" else "text/plain")
        
    raise HTTPException(status_code=404, detail="Input model file not found")

@app.api_route("/api/jobs/{job_id}/files/skeleton_obj", methods=["GET", "HEAD"])
async def get_skeleton_obj(job_id: str):
    job_dir = STORAGE_DIR / job_id
    matches = list(job_dir.glob("stage2_skel/**/skeleton.obj"))
    if matches and matches[0].exists():
        return FileResponse(str(matches[0]), media_type="text/plain", filename="skeleton.obj")
    raise HTTPException(status_code=404, detail="Skeleton OBJ not found")

@app.api_route("/api/jobs/{job_id}/files/rigged_glb", methods=["GET", "HEAD"])
async def get_rigged_glb(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = STORAGE_DIR / job_id
    # Serve the GLB this job recorded; fall back to the most recently written one. Taking
    # whatever the glob happened to yield first served a stale file on any job that had ever
    # been re-animated under the old naming.
    recorded = job.get("metadata", {}).get("rig", {}).get("glb_path")
    chosen = Path(recorded) if recorded and Path(recorded).exists() else None
    if chosen is None:
        matches = sorted(job_dir.glob("*_rigged_animated.glb"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = matches[0] if matches else None
    if chosen is not None and chosen.exists():
        return FileResponse(
            str(chosen),
            media_type="model/gltf-binary",
            filename=f"{job['input_filename'].split('.')[0]}_rigged.glb"
        )
    raise HTTPException(status_code=404, detail="Rigged GLB file not found")

@app.get("/api/jobs/{job_id}/weights/{bone_index}")
async def get_bone_weights(job_id: str, bone_index: int):
    """Returns vertex weights for the specified bone index for 3D heatmap visualization."""
    job_dir = STORAGE_DIR / job_id
    skin_npz = job_dir / "stage3_skin/skin_weights.npz"
    if not skin_npz.exists():
        raise HTTPException(status_code=404, detail="Skin weights not found")
        
    data = np.load(str(skin_npz), allow_pickle=True)
    weights = data["weights"] # (N, J)
    names = data["names"]
    
    if bone_index < 0 or bone_index >= weights.shape[1]:
        raise HTTPException(status_code=400, detail="Invalid bone index")
        
    col = weights[:, bone_index].astype(np.float32)
    return {
        "bone_index": bone_index,
        "bone_name": str(names[bone_index]),
        "num_vertices": len(col),
        "min_weight": float(col.min()),
        "max_weight": float(col.max()),
        "weights": col.tolist()
    }

@app.get("/api/facial_blendshapes/presets")
async def get_facial_blendshapes_presets():
    """Returns ARKit blendshapes list and facial expression presets."""
    from pipeline.facial_blendshapes import ARKIT_BLENDSHAPES, EXPRESSION_PRESETS
    return {
        "blendshapes": ARKIT_BLENDSHAPES,
        "presets": EXPRESSION_PRESETS
    }

# ==========================================
# 🎨 3D Statue Painting & Automation Endpoints
# ==========================================

def run_statue_job_background(job_id: str):
    """Executes the complete automated 3D statue painting pipeline in background."""
    job = database.get_statue_job(job_id)
    if not job:
        return
    
    t_start = time.time()
    job_dir = STORAGE_DIR / f"statue_jobs/{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = job["input_file_path"]
    metadata = job.get("metadata", {})
    
    def on_progress(pct: int, step_name: str, step_idx: int, total_steps: int):
        metadata["progress"] = {
            "pct": pct,
            "step_name": step_name,
            "step_idx": step_idx,
            "total_steps": total_steps
        }
        database.update_statue_job(job_id, status="processing", metadata=metadata)
        
    try:
        on_progress(5, "Đang khởi tạo pipeline tạo tượng 3D...", 1, 5)
        res = statue_runner.process_statue(
            input_path=input_file,
            output_dir=str(job_dir),
            job_id=job_id,
            generator_type=job.get("generator_type", "trellis"),
            mesh_detail=job.get("mesh_detail", "high"),
            texture_detail=job.get("texture_detail", "high"),
            target_faces=int(job.get("target_faces", 50000)),
            pedestal_shape=job.get("pedestal_shape", "round"),
            orientation=metadata.get("orientation", "auto"),
            enable_rigging=bool(job.get("enable_rigging", 0)),
            progress_callback=on_progress
        )
        
        metadata["files"] = res.get("files", {})
        metadata["mesh_stats"] = res.get("mesh_stats", {})
        metadata["progress"] = {
            "pct": 100,
            "step_name": "Tượng 3D đã hoàn thành sẵn sàng cho ứng dụng tô tượng!",
            "step_idx": 5,
            "total_steps": 5
        }
        
        database.update_statue_job(
            job_id,
            status="completed",
            duration_sec=res.get("duration_sec", 0.0),
            num_vertices=res.get("mesh_stats", {}).get("num_vertices", 0),
            num_faces=res.get("mesh_stats", {}).get("num_faces", 0),
            num_parts=res.get("mesh_stats", {}).get("num_parts", 0),
            metadata=metadata
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error in statue job {job_id}:\n{tb}")
        database.update_statue_job(
            job_id,
            status="failed",
            error_message=str(e),
            metadata={"traceback": tb, "progress": {"pct": 100, "step_name": f"Lỗi: {e}"}}
        )

@app.api_route("/statue", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/statue-studio", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_statue_studio():
    statue_file = STATIC_DIR / "statue.html"
    if not statue_file.exists():
        raise HTTPException(status_code=404, detail="Statue studio page not found")
    return HTMLResponse(
        content=statue_file.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )


@app.api_route("/painting", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_3dpainting_app():
    index_file = ROOT_DIR / "3DPainting" / "dist" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="3DPainting app not built")
    return HTMLResponse(
        content=index_file.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )

@app.post("/api/statue/generate")
async def create_statue_generation_job(
    file: UploadFile = File(...),
    generator: str = Form("trellis"),
    mesh_detail: str = Form("high"),
    texture_detail: str = Form("high"),
    target_faces: int = Form(50000),
    pedestal_shape: str = Form("round"),
    orientation: str = Form("auto"),
    enable_rigging: bool = Form(False),
    seed: int = Form(42),
    background_tasks: BackgroundTasks = None
):
    stem = Path(file.filename).stem
    clean_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', stem)
    job_id = f"statue_{int(time.time())}_{clean_stem}"
    job_dir = STORAGE_DIR / f"statue_jobs/{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    
    clean_suffix = Path(file.filename).suffix.lower()
    clean_filename = f"{clean_stem}{clean_suffix}"
    target_input = job_dir / clean_filename
    with open(target_input, "wb") as f:
        content = await file.read()
        f.write(content)
        
    metadata = {
        "original_filename": file.filename,
        "seed": seed,
        "orientation": orientation,
        "progress": {"pct": 5, "step_name": "Đã đưa vào hàng đợi xử lý...", "step_idx": 1, "total_steps": 5}
    }
    job = database.create_statue_job(
        job_id=job_id,
        title=f"Statue: {file.filename}",
        input_filename=clean_filename,
        input_file_path=str(target_input),
        generator_type=generator,
        mesh_detail=mesh_detail,
        texture_detail=texture_detail,
        target_faces=target_faces,
        pedestal_shape=pedestal_shape,
        enable_rigging=enable_rigging,
        is_automated=False,
        metadata=metadata
    )
    
    background_tasks.add_task(run_statue_job_background, job_id)
    return job

@app.get("/api/statue/jobs")
async def list_all_statue_jobs(limit: int = 50, automated: Optional[bool] = None):
    jobs = database.list_statue_jobs(limit=limit, automated_only=automated)
    return JSONResponse(
        content=jobs,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )

@app.get("/api/statue/jobs/{job_id}")
async def get_statue_job_endpoint(job_id: str):
    job = database.get_statue_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Statue job not found")
    return JSONResponse(
        content=job,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
    )

@app.delete("/api/statue/jobs/{job_id}")
async def delete_statue_job_endpoint(job_id: str):
    job = database.get_statue_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Statue job not found")
    database.delete_statue_job(job_id)
    job_dir = STORAGE_DIR / f"statue_jobs/{job_id}"
    if job_dir.exists():
        shutil.rmtree(str(job_dir), ignore_errors=True)
    return {"status": "deleted", "job_id": job_id}

class StatueRotateRequest(BaseModel):
    job_id: Optional[str] = None
    preset_key: Optional[str] = None
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    re_ground: bool = True

@app.post("/api/statue/rotate_model")
async def rotate_statue_model_endpoint(req: StatueRotateRequest):
    job_dir = None
    if req.job_id:
        job = database.get_statue_job(req.job_id)
        job_dir = STORAGE_DIR / f"statue_jobs/{req.job_id}"
    elif req.preset_key:
        job_dir = ROOT_DIR / f"playground/static/sample_presets/models/{req.preset_key}"

    if not job_dir or not job_dir.exists():
        raise HTTPException(status_code=404, detail="Mô hình hoặc Job tượng không tìm thấy")

    rad_x = np.radians(req.rx)
    rad_y = np.radians(req.ry)
    rad_z = np.radians(req.rz)
    Rx = trimesh.transformations.rotation_matrix(rad_x, [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(rad_y, [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(rad_z, [0, 0, 1])
    T_rot = trimesh.transformations.concatenate_matrices(Rz, Ry, Rx)

    glb_files = list(job_dir.glob("*.glb"))
    if not glb_files:
        raise HTTPException(status_code=404, detail="Không tìm thấy file GLB nào để xoay")

    modified_files = []
    for glb_path in glb_files:
        try:
            m = trimesh.load(str(glb_path), process=False)
            m.apply_transform(T_rot)
            if req.re_ground:
                bounds = m.bounds
                center_x = (bounds[0][0] + bounds[1][0]) / 2.0
                center_z = (bounds[0][2] + bounds[1][2]) / 2.0
                min_y = bounds[0][1]
                m.apply_translation([-center_x, -min_y, -center_z])
            m.export(str(glb_path))
            modified_files.append(glb_path.name)
        except Exception as e:
            print(f"Lỗi khi xoay {glb_path.name}: {e}")

    # Cập nhật lại file zip nếu có
    zip_matches = list(job_dir.glob("*_statue_package.zip"))
    if zip_matches:
        pkg_zip = zip_matches[0]
        try:
            with zipfile.ZipFile(str(pkg_zip), "w", zipfile.ZIP_DEFLATED) as zf:
                for f in job_dir.iterdir():
                    if f != pkg_zip and not f.is_dir():
                        zf.write(str(f), arcname=f.name)
        except Exception as ze:
            print(f"Lỗi khi cập nhật zip: {ze}")

    return {
        "success": True,
        "message": f"Đã xoay trục thành công (X: {req.rx}°, Y: {req.ry}°, Z: {req.rz}°)",
        "job_id": req.job_id,
        "preset_key": req.preset_key,
        "modified_files": modified_files
    }

@app.api_route("/api/statue/jobs/{job_id}/files/{file_type}", methods=["GET", "HEAD"])
async def get_statue_file_endpoint(job_id: str, file_type: str):
    job = database.get_statue_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Statue job not found")
        
    job_dir = STORAGE_DIR / f"statue_jobs/{job_id}"
    metadata = job.get("metadata", {})
    files_map = metadata.get("files", {})
    
    target_path = None
    if file_type in files_map:
        p = Path(files_map[file_type])
        if p.exists():
            target_path = p
            
    if target_path is None:
        # Fallbacks by glob pattern
        if file_type == "plaster_glb":
            matches = list(job_dir.glob("*_plaster.glb"))
            if matches: target_path = matches[0]
        elif file_type == "segmented_glb":
            matches = list(job_dir.glob("*_segmented.glb"))
            if matches: target_path = matches[0]
        elif file_type == "id_colored_glb":
            matches = list(job_dir.glob("*_id_colored.glb"))
            if matches: target_path = matches[0]
        elif file_type == "textured_glb":
            matches = list(job_dir.glob("*_textured.glb")) or list(job_dir.glob("stage0_*/*.glb"))
            if matches: target_path = matches[0]
        elif file_type == "rigged_glb":
            matches = list(job_dir.glob("*_rigged.glb")) or list(job_dir.glob("*_statue_rigged.glb"))
            if matches: target_path = matches[0]
        elif file_type == "manifest_json":
            matches = list(job_dir.glob("*_manifest.json"))
            if matches: target_path = matches[0]
        elif file_type == "package_zip":
            matches = list(job_dir.glob("*_statue_package.zip"))
            if matches: target_path = matches[0]
        elif file_type == "input_image":
            p = Path(job.get("input_file_path", ""))
            if p.exists(): target_path = p
            
    if target_path is None or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Requested file '{file_type}' not found for job {job_id}")
        
    stem = Path(job["input_filename"]).stem
    if target_path.suffix.lower() == ".glb":
        return FileResponse(str(target_path), media_type="model/gltf-binary", filename=f"{stem}_{file_type}.glb")
    elif target_path.suffix.lower() == ".zip":
        return FileResponse(str(target_path), media_type="application/zip", filename=f"{stem}_statue_package.zip")
    elif target_path.suffix.lower() == ".json":
        return FileResponse(str(target_path), media_type="application/json", filename=f"{stem}_manifest.json")
    elif target_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
        return FileResponse(str(target_path), media_type=f"image/{target_path.suffix.lower().removeprefix('.')}")
    else:
        return FileResponse(str(target_path))

@app.get("/api/statue/palette")
async def get_statue_palette_endpoint():
    return {
        "palette": STATUE_PALETTE,
        "recommended_tools": ["bucket_fill", "brush", "eraser"],
        "material_presets": [
            {"id": "plaster", "name": "Thạch cao mờ (Gypsum Plaster)", "roughness": 0.88, "metalness": 0.02},
            {"id": "ceramic_glossy", "name": "Gốm sứ tráng men (Glossy Ceramic)", "roughness": 0.18, "metalness": 0.08},
            {"id": "clay", "name": "Đất sét nung (Terracotta Clay)", "roughness": 0.75, "metalness": 0.0},
            {"id": "metallic_gold", "name": "Mạ vàng kim (Gold Leaf Accent)", "roughness": 0.25, "metalness": 0.95}
        ]
    }

# --- Automation & Webhook Endpoints ---

@app.get("/api/statue/automation/status")
async def get_automation_status_endpoint():
    return automation_service.get_status()

@app.get("/api/statue/automation/config")
async def get_automation_config_endpoint():
    return database.get_automation_config()

@app.post("/api/statue/automation/config")
async def update_automation_config_endpoint(config_data: Dict[str, Any]):
    updated = database.update_automation_config(config_data)
    # Restart or start worker if enabled changed
    if updated.get("enabled", False) and not automation_service.is_running:
        automation_service.start()
    elif not updated.get("enabled", False) and automation_service.is_running:
        automation_service.stop()
    return updated

@app.post("/api/statue/automation/toggle")
async def toggle_automation_endpoint():
    cfg = database.get_automation_config()
    new_state = not cfg.get("enabled", False)
    database.update_automation_config({"enabled": new_state})
    if new_state:
        automation_service.start()
    else:
        automation_service.stop()
    return {"enabled": new_state, "is_running": automation_service.is_running}

@app.post("/api/statue/automation/scan-now")
async def trigger_scan_now_endpoint():
    found_files = automation_service.trigger_scan_now()
    return {"status": "triggered", "found_files": found_files, "count": len(found_files)}

@app.post("/api/statue/automation/test-webhook")
async def test_webhook_api_endpoint(data: Dict[str, Any]):
    url = data.get("webhook_url", "").strip()
    secret = data.get("webhook_secret", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp webhook_url để kiểm tra kết nối")
    return test_webhook_endpoint(url=url, secret=secret)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    lan_ips = get_lan_ips()
    print("=" * 60)
    print("🚀 UniRig 3D Rigging & Animation Web Playground")
    print(f"Local URL:   http://localhost:{port}")
    for ip in lan_ips:
        print(f"LAN URL:     http://{ip}:{port}")
    print("=" * 60)
    uvicorn.run(app, host=host, port=port)
