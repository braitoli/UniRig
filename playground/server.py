import os
import sys
import time
import socket
import shutil
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure root dir in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.unirig_pipeline import UniRigPipeline
from playground import database

app = FastAPI(title="UniRig 3D Rigging Playground")

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

pipeline_runner = UniRigPipeline(root_dir=str(ROOT_DIR))

database.init_db()

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
    """Executes the 4 pipeline stages in background and updates DB state."""
    job = database.get_job(job_id)
    if not job:
        return
        
    t_start = time.time()
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = job["input_file_path"]
    metadata = job.get("metadata", {})
    
    try:
        # Stage 1: Preprocess
        database.update_job(job_id, status="processing_prep", stage=1)
        prep_res = pipeline_runner.preprocess_mesh(
            input_path=input_file,
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
        database.update_job(job_id, status="processing_skeleton", stage=2)
        skel_res = pipeline_runner.predict_skeleton(
            input_mesh_path=input_file,
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
        database.update_job(job_id, status="processing_skin", stage=3)
        skin_res = pipeline_runner.predict_skin(
            vertices=prep_res["vertices"],
            faces=prep_res["faces"],
            joints=skel_res["joints"],
            parents=skel_res["parents"],
            names=skel_res["names"],
            output_dir=str(job_dir / "stage3_skin"),
            use_neural=True,
            input_mesh_path=input_file,
            skel_stage_dir=skel_res["skel_npz_path"]
        )
        metadata["skin"] = {
            "bone_stats": skin_res["bone_stats"],
            "calc_time_sec": skin_res["calc_time_sec"]
        }
        database.update_job(job_id, metadata=metadata)
        
        # Stage 4: Rigged GLB & Animations
        database.update_job(job_id, status="processing_rig", stage=4)
        rigged_glb_path = str(job_dir / f"{prep_res['stem']}_rigged_animated.glb")
        rig_res = pipeline_runner.export_rigged_and_animated(
            vertices=prep_res["vertices"],
            faces=prep_res["faces"],
            joints=skel_res["joints"],
            parents=skel_res["parents"],
            skin_weights=skin_res["weights"],
            normals=prep_res["normals"],
            names=skel_res["names"],
            output_glb_path=rigged_glb_path
        )
        
        t_end = time.time()
        total_duration = round(t_end - t_start, 2)
        
        metadata["rig"] = {
            "glb_path": rigged_glb_path,
            "glb_size_bytes": rig_res["glb_size_bytes"],
            "animations": rig_res["animations"],
            "export_time_sec": rig_res["export_time_sec"],
            "total_duration_sec": total_duration
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

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

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
    return database.list_jobs()

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
    background_tasks: BackgroundTasks = None
):
    stem = Path(file.filename).stem
    job_id = f"job_{int(time.time())}_{stem}"
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    target_input = job_dir / file.filename
    with open(target_input, "wb") as f:
        content = await file.read()
        f.write(content)
        
    job = database.create_job(
        job_id=job_id,
        title=f"Custom: {file.filename}",
        input_filename=file.filename,
        input_file_path=str(target_input)
    )
    
    background_tasks.add_task(run_job_background, job_id)
    return job

@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/api/jobs/{job_id}/reanimate")
async def reanimate_job_endpoint(job_id: str):
    """
    Fast re-generation of Stage 4 animations (takes < 0.2s) without re-running
    Neural mesh & skinning models.
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
            
        stem = Path(job["input_filename"]).stem
        rigged_glb_path = str(job_dir / f"{stem}_rigged_animated.glb")
        
        rig_res = pipeline_runner.export_rigged_and_animated(
            vertices=vertices,
            faces=faces,
            joints=joints,
            parents=parents,
            skin_weights=skin_weights,
            normals=normals,
            names=names,
            output_glb_path=rigged_glb_path,
            use_pan_retargeting=True
        )
        t1 = time.time()
        
        metadata = job.get("metadata", {})
        metadata["rig"] = {
            "glb_path": rigged_glb_path,
            "glb_size_bytes": rig_res["glb_size_bytes"],
            "animations": rig_res["animations"],
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

@app.api_route("/api/jobs/{job_id}/files/input_model", methods=["GET", "HEAD"])
async def get_input_model(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = STORAGE_DIR / job_id
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
    matches = list(job_dir.glob("*_rigged_animated.glb"))
    if matches and matches[0].exists():
        return FileResponse(
            str(matches[0]),
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
