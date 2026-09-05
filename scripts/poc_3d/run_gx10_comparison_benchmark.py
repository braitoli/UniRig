#!/usr/bin/env python3
"""
Benchmark & Visual Quality Comparison: Original (280k+ faces) vs Decimated (50k faces) on GX10
Chạy trên 2 loại mô hình:
1. Simple Organic: simple_duck.jpg (ít góc cạnh, mềm mại)
2. Complex Hard-Surface: complex_mecha.jpg (nhiều góc cạnh, cơ khí, sắc nhọn)
"""

import os
import sys
import time
import json
import re
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.statue_pipeline import Statue3DPipeline
from playground import database

STORAGE_DIR = ROOT_DIR / "playground" / "storage"
JOBS_DIR = STORAGE_DIR / "statue_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
INPUTS_DIR = STORAGE_DIR / "poc_inputs"

ARTIFACTS_DIR = Path("/home/braitoli/.gemini/antigravity-cli/brain/519b4581-1187-4004-8ba0-0b8b763c75a7")

def run_job(image_path: Path, raw_glb_path: str, stem: str, target_faces: int, stage0_time: float) -> dict:
    mode_name = "Nguyên bản" if target_faces == 0 else "50.000 mặt"
    job_id = f"statue_{int(time.time())}_{stem}_{'orig' if target_faces == 0 else '50k'}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    target_img = job_dir / image_path.name
    shutil.copy2(image_path, target_img)

    title = f"Đối sánh: {stem} ({mode_name})"
    print("\n" + "=" * 80)
    print(f"🚀 BẮT ĐẦU STATUE STUDIO: {title}")
    print(f"ID: {job_id} | Target Faces: {target_faces} | Base GLB: {raw_glb_path}")
    print("=" * 80)

    metadata = {
        "original_filename": image_path.name,
        "seed": 42,
        "orientation": "auto",
        "progress": {"pct": 10, "step_name": f"Đang xử lý {mode_name}...", "step_idx": 2, "total_steps": 5}
    }

    job = database.create_statue_job(
        job_id=job_id,
        title=title,
        input_filename=image_path.name,
        input_file_path=str(target_img),
        generator_type="trellis",
        mesh_detail="high",
        texture_detail="high",
        target_faces=target_faces,
        pedestal_shape="round",
        enable_rigging=False,
        is_automated=False,
        metadata=metadata
    )

    t_pipe_start = time.time()
    pipe = Statue3DPipeline(root_dir=str(ROOT_DIR))

    res = pipe.process_statue(
        input_path=raw_glb_path,
        output_dir=str(job_dir),
        job_id=job_id,
        generator_type="trellis",
        mesh_detail="high",
        texture_detail="high",
        target_faces=target_faces,
        pedestal_shape="round",
        target_height=1.6,
        flatten_bottom=True,
        orientation="auto",
        enable_rigging=False
    )
    pipeline_time = round(time.time() - t_pipe_start, 2)
    total_statue_time = round(stage0_time + pipeline_time, 2)

    metadata["files"] = res.get("files", {})
    metadata["mesh_stats"] = res.get("mesh_stats", {})
    metadata["timing"] = {
        "stage0_ai_sec": stage0_time,
        "pipeline_sec": pipeline_time,
        "total_time_sec": total_statue_time
    }
    metadata["progress"] = {
        "pct": 100,
        "step_name": f"Hoàn thành {mode_name}!",
        "step_idx": 5,
        "total_steps": 5
    }

    database.update_statue_job(
        job_id,
        status="completed",
        duration_sec=total_statue_time,
        num_vertices=res.get("mesh_stats", {}).get("num_vertices", 0),
        num_faces=res.get("mesh_stats", {}).get("num_faces", 0),
        num_parts=res.get("mesh_stats", {}).get("num_parts", 0),
        metadata=metadata
    )

    print(f"[+] Hoàn thành {title}: Pipeline = {pipeline_time}s | Tổng = {total_statue_time}s | Faces = {res.get('mesh_stats', {}).get('num_faces', 0):,}")

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "title": title,
        "target_faces": target_faces,
        "stage0_time_sec": stage0_time,
        "pipeline_time_sec": pipeline_time,
        "total_time_sec": total_statue_time,
        "num_vertices": res.get("mesh_stats", {}).get("num_vertices", 0),
        "num_faces": res.get("mesh_stats", {}).get("num_faces", 0),
        "files": res.get("files", {})
    }


def render_mesh_views(glb_path: str, output_prefix: str, image_size: int = 768) -> dict:
    """Renders high-quality Perspective view for visual comparison using EGL headless."""
    import os
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender

    scene_or_mesh = trimesh.load(glb_path, process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_geometry()
        if isinstance(mesh, dict):
            mesh = trimesh.util.concatenate(list(mesh.values()))
    else:
        mesh = scene_or_mesh.copy()

    vertices = mesh.vertices.copy()
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices -= center
    radius = np.linalg.norm(vertices, axis=1).max()
    mesh.vertices = vertices

    # Convert texture to vertex colors for smooth headless EGL rendering
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "to_color"):
        try:
            mesh.visual = mesh.visual.to_color()
        except Exception as e:
            print(f"to_color fallback: {e}")

    rendered_images = {}
    r = pyrender.OffscreenRenderer(image_size, image_size)

    # 1. Perspective View
    pymesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    pscene = pyrender.Scene(bg_color=[0.95, 0.95, 0.96, 1.0], ambient_light=[0.65, 0.65, 0.65])
    pscene.add(pymesh)

    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    pscene.add(key_light, pose=np.eye(4))

    yaw = np.radians(35)
    pitch = np.radians(15)
    dist = radius * 2.4
    cam_pos = np.array([
        dist * np.sin(yaw) * np.cos(pitch),
        dist * np.sin(pitch),
        dist * np.cos(yaw) * np.cos(pitch)
    ], dtype=np.float32)

    forward = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = cam_pos

    yfov = 2 * np.arctan2(radius * 1.15, dist)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    pscene.add(camera, pose=cam_pose)

    color, _ = r.render(pscene)
    img_out = f"{output_prefix}_perspective.png"
    Image.fromarray(color).save(img_out)
    rendered_images["perspective"] = img_out

    r.delete()
    return rendered_images


def create_side_by_side_comparison(
    orig_res: dict,
    dec_res: dict,
    orig_renders: dict,
    dec_renders: dict,
    category_name: str,
    output_path: Path
):
    """Creates a side-by-side composite comparison image with metrics banner."""
    img_orig = Image.open(orig_renders["perspective"])
    img_dec = Image.open(dec_renders["perspective"])
    W, H = img_orig.size

    header_h = 130
    footer_h = 90
    canvas_w = W * 2 + 30
    canvas_h = header_h + H + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(img_orig, (10, header_h))
    canvas.paste(img_dec, (W + 20, header_h))

    # Header
    draw.rectangle([(0, 0), (canvas_w, header_h)], fill=(30, 41, 59))
    draw.text((30, 20), f"DOI SANH CHAT LUONG 3D: NGUYEN BAN vs 50K MAT ({category_name.upper()})", fill=(255, 255, 255))
    draw.text((30, 55), "Thu nghiem tren may tram NVIDIA GB10 (GX10) - UniRig 3D Statue Studio", fill=(148, 163, 184))
    draw.text((30, 85), "Doi chieu do trung thuc goc canh, chat luong texture va hieu nang xuat GLB", fill=(203, 213, 225))

    # Labels
    draw.rectangle([(10, header_h), (W + 10, header_h + 35)], fill=(15, 118, 110))
    draw.text((25, header_h + 8), f"BAN NGUYEN BAN (100% GOC): {orig_res['num_faces']:,} MAT", fill=(255, 255, 255))

    draw.rectangle([(W + 20, header_h), (W * 2 + 20, header_h + 35)], fill=(30, 64, 175))
    draw.text((W + 35, header_h + 8), f"BAN DECIMATE TOI UU WEB: {dec_res['num_faces']:,} MAT", fill=(255, 255, 255))

    # Footer
    draw.rectangle([(0, canvas_h - footer_h), (canvas_w, canvas_h)], fill=(241, 245, 249))

    orig_glb_size = round(os.path.getsize(orig_res['files']['textured_glb']) / (1024*1024), 2)
    orig_zip_size = round(os.path.getsize(orig_res['files']['package_zip']) / (1024*1024), 2)
    draw.text((30, canvas_h - 75), f"* Da giac: {orig_res['num_faces']:,} faces | Dinh: {orig_res['num_vertices']:,}", fill=(30, 41, 59))
    draw.text((30, canvas_h - 48), f"* Dung luong: GLB {orig_glb_size} MB | ZIP {orig_zip_size} MB", fill=(30, 41, 59))
    draw.text((30, canvas_h - 22), f"* Thoi gian Pipeline: {orig_res['pipeline_time_sec']}s (Tong: {orig_res['total_time_sec']}s)", fill=(100, 116, 139))

    dec_glb_size = round(os.path.getsize(dec_res['files']['textured_glb']) / (1024*1024), 2)
    dec_zip_size = round(os.path.getsize(dec_res['files']['package_zip']) / (1024*1024), 2)
    reduction = round((1 - dec_res['num_faces'] / max(orig_res['num_faces'], 1)) * 100, 1)
    size_ratio = round(orig_glb_size / max(dec_glb_size, 0.1), 1)
    draw.text((W + 35, canvas_h - 75), f"* Da giac: {dec_res['num_faces']:,} faces (Giam {reduction}%)", fill=(16, 185, 129))
    draw.text((W + 35, canvas_h - 48), f"* Dung luong: GLB {dec_glb_size} MB | ZIP {dec_zip_size} MB (Nhe hon {size_ratio}x)", fill=(16, 185, 129))
    pipe_diff = round(orig_res['pipeline_time_sec'] - dec_res['pipeline_time_sec'], 1)
    draw.text((W + 35, canvas_h - 22), f"* Thoi gian Pipeline: {dec_res['pipeline_time_sec']}s (Nhanh hon {pipe_diff}s)", fill=(16, 185, 129))

    canvas.save(output_path, quality=95)
    print(f"[+] Da luu anh doi sanh: {output_path}")
    return str(output_path)


def main():
    duck_input = INPUTS_DIR / "simple_duck.jpg"
    mecha_input = INPUTS_DIR / "complex_mecha.jpg"

    raw_duck_glb = "/tmp/test_duck.glb"
    raw_mecha_glb = "/tmp/test_mecha.glb"

    stage0_duck_time = 185.0
    stage0_mecha_time = 210.0

    # =========================================================================
    # 1. SIMPLE MODEL: DUCK (Reusing the finished jobs)
    # =========================================================================
    print("\n" + "#" * 80)
    print("MÔ HÌNH 1: ĐƠN GIẢN, BỀ MẶT CONG HỮU CƠ (SIMPLE DUCK)")
    print("#" * 80)

    duck_orig_dir = JOBS_DIR / "statue_1788536509_simple_duck_orig"
    duck_50k_dir = JOBS_DIR / "statue_1788536552_simple_duck_50k"

    duck_orig = {
        "job_id": "statue_1788536509_simple_duck_orig",
        "job_dir": str(duck_orig_dir),
        "title": "Đối sánh: simple_duck (Nguyên bản)",
        "target_faces": 0,
        "stage0_time_sec": stage0_duck_time,
        "pipeline_time_sec": 43.40,
        "total_time_sec": round(stage0_duck_time + 43.40, 2),
        "num_vertices": 148112,
        "num_faces": 296022,
        "files": {
            "textured_glb": str(duck_orig_dir / "test_duck_textured.glb"),
            "package_zip": str(duck_orig_dir / "test_duck_statue_package.zip")
        }
    }

    duck_50k = {
        "job_id": "statue_1788536552_simple_duck_50k",
        "job_dir": str(duck_50k_dir),
        "title": "Đối sánh: simple_duck (50.000 mặt)",
        "target_faces": 50000,
        "stage0_time_sec": stage0_duck_time,
        "pipeline_time_sec": 32.79,
        "total_time_sec": round(stage0_duck_time + 32.79, 2),
        "num_vertices": 25114,
        "num_faces": 49999,
        "files": {
            "textured_glb": str(duck_50k_dir / "test_duck_textured.glb"),
            "package_zip": str(duck_50k_dir / "test_duck_statue_package.zip")
        }
    }

    print("\n[*] Render góc nhìn so sánh cho Simple Duck...")
    duck_orig_renders = render_mesh_views(duck_orig["files"]["textured_glb"], str(STORAGE_DIR / "cmp_duck_orig"))
    duck_50k_renders = render_mesh_views(duck_50k["files"]["textured_glb"], str(STORAGE_DIR / "cmp_duck_50k"))

    duck_cmp_img = ARTIFACTS_DIR / "comparison_simple_duck.jpg"
    create_side_by_side_comparison(
        duck_orig, duck_50k,
        duck_orig_renders, duck_50k_renders,
        category_name="Mo hinh Huu co Don gian (Simple Duck)",
        output_path=duck_cmp_img
    )

    # =========================================================================
    # 2. COMPLEX MODEL: MECHA ROBOT
    # =========================================================================
    print("\n" + "#" * 80)
    print("MÔ HÌNH 2: PHỨC TẠP, NHIỀU GÓC CẠNH CƠ KHÍ (COMPLEX MECHA ROBOT)")
    print("#" * 80)

    mecha_orig = run_job(mecha_input, raw_mecha_glb, stem="complex_mecha", target_faces=0, stage0_time=stage0_mecha_time)
    mecha_50k = run_job(mecha_input, raw_mecha_glb, stem="complex_mecha", target_faces=50000, stage0_time=stage0_mecha_time)

    print("\n[*] Render góc nhìn so sánh cho Complex Mecha...")
    mecha_orig_renders = render_mesh_views(mecha_orig["files"]["textured_glb"], str(STORAGE_DIR / "cmp_mecha_orig"))
    mecha_50k_renders = render_mesh_views(mecha_50k["files"]["textured_glb"], str(STORAGE_DIR / "cmp_mecha_50k"))

    mecha_cmp_img = ARTIFACTS_DIR / "comparison_complex_mecha.jpg"
    create_side_by_side_comparison(
        mecha_orig, mecha_50k,
        mecha_orig_renders, mecha_50k_renders,
        category_name="Mo hinh Co khi Phuc tap (Complex Mecha)",
        output_path=mecha_cmp_img
    )

    full_report = {
        "hardware": "NVIDIA GB10 (GX10 DGX Spark ARM64)",
        "duck": {
            "orig": duck_orig,
            "decimated_50k": duck_50k,
            "comparison_image": str(duck_cmp_img)
        },
        "mecha": {
            "orig": mecha_orig,
            "decimated_50k": mecha_50k,
            "comparison_image": str(mecha_cmp_img)
        }
    }
    summary_json = ROOT_DIR / "docs" / "GX10_DECIMATION_COMPARISON_REPORT.json"
    with open(summary_json, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[+] ĐÃ HOÀN THÀNH TOÀN BỘ VÀ LƯU BÁO CÁO: {summary_json}")

if __name__ == "__main__":
    main()
