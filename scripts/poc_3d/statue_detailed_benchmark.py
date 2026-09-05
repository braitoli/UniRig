#!/usr/bin/env python3
"""
Detailed Statue Studio Benchmarking Tool (CPU vs GPU Profiling)
Đo đạc chi tiết từng step trong luồng tạo ảnh TÔ TƯỢNG 3D:
- Thời gian thực tế (Wall-clock time)
- Thời gian CPU (Process time)
- Thời gian GPU (CUDA events timing)
- Bộ nhớ VRAM đỉnh (Peak VRAM allocated & reserved)
- Kích thước & thông số mesh sau từng step
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import torch
import trimesh
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.statue_pipeline import Statue3DPipeline
from pipeline.statue_optimizer import (
    clean_and_repair_mesh,
    auto_ground_and_orient,
    add_statue_pedestal,
    decimate_mesh_for_statue,
    segment_statue_parts,
    export_all_statue_variants
)

def get_vram_mb():
    if torch.cuda.is_available():
        return {
            "allocated": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
            "reserved": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
            "max_allocated": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "max_reserved": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
        }
    return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}

class StepTimer:
    def __init__(self, step_name):
        self.step_name = step_name
        self.wall_start = 0
        self.cpu_start = 0
        self.gpu_start_event = None
        self.gpu_end_event = None
        self.vram_start = {}

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.gpu_start_event = torch.cuda.Event(enable_timing=True)
            self.gpu_end_event = torch.cuda.Event(enable_timing=True)
            self.gpu_start_event.record()
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.vram_start = get_vram_mb()
        print(f"\n---> [BẮT ĐẦU] {self.step_name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        gpu_time_ms = 0.0
        if torch.cuda.is_available():
            self.gpu_end_event.record()
            torch.cuda.synchronize()
            gpu_time_ms = self.gpu_start_event.elapsed_time(self.gpu_end_event)
        
        self.wall_time_sec = round(wall_end - self.wall_start, 3)
        self.cpu_time_sec = round(cpu_end - self.cpu_start, 3)
        self.gpu_time_sec = round(gpu_time_ms / 1000.0, 3)
        self.vram_end = get_vram_mb()

        print(f"---> [HOÀN THÀNH] {self.step_name}")
        print(f"     * Wall-clock Time: {self.wall_time_sec}s")
        print(f"     * CPU Time:        {self.cpu_time_sec}s")
        print(f"     * GPU Time:        {self.gpu_time_sec}s")
        print(f"     * Peak VRAM:       {self.vram_end['max_allocated']} MB (Alloc) / {self.vram_end['max_reserved']} MB (Reserved)")

    def get_metrics(self):
        return {
            "step": self.step_name,
            "wall_time_sec": self.wall_time_sec,
            "cpu_time_sec": self.cpu_time_sec,
            "gpu_time_sec": self.gpu_time_sec,
            "vram_peak_allocated_mb": self.vram_end.get("max_allocated", 0),
            "vram_peak_reserved_mb": self.vram_end.get("max_reserved", 0),
        }

def run_statue_detailed_benchmark(
    image_path: str,
    output_dir: str,
    generator_type: str = "trellis",
    mesh_detail: str = "high",
    texture_detail: str = "high",
    target_faces: int = 50000,
    pedestal_shape: str = "round",
    seed: int = 42
):
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    job_id = f"statue_bench_{generator_type}_{int(time.time())}_{stem}"
    job_dir = out_p / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🏛️ ĐO ĐẠC CHI TIẾT HIỆU NĂNG TÔ TƯỢNG 3D (STATUE STUDIO)")
    print(f"Mô hình:            {generator_type.upper()}")
    print(f"Ảnh đầu vào:        {image_path}")
    print(f"Mesh Detail:        {mesh_detail} (Nguyên bản cao nhất)")
    print(f"Texture Detail:     {texture_detail} (Bake 4096)")
    print(f"Target Polygons:    {target_faces} faces")
    print(f"Thư mục lưu:        {job_dir}")
    print("=" * 80)

    steps_metrics = []
    t_global_wall_start = time.perf_counter()
    t_global_cpu_start = time.process_time()

    statue_pipe = Statue3DPipeline(root_dir=str(ROOT_DIR))

    # -------------------------------------------------------------
    # STEP 1: AI 3D RECONSTRUCTION (STAGE 0)
    # -------------------------------------------------------------
    with StepTimer("Step 1: AI 3D Reconstruction (Stage 0: Image-to-3D)") as t1:
        stage0_dir = job_dir / "stage0_raw"
        stage0_dir.mkdir(parents=True, exist_ok=True)
        gen_res = statue_pipe.unirig.generate_3d_from_image(
            image_path=image_path,
            output_dir=str(stage0_dir),
            seed=seed,
            generator_type=generator_type,
            mesh_detail=mesh_detail,
            texture_detail=texture_detail
        )
        raw_glb_path = gen_res["output_glb_path"]
    m1 = t1.get_metrics()
    m1["raw_glb_size_mb"] = round(os.path.getsize(raw_glb_path) / (1024 * 1024), 2)
    steps_metrics.append(m1)

    # Nạp mesh từ file GLB thô vừa sinh
    raw_scene_or_mesh = trimesh.load(raw_glb_path, process=False)
    if isinstance(raw_scene_or_mesh, trimesh.Scene):
        raw_mesh = raw_scene_or_mesh.dump(concatenate=True)
        textured_mesh = raw_scene_or_mesh.dump(concatenate=True)
    else:
        raw_mesh = raw_scene_or_mesh.copy()
        textured_mesh = raw_scene_or_mesh.copy()

    orig_vertices = len(raw_mesh.vertices)
    orig_faces = len(raw_mesh.faces)
    print(f"[*] Mesh thô ban đầu: {orig_vertices:,} đỉnh, {orig_faces:,} mặt tam giác.")

    # -------------------------------------------------------------
    # STEP 2: CLEAN, REPAIR & AUTO GROUND Y-UP
    # -------------------------------------------------------------
    with StepTimer("Step 2: Clean, Repair Manifold & Auto Ground Y-Up") as t2:
        mesh = clean_and_repair_mesh(raw_mesh, cull_hidden=True)
        mesh = auto_ground_and_orient(
            mesh,
            target_height=1.6,
            flatten_bottom=True,
            orientation="auto"
        )
        if textured_mesh is not None:
            textured_mesh = auto_ground_and_orient(
                textured_mesh,
                target_height=1.6,
                flatten_bottom=False,
                orientation="auto"
            )
    m2 = t2.get_metrics()
    m2["clean_vertices"] = len(mesh.vertices)
    m2["clean_faces"] = len(mesh.faces)
    steps_metrics.append(m2)

    # -------------------------------------------------------------
    # STEP 3: PEDESTAL BASE (ĐẾ TƯỢNG)
    # -------------------------------------------------------------
    has_pedestal = False
    with StepTimer(f"Step 3: Thêm đế tượng vững chãi ({pedestal_shape})") as t3:
        if pedestal_shape in ("round", "square", "chamfered", "disc"):
            mesh, ped = add_statue_pedestal(
                mesh,
                shape=pedestal_shape,
                pedestal_height=0.05
            )
            has_pedestal = (ped is not None)
    m3 = t3.get_metrics()
    m3["has_pedestal"] = has_pedestal
    steps_metrics.append(m3)

    # -------------------------------------------------------------
    # STEP 4: POLYGON DECIMATION (NGUYÊN BẢN HOẶC WEBGL)
    # -------------------------------------------------------------
    with StepTimer(f"Step 4: Xử lý Polygon Mesh ({'Giữ NGUYÊN BẢN' if target_faces <= 0 else f'Decimate về {target_faces:,} faces'})") as t4:
        if target_faces > 0 and len(mesh.faces) > target_faces:
            print(f"[*] Decimate mesh từ {len(mesh.faces):,} xuống {target_faces:,} mặt...")
            mesh = decimate_mesh_for_statue(mesh, target_faces=target_faces)
        else:
            print(f"[*] GIỮ NGUYÊN BẢN 100% SỐ MẶT GỐC: {len(mesh.faces):,} mặt tam giác (không decimate).")
    m4 = t4.get_metrics()
    m4["final_mesh_faces"] = len(mesh.faces)
    m4["is_original_faces"] = (target_faces <= 0)
    steps_metrics.append(m4)

    # -------------------------------------------------------------
    # STEP 5: MULTI-PART TEXTURE COLOR SEGMENTATION (CHO TÔ TƯỢNG)
    # -------------------------------------------------------------
    with StepTimer("Step 5: Phân vùng tô màu Bucket-fill theo Texture") as t5:
        seg_data = segment_statue_parts(mesh, has_pedestal=has_pedestal, texture_source=textured_mesh)
    m5 = t5.get_metrics()
    m5["num_segments"] = len(seg_data.get("segments", []))
    steps_metrics.append(m5)

    # -------------------------------------------------------------
    # STEP 6: EXPORT 6 BIẾN THỂ GLB & STATUE PACKAGE ZIP
    # -------------------------------------------------------------
    with StepTimer("Step 6: Xuất bản 6 biến thể GLB + ZIP Package") as t6:
        exported_files = export_all_statue_variants(
            base_mesh=mesh,
            segmented_data=seg_data,
            output_dir=job_dir,
            stem=stem,
            original_texture_mesh=textured_mesh
        )
    m6 = t6.get_metrics()
    m6["exported_files_count"] = len(exported_files)
    steps_metrics.append(m6)

    t_global_wall = round(time.perf_counter() - t_global_wall_start, 3)
    t_global_cpu = round(time.process_time() - t_global_cpu_start, 3)
    t_global_gpu = round(sum(m.get("gpu_time_sec", 0) for m in steps_metrics), 3)

    print("\n" + "=" * 80)
    print("📊 TỔNG HỢP KẾT QUẢ ĐO ĐẠC PIPELINE TÔ TƯỢNG 3D:")
    print(f"Tổng Wall-clock Time:  {t_global_wall}s ({t_global_wall/60:.2f} phút)")
    print(f"Tổng CPU Time:         {t_global_cpu}s")
    print(f"Tổng GPU Time:         {t_global_gpu}s")
    print("\nBảng chi tiết từng step:")
    print(f"{'Step Name':<50} | {'Wall (s)':<10} | {'CPU (s)':<10} | {'GPU (s)':<10} | {'Peak VRAM (MB)':<15}")
    print("-" * 105)
    for m in steps_metrics:
        print(f"{m['step']:<50} | {m['wall_time_sec']:<10} | {m['cpu_time_sec']:<10} | {m['gpu_time_sec']:<10} | {m['vram_peak_allocated_mb']:<15}")
    print("=" * 80)

    final_report = {
        "job_id": job_id,
        "generator_type": generator_type,
        "input_image": image_path,
        "mesh_detail": mesh_detail,
        "texture_detail": texture_detail,
        "summary": {
            "total_wall_time_sec": t_global_wall,
            "total_cpu_time_sec": t_global_cpu,
            "total_gpu_time_sec": t_global_gpu,
            "orig_vertices": orig_vertices,
            "orig_faces": orig_faces,
            "final_faces": len(mesh.faces)
        },
        "steps": steps_metrics,
        "exported_artifacts": exported_files
    }

    report_path = job_dir / f"detailed_benchmark_{generator_type}.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"[+] Đã lưu JSON kết quả chi tiết: {report_path}")
    return final_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed Statue Studio Benchmark")
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, default="results/statue_detailed_poc")
    parser.add_argument("--generator", "-g", type=str, default="trellis", choices=["trellis", "pixal3d"])
    parser.add_argument("--mesh_detail", type=str, default="high")
    parser.add_argument("--texture_detail", type=str, default="high")
    parser.add_argument("--target_faces", type=int, default=50000, help="0 để giữ nguyên bản polygon mesh, > 0 để decimate (mặc định: 50,000)")
    parser.add_argument("--pedestal", type=str, default="round")
    args = parser.parse_args()

    run_statue_detailed_benchmark(
        image_path=args.image,
        output_dir=args.output_dir,
        generator_type=args.generator,
        mesh_detail=args.mesh_detail,
        texture_detail=args.texture_detail,
        target_faces=args.target_faces,
        pedestal_shape=args.pedestal
    )
