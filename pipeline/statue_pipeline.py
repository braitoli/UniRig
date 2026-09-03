"""
Master Statue 3D Pipeline.
End-to-end 2D image to production-ready 3D statue generation with:
- AI 3D Reconstruction (TRELLIS.2-4B / Tencent Hunyuan3D-2.1)
- Clean manifold geometry & standard Y-up ground alignment
- Optional Pedestal Base (Đế tượng đứng vững)
- Polygon Decimation for smooth 60fps WebGL/Mobile apps
- Multi-Part Anatomical / Spatial Segmentation (Ready for Bucket Fill Tool)
- Optional UniRig skeleton rigging & dancing animations
- Full package export (Plaster, Segmented, Textured, Rigged, ZIP, Manifest)
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

from .unirig_pipeline import UniRigPipeline
from .statue_optimizer import (
    clean_and_repair_mesh,
    auto_ground_and_orient,
    add_statue_pedestal,
    decimate_mesh_for_statue,
    segment_statue_parts,
    export_all_statue_variants,
    STATUE_PALETTE
)

class Statue3DPipeline:
    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent
        else:
            self.root_dir = Path(root_dir)
            
        self.unirig = UniRigPipeline(root_dir=str(self.root_dir))

    def process_statue(
        self,
        input_path: str,
        output_dir: str,
        job_id: str,
        generator_type: str = "trellis",
        mesh_detail: str = "high",
        texture_detail: str = "high",
        target_faces: int = 50000,
        pedestal_shape: str = "round",
        pedestal_height: float = 0.05,
        target_height: float = 1.6,
        flatten_bottom: bool = True,
        orientation: str = "auto",
        enable_rigging: bool = False,
        seed: int = 42,
        progress_callback: Optional[Callable[[int, str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full automated statue pipeline.
        """
        t_start = time.time()
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem

        def report(pct: int, msg: str, step_idx: int, total_steps: int = 5):
            if progress_callback:
                progress_callback(pct, msg, step_idx, total_steps)
            print(f"[{job_id}] [{pct}%] (Step {step_idx}/{total_steps}): {msg}")

        input_ext = Path(input_path).suffix.lower()
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        # Step 1: 3D Model Generation / Loading
        if input_ext in image_exts:
            report(10, "Đang tạo mô hình 3D từ ảnh 2D qua mạng Neural AI...", 1, 5)
            stage0_dir = out_p / "stage0_generated"
            gen_res = self.unirig.generate_3d_from_image(
                image_path=input_path,
                output_dir=str(stage0_dir),
                seed=seed,
                generator_type=generator_type,
                mesh_detail=mesh_detail,
                texture_detail=texture_detail,
                progress_callback=lambda p, m, s, t: report(int(5 + p * 0.35), f"Tạo 3D: {m}", 1, 5)
            )
            raw_glb_path = gen_res["output_glb_path"]
        else:
            report(35, "Đang nạp mô hình 3D đầu vào...", 1, 5)
            raw_glb_path = input_path

        # Load mesh via trimesh
        report(45, "Đang tối ưu hình học: Chuẩn hóa trục Y-up, tiếp đất và sửa lỗi Mesh...", 2, 5)
        raw_scene_or_mesh = trimesh.load(raw_glb_path, process=False)
        if isinstance(raw_scene_or_mesh, trimesh.Scene):
            raw_mesh = raw_scene_or_mesh.dump(concatenate=True)
            textured_mesh = raw_scene_or_mesh.dump(concatenate=True)
        else:
            raw_mesh = raw_scene_or_mesh.copy()
            textured_mesh = raw_scene_or_mesh.copy()

        # Step 2: Clean, orient, and ground
        mesh = clean_and_repair_mesh(raw_mesh, cull_hidden=True)
        mesh = auto_ground_and_orient(
            mesh,
            target_height=target_height,
            flatten_bottom=flatten_bottom,
            orientation=orientation
        )
        if textured_mesh is not None:
            textured_mesh = auto_ground_and_orient(
                textured_mesh,
                target_height=target_height,
                flatten_bottom=False,
                orientation=orientation
            )

        # Step 3: Pedestal Base
        has_pedestal = False
        if pedestal_shape in ("round", "square", "chamfered", "disc"):
            report(55, f"Đang tạo đế tượng vững chãi (kiểu {pedestal_shape})...", 3, 5)
            mesh, ped = add_statue_pedestal(
                mesh,
                shape=pedestal_shape,
                pedestal_height=pedestal_height
            )
            has_pedestal = (ped is not None)

        # Step 4: Decimation for WebGL & Mobile 60fps
        if target_faces > 0 and len(mesh.faces) > target_faces:
            report(65, f"Đang tối ưu số lượng Polygon ({len(mesh.faces)} -> {target_faces}) cho WebGL...", 3, 5)
            mesh = decimate_mesh_for_statue(mesh, target_faces=target_faces)

        # Step 5: Intelligent Anatomical Part Segmentation
        report(75, "Đang phân tích và phân vùng giải phẫu (Đầu, Thân, Tóc, Tay, Chân) cho tính năng Đổ Màu...", 4, 5)
        seg_data = segment_statue_parts(mesh, has_pedestal=has_pedestal)

        # Step 6: Export Statue GLBs
        report(88, "Đang xuất bản các định dạng Tượng (Thạch cao trắng, Phân vùng, Texture, ZIP)...", 5, 5)
        exported_files = export_all_statue_variants(
            base_mesh=mesh,
            segmented_data=seg_data,
            output_dir=out_p,
            stem=stem,
            original_texture_mesh=textured_mesh
        )

        # Optional Step 7: Auto-Rigging with UniRig
        rigged_glb_path = None
        if enable_rigging:
            try:
                report(92, "Đang tự động tích hợp khung xương UniRig và các điệu nhảy Mocap...", 5, 5)
                rig_dir = out_p / "rigging"
                rig_dir.mkdir(parents=True, exist_ok=True)
                
                # Run skeleton inference
                prep_res = self.unirig.preprocess_mesh(
                    input_path=exported_files["plaster_glb"],
                    output_dir=str(rig_dir / "prep")
                )
                skel_res = self.unirig.predict_skeleton(
                    input_mesh_path=exported_files["plaster_glb"],
                    npz_dir=str(rig_dir / "prep"),
                    output_dir=str(rig_dir / "skel")
                )
                skin_res = self.unirig.predict_skin(
                    vertices=prep_res["vertices"],
                    faces=prep_res["faces"],
                    joints=skel_res["joints"],
                    parents=skel_res["parents"],
                    names=skel_res["names"],
                    output_dir=str(rig_dir / "skin"),
                    use_neural=True,
                    input_mesh_path=exported_files["plaster_glb"],
                    skel_stage_dir=skel_res["skel_npz_path"]
                )
                rigged_out = str(out_p / f"{stem}_statue_rigged.glb")
                rig_exp = self.unirig.export_rigged_and_animated(
                    vertices=prep_res["vertices"],
                    faces=prep_res["faces"],
                    joints=skel_res["joints"],
                    parents=skel_res["parents"],
                    skin_weights=skin_res["weights"],
                    normals=prep_res["normals"],
                    names=skel_res["names"],
                    output_glb_path=rigged_out
                )
                rigged_glb_path = rigged_out
                exported_files["rigged_glb"] = rigged_glb_path
            except Exception as e:
                print(f"[{job_id}] Rigging warning: {e}")

        total_duration = round(time.time() - t_start, 2)
        report(100, "Hoàn thành toàn bộ quy trình tạo Tượng 3D!", 5, 5)

        return {
            "job_id": job_id,
            "status": "completed",
            "duration_sec": total_duration,
            "files": exported_files,
            "mesh_stats": {
                "num_vertices": int(len(mesh.vertices)),
                "num_faces": int(len(mesh.faces)),
                "num_parts": int(seg_data["num_parts_detected"]),
                "parts": seg_data["part_info"]
            },
            "settings": {
                "generator_type": generator_type,
                "mesh_detail": mesh_detail,
                "texture_detail": texture_detail,
                "target_faces": target_faces,
                "pedestal_shape": pedestal_shape,
                "enable_rigging": enable_rigging
            }
        }
