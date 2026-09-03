"""
Statue Optimizer & Exporter for Online 3D Statue Painting Applications.

Provides:
1. Mesh clean-up, repair, and normalization.
2. Auto-grounding (Y-up, centered X/Z, base at Y=0) and bottom flattening.
3. Pedestal generation (round / square / chamfered base stands).
4. Quadric mesh decimation (optimizing polycount for WebGL 60fps & mobile).
5. Intelligent Anatomical / Spatial Part Segmentation (Head, Body, Limbs, Hair, Base)
   for instant "Bucket Fill" (Đổ màu từng vùng) in online painting apps.
6. Multi-material / Multi-part GLB and Pure Plaster White GLB exports.
7. Complete packaging for API/online app integration.
"""

import os
import time
import json
import zipfile
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

STATUE_PALETTE = [
    {"name": "Đầu / Khuôn mặt (Head)", "hex": "#FFE0BD", "rgb": [255, 224, 189]},
    {"name": "Tóc / Mũ (Hair/Headwear)", "hex": "#795548", "rgb": [121, 85, 72]},
    {"name": "Thân trên / Áo (Upper Body)", "hex": "#4FC3F7", "rgb": [79, 195, 247]},
    {"name": "Thân dưới / Quần (Lower Body)", "hex": "#81C784", "rgb": [129, 199, 132]},
    {"name": "Tay / Cánh tay (Arms/Hands)", "hex": "#FFB74D", "rgb": [255, 183, 77]},
    {"name": "Chân / Giày (Legs/Feet)", "hex": "#BA68C8", "rgb": [186, 104, 200]},
    {"name": "Phụ kiện / Trang sức (Accessories)", "hex": "#FFD54F", "rgb": [255, 213, 79]},
    {"name": "Đế tượng (Pedestal Base)", "hex": "#90A4AE", "rgb": [144, 164, 174]},
    {"name": "Chi tiết nhỏ (Fine Accents)", "hex": "#E57373", "rgb": [229, 115, 115]},
    {"name": "Màu xanh ngọc (Cyan Pearl)", "hex": "#4DD0E1", "rgb": [77, 208, 225]},
]

def clean_and_repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Removes duplicate vertices, unreferenced vertices, and repairs normals."""
    m = mesh.copy()
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    m.remove_infinite_values()
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_winding(m)
    return m

def auto_ground_and_orient(
    mesh: trimesh.Trimesh,
    target_height: float = 1.6,
    flatten_bottom: bool = True,
    orientation: str = "auto"
) -> trimesh.Trimesh:
    """
    Ensures model is properly oriented (supporting horizontal mode for vehicles/quadrupeds),
    centered at (0, 0) in X/Z, scaled to standard height, and resting flat on Y = 0.
    """
    m = mesh.copy()
    v = m.vertices.copy().astype(np.float32)
    extents = v.max(axis=0) - v.min(axis=0)

    if orientation == "horizontal":
        # Force model to lie horizontally flat:
        # If height (Y) is greater than length (Z), rotate -90 deg on X so it lies flat
        if extents[1] > extents[2]:
            x = v[:, 0].copy()
            y = v[:, 1].copy()
            z = v[:, 2].copy()
            v[:, 0] = x
            v[:, 1] = -z
            v[:, 2] = y
    elif orientation == "upright":
        # Force model to stand upright
        if extents[2] > extents[1] * 1.5:
            x = v[:, 0].copy()
            y = v[:, 1].copy()
            z = v[:, 2].copy()
            v[:, 0] = x
            v[:, 1] = z
            v[:, 2] = -y

    # Center in X and Z
    b_min = v.min(axis=0)
    b_max = v.max(axis=0)
    center_x = (b_min[0] + b_max[0]) / 2.0
    center_z = (b_min[2] + b_max[2]) / 2.0
    v[:, 0] -= center_x
    v[:, 2] -= center_z

    # Ground feet / base at Y = 0
    v[:, 1] -= b_min[1]

    # Scale to standard height if specified
    current_height = v[:, 1].max()
    if current_height > 1e-4 and target_height > 0:
        scale_factor = target_height / current_height
        v *= scale_factor

    # Flatten the very bottom vertices (lowest 1.5% Y) so statue stands solidly without wobble
    if flatten_bottom:
        y_threshold = v[:, 1].min() + (v[:, 1].max() - v[:, 1].min()) * 0.015
        bottom_mask = v[:, 1] <= y_threshold
        if bottom_mask.sum() > 0:
            v[bottom_mask, 1] = 0.0

    m.vertices = v
    trimesh.repair.fix_normals(m)
    return m

def add_statue_pedestal(
    mesh: trimesh.Trimesh,
    shape: str = "round",
    pedestal_height: float = 0.05,
    margin_ratio: float = 0.12
) -> Tuple[trimesh.Trimesh, Optional[trimesh.Trimesh]]:
    """
    Creates an elegant, solid statue pedestal (đế tượng) at the bottom.
    Returns: (combined_mesh, pedestal_mesh_only)
    """
    if shape not in ("round", "square", "chamfered", "disc"):
        return mesh, None

    v = mesh.vertices
    b_min = v.min(axis=0)
    b_max = v.max(axis=0)
    
    lower_verts = v[v[:, 1] <= b_min[1] + (b_max[1] - b_min[1]) * 0.2]
    if len(lower_verts) == 0:
        lower_verts = v
    
    rad_x = (lower_verts[:, 0].max() - lower_verts[:, 0].min()) / 2.0
    rad_z = (lower_verts[:, 2].max() - lower_verts[:, 2].min()) / 2.0
    base_radius = max(rad_x, rad_z) * (1.0 + margin_ratio)
    base_radius = max(base_radius, 0.25)

    if shape in ("round", "disc"):
        pedestal = trimesh.creation.cylinder(
            radius=base_radius,
            height=pedestal_height,
            sections=48
        )
        pedestal.apply_translation([0, -pedestal_height / 2.0, 0])
    elif shape == "chamfered":
        p1 = trimesh.creation.cylinder(radius=base_radius * 1.08, height=pedestal_height * 0.5, sections=48)
        p1.apply_translation([0, -pedestal_height * 0.75, 0])
        p2 = trimesh.creation.cylinder(radius=base_radius, height=pedestal_height * 0.5, sections=48)
        p2.apply_translation([0, -pedestal_height * 0.25, 0])
        pedestal = trimesh.util.concatenate([p1, p2])
    else:
        pedestal = trimesh.creation.box(
            extents=[base_radius * 2.0, pedestal_height, base_radius * 2.0]
        )
        pedestal.apply_translation([0, -pedestal_height / 2.0, 0])

    pedestal.apply_translation([0, pedestal_height, 0])
    mesh_raised = mesh.copy()
    mesh_raised.apply_translation([0, pedestal_height, 0])

    combined = trimesh.util.concatenate([mesh_raised, pedestal])
    trimesh.repair.fix_normals(combined)
    return combined, pedestal

def decimate_mesh_for_statue(
    mesh: trimesh.Trimesh,
    target_faces: int = 50000
) -> trimesh.Trimesh:
    """
    Decimates mesh to target polygon count for fast mobile/browser 60fps rendering.
    Preserves UVs and vertex colors if present.
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh

    try:
        decimated = mesh.simplify_quadric_decimation(face_count=target_faces)
        trimesh.repair.fix_normals(decimated)
        print(f"[StatueOptimizer] Decimated from {len(mesh.faces)} -> {len(decimated.faces)} faces")
        return decimated
    except Exception as e:
        print(f"[StatueOptimizer] Warning: Decimation failed ({e}), returning original mesh")
        return mesh

def segment_statue_parts(
    mesh: trimesh.Trimesh,
    has_pedestal: bool = False
) -> Dict[str, Any]:
    """
    Intelligently segments the statue mesh into paintable anatomical/spatial parts:
    - Head & Face (Đầu & Khuôn mặt)
    - Hair / Headwear (Tóc / Mũ)
    - Upper Torso / Clothing (Thân trên / Áo)
    - Lower Torso / Legs (Thân dưới / Quần / Chân)
    - Arms & Hands (Tay & Cánh tay)
    - Pedestal / Base (Đế tượng nếu có)
    """
    v = mesh.vertices.copy()
    f = mesh.faces.copy()
    num_verts = len(v)
    num_faces = len(f)

    face_centers = v[f].mean(axis=1)
    face_normals = mesh.face_normals

    b_min = v.min(axis=0)
    b_max = v.max(axis=0)
    total_height = max(b_max[1] - b_min[1], 1e-4)

    norm_y = (face_centers[:, 1] - b_min[1]) / total_height
    norm_x = (face_centers[:, 0] - (b_min[0] + b_max[0]) / 2.0)
    norm_z = (face_centers[:, 2] - (b_min[2] + b_max[2]) / 2.0)
    radial_dist = np.sqrt(norm_x**2 + norm_z**2)

    face_part_ids = np.zeros(num_faces, dtype=int)

    if has_pedestal or b_min[1] < 0.05:
        base_mask = (norm_y <= 0.06)
        face_part_ids[base_mask] = 7

    top_mask = (norm_y > 0.72) & (face_part_ids == 0)
    if top_mask.sum() > 0:
        face_mask = top_mask & (face_normals[:, 2] > 0.1) & (radial_dist < radial_dist[top_mask].mean() * 1.1)
        hair_mask = top_mask & (~face_mask)
        face_part_ids[face_mask] = 0
        face_part_ids[hair_mask] = 1

    mid_upper_mask = (norm_y > 0.42) & (norm_y <= 0.72) & (face_part_ids == 0)
    if mid_upper_mask.sum() > 0:
        p70 = np.percentile(radial_dist[mid_upper_mask], 70)
        arms_mask = mid_upper_mask & (radial_dist > p70)
        torso_mask = mid_upper_mask & (~arms_mask)
        face_part_ids[torso_mask] = 2
        face_part_ids[arms_mask] = 4

    lower_mask = (norm_y <= 0.42) & (face_part_ids == 0)
    legs_mask = lower_mask & (norm_y > 0.15)
    feet_mask = lower_mask & (~legs_mask)
    face_part_ids[legs_mask] = 3
    face_part_ids[feet_mask] = 5

    unassigned = (face_part_ids == 0) & (~top_mask)
    face_part_ids[unassigned] = 2

    unique_ids = sorted(np.unique(face_part_ids).tolist())
    submeshes: Dict[str, trimesh.Trimesh] = {}
    part_info = []

    vertex_colors = np.ones((num_verts, 4), dtype=np.uint8) * 255
    vert_counts = np.zeros(num_verts, dtype=int)
    vert_colors_acc = np.zeros((num_verts, 3), dtype=float)

    for pid in unique_ids:
        palette_entry = STATUE_PALETTE[pid % len(STATUE_PALETTE)]
        p_name = palette_entry["name"]
        p_hex = palette_entry["hex"]
        p_rgb = palette_entry["rgb"]

        p_faces_mask = (face_part_ids == pid)
        if p_faces_mask.sum() == 0:
            continue

        try:
            subm = mesh.submesh([p_faces_mask], append=True)
            subm_name = f"Part_{pid:02d}_{p_name.split(' ')[0]}"
            submeshes[subm_name] = subm

            part_info.append({
                "part_id": int(pid),
                "name": p_name,
                "submesh_name": subm_name,
                "hex_color": p_hex,
                "rgb_color": p_rgb,
                "face_count": int(p_faces_mask.sum()),
                "vertex_count": int(len(subm.vertices))
            })
        except Exception as e:
            print(f"[StatueOptimizer] Submesh extract error {pid}: {e}")

        p_faces = f[p_faces_mask]
        for face in p_faces:
            for vi in face:
                vert_colors_acc[vi] += p_rgb
                vert_counts[vi] += 1

    valid_verts = vert_counts > 0
    vert_colors_acc[valid_verts] /= vert_counts[valid_verts, None]
    vertex_colors[valid_verts, :3] = np.clip(vert_colors_acc[valid_verts], 0, 255).astype(np.uint8)

    return {
        "face_part_ids": face_part_ids,
        "submeshes": submeshes,
        "part_info": part_info,
        "vertex_colors": vertex_colors,
        "num_parts_detected": len(part_info)
    }

def extract_and_optimize_outer_shell(
    mesh: trimesh.Trimesh,
    max_texture_dim: int = 2048
) -> trimesh.Trimesh:
    """
    Extracts and optimizes the outer shell of a 3D model for lightweight painting & WebGL:
    1. Removes degenerate (zero-area) triangles.
    2. Removes duplicate / coincident overlapping faces.
    3. Retains clean exterior surface shell with valid normals.
    4. Optimizes texture size (resizing massive 4K textures down to standard 2K)
       and converts opaque RGBA textures to RGB for smaller footprint.
    """
    m = mesh.copy()

    # 1. Advanced Hidden & Interior Geometry Removal (via FaceParsing mesh_cleanup)
    cleaned_by_faceparsing = False
    try:
        import sys
        faceparsing_dir = Path(__file__).resolve().parent.parent / "FaceParsing"
        if faceparsing_dir.exists() and str(faceparsing_dir) not in sys.path:
            sys.path.insert(0, str(faceparsing_dir))
        from app.mesh_cleanup import clean_mesh as faceparsing_clean_mesh
        cleaned_mesh, report, _ = faceparsing_clean_mesh(m, remove_hidden=True, views=16, resolution=256)
        if report.get("applied", False):
            m = cleaned_mesh
            cleaned_by_faceparsing = True
            print(f"[StatueOptimizer] FaceParsing clean_mesh applied: removed {report.get('removed_faces', 0)} faces "
                  f"({report.get('hidden', 0)} hidden, {report.get('degenerate', 0)} degen, {report.get('duplicate', 0)} dup), "
                  f"patched {report.get('patched', 0)} faces.")
    except Exception as err:
        print(f"[StatueOptimizer] FaceParsing clean_mesh fallback ({err})")

    if not cleaned_by_faceparsing:
        # Native fallback: Clean degenerate & duplicate triangles
        vertices = np.asarray(m.vertices, dtype=np.float64)
        faces = np.asarray(m.faces, dtype=np.int64)
        span = float(np.ptp(vertices, axis=0).max()) or 1.0

        corners = vertices[faces]
        twice_area = np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
        valid_mask = twice_area > (span ** 2) * 1e-14

        grid_key = np.round(vertices / max(span * 1e-5, 1e-12)).astype(np.int64)
        _, welded_ids = np.unique(grid_key, axis=0, return_inverse=True)

        identity = welded_ids[:, None]
        if hasattr(m, "visual") and hasattr(m.visual, "uv") and m.visual.uv is not None and len(m.visual.uv) == len(vertices):
            uv_discrete = np.round(np.asarray(m.visual.uv, dtype=np.float64) * 1e5).astype(np.int64)
            identity = np.concatenate([identity, uv_discrete], axis=1)

        _, fine_ids = np.unique(identity, axis=0, return_inverse=True)
        sorted_tri_ids = np.sort(fine_ids[faces], axis=1)
        _, first_seen_idx = np.unique(sorted_tri_ids, axis=0, return_index=True)
        is_not_duplicate = np.zeros(len(faces), dtype=bool)
        is_not_duplicate[first_seen_idx] = True

        keep_faces = valid_mask & is_not_duplicate
        if keep_faces.sum() > 0 and keep_faces.sum() < len(faces):
            m.update_faces(keep_faces)
            m.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(m)
    _ = m.vertex_normals

    # 3. Optimize texture map (downscale from 4K to 2K, compress)
    if hasattr(m, "visual") and hasattr(m.visual, "material") and m.visual.material:
        mat = m.visual.material
        img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is not None and isinstance(img, Image.Image):
            orig_w, orig_h = img.size
            if max(orig_w, orig_h) > max_texture_dim:
                scale = max_texture_dim / max(orig_w, orig_h)
                new_size = (int(orig_w * scale), int(orig_h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            if img.mode == "RGBA":
                extrema = img.getextrema()
                if len(extrema) == 4 and extrema[3][0] == 255 and extrema[3][1] == 255:
                    img = img.convert("RGB")
            mat.image = img
            mat.baseColorTexture = img

    return m


def export_all_statue_variants(
    base_mesh: trimesh.Trimesh,
    segmented_data: Dict[str, Any],
    output_dir: Path,
    stem: str,
    original_texture_mesh: Optional[trimesh.Trimesh] = None
) -> Dict[str, str]:
    """
    Exports all production-ready GLB variants into the output directory:
    - Pure plaster white GLB
    - Multi-part segmented GLB
    - ID-colored vertex GLB
    - Original Textured GLB
    - Lightweight Outer Shell GLB (Chỉ lấy phần vỏ ngoài siêu nhẹ)
    - Metadata manifest JSON
    - Complete ZIP package
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files = {}

    # Ensure base mesh has valid normals
    base_mesh.fix_normals()
    _ = base_mesh.vertex_normals

    # 1. Pure Plaster White GLB
    plaster_mesh = base_mesh.copy()
    plaster_mesh.fix_normals()
    _ = plaster_mesh.vertex_normals
    plaster_mat = trimesh.visual.material.PBRMaterial(
        name="Plaster_Statue_Material",
        baseColorFactor=[0.94, 0.93, 0.91, 1.0],
        roughnessFactor=0.88,
        metallicFactor=0.02
    )
    plaster_mesh.visual.material = plaster_mat
    plaster_glb_path = output_dir / f"{stem}_plaster.glb"
    plaster_scene = trimesh.Scene({"Statue_Plaster": plaster_mesh})
    plaster_glb_bytes = trimesh.exchange.gltf.export_glb(plaster_scene, include_normals=True)
    with open(plaster_glb_path, "wb") as fp:
        fp.write(plaster_glb_bytes)
    exported_files["plaster_glb"] = str(plaster_glb_path)

    # 2. Multi-Part Segmented GLB
    submeshes = segmented_data.get("submeshes", {})
    part_info = segmented_data.get("part_info", [])
    scene_parts = {}

    for info in part_info:
        sub_name = info["submesh_name"]
        if sub_name in submeshes:
            s_mesh = submeshes[sub_name].copy()
            s_mesh.fix_normals()
            _ = s_mesh.vertex_normals
            rgb = info["rgb_color"]
            part_mat = trimesh.visual.material.PBRMaterial(
                name=f"Mat_{sub_name}",
                baseColorFactor=[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, 1.0],
                roughnessFactor=0.75,
                metallicFactor=0.05
            )
            s_mesh.visual.material = part_mat
            scene_parts[sub_name] = s_mesh

    if scene_parts:
        segmented_scene = trimesh.Scene(scene_parts)
        segmented_glb_path = output_dir / f"{stem}_segmented.glb"
        segmented_glb_bytes = trimesh.exchange.gltf.export_glb(segmented_scene, include_normals=True)
        with open(segmented_glb_path, "wb") as fp:
            fp.write(segmented_glb_bytes)
        exported_files["segmented_glb"] = str(segmented_glb_path)

    # 3. ID Colored Mesh
    id_mesh = base_mesh.copy()
    id_mesh.fix_normals()
    _ = id_mesh.vertex_normals
    v_colors = segmented_data.get("vertex_colors")
    if v_colors is not None and len(v_colors) == len(id_mesh.vertices):
        id_mesh.visual = trimesh.visual.ColorVisuals(mesh=id_mesh, vertex_colors=v_colors)
        id_glb_path = output_dir / f"{stem}_id_colored.glb"
        id_scene = trimesh.Scene({"Statue_ID_Mask": id_mesh})
        id_glb_bytes = trimesh.exchange.gltf.export_glb(id_scene, include_normals=True)
        with open(id_glb_path, "wb") as fp:
            fp.write(id_glb_bytes)
        exported_files["id_colored_glb"] = str(id_glb_path)

    # 4. Textured GLB
    if original_texture_mesh is not None:
        original_texture_mesh.fix_normals()
        _ = original_texture_mesh.vertex_normals
        textured_glb_path = output_dir / f"{stem}_textured.glb"
        tex_scene = trimesh.Scene({"Statue_Textured": original_texture_mesh})
        tex_glb_bytes = trimesh.exchange.gltf.export_glb(tex_scene, include_normals=True)
        with open(textured_glb_path, "wb") as fp:
            fp.write(tex_glb_bytes)
        exported_files["textured_glb"] = str(textured_glb_path)

        # 5. Lightweight Outer Shell GLB (Chỉ lấy vỏ ngoài, đã làm sạch ruột và nén WebP)
        shell_mesh = extract_and_optimize_outer_shell(original_texture_mesh, max_texture_dim=2048)
        shell_glb_path = output_dir / f"{stem}_shell.glb"
        shell_scene = trimesh.Scene({"Statue_Outer_Shell": shell_mesh})
        shell_glb_bytes = trimesh.exchange.gltf.export_glb(shell_scene, include_normals=True, extension_webp=True)
        with open(shell_glb_path, "wb") as fp:
            fp.write(shell_glb_bytes)
        exported_files["shell_glb"] = str(shell_glb_path)
    else:
        shell_mesh = extract_and_optimize_outer_shell(base_mesh)
        shell_glb_path = output_dir / f"{stem}_shell.glb"
        shell_scene = trimesh.Scene({"Statue_Outer_Shell": shell_mesh})
        shell_glb_bytes = trimesh.exchange.gltf.export_glb(shell_scene, include_normals=True)
        with open(shell_glb_path, "wb") as fp:
            fp.write(shell_glb_bytes)
        exported_files["shell_glb"] = str(shell_glb_path)

    # 5. Metadata Manifest JSON
    manifest = {
        "statue_id": stem,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mesh_stats": {
            "num_vertices": int(len(base_mesh.vertices)),
            "num_faces": int(len(base_mesh.faces)),
            "bounding_box": {
                "min": [round(float(x), 4) for x in base_mesh.bounds[0]],
                "max": [round(float(x), 4) for x in base_mesh.bounds[1]],
                "height": round(float(base_mesh.bounds[1][1] - base_mesh.bounds[0][1]), 4)
            }
        },
        "painting_parts": part_info,
        "files": {k: Path(v).name for k, v in exported_files.items()}
    }
    manifest_path = output_dir / f"{stem}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    exported_files["manifest_json"] = str(manifest_path)

    # 6. Complete ZIP Package
    zip_path = output_dir / f"{stem}_statue_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fkey, fpath_str in exported_files.items():
            fpath = Path(fpath_str)
            if fpath.exists():
                zf.write(fpath, arcname=fpath.name)
    exported_files["package_zip"] = str(zip_path)

    return exported_files
