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
from io import BytesIO

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

def clean_and_repair_mesh(mesh: trimesh.Trimesh, cull_hidden: bool = True) -> trimesh.Trimesh:
    """Removes duplicate vertices, unreferenced vertices, repairs normals, and culls interior buried geometry."""
    m = mesh.copy()
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    m.remove_infinite_values()
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_winding(m)

    if cull_hidden and len(m.faces) > 500:
        try:
            from pipeline.culling_engine import clean_and_cull_mesh
            cleaned, report, _ = clean_and_cull_mesh(m, remove_hidden=True, views=16, resolution=256)
            if report.get("applied", False):
                m = cleaned
                print(f"[StatueOptimizer] culling_engine: removed {report.get('removed_faces', 0)} faces "
                      f"({report.get('hidden', 0)} hidden, {report.get('degenerate', 0)} degen, {report.get('duplicate', 0)} dup), "
                      f"patched {report.get('patched', 0)} faces.")
        except Exception as e:
            print(f"[StatueOptimizer] culling_engine notice: {e}")

    trimesh.repair.fix_normals(m)
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
        # Vỏ rỗng có hai lớp đáy rất gần nhau (đo được 0.195% chiều cao, trong khi dải
        # 1.5% rộng gấp 7.6 lần). Ép cả hai về y=0 làm chúng trùng khít -> z-fighting.
        # Chỉ ép lớp đáy NGOÀI: bỏ qua đỉnh nào có mặt kề hướng lên.
        max_ny = np.full(len(v), -2.0, dtype=np.float32)
        face_ny = m.face_normals[:, 1]
        for k in range(3):
            np.maximum.at(max_ny, m.faces[:, k], face_ny)
        bottom_mask &= (max_ny <= 0.5)
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

    # trimesh.creation.cylinder dựng hình trụ theo trục Z, trong khi tượng là Y-up,
    # nên phải xoay 90° quanh X thì đế mới nằm ngang trên mặt phẳng.
    lay_flat = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])

    if shape in ("round", "disc"):
        pedestal = trimesh.creation.cylinder(
            radius=base_radius,
            height=pedestal_height,
            sections=48,
            transform=lay_flat
        )
        pedestal.apply_translation([0, -pedestal_height / 2.0, 0])
    elif shape == "chamfered":
        p1 = trimesh.creation.cylinder(radius=base_radius * 1.08, height=pedestal_height * 0.5,
                                       sections=48, transform=lay_flat)
        p1.apply_translation([0, -pedestal_height * 0.75, 0])
        p2 = trimesh.creation.cylinder(radius=base_radius, height=pedestal_height * 0.5,
                                       sections=48, transform=lay_flat)
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

    # Topological smoothing of part labels (FaceParsing mesh_segmenter approach)
    if hasattr(mesh, "face_adjacency") and len(mesh.face_adjacency) > 0:
        try:
            import scipy.sparse as sp
            adj = mesh.face_adjacency
            num_f = len(face_part_ids)
            a, b = adj[:, 0], adj[:, 1]
            self_idx = np.arange(num_f)
            rows = np.concatenate([a, b, self_idx])
            cols = np.concatenate([b, a, self_idx])
            data = np.ones(len(rows), dtype=np.float32)
            adj_mat = sp.csr_matrix((data, (rows, cols)), shape=(num_f, num_f))
            num_classes = int(face_part_ids.max()) + 1
            cur_labels = face_part_ids.copy()
            for _ in range(5):
                one_hot = sp.csr_matrix(
                    (np.ones(num_f, dtype=np.float32), (self_idx, cur_labels)),
                    shape=(num_f, num_classes)
                )
                neighborhood_votes = adj_mat @ one_hot
                cur_labels = np.asarray(neighborhood_votes.argmax(axis=1)).ravel()
            face_part_ids = cur_labels
        except Exception as e:
            print(f"[StatueOptimizer] Topology label smoothing notice: {e}")

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

def optimize_material_textures(
    mesh: trimesh.Trimesh,
    max_texture_dim: int = 2048,
    jpeg_quality: int = 92
) -> trimesh.Trimesh:
    """
    Thu nhỏ mọi texture của vật liệu về tối đa `max_texture_dim` và mã hoá lại sang JPEG.

    Phải là JPEG chứ không phải WebP: trimesh chỉ ghi được WebP qua phần mở rộng
    EXT_texture_webp và nó nằm trong `extensionsRequired`, nên các viewer phổ thông
    (macOS Preview/Quick Look, Windows 3D Viewer...) bỏ luôn texture — model hiện ra
    đen bóng vì metallicFactor = 1.0. Định dạng nào khác JPEG thì trimesh ghi ra PNG,
    nặng gấp ~5 lần. WebP mà trimesh ghi cũng chỉ là lossy quality 80 nên không mất chất.
    """
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    if mat is None:
        return mesh

    opaque = str(getattr(mat, "alphaMode", None) or "OPAQUE").upper() == "OPAQUE"
    for slot in ("baseColorTexture", "metallicRoughnessTexture", "emissiveTexture", "normalTexture"):
        img = getattr(mat, slot, None)
        if not isinstance(img, Image.Image):
            continue

        if max(img.size) > max_texture_dim:
            scale = max_texture_dim / max(img.size)
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.Resampling.LANCZOS)

        # Kênh alpha ở đây chỉ là vùng đệm giữa các mảnh UV atlas; alphaMode OPAQUE
        # nghĩa là renderer bỏ qua nó, nên khử được để dùng JPEG.
        if img.mode == "RGBA" and opaque:
            img = img.convert("RGB")

        if img.mode == "RGB":
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
            buf.seek(0)
            img = Image.open(buf)
            img.load()

        setattr(mat, slot, img)

    # Tượng để tô màu thì không nên là vật liệu kim loại: khi model đi qua chuỗi
    # chuyển đổi USDZ của macOS (Preview/Quick Look), metallicRoughnessTexture hay
    # bị rớt và metallic mặc định về 1.0 toàn khối -> tượng hiện xám kim loại, mất
    # hết màu khuếch tán từ baseColorTexture. Đo roughness trung bình từ map (kênh
    # G) rồi bỏ hẳn map, chuyển sang hệ số vô hướng để MỌI viewer (three.js,
    # Blender, USDZ, Windows 3D Viewer) render giống nhau, đồng thời giảm dung lượng.
    mr_img = getattr(mat, "metallicRoughnessTexture", None)
    if isinstance(mr_img, Image.Image):
        # Giá trị thật nằm trong kênh G của map; đo trước khi bỏ map đi.
        mat.roughnessFactor = float(np.asarray(mr_img.convert("RGB"))[..., 1].mean() / 255.0)
        mat.metallicRoughnessTexture = None
    mat.metallicFactor = 0.0

    return mesh


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

    # 1. Advanced Hidden & Interior Geometry Removal (via pipeline.culling_engine)
    cleaned_by_culling = False
    try:
        from pipeline.culling_engine import clean_and_cull_mesh
        cleaned_mesh, report, _ = clean_and_cull_mesh(m, remove_hidden=True, views=16, resolution=256)
        if report.get("applied", False):
            m = cleaned_mesh
            cleaned_by_culling = True
            print(f"[StatueOptimizer] culling_engine applied: removed {report.get('removed_faces', 0)} faces "
                  f"({report.get('hidden', 0)} hidden, {report.get('degenerate', 0)} degen, {report.get('duplicate', 0)} dup), "
                  f"patched {report.get('patched', 0)} faces.")
    except Exception as err:
        print(f"[StatueOptimizer] culling_engine fallback ({err})")

    if not cleaned_by_culling:
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

    # 2. Optimize texture maps (downscale + re-encode)
    optimize_material_textures(m, max_texture_dim=max_texture_dim)

    return m


def _bake_uv_from_source(
    dec_mesh: trimesh.Trimesh,
    source: trimesh.Trimesh,
    source_uv: np.ndarray,
    max_texture_dim: int = 1536
) -> trimesh.Trimesh:
    """
    Gán lại UV cho mesh đã rút gọn bằng cách chiếu ngược lên mesh nguồn:
    mỗi mặt lấy UV từ đúng MỘT tam giác nguồn (tam giác gần tâm mặt nhất) rồi nội suy
    barycentric không kẹp cho cả 3 đỉnh, nên không có mặt nào vắt qua hai mảnh UV rời nhau.
    """
    corners = dec_mesh.triangles
    _, _, src_id = trimesh.proximity.closest_point(source, corners.mean(axis=1))
    src_tri = np.repeat(source.triangles[src_id], 3, axis=0)
    pts = corners.reshape(-1, 3)

    a = src_tri[:, 0]
    e0, e1, dv = src_tri[:, 1] - a, src_tri[:, 2] - a, pts - a
    d00 = (e0 * e0).sum(1); d01 = (e0 * e1).sum(1); d11 = (e1 * e1).sum(1)
    d20 = (dv * e0).sum(1); d21 = (dv * e1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bary = np.stack([1.0 - v - w, v, w], axis=1)
    uv = (np.repeat(source_uv[source.faces[src_id]], 3, axis=0) * bary[:, :, None]).sum(axis=1)

    # Hàn góc thành đỉnh theo DUNG SAI UV thật thay vì làm tròn theo lưới:
    # làm tròn bỏ sót cặp sát nhau nhưng rơi khác bin, nên phình 22.438 -> 61.214 đỉnh.
    # Gom các góc quanh cùng một đỉnh khi UV cách nhau <= uv_tol; đỉnh chỉ tách khi
    # thật sự vắt qua ranh giới hai mảnh atlas.
    uv_tol = 0.5 / float(max_texture_dim)   # 0,5 pixel -> dưới 1 texel, không thể nhìn thấy
    corner_vert = dec_mesh.faces.reshape(-1)
    order = np.argsort(corner_vert, kind="stable")
    cv = corner_vert[order]
    bounds = np.flatnonzero(np.r_[True, cv[1:] != cv[:-1], True])
    label = np.empty(len(order), dtype=np.int64)
    n_out = 0
    for i in range(len(bounds) - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        reps = []
        for j in idx:
            u = uv[j]
            hit = -1
            for k, r in enumerate(reps):
                if abs(u[0] - r[0]) <= uv_tol and abs(u[1] - r[1]) <= uv_tol:
                    hit = k
                    break
            if hit < 0:
                reps.append(u)
                hit = len(reps) - 1
            label[j] = n_out + hit
        n_out += len(reps)

    cnt = np.bincount(label, minlength=n_out).astype(np.float64)[:, None]
    vuv = np.zeros((n_out, 2)); np.add.at(vuv, label, uv);  vuv /= cnt
    vpos = np.zeros((n_out, 3)); np.add.at(vpos, label, pts); vpos /= cnt

    baked = trimesh.Trimesh(vertices=vpos, faces=label.reshape(-1, 3), process=False)
    baked.visual = trimesh.visual.TextureVisuals(uv=vuv, material=source.visual.material)
    return baked


def create_max_optimized_shell(
    mesh: trimesh.Trimesh,
    target_faces: int = 45000,
    max_texture_dim: int = 1536
) -> trimesh.Trimesh:
    """
    Creates the ultimate lightweight, maximum-optimized outer shell model:
    1. Bóc sạch 100% ruột và các khoang ẩn bằng culling_engine (FaceParsing clearance culling).
    2. Rút gọn số lượng mặt (Polygon decimation) từ ~280k xuống ~45k mặt bằng fast_simplification.
    3. Chiếu ngược UV từ mesh nguồn để giữ nguyên Texture gốc AI sau khi rút gọn.
    4. Tối ưu kích thước texture sang chuẩn 1.5K sắc nét và khử kênh Alpha dư thừa.
    5. Xuất định dạng GLB WebP siêu nhẹ (~3 - 4 MB) tải tức thì trên Mobile và Web.
    """
    shell = extract_and_optimize_outer_shell(mesh, max_texture_dim=max_texture_dim)

    # If shell already has fewer faces than target, return as is
    if len(shell.faces) <= target_faces or target_faces <= 0:
        return shell

    try:
        import fast_simplification

        # Hàn các đỉnh trùng vị trí TRƯỚC khi rút gọn: mesh do AI sinh ra bị tách đỉnh tại
        # đường khâu UV (~2.2 đỉnh / vị trí), rút gọn thẳng sẽ xé rách bề mặt (mất ~39%
        # diện tích) và kéo theo texture bị bệt.
        welded = shell.copy()
        welded.merge_vertices(merge_tex=True, merge_norm=True)

        v_dec, f_dec = fast_simplification.simplify(
            np.ascontiguousarray(welded.vertices, dtype=np.float32),
            np.ascontiguousarray(welded.faces, dtype=np.int64),
            target_count=target_faces,
            agg=5.0
        )
        dec_mesh = trimesh.Trimesh(vertices=v_dec, faces=f_dec, process=False)

        # Lấy lại UV từ mesh nguồn (hàn đỉnh đã làm mất UV gốc)
        orig_uv = getattr(getattr(shell, "visual", None), "uv", None)
        if orig_uv is not None and len(orig_uv) == len(shell.vertices):
            dec_mesh = _bake_uv_from_source(dec_mesh, shell, np.asarray(orig_uv, dtype=np.float64), max_texture_dim=max_texture_dim)
        elif hasattr(shell, "visual"):
            dec_mesh.visual = shell.visual.copy()

        trimesh.repair.fix_normals(dec_mesh)
        _ = dec_mesh.vertex_normals
        print(f"[StatueOptimizer] Max-optimized shell: decimated {len(shell.faces)} -> {len(dec_mesh.faces)} faces")
        return dec_mesh
    except Exception as e:
        print(f"[StatueOptimizer] Max-optimized shell decimation fallback ({e})")
        return shell


def _strip_dead_uv(mesh):
    """Bỏ UV khi material không lấy màu từ texture map nào — UV chỉ là dữ liệu chết."""
    mat = getattr(mesh.visual, "material", None)
    if mat is None or getattr(mesh.visual, "uv", None) is None:
        return mesh
    if any(getattr(mat, a, None) is not None for a in
           ("baseColorTexture", "emissiveTexture", "metallicRoughnessTexture",
            "normalTexture", "occlusionTexture")):
        return mesh
    mesh.visual = trimesh.visual.TextureVisuals(uv=None, material=mat)
    return mesh


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
    - Original Textured GLB (with WebP compression)
    - Lightweight Outer Shell GLB (Chỉ lấy phần vỏ ngoài siêu nhẹ)
    - Ultra-Optimized Outer Shell GLB (Vỏ ngoài + Tối ưu tối đa ~45k mặt + Texture gốc AI)
    - Metadata manifest JSON with Mesh Integrity metrics
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
    plaster_mesh = _strip_dead_uv(plaster_mesh)
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
            s_mesh = _strip_dead_uv(s_mesh)
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
        optimize_material_textures(original_texture_mesh, max_texture_dim=4096)
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
        shell_glb_bytes = trimesh.exchange.gltf.export_glb(shell_scene, include_normals=True)
        with open(shell_glb_path, "wb") as fp:
            fp.write(shell_glb_bytes)
        exported_files["shell_glb"] = str(shell_glb_path)

        # 6. Ultra-Optimized Outer Shell GLB (Vỏ ngoài + Tối ưu tối đa ~45k mặt + Texture gốc AI)
        shell_opt_mesh = create_max_optimized_shell(original_texture_mesh, target_faces=45000, max_texture_dim=1536)
        shell_opt_glb_path = output_dir / f"{stem}_shell_optimized.glb"
        shell_opt_scene = trimesh.Scene({"Statue_Shell_Max_Optimized": shell_opt_mesh})
        shell_opt_glb_bytes = trimesh.exchange.gltf.export_glb(shell_opt_scene, include_normals=True)
        with open(shell_opt_glb_path, "wb") as fp:
            fp.write(shell_opt_glb_bytes)
        exported_files["shell_optimized_glb"] = str(shell_opt_glb_path)
    else:
        shell_mesh = extract_and_optimize_outer_shell(base_mesh)
        shell_glb_path = output_dir / f"{stem}_shell.glb"
        shell_scene = trimesh.Scene({"Statue_Outer_Shell": shell_mesh})
        shell_glb_bytes = trimesh.exchange.gltf.export_glb(shell_scene, include_normals=True)
        with open(shell_glb_path, "wb") as fp:
            fp.write(shell_glb_bytes)
        exported_files["shell_glb"] = str(shell_glb_path)

        shell_opt_mesh = create_max_optimized_shell(base_mesh, target_faces=45000)
        shell_opt_glb_path = output_dir / f"{stem}_shell_optimized.glb"
        shell_opt_scene = trimesh.Scene({"Statue_Shell_Max_Optimized": shell_opt_mesh})
        shell_opt_glb_bytes = trimesh.exchange.gltf.export_glb(shell_opt_scene, include_normals=True)
        with open(shell_opt_glb_path, "wb") as fp:
            fp.write(shell_opt_glb_bytes)
        exported_files["shell_optimized_glb"] = str(shell_opt_glb_path)

    # 5. Metadata Manifest JSON with Mesh Integrity Metrics
    from pipeline.mesh_integrity import evaluate_mesh_integrity
    integrity_report = evaluate_mesh_integrity(base_mesh)

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
        "mesh_integrity": integrity_report,
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
