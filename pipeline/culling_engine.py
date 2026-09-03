"""
Culling & Geometry Cleaning Engine for UniRig 3D Statue Pipeline.

Provides high-performance, robust geometry preprocessing:
1. Degenerate Face Elimination (zero-area triangles).
2. Decal-Aware Coincident Triangle Culling (preserves UV & vertex color layers).
3. Multi-View Spherical Clearance Depth Probing (removes unseen interior cavities / buried geometry).
4. Silhouette Rim Protection & Sheet Cluster Thresholding (prevents surface holes).
5. Watertight Boundary Hole Repair (fans patches from existing vertices, 100% preserving UVs).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
import trimesh

_WELD_TOLERANCE = 1e-5
_VIEWS = 26
_RESOLUTION = 512
_BURIED_DEPTH = 5.0
_CLUSTER_FLOOR = 120
_PATCH_MAX_RIM = 256
_PATCH_ROUNDS = 6
_PATCH_ARRIVED_RIM = 8
_MAX_REMOVED_SHARE = 0.5


def _welded(vertices: np.ndarray, span: float) -> np.ndarray:
    """Vertices at the same point in space, given one id."""
    key = np.round(vertices / max(span * _WELD_TOLERANCE, 1e-12)).astype(np.int64)
    return np.unique(key, axis=0, return_inverse=True)[1].astype(np.int64)


def _spread(count: int) -> np.ndarray:
    """Fibonacci sphere directions evenly spread over a sphere."""
    i = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * i / count)
    around = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([
        np.cos(around) * np.sin(polar),
        np.cos(polar),
        np.sin(around) * np.sin(polar)
    ], axis=1)


def _rasterize(
    v_screen: np.ndarray,
    v_depth: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Z-buffered software triangle rasterizer.
    Returns (face_buf, bary_buf) flattened to height * width.
    """
    npix = height * width
    empty = (np.full(npix, -1, np.int64), np.zeros((npix, 3), np.float32))
    if len(faces) == 0:
        return empty

    tri = v_screen[faces]
    depth = v_depth[faces]

    x0, y0 = tri[:, 0, 0], tri[:, 0, 1]
    x1, y1 = tri[:, 1, 0], tri[:, 1, 1]
    x2, y2 = tri[:, 2, 0], tri[:, 2, 1]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    xmin = np.clip(np.floor(tri[:, :, 0].min(1)), 0, width - 1).astype(np.int32)
    xmax = np.clip(np.ceil(tri[:, :, 0].max(1)), 0, width - 1).astype(np.int32)
    ymin = np.clip(np.floor(tri[:, :, 1].min(1)), 0, height - 1).astype(np.int32)
    ymax = np.clip(np.ceil(tri[:, :, 1].max(1)), 0, height - 1).astype(np.int32)

    span = np.maximum(xmax - xmin + 1, ymax - ymin + 1)
    max_span = min(int(span.max()) if span.size else 1, max(width, height))
    keep = (
        (np.abs(area) > 1e-12)
        & (depth.min(1) > 1e-6)
        & (xmax >= xmin)
        & (ymax >= ymin)
        & (span <= max_span)
    )
    face_ids = np.where(keep)[0]
    if len(face_ids) == 0:
        return empty

    spans = span[face_ids]
    flat_parts, depth_parts, face_parts, bary_parts = [], [], [], []

    n_tiles = max(7, int(np.ceil(np.log2(max(max_span, 1)))) + 1)
    for tile in (2 ** np.arange(0, n_tiles)):
        sel = face_ids[spans <= 1] if tile == 1 else face_ids[(spans > tile // 2) & (spans <= tile)]
        if len(sel) == 0:
            continue

        oy, ox = np.meshgrid(np.arange(tile), np.arange(tile), indexing="ij")
        px = xmin[sel][:, None] + ox.ravel()[None, :]
        py = ymin[sel][:, None] + oy.ravel()[None, :]
        inside = (px <= xmax[sel][:, None]) & (py <= ymax[sel][:, None])

        cx, cy = px + 0.5, py + 0.5
        ax, ay = x0[sel][:, None], y0[sel][:, None]
        bx, by = x1[sel][:, None], y1[sel][:, None]
        gx, gy = x2[sel][:, None], y2[sel][:, None]
        inv_area = area[sel][:, None]
        w0 = ((bx - cx) * (gy - cy) - (gx - cx) * (by - cy)) / inv_area
        w1 = ((gx - cx) * (ay - cy) - (ax - cx) * (gy - cy)) / inv_area
        w2 = 1.0 - w0 - w1
        inside &= (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        d = (
            w0 * depth[sel, 0][:, None]
            + w1 * depth[sel, 1][:, None]
            + w2 * depth[sel, 2][:, None]
        )
        flat_parts.append((py * width + px)[inside])
        depth_parts.append(d[inside])
        face_parts.append(np.repeat(sel[:, None], px.shape[1], axis=1)[inside])
        bary_parts.append(np.stack([w0[inside], w1[inside], w2[inside]], axis=1))

    if not flat_parts:
        return empty

    flat = np.concatenate(flat_parts)
    d = np.concatenate(depth_parts)
    fid = np.concatenate(face_parts)
    bary = np.concatenate(bary_parts)

    zbuf = np.full(npix, np.inf)
    np.minimum.at(zbuf, flat, d)
    hit = d <= zbuf[flat] + 1e-12

    face_buf = np.full(npix, -1, np.int64)
    bary_buf = np.zeros((npix, 3), np.float32)
    face_buf[flat[hit]] = fid[hit]
    bary_buf[flat[hit]] = bary[hit]
    return face_buf, bary_buf


def _clearance(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: int = 16,
    resolution: int = 256,
) -> np.ndarray:
    """Calculates clearance distance behind outer front-most surface for each face."""
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    radius = float(np.linalg.norm(vertices - centre, axis=1).max()) * 1.05
    if radius <= 0:
        return np.full(len(faces), -np.inf)

    corners = vertices[faces]
    middle = corners.mean(axis=1)
    probes = np.concatenate(
        [middle] + [0.7 * corners[:, i] + 0.3 * middle for i in range(3)]
    )

    behind = np.full(len(faces), np.inf)
    for direction in _spread(views):
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(direction @ up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(direction, up)
        right /= np.linalg.norm(right)
        frame = np.stack([right, np.cross(right, direction), direction])

        def project(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            local = (points - centre) @ frame.T
            screen = np.stack([
                (local[:, 0] / radius * 0.5 + 0.5) * (resolution - 1),
                (0.5 - local[:, 1] / radius * 0.5) * (resolution - 1),
            ], axis=1)
            return screen, radius * 2 - local[:, 2]

        screen, depth = project(vertices)
        drawn, weights = _rasterize(screen, depth, faces, resolution, resolution)
        covered = drawn >= 0
        if not covered.any():
            continue
        buffer = np.full(resolution * resolution, np.inf)
        buffer[covered] = (depth[faces[drawn[covered]]] * weights[covered]).sum(axis=1)

        here, own = project(probes)
        column = np.clip(here[:, 0].astype(np.int32), 0, resolution - 1)
        row = np.clip(here[:, 1].astype(np.int32), 0, resolution - 1)
        gap = own - buffer[row * resolution + column]
        behind = np.minimum(behind, gap.reshape(4, len(faces)).min(axis=0))
    return behind


def _off_the_rim(faces: np.ndarray, hidden: np.ndarray, vertex_count: int) -> np.ndarray:
    """Stops one ring short of the visible surface to protect silhouettes."""
    if not hidden.any():
        return hidden
    touched = np.zeros(vertex_count, bool)
    touched[faces[~hidden].ravel()] = True
    return hidden & ~touched[faces].any(axis=1)


def _in_company(faces: np.ndarray, hidden: np.ndarray, floor: int) -> np.ndarray:
    """Keeps only the hidden faces that are part of a continuous run/cluster >= floor."""
    if not hidden.any():
        return hidden
    index = np.where(hidden)[0]
    kept = faces[index]
    rows = np.repeat(np.arange(len(index)), 3)
    incidence = sparse.csr_matrix(
        (np.ones(kept.size), (rows, kept.ravel())),
        shape=(len(index), int(faces.max()) + 1),
    )
    _, group = connected_components(incidence @ incidence.T, directed=False)
    big = np.bincount(group) >= floor
    survivors = np.zeros(len(faces), bool)
    survivors[index[big[group]]] = True
    return survivors


def _boundary(faces: np.ndarray) -> set:
    """Directed edges with no face on the opposite side."""
    if len(faces) == 0:
        return set()
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    seen = set(map(tuple, directed))
    return {(int(b), int(a)) for a, b in directed if (b, a) not in seen}


def _fill_new_holes(
    faces: np.ndarray,
    keep: np.ndarray,
    welded: np.ndarray,
    rim: int = _PATCH_MAX_RIM,
) -> np.ndarray:
    """Sews shut any opening that removing faces has opened without inventing new vertices."""
    kept = faces[keep]
    if len(kept) == 0:
        return np.empty((0, 3), np.int64)

    before = _boundary(welded[faces])
    joined = welded[kept]
    seam_w = np.vstack([joined[:, [0, 1]], joined[:, [1, 2]], joined[:, [2, 0]]])
    seam_r = np.vstack([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]])
    present = set(map(tuple, seam_w))

    step: Dict[int, List[Tuple[int, int, int]]] = {}
    arrived = set()
    for (wa, wb), (ra, rb) in zip(map(tuple, seam_w), map(tuple, seam_r)):
        if (wb, wa) in present:
            continue
        rim_edge = (int(wb), int(wa))
        if rim_edge in before:
            # An opening the model arrived with. Closed as well, but only when small enough
            arrived.add(rim_edge)
        step.setdefault(int(wb), []).append((int(wa), int(rb), int(ra)))
    if not step:
        return np.empty((0, 3), np.int64)

    patch: List[List[int]] = []
    used = set()
    for origin in list(step):
        for onward, first_real, next_real in list(step[origin]):
            if (origin, onward) in used:
                continue
            used.add((origin, onward))
            corners = [first_real]
            walked = [(origin, onward)]
            here, carried = onward, next_real
            while here != origin and len(corners) <= _PATCH_MAX_RIM:
                corners.append(carried)
                ahead = [o for o in step.get(here, []) if (here, o[0]) not in used]
                if not ahead:
                    break
                used.add((here, ahead[0][0]))
                walked.append((here, ahead[0][0]))
                here, carried = ahead[0][0], ahead[0][2]
            if here != origin or len(corners) < 3:
                continue
            touches_own = any(edge in arrived for edge in walked)
            allowed = _PATCH_ARRIVED_RIM if touches_own else rim
            if len(corners) <= allowed:
                for i in range(1, len(corners) - 1):
                    patch.append([corners[0], corners[i], corners[i + 1]])
    return np.asarray(patch, np.int64) if patch else np.empty((0, 3), np.int64)


def clean_floating_debris(
    mesh: trimesh.Trimesh,
    min_faces: int = 30
) -> Tuple[trimesh.Trimesh, int]:
    """
    Detects and eliminates small floating fragments / speck artifacts produced by
    Marching Cubes / FlexiCubes that are disconnected from the primary mesh body.
    Preserves all UVs, vertex colors, and materials.
    """
    if len(mesh.faces) == 0:
        return mesh, 0

    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    span = float(np.ptp(vertices, axis=0).max()) or 1.0
    welded = _welded(vertices, span)
    wf = welded[faces]

    rows = np.repeat(np.arange(len(faces)), 3)
    incidence = sparse.csr_matrix(
        (np.ones(wf.size), (rows, wf.ravel())),
        shape=(len(faces), int(wf.max()) + 1),
    )
    n_comp, labels = connected_components(incidence @ incidence.T, directed=False)
    counts = np.bincount(labels)

    keep_components = np.where(counts >= min_faces)[0]
    if len(keep_components) == 0:
        keep_components = [np.argmax(counts)]

    keep_faces = np.isin(labels, keep_components)
    removed_count = int((~keep_faces).sum())
    if removed_count == 0:
        return mesh, 0

    cleaned = mesh.copy()
    cleaned.update_faces(keep_faces)
    cleaned.remove_unreferenced_vertices()
    return cleaned, removed_count


def clean_and_cull_mesh(
    mesh: trimesh.Trimesh,
    remove_hidden: bool = True,
    remove_debris: bool = True,
    views: int = _VIEWS,
    resolution: int = _RESOLUTION,
    buried_depth: float = _BURIED_DEPTH,
    cluster_floor: int = _CLUSTER_FLOOR,
) -> Tuple[trimesh.Trimesh, Dict[str, Any], np.ndarray]:
    """
    Comprehensive geometry sanitation & exterior shell extraction:
    1. Removes degenerate zero-area triangles.
    2. Removes duplicate / back-to-back inverted triangles (preserving decal & UV layers).
    3. Multi-view Z-buffered spherical clearance probing (26 Fibonacci views at 512x512).
    4. Silhouette rim protection and contiguous cluster thresholding.
    5. Watertight boundary hole patching preserving UVs.
    6. Floating debris artifact elimination.
    7. Consistent outward normal orientation.
    """
    started = time.perf_counter()
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    before_faces, before_vertices = len(faces), len(vertices)
    span = float(np.ptp(vertices, axis=0).max()) or 1.0

    keep = np.ones(len(faces), bool)

    # 1. Degenerate triangles (zero or negligible area)
    corners = vertices[faces]
    twice_area = np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    degenerate = twice_area <= (span ** 2) * 1e-14
    keep &= ~degenerate

    # 2. Duplicate coincident & back-to-back inverted triangles (UV & vertex color aware)
    identity = _welded(vertices, span)[:, None]
    for attribute in (getattr(mesh.visual, "uv", None),
                      getattr(mesh.visual, "vertex_colors", None)):
        if attribute is not None and len(attribute) == len(vertices):
            fine = np.round(np.asarray(attribute, np.float64) * 1e6).astype(np.int64)
            identity = np.concatenate([identity, fine], axis=1)
    identity = np.unique(identity, axis=0, return_inverse=True)[1]

    signature = np.sort(identity[faces], axis=1)
    _, first = np.unique(signature, axis=0, return_index=True)
    repeated = np.ones(len(faces), bool)
    repeated[first] = False
    keep &= ~repeated

    # 3. Unseen interior / buried geometry removal (26 spherical views, 512 resolution)
    hidden = np.zeros(len(faces), bool)
    checked = 0
    if remove_hidden and keep.any():
        live = np.where(keep)[0]
        pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edge = np.linalg.norm(vertices[pairs[:, 1]] - vertices[pairs[:, 0]], axis=1)
        median_edge = float(np.median(edge[edge > 0])) if (edge > 0).any() else span * 1e-3
        behind = _clearance(vertices, faces[live], views=views, resolution=resolution)
        candidate = np.zeros(len(faces), bool)
        candidate[live[behind > buried_depth * median_edge]] = True
        candidate |= ~keep
        hidden = _off_the_rim(faces, candidate, len(vertices))
        hidden = _in_company(faces, hidden, cluster_floor)
        hidden &= keep
        checked = len(live)
        keep &= ~hidden

    # 4. Patch new boundary openings
    welded = _welded(vertices, span)
    original_rim = _boundary(welded[faces])
    added = _fill_new_holes(faces, keep, welded)
    for _ in range(_PATCH_ROUNDS):
        surface = np.vstack([faces[keep], added]) if len(added) else faces[keep]
        still_open = _boundary(welded[surface]) - original_rim
        if not still_open:
            break
        exposed = {corner for edge in still_open for corner in edge}
        restore = (~keep) & np.isin(welded[faces], list(exposed)).any(axis=1)
        if not restore.any():
            break
        keep |= restore
        added = _fill_new_holes(faces, keep, welded)

    gone = ~keep
    report: Dict[str, Any] = {
        "faces_before": before_faces,
        "vertices_before": before_vertices,
        "patched": int(len(added)),
        "degenerate": int((degenerate & gone).sum()),
        "duplicate": int((repeated & gone).sum()),
        "hidden": int((hidden & gone).sum()),
        "kept_back": int((hidden & keep).sum()),
        "checked": checked,
        "debris_removed": 0,
        "applied": True,
    }

    removed = int((~keep).sum())
    if removed == 0 or removed > _MAX_REMOVED_SHARE * before_faces:
        report.update(
            applied=False,
            faces_after=before_faces,
            vertices_after=before_vertices,
            removed_faces=0,
            removed_vertices=0,
            patched=0,
            kept_back=0,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2)
        )
        return mesh, report, np.arange(before_vertices)

    cleaned = mesh.copy()
    if len(added):
        cleaned.faces = np.vstack([faces[keep], added])
    else:
        cleaned.update_faces(keep)

    survived = np.zeros(before_vertices, bool)
    survived[np.asarray(cleaned.faces).ravel()] = True
    cleaned.update_vertices(survived)

    # 5. Remove disconnected floating debris specks
    if remove_debris:
        cleaned, debris_removed = clean_floating_debris(cleaned, min_faces=30)
        report["debris_removed"] = debris_removed

    # 6. Ensure normals and winding are consistent
    trimesh.repair.fix_normals(cleaned)
    trimesh.repair.fix_winding(cleaned)

    report.update(
        faces_after=int(len(cleaned.faces)),
        vertices_after=int(len(cleaned.vertices)),
        removed_faces=before_faces - int(len(cleaned.faces)),
        removed_vertices=before_vertices - int(len(cleaned.vertices)),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return cleaned, report, np.where(survived)[0]
