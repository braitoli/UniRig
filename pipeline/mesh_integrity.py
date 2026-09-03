"""
Mesh Integrity & Geometric Quality Inspector for UniRig 3D Statue Pipeline.

Evaluates:
1. Watertightness & Boundary Seams.
2. Degenerate Triangles (zero or sliver area).
3. Normal Consistency & Flipped Faces.
4. Edge Stretch / Aspect Ratio distribution.
5. Overall Mesh Health Score (0 - 100%).
"""

from typing import Any, Dict
import numpy as np
import trimesh


def evaluate_mesh_integrity(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Analyzes mesh topological and geometric health.
    Returns structured metrics and a quality score for manufacturing & 3D painting.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_verts = len(vertices)
    n_faces = len(faces)

    if n_verts == 0 or n_faces == 0:
        return {
            "is_valid": False,
            "quality_score": 0,
            "error": "Empty mesh"
        }

    span = float(np.ptp(vertices, axis=0).max()) or 1.0

    # 1. Degenerate faces check
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    degenerate_count = int((twice_area <= (span ** 2) * 1e-14).sum())

    # 2. Edge statistics & aspect ratio
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge_lengths = np.linalg.norm(vertices[pairs[:, 1]] - vertices[pairs[:, 0]], axis=1)
    valid_edges = edge_lengths[edge_lengths > 1e-12]
    median_edge = float(np.median(valid_edges)) if len(valid_edges) > 0 else span * 1e-3

    # Triangular aspect ratio (longest edge / shortest altitude)
    shortest_alt = twice_area / np.maximum(edge_lengths.reshape(-1, 3).max(axis=1), 1e-12)
    sliver_count = int((shortest_alt < 0.05 * median_edge).sum())

    # 3. Watertight & Boundary analysis
    is_watertight = bool(mesh.is_watertight)
    euler_number = int(mesh.euler_number) if hasattr(mesh, "euler_number") else 2

    # 4. Flipped normals / normal consistency
    has_consistent_winding = bool(getattr(mesh, "is_winding_consistent", True))

    # 5. Composite Quality Score calculation (0 - 100)
    deductions = 0
    if not is_watertight:
        deductions += 10
    if degenerate_count > 0:
        deductions += min(25, int(degenerate_count / max(1, n_faces) * 500))
    if sliver_count > 0:
        deductions += min(15, int(sliver_count / max(1, n_faces) * 200))
    if not has_consistent_winding:
        deductions += 20

    score = max(50, 100 - deductions)

    return {
        "is_valid": True,
        "quality_score": score,
        "is_watertight": is_watertight,
        "has_consistent_winding": has_consistent_winding,
        "euler_number": euler_number,
        "num_vertices": n_verts,
        "num_faces": n_faces,
        "degenerate_faces": degenerate_count,
        "sliver_faces": sliver_count,
        "median_edge_length": round(median_edge, 6),
        "bounding_box_span": round(span, 4),
    }
