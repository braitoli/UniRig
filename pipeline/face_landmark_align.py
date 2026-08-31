"""
Locates facial landmark categories (eyes, mouth, nose, eyebrows) on an arbitrary
triangle mesh by rendering it from multiple yaw angles and running MediaPipe's
FaceLandmarker on each render, then computes the similarity transform (rotation +
uniform scale + translation) between two matched landmark sets via the Umeyama
algorithm.

Used to align the ARKit blendshape template onto a target head using real detected
facial features instead of a naive axis-aligned bounding-sphere fit -- this fixes
both the missing front-facing (yaw) alignment and inaccurate per-feature placement
that a spatial-only nearest-neighbor correspondence cannot recover on its own.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "mediapipe" / "face_landmarker.task"

_landmarker = None
_feature_groups = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import vision, BaseOptions
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
            num_faces=1,
            # MediaPipe's default (0.5). A lower threshold was tested and rejected: on a
            # plain gray sphere with no facial features at all, 0.15 produced "detections"
            # on 21/24 candidate angles and 0.3 still produced 8/24 -- the detector stage
            # is a lightweight blob classifier that false-positives easily on round shaded
            # geometry. At 0.5 the sphere gets 0/24 while real character renders still
            # detect reliably across several consecutive angles.
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _get_feature_groups() -> Dict[str, List[int]]:
    global _feature_groups
    if _feature_groups is None:
        from mediapipe.tasks.python import vision
        conn = vision.FaceLandmarksConnections
        raw = {
            "left_eye": conn.FACE_LANDMARKS_LEFT_EYE,
            "right_eye": conn.FACE_LANDMARKS_RIGHT_EYE,
            "mouth": conn.FACE_LANDMARKS_LIPS,
            "nose": conn.FACE_LANDMARKS_NOSE,
            "left_eyebrow": conn.FACE_LANDMARKS_LEFT_EYEBROW,
            "right_eyebrow": conn.FACE_LANDMARKS_RIGHT_EYEBROW,
        }
        groups = {}
        for name, connections in raw.items():
            idxs = set()
            for c in connections:
                idxs.add(c.start)
                idxs.add(c.end)
            groups[name] = sorted(idxs)
        _feature_groups = groups
    return _feature_groups


def sample_vertex_colors(vertices: np.ndarray, faces: np.ndarray, uvs: Optional[np.ndarray],
                          base_color_texture) -> Optional[np.ndarray]:
    """Bakes a UV texture down to per-vertex RGBA colors so the mesh can be rendered
    without a GL texture object (works around a pyrender/PyOpenGL texture bug on some
    platforms, and is all MediaPipe needs anyway)."""
    if uvs is None or base_color_texture is None or len(uvs) != len(vertices):
        return None
    try:
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        tmesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=base_color_texture)
        return np.asarray(tmesh.visual.to_color().vertex_colors)
    except Exception:
        return None


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _render_yaw(pymesh, center: np.ndarray, radius: float, yaw_deg: float, image_size: int):
    import pyrender
    yaw = np.radians(yaw_deg)
    dist = radius * 2.8
    cam_pos = center + np.array([dist * np.sin(yaw), 0.0, dist * np.cos(yaw)], dtype=np.float32)
    cam_pose = _look_at_pose(cam_pos, center)
    yfov = 2 * np.arctan2(radius * 1.5, dist)

    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pymesh)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    scene.add(camera, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()
    return color, depth, cam_pose, yfov


def _backproject(px: float, py: float, depth_value: float, cam_pose: np.ndarray,
                  yfov: float, image_size: int) -> np.ndarray:
    fx = fy = (image_size / 2.0) / np.tan(yfov / 2.0)
    cx = cy = image_size / 2.0
    x_ndc = (px - cx) / fx
    y_ndc = (cy - py) / fy  # image row 0 is top; camera +Y is up -> flip
    z_cam = -depth_value    # camera looks down -Z; a point `depth_value` in front has Z_cam = -depth_value
    p_cam = np.array([x_ndc * depth_value, y_ndc * depth_value, z_cam, 1.0], dtype=np.float32)
    return (cam_pose @ p_cam)[:3]


def _valid_depth_near(depth: np.ndarray, px: int, py: int, window: int = 3) -> Optional[float]:
    h, w = depth.shape
    best = None
    for r in range(max(0, py - window), min(h, py + window + 1)):
        for c in range(max(0, px - window), min(w, px + window + 1)):
            d = depth[r, c]
            if d > 0:
                dist = abs(r - py) + abs(c - px)
                if best is None or dist < best[0]:
                    best = (dist, d)
    return best[1] if best is not None else None


def locate_face_landmarks_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    radius: float,
    vertex_colors: Optional[np.ndarray] = None,
    num_angles: int = 24,
    image_size: int = 512,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Sweeps `num_angles` yaw angles around `center`, rendering the mesh and running
    MediaPipe FaceLandmarker on each view. A real face is visible across a range of
    nearby angles, not just one -- so this picks the longest run of consecutive
    successful detections (an isolated single-angle hit is more likely noise) and
    backprojects each detected feature group's centroid to 3D through that view's
    depth buffer. Returns None if no face was found at any angle.
    """
    import pyrender
    import mediapipe as mp

    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        tmesh.visual = trimesh.visual.color.ColorVisuals(tmesh, vertex_colors=vertex_colors)
    pymesh = pyrender.Mesh.from_trimesh(tmesh, smooth=True)

    landmarker = _get_landmarker()
    groups = _get_feature_groups()

    yaw_step = 360.0 / num_angles
    renders = {}
    hits = []
    for i in range(num_angles):
        yaw_deg = i * yaw_step
        color, depth, cam_pose, yfov = _render_yaw(pymesh, center, radius, yaw_deg, image_size)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color))
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            renders[i] = (result.face_landmarks[0], depth, cam_pose, yfov)
            hits.append(i)

    if not hits:
        return None

    MIN_RUN_LENGTH = 2  # require the face to hold up across >= 2 consecutive angles, not a single fluke frame

    hit_set = set(hits)
    visited = set()
    best_run: List[int] = []
    for start in hits:
        if start in visited:
            continue
        run = [start]
        visited.add(start)
        cur = (start + 1) % num_angles
        while cur in hit_set and cur not in visited:
            run.append(cur)
            visited.add(cur)
            cur = (cur + 1) % num_angles
        if len(run) > len(best_run):
            best_run = run

    if len(best_run) < MIN_RUN_LENGTH:
        return None

    best_idx = best_run[len(best_run) // 2]
    landmarks, depth, cam_pose, yfov = renders[best_idx]

    out: Dict[str, np.ndarray] = {}
    for name, idxs in groups.items():
        pts = []
        for li in idxs:
            if li >= len(landmarks):
                continue
            lm = landmarks[li]
            px, py = lm.x * image_size, lm.y * image_size
            d = _valid_depth_near(depth, int(round(px)), int(round(py)))
            if d is not None:
                pts.append(_backproject(px, py, d, cam_pose, yfov, image_size))
        if pts:
            out[name] = np.mean(pts, axis=0).astype(np.float32)

    return out if len(out) >= 3 else None


def compute_similarity_transform(src_pts: np.ndarray, tgt_pts: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Umeyama's method: least-squares similarity transform (R, s, t) minimizing
    sum ||s * R @ src_i + t - tgt_i||^2 given >= 3 matched point correspondences.
    """
    assert src_pts.shape == tgt_pts.shape and len(src_pts) >= 3
    src_mean = src_pts.mean(axis=0)
    tgt_mean = tgt_pts.mean(axis=0)
    src_c = src_pts - src_mean
    tgt_c = tgt_pts - tgt_mean

    cov = (tgt_c.T @ src_c) / len(src_pts)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_src = (src_c ** 2).sum(axis=1).mean()
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0

    t = tgt_mean - scale * (R @ src_mean)
    return R.astype(np.float32), scale, t.astype(np.float32)
