"""
Visual check for head and eye detection.

For each input mesh: prints the scale-relative measurements the detectors keyed off, and
writes two contact sheets --

    <name>_input.png   what the eye detector actually sees (textured head renders)
    <name>_masks.png   the detected regions painted on (head green, character-left eye
                       red, character-right eye blue)

Geometric detection cannot be verified by assertions alone; look at the images.

Usage:
    python scripts/debug_head_eye.py OUT_DIR MESH.glb [MESH.glb ...]
"""
import os
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
import cv2

from pipeline.unirig_pipeline import auto_orient_and_center_mesh
from pipeline.mesh_segmentation import detect_head_region
from pipeline.eye_detection import detect_eye_regions, detect_eyes_2d
from pipeline.head_views import HeadViewRenderer

BASE_COLOR = (170, 170, 175)
HEAD_COLOR = (60, 200, 90)
LEFT_COLOR = (225, 60, 60)
RIGHT_COLOR = (60, 110, 235)

HEAD_ANGLES = [(0, 0), (45, 0), (-45, 0), (90, 0), (-90, 0), (180, 0), (0, 25), (0, -25)]


def _load(path):
    scene = trimesh.load(path, process=False)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.uint8)
        if len(colors) != len(v):
            colors = None
    except Exception as e:
        print(f"  (no texture colors: {e})")
        colors = None
    v, _normals, _tm = auto_orient_and_center_mesh(v, f)
    return v, f, colors


def _paint(n, head, eyes):
    colors = np.zeros((n, 4), dtype=np.uint8)
    colors[:, :3] = BASE_COLOR
    colors[:, 3] = 255
    if head is not None:
        colors[head.mask, :3] = HEAD_COLOR
    if eyes is not None:
        colors[eyes.left_mask, :3] = LEFT_COLOR
        colors[eyes.right_mask, :3] = RIGHT_COLOR
    return colors


def _tile(images, labels, size):
    frames = []
    for img, label in zip(images, labels):
        frame = np.ascontiguousarray(img.copy())
        cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(frame)
    while len(frames) % 4:
        frames.append(np.full((size, size, 3), 40, dtype=np.uint8))
    rows = [np.hstack(frames[i:i + 4]) for i in range(0, len(frames), 4)]
    return np.vstack(rows)


def _write(out_dir, stem, suffix, images, labels, size):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}_{suffix}.png")
    cv2.imwrite(path, cv2.cvtColor(_tile(images, labels, size), cv2.COLOR_RGB2BGR))
    print(f"  wrote {path}")


def run(path, out_dir, size=512):
    stem = os.path.splitext(os.path.basename(path))[0][:44]
    print(f"\n{'=' * 78}\n{stem}\n{'=' * 78}")

    v, f, tex_colors = _load(path)
    extents = v.max(axis=0) - v.min(axis=0)
    diag = float(np.linalg.norm(extents))
    print(f"  mesh: {len(v)} verts, {len(f)} faces, extents={np.round(extents, 4).tolist()}, "
          f"texture_colors={'yes' if tex_colors is not None else 'NO'}")

    t0 = time.time()
    head = detect_head_region(v, f)
    t_head = time.time() - t0
    if head is None:
        print(f"  HEAD: not detected ({t_head:.2f}s)")
        return
    print(f"  HEAD: {head.n_vertices} verts ({head.n_vertices / len(v):.1%} of mesh), "
          f"radius={head.radius:.4f} ({head.radius / diag:.1%} of bbox diagonal), {t_head:.2f}s")

    t0 = time.time()
    eyes = detect_eye_regions(v, f, head, vertex_colors=tex_colors)
    t_eye = time.time() - t0
    if eyes is None:
        print(f"  EYES: not detected ({t_eye:.2f}s)")
    else:
        print(f"  EYES: L={int(eyes.left_mask.sum())} R={int(eyes.right_mask.sum())} verts, "
              f"separation={eyes.separation / head.radius:.2f} head radii, "
              f"conf={eyes.confidence:.3f}, {eyes.n_detections} detections / "
              f"{eyes.n_views} views, {t_eye:.2f}s")
        print(f"        forward={np.round(eyes.forward, 3).tolist()}  "
              f"L_center={np.round(eyes.left_center, 4).tolist()}  "
              f"R_center={np.round(eyes.right_center, 4).tolist()}")

    front_yaw = float(np.degrees(np.arctan2(eyes.forward[0], eyes.forward[2]))) if eyes is not None else 0.0

    # What the detector sees, with every box it found drawn on.
    images, labels = [], []
    with HeadViewRenderer(v, f, head.center, head.radius, vertex_colors=tex_colors,
                          image_size=size, framing=1.25) as r:
        for dyaw, pitch in HEAD_ANGLES:
            view = r.render(front_yaw + dyaw, pitch)
            frame = np.ascontiguousarray(view.color.copy())
            boxes = detect_eyes_2d([view.color])[0]
            for (x1, y1, x2, y2, conf) in boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            images.append(frame)
            labels.append(f"yaw{dyaw:+d} pitch{pitch:+d} n={len(boxes)}")
    _write(out_dir, stem, "input", images, labels, size)

    # The regions painted over the character's own texture, so a mask that is near the
    # eye but not on it is obvious.
    if tex_colors is not None:
        overlay = tex_colors.copy()
        if head is not None:
            overlay[head.mask, :3] = (0.65 * overlay[head.mask, :3].astype(np.float32)
                                      + 0.35 * np.array(HEAD_COLOR, np.float32)).astype(np.uint8)
        if eyes is not None:
            overlay[eyes.left_mask, :3] = LEFT_COLOR
            overlay[eyes.right_mask, :3] = RIGHT_COLOR
        images, labels = [], []
        with HeadViewRenderer(v, f, head.center, head.radius, vertex_colors=overlay,
                              image_size=size, framing=1.25) as r:
            for dyaw, pitch in HEAD_ANGLES:
                images.append(r.render(front_yaw + dyaw, pitch).color)
                labels.append(f"overlay yaw{dyaw:+d} pitch{pitch:+d}")
        _write(out_dir, stem, "overlay", images, labels, size)

    # The regions that came out of it.
    mask_colors = _paint(len(v), head, eyes)
    images, labels = [], []
    body_center = (v.max(axis=0) + v.min(axis=0)) / 2.0
    with HeadViewRenderer(v, f, body_center, diag / 2.0, vertex_colors=mask_colors,
                          image_size=size, framing=1.05) as r:
        for dyaw in (0, 90, 180, 270):
            images.append(r.render(front_yaw + dyaw).color)
            labels.append(f"body yaw{dyaw:+d}")
    with HeadViewRenderer(v, f, head.center, head.radius, vertex_colors=mask_colors,
                          image_size=size, framing=1.25) as r:
        for dyaw, pitch in HEAD_ANGLES:
            images.append(r.render(front_yaw + dyaw, pitch).color)
            labels.append(f"head yaw{dyaw:+d} pitch{pitch:+d}")
    _write(out_dir, stem, "masks", images, labels, size)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    for mesh_path in sys.argv[2:]:
        run(mesh_path, sys.argv[1])
