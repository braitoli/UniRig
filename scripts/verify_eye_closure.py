#!/usr/bin/env python3
"""
Measures whether a blink actually closes the eye, by rendering it.

Every other check in this repo measures the eye machinery against its own intermediate
values -- mask sizes, vote ratios, morph magnitudes -- and all of them passed on a
character whose blink visibly tore the eye open instead of shutting it. This measures the
thing a viewer actually sees:

  coverage   Of the pixels that show eye (bright sclera or dark iris) in the rest pose,
             what fraction still shows eye at full blink? A working lid drives this to
             roughly zero. A lid that covers part of the eye leaves the rest on screen.

  ragged     The eye's remaining silhouette at full blink, as a perimeter^2/area ratio
             normalised so a circle scores 1.0. A lid copied from a ragged vertex sample
             leaves a sawtooth edge, and that edge is what reads as a crack.

Usage:
    python scripts/verify_eye_closure.py [mesh.glb ...]

Defaults to the most recent playground job that produced eyelids. Needs PYTHONNOUSERSITE=1
or the face parser silently fails and no lids get built at all.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COVERAGE_LIMIT = 0.15   # of the eye may still be showing at full blink
RAGGED_LIMIT = 4.0      # perimeter^2 / (4*pi*area) of whatever is still showing


def camera(centre, forward, radius, size):
    """The pose and intrinsics `render` uses, so callers can project into its frame."""
    from pipeline.face_landmark_align import _look_at_pose
    cam_pos = centre + forward * radius * 3.0
    pose = _look_at_pose(cam_pos.astype(np.float32), centre.astype(np.float32))
    yfov = 2 * np.arctan2(radius, radius * 3.0)
    return pose, yfov


def project(points, pose, yfov, size):
    """World points to pixel coordinates in `render`'s frame."""
    inv = np.linalg.inv(pose)
    cam = points @ inv[:3, :3].T + inv[:3, 3]
    depth = np.maximum(-cam[:, 2], 1e-6)
    f = (size / 2.0) / np.tan(yfov / 2.0)
    c = size / 2.0
    return np.stack([cam[:, 0] / depth * f + c, c - cam[:, 1] / depth * f], axis=1)


def render(vertices, faces, colors, centre, forward, radius, size=800):
    import pyrender
    import trimesh
    from pipeline.face_landmark_align import _look_at_pose

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.visual = trimesh.visual.color.ColorVisuals(tm, vertex_colors=colors)
    # Flat, unlit: ambient only and no directional light, so a pixel's value is the
    # surface's own colour. Lighting it properly blew the skin out to a median luma of
    # 0.927 and made sclera indistinguishable from cheek, which silently turned the
    # measurement below into "count bright pixels".
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))

    cam_pos = centre + forward * radius * 3.0
    pose = _look_at_pose(cam_pos.astype(np.float32), centre.astype(np.float32))
    scene.add(pyrender.PerspectiveCamera(yfov=2 * np.arctan2(radius, radius * 3.0),
                                         aspectRatio=1.0), pose=pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return color


def eye_pixels(image, skin_luma):
    """
    Pixels that read as eye rather than as skin, in a flat-lit render.

    Thresholds are derived from the character's own skin rather than fixed: `skin_luma` is
    the median albedo over the face, and an eye is what departs far enough from it in
    either direction -- the sclera above, the iris below. Fixed thresholds do not survive
    the range of characters this pipeline generates.
    """
    rgb = image.astype(np.float32) / 255.0
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lit = luma > 0.02                      # exclude the background
    sclera = lit & (luma > skin_luma + 0.22)
    iris = lit & (luma < skin_luma - 0.22)
    return sclera | iris


def raggedness(mask):
    """perimeter^2 / (4*pi*area), 1.0 for a disc. Higher means a more broken outline."""
    area = int(mask.sum())
    if area < 16:
        return 0.0
    horizontal = mask[:, 1:] ^ mask[:, :-1]
    vertical = mask[1:, :] ^ mask[:-1, :]
    perimeter = int(horizontal.sum() + vertical.sum())
    return perimeter ** 2 / (4.0 * np.pi * area)


def run(path: Path, save_dir: Path):
    from scripts.verify_expression_transfer import load_mesh
    from pipeline.eye_detection import detect_eye_regions
    from pipeline.eyelid_patch import attach_eyelids
    from pipeline.face_landmark_align import sample_vertex_colors
    from pipeline.mesh_segmentation import detect_head_region
    from PIL import Image

    vertices, faces, colors, uvs, texture = load_mesh(path)
    print(f"    {len(vertices)} vertices, {len(faces)} faces")

    head = detect_head_region(vertices, faces)
    if head is None:
        print("    no head region"); return None
    appearance = colors if colors is not None else sample_vertex_colors(
        vertices, faces, uvs, texture)
    regions = detect_eye_regions(vertices, faces, head, vertex_colors=appearance)
    if regions is None:
        print("    eye detection found nothing -- no lids to measure"); return None

    lids = attach_eyelids(vertices, faces, regions, colors=colors, uvs=uvs,
                          normals=None, skin_weights=None, appearance=appearance)
    if lids is None:
        print("    no eyelids built"); return None

    base_colors = appearance if appearance is not None else colors
    lid_colors = lids.colors if lids.colors is not None else np.vstack([
        base_colors, np.tile(base_colors.mean(axis=0).astype(np.uint8), (lids.n_added, 1))])

    centre = (vertices[regions.left_mask].mean(axis=0)
              + vertices[regions.right_mask].mean(axis=0)) / 2.0
    forward = regions.forward / np.linalg.norm(regions.forward)
    view_radius = float(regions.separation)

    rest = render(lids.vertices, lids.faces, lid_colors, centre, forward, view_radius)
    shut = lids.vertices.copy()
    for side in ("Left", "Right"):
        delta = lids.morph_targets.get(f"eyeBlink{side}")
        if delta is not None:
            shut = shut + delta
    closed = render(shut, lids.faces, lid_colors, centre, forward, view_radius)

    # The measurement region has to be the eyes and nothing else. A horizontal band across
    # the face was tried and is wrong: on a character with hair and a moustache it selected
    # 55802 dark pixels of which not one was sclera, so the score was dominated by features
    # that a blink cannot change and sat at 99% no matter what the lid did. Project each
    # eye's own vertices instead and measure inside their boxes.
    size = rest.shape[0]
    pose, yfov = camera(centre, forward, view_radius, size)
    box = np.zeros((size, size), dtype=bool)
    for mask in (regions.left_mask, regions.right_mask):
        uv = project(vertices[mask], pose, yfov, size)
        lo = np.clip(uv.min(axis=0), 0, size - 1).astype(int)
        hi = np.clip(uv.max(axis=0), 0, size - 1).astype(int)
        pad = int(0.35 * max(hi[0] - lo[0], hi[1] - lo[1], 1))
        box[max(lo[1] - pad, 0): hi[1] + pad, max(lo[0] - pad, 0): hi[0] + pad] = True

    rgb = rest.astype(np.float32) / 255.0
    luma_rest = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Skin is calibrated from the face at large, not from inside the boxes -- the boxes are
    # mostly eye, so their own median is not a skin reference.
    skin_luma = float(np.median(luma_rest[(luma_rest > 0.02) & ~box]))

    rest_eye = eye_pixels(rest, skin_luma) & box
    still = eye_pixels(closed, skin_luma) & rest_eye
    print(f"    skin albedo {skin_luma:.3f}; measuring inside {int(box.sum())} px "
          f"of eye boxes")
    coverage = float(still.sum()) / max(int(rest_eye.sum()), 1)
    ragged = raggedness(still)

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    Image.fromarray(rest).save(save_dir / f"{stem}_rest.png")
    Image.fromarray(closed).save(save_dir / f"{stem}_blink.png")

    print(f"    eye pixels at rest: {int(rest_eye.sum())}")
    print(f"    still showing at full blink: {int(still.sum())} "
          f"({100 * coverage:.1f}%), raggedness {ragged:.1f}")
    print(f"    images -> {save_dir}/{stem}_{{rest,blink}}.png")

    failures = []
    if coverage > COVERAGE_LIMIT:
        failures.append(f"{100 * coverage:.0f}% of the eye still visible at full blink "
                        f"(limit {100 * COVERAGE_LIMIT:.0f}%)")
    if ragged > RAGGED_LIMIT:
        failures.append(f"remaining silhouette raggedness {ragged:.1f} > {RAGGED_LIMIT}")
    for reason in failures:
        print(f"    FAIL: {reason}")
    return not failures


def default_meshes():
    storage = ROOT / "playground" / "storage"
    found = sorted(storage.glob("*/stage1_prep/*_input.glb"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return found[:1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("meshes", nargs="*")
    parser.add_argument("-o", "--out", default=str(ROOT / "test_results" / "eye_closure"))
    args = parser.parse_args()

    paths = [Path(m) for m in args.meshes] or default_meshes()
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No mesh to check.")
        return 1

    results = []
    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}")
        results.append(run(path, Path(args.out)))

    ok = [r for r in results if r]
    skipped = [r for r in results if r is None]
    print(f"\n{len(ok)}/{len(results) - len(skipped)} passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    return 0 if len(ok) == len(results) - len(skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
