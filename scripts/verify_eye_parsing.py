#!/usr/bin/env python3
"""
Checks face-parsing eye detection on real meshes, and builds a GLB you can open to see it.

For each input mesh this reports whether the parser responded at all, whether the response
survived the trust gate, and how long each stage took -- then writes:

    <name>_parse.png   contact sheet: every scanned angle with its parsed labels painted on
                       (eyes red, glasses orange, brows blue, skin faint)
    <name>_eyes.glb    the mesh with eyelids welded in, the eye morph targets, and the
                       auto-blink clip -- drop this into the playground viewer

Whether a parser trained on photographs of human faces responds to a shaded 3D
reconstruction is the one thing that cannot be settled by reading the code, and it decides
whether this approach is usable at all. Look at the contact sheet.

Usage:
    python scripts/verify_eye_parsing.py OUT_DIR MESH.glb [MESH.glb ...]
"""
import os
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from pipeline.animation import generate_blink_animation
from pipeline.eye_detection import detect_eye_regions
from pipeline.eyelid_patch import attach_eyelids
from pipeline.face_landmark_align import sample_vertex_colors
from pipeline.face_parsing import (
    BROW_LABELS, EYE_LABELS, GLASSES_LABELS, parse_faces, summarise,
)
from pipeline.head_views import HeadViewRenderer, spiral_yaws
from pipeline.mesh_segmentation import detect_head_region
from pipeline.rig_export import create_rigged_glb
from pipeline.unirig_pipeline import auto_orient_and_center_mesh

TILE = 256
COLUMNS = 4
LABEL_COLORS = {
    **{l: (235, 60, 60) for l in EYE_LABELS},
    **{l: (250, 160, 40) for l in GLASSES_LABELS},
    **{l: (70, 120, 240) for l in BROW_LABELS},
    1: (120, 200, 130),   # skin, drawn faintly so you can see whether the face registered
}
SKIN_ALPHA = 0.25
FEATURE_ALPHA = 0.75


def load(path):
    """
    Loads a mesh with its colour baked per vertex, the same way the pipeline does.

    Reading `vertex_colors` alone is not enough and quietly ruins the test: every example
    mesh in this repo carries its appearance in a UV texture, and asking trimesh for vertex
    colours on one of those returns flat grey. The parser then sees an untextured blob and
    reports no face anywhere -- which looks exactly like the parser failing, and is not.
    """
    scene = trimesh.load(path, process=False)
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)

    uvs, texture = None, None
    try:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
        texture = mesh.visual.material.baseColorTexture
    except Exception:
        pass

    colors = sample_vertex_colors(v, f, uvs, texture)
    source = "texture"
    if colors is None:
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == len(v):
                colors = np.asarray(vc, dtype=np.uint8)
                source = "vertex colors"
        except Exception:
            pass
    if colors is None:
        source = "none (flat grey)"

    v, _normals, _tm = auto_orient_and_center_mesh(v, f)
    return v, f, colors, source


def overlay(color, labels):
    out = color.astype(np.float32)
    for label_id, rgb in LABEL_COLORS.items():
        m = labels == label_id
        if not m.any():
            continue
        a = SKIN_ALPHA if label_id == 1 else FEATURE_ALPHA
        out[m] = out[m] * (1.0 - a) + np.array(rgb, np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def contact_sheet(tiles, path):
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGB", (COLUMNS * TILE, rows * (TILE + 16)), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    for i, (image, caption) in enumerate(tiles):
        x, y = (i % COLUMNS) * TILE, (i // COLUMNS) * (TILE + 16)
        sheet.paste(Image.fromarray(image).resize((TILE, TILE)), (x, y))
        draw.text((x + 4, y + TILE + 2), caption, fill=(215, 215, 220))
    sheet.save(path)


def run(mesh_path, out_dir):
    stem = os.path.splitext(os.path.basename(mesh_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78)
    print(stem)
    print("=" * 78)

    v, f, colors, source = load(mesh_path)
    print(f"  mesh: {len(v)} verts, {len(f)} faces, colour from {source}")

    t0 = time.time()
    head = detect_head_region(v, f)
    print(f"  [1/4] head region: "
          f"{'not detected' if head is None else f'{head.n_vertices} verts, radius {head.radius:.3f}'}"
          f"  ({time.time() - t0:.2f}s)")
    if head is None:
        return

    # --- what the parser sees, angle by angle ---
    t0 = time.time()
    tiles = []
    hits = 0
    yaws = sorted(spiral_yaws(8))
    from pipeline.eye_detection import _FRAMING
    with HeadViewRenderer(v, f, head.center, head.radius, vertex_colors=colors,
                          image_size=768, framing=_FRAMING) as renderer:
        views = [renderer.render(y) for y in yaws]
    labels, _conf = parse_faces([view.color for view in views])
    for view, lab in zip(views, labels):
        counts = summarise(lab)
        eye_px = sum(counts.get(n, 0) for n in ("l_eye", "r_eye", "eye_g"))
        brow_px = sum(counts.get(n, 0) for n in ("l_brow", "r_brow"))
        hits += 1 if eye_px > 0 else 0
        tiles.append((overlay(view.color, lab),
                      f"yaw {view.yaw_deg:+.0f}  eye {eye_px}px  brow {brow_px}px"))
        print(f"        yaw {view.yaw_deg:+7.1f}: eye {eye_px:6d}px  brow {brow_px:5d}px  "
              f"skin {counts.get('skin', 0):6d}px")
    sheet_path = os.path.join(out_dir, f"{stem}_parse.png")
    contact_sheet(tiles, sheet_path)
    print(f"  [2/4] parsed {len(views)} angles, {hits} with eye pixels  "
          f"({time.time() - t0:.2f}s)  ->  {sheet_path}")

    # --- the full detector, with cross-view agreement and mirror completion ---
    t0 = time.time()
    eyes = detect_eye_regions(v, f, head, vertex_colors=colors)
    print(f"  [3/4] eye regions: {'REJECTED' if eyes is None else 'ACCEPTED'}  "
          f"({time.time() - t0:.2f}s)")
    if eyes is None:
        return

    # --- eyelids and a loadable GLB ---
    t0 = time.time()
    lids = attach_eyelids(v, f, eyes, colors=colors)
    if lids is None:
        print("  [4/4] no eyelid could be built")
        return

    skin = np.ones((len(lids.vertices), 1), dtype=np.float32)
    morphs = lids.morph_targets
    clip = generate_blink_animation(list(morphs.keys()))
    glb_path = os.path.join(out_dir, f"{stem}_eyes.glb")
    create_rigged_glb(
        vertices=lids.vertices, faces=lids.faces,
        joints=np.zeros((1, 3), np.float32), parents=[None],
        skin_weights=skin, normals=lids.normals, colors=lids.colors,
        joint_names=["Root"], morph_targets=morphs,
        animations={"Blink": clip} if clip else None,
        output_path=glb_path,
    )
    print(f"  [4/4] {lids.n_added} lid verts, morphs {sorted(morphs)}, "
          f"blink clip {'yes' if clip else 'no'}  ({time.time() - t0:.2f}s)")
    print(f"        ->  {glb_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    for path in sys.argv[2:]:
        try:
            run(path, sys.argv[1])
        except Exception as e:
            import traceback
            print(f"  ERROR on {path}: {e}")
            traceback.print_exc()
        print()
