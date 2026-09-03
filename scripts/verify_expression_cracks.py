#!/usr/bin/env python3
"""
Measures whether an expression tears the mesh open.

`verify_expression_transfer.py` already measures how much a morph changes across one EDGE.
That metric cannot see the failure this script exists for. A mesh with a UV seam carries
several vertices at the identical position, one per side of the seam, and they are NOT
joined by an edge -- that is what a seam is. Any per-vertex quantity computed without
reference to which vertices are coincident can therefore hand the two copies different
values, and the surface pulls apart along the seam while every edge-based check stays
green. The character's face splits open along a network of thin cracks and nothing in the
repo reports a number above its limit.

So two things are measured, both in units of the mesh's own median edge, because that is
the unit a crack is visible in -- a gap of one edge is a hole the size of a triangle:

  seam_gap   Over every set of vertices sharing a position, the spread of the morph's
             displacement inside that set. Non-zero means the morph is pulling coincident
             vertices apart, and the surface has an open crack exactly there.

  step       The largest displacement on a vertex that sits directly across an edge from
             the region `eyelid_patch` owns. The caller zeroes transferred deltas on that
             region, so this is the height of the cliff at the handover: skin moving at
             full amplitude flush against skin pinned at zero.

Run:
    PYTHONNOUSERSITE=1 python scripts/verify_expression_cracks.py [mesh.glb ...]

Without PYTHONNOUSERSITE=1 the face parser fails silently, no lids are built, and the
`step` column is measured against nothing.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A crack narrower than this reads as shading, not as a hole. One tenth of a triangle.
SEAM_LIMIT = 0.10
# The handover cliff, in the same unit. A step of a quarter triangle is a visible crease.
STEP_LIMIT = 0.25

DEFAULT_MESH = ("playground/storage/job_1788109378_3d_style_3/stage1_prep/"
                "3d_style_3_generated_3d_input.glb")


def median_edge_length(vertices, faces):
    tri = faces if len(faces) <= 20000 else faces[
        np.linspace(0, len(faces) - 1, 20000).astype(np.int64)]
    return float(np.median(np.concatenate([
        np.linalg.norm(vertices[tri[:, 0]] - vertices[tri[:, 1]], axis=1),
        np.linalg.norm(vertices[tri[:, 1]] - vertices[tri[:, 2]], axis=1),
    ])))


def coincident_groups(vertices, median_edge):
    """
    Vertices that occupy the same point, grouped.

    Quantised to a ten-thousandth of a median edge rather than compared exactly: an
    exporter that round-trips a seam through float32 leaves the copies differing in the
    last bit, and an exact comparison then reports no duplicates on a mesh made almost
    entirely of them.

    Returns (group id per vertex, mask of vertices in a group of more than one).
    """
    key = np.round(vertices / max(1e-4 * median_edge, 1e-12)).astype(np.int64)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return inverse, counts[inverse] > 1


def seam_gap(delta, inverse, dup, n_groups):
    """
    The widest a morph pulls one set of coincident vertices apart, and where.

    The spread is taken per axis and combined, which is the diagonal of the set's bounding
    box -- an upper bound on the true spread, and the cheap one: the exact diameter needs
    every pair inside every group.
    """
    idx = np.nonzero(dup)[0]
    if len(idx) == 0:
        return 0.0, -1
    g = inverse[idx]
    hi = np.full((n_groups, 3), -np.inf, dtype=np.float64)
    lo = np.full((n_groups, 3), np.inf, dtype=np.float64)
    np.maximum.at(hi, g, delta[idx])
    np.minimum.at(lo, g, delta[idx])
    live = np.isfinite(hi[:, 0])
    spread = np.linalg.norm(hi[live] - lo[live], axis=1)
    if len(spread) == 0:
        return 0.0, -1
    worst = int(np.argmax(spread))
    # A representative vertex of the worst group, so the caller can say where it is.
    where = idx[np.nonzero(g == np.nonzero(live)[0][worst])[0][0]]
    return float(spread[worst]), int(where)


def protected_step(delta, edges, protected):
    """Largest displacement on a vertex sitting across an edge from the frozen region."""
    if protected is None or not protected.any():
        return float("nan"), -1
    a, b = edges[:, 0], edges[:, 1]
    crossing = protected[a] ^ protected[b]
    outside = np.unique(np.concatenate(
        [a[crossing & ~protected[a]], b[crossing & ~protected[b]]]))
    if len(outside) == 0:
        return float("nan"), -1
    mag = np.linalg.norm(delta[outside], axis=1)
    k = int(np.argmax(mag))
    return float(mag[k]), int(outside[k])


def edge_array(faces):
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def describe(where, vertices, landmarks, edge):
    """Which feature the worst offender sits nearest, in median edges."""
    if where < 0 or not landmarks:
        return ""
    p = vertices[where]
    name, d = min(((k, float(np.linalg.norm(p - v))) for k, v in landmarks.items()),
                  key=lambda kv: kv[1])
    return f"{d / edge:.0f} edges from {name}"


def run(path: Path, verbose: bool):
    from pipeline.unirig_pipeline import UniRigPipeline
    from scripts.verify_expression_transfer import load_mesh

    print(f"\n=== {path.name} ===", flush=True)
    vertices, faces, colors, uvs, texture = load_mesh(path)
    print(f"    mesh: {len(vertices)} vertices, {len(faces)} faces", flush=True)

    t0 = time.time()
    pipeline = UniRigPipeline()
    v, f, _c, _u, _n, _w, morphs = pipeline._build_facial_morphs(
        vertices=vertices, faces=faces, colors=colors, uvs=uvs,
        normals=None, skin_weights=None, base_color_texture=texture)
    print(f"    built {len(morphs)} morph targets in {time.time() - t0:.1f}s", flush=True)
    if not morphs:
        print("    nothing to measure")
        return False

    # The eyelid patch appends its geometry after the original vertices, so anything past
    # the original count is patch-owned. That is the same reconstruction
    # `verify_expression_transfer.py` uses, and it is used here rather than re-running the
    # detector because the detector is not deterministic: two runs over this mesh produced
    # 458 and 456 lid vertices, so a second pass returns a mask of the wrong length.
    edge = median_edge_length(v, f)
    protected = None
    landmarks = {}
    if len(v) > len(vertices):
        protected = np.zeros(len(v), dtype=bool)
        protected[len(vertices):] = True
        landmarks = {"the eye patch": v[len(vertices):].mean(axis=0)}

    edges = edge_array(f)
    inverse, dup = coincident_groups(v, edge)
    n_groups = int(inverse.max()) + 1
    print(f"    median edge {edge:.5f};  {int(dup.sum())} of {len(v)} vertices "
          f"({100.0 * dup.mean():.1f}%) share a position with another", flush=True)

    rows = []
    for name in sorted(morphs):
        delta = np.asarray(morphs[name], dtype=np.float64)
        if len(delta) != len(v):
            continue
        peak = float(np.linalg.norm(delta, axis=1).max())
        if peak <= 1e-12:
            continue
        gap, gap_at = seam_gap(delta, inverse, dup, n_groups)
        step, step_at = protected_step(delta, edges, protected)
        rows.append((name, peak / edge, gap / edge, gap_at, step / edge, step_at))

    rows.sort(key=lambda r: -max(r[2], 0.0 if np.isnan(r[4]) else r[4]))
    print(f"    {'shape':<24} {'peak':>7} {'seam_gap':>9} {'step':>7}   (median edges)")
    for name, peak, gap, gap_at, step, step_at in (rows if verbose else rows[:12]):
        flag = "  <-- CRACK" if gap > SEAM_LIMIT else ("  <-- STEP" if step > STEP_LIMIT else "")
        print(f"    {name:<24} {peak:7.2f} {gap:9.3f} {step:7.3f}{flag}")

    worst_gap = max((r[2] for r in rows), default=0.0)
    worst_step = max((r[4] for r in rows if not np.isnan(r[4])), default=float("nan"))
    gap_row = max(rows, key=lambda r: r[2], default=None)
    print()
    print(f"    worst seam_gap : {worst_gap:.3f} edges (limit {SEAM_LIMIT})"
          + (f"  [{gap_row[0]}, {describe(gap_row[3], v, landmarks, edge)}]" if gap_row else ""))
    print(f"    worst step     : {worst_step:.3f} edges (limit {STEP_LIMIT})")

    ok = worst_gap <= SEAM_LIMIT and (np.isnan(worst_step) or worst_step <= STEP_LIMIT)
    print(f"    {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meshes", nargs="*", default=[DEFAULT_MESH])
    ap.add_argument("--verbose", action="store_true", help="every shape, not just the worst")
    args = ap.parse_args()

    results = [run(Path(m) if Path(m).is_absolute() else ROOT / m, args.verbose)
               for m in args.meshes]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
