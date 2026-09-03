#!/usr/bin/env python3
"""
Checks the eyelid patch and the auto-blink clip without needing a GPU or a model download.

These are the properties that decide whether a blink reads as real rather than as a sliding
shutter, and none of them is visible by reading the code:

1. The lid stays in front of the eye at EVERY weight, not just at the ends -- a lid that
   dips behind the surface part-way through is invisible there and then pops out.
2. At rest it is a crease along the lash line, and closed it covers the eye with no holes
   along the seam where the two lids meet.
3. The corners barely travel, falling off exactly as steeply as the eye's outline does.
4. The two lids meet low on the eye: the upper does about 75% of the closure.
5. `eyeSquint` is AU7 -- narrowing driven by the LOWER lid -- not a scaled-down blink.
6. The blink closes 3-6x faster than it opens, which is the physiological asymmetry.
7. The blink track's morph indices line up with the order `create_rigged_glb` declares the
   targets in. An off-by-one there animates the wrong shape and is invisible in the code.

Run: python scripts/verify_eyelid_blink.py
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_head():
    """
    An icosphere standing in for a head, with two almond-shaped patches standing in for eyes.

    The outline has to be an almond and not a disc. Half of what makes a blink read
    correctly is that the lid corners barely move, and that falls out of the eye's own
    outline pinching to a point at each corner -- a disc has no corners to pinch, so it
    cannot exhibit the property at all, and testing against one silently passes an
    implementation that gets it wrong.
    """
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    n = np.asarray(mesh.vertex_normals, dtype=np.float32)

    half_width, half_height = 0.30, 0.11

    def almond(direction):
        d = np.asarray(direction, np.float32)
        d /= np.linalg.norm(d)
        world_up = np.array([0.0, 1.0, 0.0], np.float32)
        up = world_up - float(world_up @ d) * d
        up /= np.linalg.norm(up)
        across = np.cross(d, up)
        rel = v - d
        h, u = rel @ across, rel @ up
        front = (v @ d) > 0.85
        return front & ((h / half_width) ** 2 + (u / half_height) ** 2 < 1.0)

    def iris(direction, radius=0.085):
        d = np.asarray(direction, np.float32)
        d /= np.linalg.norm(d)
        return ((v @ d) > 0.85) & (np.linalg.norm(v - d, axis=1) < radius)

    left_dir = np.array([0.35, 0.15, 0.92], dtype=np.float32)
    right_dir = np.array([-0.35, 0.15, 0.92], dtype=np.float32)
    left_dir /= np.linalg.norm(left_dir)
    right_dir /= np.linalg.norm(right_dir)
    return (v, f, n, almond(left_dir), almond(right_dir), left_dir, right_dir,
            iris(left_dir) | iris(right_dir))


def main():
    from scipy.spatial import cKDTree

    from pipeline.animation import generate_blink_animation
    from pipeline.eye_detection import EyeRegions
    from pipeline.eyelid_patch import attach_eyelids, eye_frame, merge_morph_targets
    from pipeline.rig_export import create_rigged_glb

    print("\n=== 1. Eyelid geometry ===")
    v, f, vn, left, right, lc, rc, iris = make_head()
    n0 = len(v)
    colors = np.tile(np.array([[200, 170, 150, 255]], np.uint8), (n0, 1))
    colors[left | right] = [238, 238, 242, 255]       # sclera
    colors[iris] = [28, 28, 46, 255]                  # iris and pupil
    uvs = np.zeros((n0, 2), np.float32)
    skin_weights = np.zeros((n0, 2), np.float32)
    skin_weights[:, 0] = 1.0

    regions = EyeRegions(
        left_mask=left, right_mask=right, left_center=lc, right_center=rc,
        forward=np.array([0.0, 0.0, 1.0], np.float32),
        separation=float(np.linalg.norm(lc - rc)),
        confidence=0.9, n_detections=4, n_views=4,
    )

    result = attach_eyelids(v, f, regions, colors=colors, uvs=uvs, skin_weights=skin_weights)
    check("attach_eyelids returns a result", result is not None)
    if result is None:
        return 1

    check("lid vertices were added", result.n_added > 0,
          f"{n0} -> {len(result.vertices)} vertices")
    check("every per-vertex array grew in step",
          len(result.colors) == len(result.uvs) == len(result.skin_weights)
          == len(result.normals) == len(result.vertices),
          f"all {len(result.vertices)}")
    check("padded eye masks match the new vertex count",
          len(result.eye_regions.left_mask) == len(result.vertices))
    check("original vertices are untouched", np.allclose(result.vertices[:n0], v))

    blink_l = result.morph_targets["eyeBlinkLeft"]
    blink_r = result.morph_targets["eyeBlinkRight"]
    check("blink morphs move only lid vertices",
          not np.any(blink_l[:n0]) and not np.any(blink_r[:n0]))

    lid_idx = np.arange(n0, len(result.vertices))
    rest = result.vertices[lid_idx]
    closed = (result.vertices + blink_l + blink_r)[lid_idx]

    # --- properties 1 and 2: in front at every weight, a crease at rest, sealed when shut ---
    tree = cKDTree(v)
    def signed_height(points):
        """How far each point sits outside the original surface, along the local normal."""
        _d, near = tree.query(points, k=1)
        return np.einsum("ij,ij->i", points - v[near], vn[near])

    # The lid must be in front of the eye for the WHOLE of its travel, not just at the ends.
    # A lid that dips behind the surface part-way through is invisible there, and because a
    # morph is linear every vertex dips at the same weight -- so the eye looks untouched for
    # part of the blink and the lid then pops out. Sampling the interior weights is the only
    # way to see that; both endpoints can be perfect while the middle is not.
    worst = None
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        h = signed_height(result.vertices[lid_idx] + w * (blink_l + blink_r)[lid_idx])
        if worst is None or float(h.min()) < worst[1]:
            worst = (w, float(h.min()))
    check("the lid stays in front of the eye at every weight", worst[1] > 0.0,
          f"closest approach {worst[1]:+.5f} at weight {worst[0]:.1f}")

    # At rest the lid is collapsed onto the eye's outline, so it should have almost no area:
    # a crease along the lash line rather than a sheet over the eye.
    def area(points, tris):
        a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
        return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 0.5)

    # Only the LIDS, not the iris caps: both are appended geometry, but the iris is a full
    # disc at rest by design and would swamp the crease measurement below.
    added_faces = result.faces[len(f):]
    lid_moved = np.zeros(len(result.vertices), bool)
    lid_moved[np.linalg.norm(blink_l + blink_r, axis=1) > 1e-9] = True
    lid_faces = added_faces[lid_moved[added_faces].any(axis=1)]
    eye_faces = f[left[f].sum(axis=1) == 3]
    rest_area = area(result.vertices, lid_faces)
    closed_area = area(result.vertices + blink_l + blink_r, lid_faces)
    eye_area = max(area(v, eye_faces) * 2.0, 1e-9)      # both eyes
    check("the resting lid is a crease, not a sheet", rest_area / eye_area < 0.15,
          f"rest area is {100 * rest_area / eye_area:.1f}% of the eyes' own area")
    check("the closed lid covers the eyes", closed_area / eye_area > 0.80,
          f"closed area is {100 * closed_area / eye_area:.0f}% of the eyes' own area")

    # Area alone does not prove coverage -- two lids can add up to the right total and still
    # leave a row of holes where they meet. Every triangle of the eye has to be claimed by
    # some lid, which is exact and checkable: the lid faces are copies, so the eye triangle
    # they came from is recoverable by matching their vertices' original positions.
    # Match on the CLOSED pose, not the rest pose: at rest each lid vertex is collapsed onto
    # the eye's outline, so its nearest original vertex is not the one it was copied from.
    lid_src = cKDTree(v).query(closed, k=1)[1]
    covered = {tuple(sorted(t)) for t in lid_src[lid_faces - len(v)]}
    eye_tris = {tuple(sorted(t)) for t in f[(left | right)[f].sum(axis=1) == 3]}
    missing = eye_tris - covered
    check("no eye triangle is left uncovered", len(missing) == 0,
          f"{len(eye_tris) - len(missing)}/{len(eye_tris)} eye triangles claimed by a lid")

    # --- property 3: the corners barely travel ---
    # Which lid vertices belong to the left eye is decided by which morph moves them, not by
    # distance: the two eyes are close enough on this head that a radius test catches both,
    # and measuring one eye's horizontal coordinate against the other's centre is noise.
    _out, up, across = eye_frame(v[left], vn[left], regions.forward)
    centre_l = v[left].mean(axis=0)
    own = np.linalg.norm(blink_l[lid_idx], axis=1) > 1e-9
    own_idx = lid_idx[own]
    travel_l = np.linalg.norm(blink_l[own_idx], axis=1)
    # The upper lid only: the lower one travels a short distance everywhere by design, so
    # including it would flatten the very falloff being measured.
    upper_sel = (blink_l[own_idx] @ up) < 0
    h_coord = (result.vertices[own_idx] - centre_l) @ across
    # What has to fall off at the corners is the LEADING EDGE of the lid, not every vertex
    # of it. A lid is a flap hinged along its top: the vertices at its root sit under the
    # brow and barely move wherever they are, while the free edge sweeps the opening. So the
    # measurement is per column -- the furthest-travelling vertex in each -- against the
    # eye's own opening height there, which the fixture defines analytically. Correlating
    # every vertex instead mixes in the hinge and measures nothing.
    # The span is the ORBIT ELLIPSE's half-width. The lid is built over the ellipse
    # `_orbit_region` fits around the eye, not over the painted eye itself -- the parser's
    # mask is a sparse sample of the opening, and a lid confined to it covered 21% of the
    # eye. So the outline that fixes the falloff is the ellipse's, and it is read off the
    # region the patch protected rather than off the fixture's almond, which describes a
    # lid an earlier design built.
    orbit = result.protected[:len(v)] & (
        np.linalg.norm(v - centre_l, axis=1) < 0.5 * regions.separation)
    span = float(np.abs((v[orbit] - centre_l) @ across).max())
    within = np.abs(h_coord) <= span
    sel_mask = upper_sel & within
    h_sel, t_sel = h_coord[sel_mask], travel_l[sel_mask]
    n_bins = 12
    bin_edges = np.linspace(-span, span, n_bins + 1)
    bin_id = np.clip(np.digitize(h_sel, bin_edges) - 1, 0, n_bins - 1)
    edge_travel, edge_h = [], []
    for b in range(n_bins):
        sel = np.nonzero(bin_id == b)[0]
        if len(sel) == 0:
            continue
        edge_travel.append(float(t_sel[sel].max()))
        edge_h.append(float(0.5 * (bin_edges[b] + bin_edges[b + 1])))
    edge_travel = np.asarray(edge_travel)
    edge_h = np.asarray(edge_h)
    opening = np.sqrt(np.clip(1.0 - (edge_h / span) ** 2, 0.0, 1.0))
    corr = float(np.corrcoef(edge_travel, opening)[0, 1]) if len(edge_travel) > 2 else 0.0
    check("the lid's leading edge follows the eye's outline", corr >= 0.75,
          f"correlation with opening height = {corr:.2f} over {len(edge_travel)} columns")
    # How much the corner is allowed to move is not a free parameter -- the outline fixes
    # it. On this almond a column at 0.85 of the span is still 53% as tall as the centre, so
    # demanding "0.2 at the corners" of a band that wide is unreachable by geometry, not a
    # defect. What the falloff must match is the outline's own, so the measured ratio is
    # pinned to the analytic one from both sides. That is stricter than any single ceiling:
    # a lid that moves too little at the corners fails it just as a shutter does.
    outer = np.abs(edge_h) > 0.85 * span
    inner = np.abs(edge_h) < 0.25 * span
    ratio = float(edge_travel[outer].mean() / max(edge_travel[inner].mean(), 1e-9))
    expected = float(opening[outer].mean() / max(opening[inner].mean(), 1e-9))
    check("corner travel falls off exactly as the outline does",
          abs(ratio - expected) <= 0.15,
          f"measured {ratio:.2f} vs {expected:.2f} implied by the outline "
          f"({int(outer.sum())} outer / {int(inner.sum())} inner columns)")
    tip = edge_travel[np.abs(edge_h) > 0.95 * span]
    if len(tip):
        check("the very corner barely moves",
              float(tip.mean() / max(edge_travel[inner].mean(), 1e-9)) <= 0.35,
              f"tip/centre = {float(tip.mean() / max(edge_travel[inner].mean(), 1e-9)):.2f}")

    # --- property 4: two lids, meeting low on the eye ---
    # An upper lid travels down the eye and a lower lid travels up it, so the sign of each
    # lid vertex's displacement along the eye's own up axis says which lid it belongs to --
    # no need to infer it from where the vertex happens to sit.
    u_eye = (v[left] - centre_l) @ up
    eye_h = float(np.ptp(u_eye))
    own_idx = lid_idx[own]
    drop = blink_l[own_idx] @ up
    upper_lid = own_idx[drop < -1e-9]
    lower_lid = own_idx[drop > 1e-9]
    check("both an upper and a lower lid were built",
          len(upper_lid) > 0 and len(lower_lid) > 0,
          f"{len(upper_lid)} upper / {len(lower_lid)} lower lid vertices")
    if len(upper_lid) and len(lower_lid):
        u_up = (result.vertices[upper_lid] + blink_l[upper_lid] - centre_l) @ up
        share = float(u_eye.max() - u_up.min()) / max(eye_h, 1e-9)
        check("the upper lid covers about three quarters of the eye", 0.60 <= share <= 0.95,
              f"upper lid share = {share:.2f}")
        up_travel = float(np.linalg.norm(blink_l[upper_lid], axis=1).mean())
        low_travel = float(np.linalg.norm(blink_l[lower_lid], axis=1).mean())
        check("the lower lid travels much less than the upper",
              low_travel < 0.75 * up_travel,
              f"upper {up_travel:.4f} vs lower {low_travel:.4f}")

    # --- property 5: squint is lower-lid dominant, not a scaled blink ---
    squint_l = result.morph_targets["eyeSquintLeft"]
    moving_idx = lid_idx[np.linalg.norm(blink_l[lid_idx], axis=1) > 1e-9]
    frac = (np.linalg.norm(squint_l[moving_idx], axis=1)
            / np.linalg.norm(blink_l[moving_idx], axis=1))
    check("squint drives the two lids by different amounts",
          float(frac.max() - frac.min()) > 0.2,
          f"per-vertex squint/blink ranges {frac.min():.2f}-{frac.max():.2f}, "
          f"not one constant")
    dominant = moving_idx[frac > 0.5 * (frac.max() + frac.min())]
    lower_dir = float(np.mean(squint_l[dominant] @ up)) if len(dominant) else 0.0
    check("the dominant squint lid travels upward", lower_dir > 0.0,
          f"mean displacement along the eye's up axis = {lower_dir:+.4f}")
    check("cheekSquint accompanies it", "cheekSquintLeft" in result.morph_targets)
    cheek = result.morph_targets.get("cheekSquintLeft")
    if cheek is not None:
        active = np.linalg.norm(cheek, axis=1) > 1e-6
        check("the cheek rises", float(np.mean(cheek[active] @ up)) > 0.0,
              f"{int(active.sum())} vertices, mean rise "
              f"{float(np.mean(cheek[active] @ up)):+.4f}")

    print("\n=== 1b. Iris, gaze and the rest of the eye family ===")
    names_built = sorted(result.morph_targets)
    gaze = [n for n in names_built if n.startswith("eyeLook")]
    check("gaze morphs were built", len(gaze) == 8, f"{len(gaze)}: {gaze}")

    protected = result.protected
    # The frozen region is the eye OPENING plus every added piece, and the opening is the
    # fitted ellipse -- which contains the parser's sample and is normally larger than it.
    # Requiring equality with the sample, as this did, pins the check to the old design in
    # which the lid could only ever cover the sampled vertices.
    check("the painted eye and every added piece are marked protected",
          protected[n0:].all() and protected[:n0][left | right].all(),
          f"{int(protected.sum())} vertices ({int(protected[:n0].sum())} eye opening, "
          f"covering all {int((left | right).sum())} painted "
          f"+ {len(protected) - n0} added)")

    if gaze:
        look = result.morph_targets["eyeLookUpLeft"]
        moved = np.nonzero(np.linalg.norm(look, axis=1) > 1e-9)[0]
        check("gaze moves only iris-cap vertices",
              moved.min() >= n0 and not np.any(look[:n0]),
              f"{len(moved)} vertices, all above the original {n0}")

        # Rigidity is the whole point. A rotation preserves every distance inside the cap;
        # a displacement field does not, and that difference is exactly what turns a round
        # pupil into an oval. Checking the pairwise distances is checking the pupil is still
        # a circle, without having to measure a circle.
        before = result.vertices[moved]
        after = before + look[moved]
        sample = moved[:: max(1, len(moved) // 40)]
        b = result.vertices[sample]
        a = b + look[sample]
        d_before = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)
        d_after = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
        scale = max(float(d_before.max()), 1e-9)
        drift = float(np.abs(d_after - d_before).max()) / scale
        check("gaze is a rigid rotation, so the iris stays round", drift < 1e-4,
              f"largest distance change is {drift:.2e} of the iris diameter")

        # And it has to actually go somewhere, in the right direction.
        _out, up_l, _across = eye_frame(v[left], vn[left], regions.forward)
        travelled = float(np.linalg.norm(after.mean(axis=0) - before.mean(axis=0)))
        check("gaze actually moves the iris", travelled > 1e-4,
              f"iris centre travels {travelled:.4f}")
        check("eyeLookUp moves the iris upward",
              float((after.mean(axis=0) - before.mean(axis=0)) @ up_l) > 0)
        down = result.morph_targets["eyeLookDownLeft"]
        check("eyeLookDown moves it the other way",
              float((down[moved].mean(axis=0)) @ up_l) < 0)

        # It must not slide off the opening and onto the cheek. The opening is the region
        # the patch itself built over -- the orbit ellipse, recoverable as the protected
        # vertices around this eye -- rather than the parser's sample. That is the binding
        # invariant: an iris that leaves the region the LIDS cover is an iris that stays
        # visible through a closed eye, which is the defect this bound exists to prevent.
        # Measuring against the sample instead bounds the gaze by a sparse subset of the
        # eye, and on a real character that collapsed the travel to nothing -- eyeLookUp and
        # eyeLookDown were not built at all.
        eye_region = result.protected[:len(v)] & (
            np.linalg.norm(v - v[left].mean(axis=0), axis=1) < 0.5 * regions.separation)
        centre_eye = v[eye_region].mean(axis=0)
        eye_reach = float(np.linalg.norm(v[eye_region] - centre_eye, axis=1).max())
        worst_reach = 0.0
        for name in gaze:
            if not name.endswith("Left"):
                continue
            d = result.morph_targets[name]
            pts = result.vertices[moved] + d[moved]
            worst_reach = max(worst_reach, float(np.linalg.norm(pts - centre_eye, axis=1).max()))
        check("the iris stays inside the eye opening", worst_reach <= eye_reach * 1.02,
              f"furthest iris point reaches {worst_reach / eye_reach:.2f} of the eye's radius")

    wide = result.morph_targets.get("eyeWideLeft")
    check("eyeWide was built", wide is not None)
    if wide is not None:
        touched = np.nonzero(np.linalg.norm(wide, axis=1) > 1e-9)[0]
        check("eyeWide never touches the painted eye or the iris",
              not np.any(protected[touched]),
              f"{len(touched)} skin vertices moved, none protected")

    print("\n=== 2. Auto-blink clip ===")
    merged = merge_morph_targets(
        {"mouthSmileLeft": np.zeros((len(result.vertices), 3), np.float32)},
        result.morph_targets,
    )
    names = list(merged.keys())
    clip = generate_blink_animation(names, duration=12.0, seed=0)
    check("a clip was produced", clip is not None)
    if clip is None:
        return 1

    track = clip["tracks"][0]
    times, values = track["times"], track["values"]
    check("track targets morph weights", track["path"] == "weights")
    check("times increase monotonically", bool(np.all(np.diff(times) > 0)),
          f"{len(times)} keyframes over {clip['duration']}s")
    check("values are one column per morph target", values.shape == (len(times), len(names)))
    check("weights stay in [0, 1]", float(values.min()) >= 0.0 and float(values.max()) <= 1.0)
    check("clip opens and closes shut",
          float(values[0].max()) == 0.0 and float(values[-1].max()) == 0.0)

    col_l = names.index("eyeBlinkLeft")
    col_r = names.index("eyeBlinkRight")
    non_blink = [i for i, n in enumerate(names) if not n.startswith("eyeBlink")]
    check("only the blink morphs are driven",
          float(values[:, non_blink].max()) == 0.0 if non_blink else True)

    curve = values[:, col_l]
    rises = np.nonzero(np.diff((curve > 0.5).astype(int)) == 1)[0]
    check("blinks fire at a plausible rate", 2 <= len(rises) <= 6,
          f"{len(rises)} blinks in 12s (a resting human does 3-4)")

    # --- property 6: closes far faster than it opens ---
    first = rises[0]
    peak = first + 1 + int(np.argmax(curve[first + 1:first + 6]))
    close_t = float(times[peak] - times[first])
    tail = peak + int(np.argmax(curve[peak:] < 1e-6)) if (curve[peak:] < 1e-6).any() else -1
    open_t = float(times[tail] - times[peak])
    check("the blink closes 3-6x faster than it opens", 3.0 <= open_t / close_t <= 6.0,
          f"close {close_t * 1000:.0f}ms, open {open_t * 1000:.0f}ms, "
          f"ratio {open_t / close_t:.1f}")
    check("the closing phase lasts 30-60ms", 0.030 <= close_t <= 0.060,
          f"{close_t * 1000:.0f}ms")
    check("the two eyes are not perfectly synchronous",
          not np.array_equal(values[:, col_l], values[:, col_r]),
          "one eye leads the other")

    same = generate_blink_animation(names, duration=12.0, seed=0)
    check("the clip is deterministic", np.allclose(same["tracks"][0]["times"], times))

    print("\n=== 2b. Eye expressions as clips ===")
    from pipeline.animation import generate_expression_animations
    from pipeline.facial_blendshapes import EXPRESSION_PRESETS

    exprs = generate_expression_animations(names)
    drivable = [k for k, p in EXPRESSION_PRESETS.items()
                if any(m in names for m in p.get("weights", {}))]
    check("every expression this mesh can drive became a clip",
          len(exprs) == len(drivable),
          f"{len(exprs)} clips for {len(drivable)} drivable presets "
          f"(of {len(EXPRESSION_PRESETS)} defined)")

    undrivable = [k for k in EXPRESSION_PRESETS if k not in drivable]
    check("presets with no morphs on this mesh produce no clip",
          all(EXPRESSION_PRESETS[k]["name"] not in exprs for k in undrivable),
          f"{len(undrivable)} skipped: {undrivable}" if undrivable else "none to skip")

    for label, clip in exprs.items():
        track = clip["tracks"][0]
        t, val = track["times"], track["values"]
        if not (track["path"] == "weights" and bool(np.all(np.diff(t) > 0))
                and val.shape == (len(t), len(names))
                and float(val[0].max()) == 0.0 and float(val[-1].max()) == 0.0
                and 0.0 <= float(val.min()) and float(val.max()) <= 1.0):
            check(f"clip '{label}' is well formed", False)
            break
    else:
        check("every clip is well formed", True,
              "monotonic times, one column per target, starts and ends neutral, "
              "weights within [0,1]")

    # A saccade is ballistic; a squint is muscular. If every expression shared one envelope
    # a glance would read as a head turn, so the clips have to differ where the physiology
    # does.
    def attack_of(label):
        clip = exprs[label]
        t, val = clip["tracks"][0]["times"], clip["tracks"][0]["values"]
        peak = int(np.argmax(val.max(axis=1)))
        return float(t[peak])

    look = EXPRESSION_PRESETS["look_up"]["name"]
    squint = EXPRESSION_PRESETS["squint"]["name"]
    if look in exprs and squint in exprs:
        check("a glance arrives far faster than a squint",
              attack_of(look) < 0.5 * attack_of(squint),
              f"gaze reaches full at {attack_of(look) * 1000:.0f}ms, "
              f"squint at {attack_of(squint) * 1000:.0f}ms")

    check("clips return to neutral and pause before looping",
          all(float(c["tracks"][0]["values"][-1].max()) == 0.0 for c in exprs.values()))

    print("\n=== 3. glTF export ===")
    glb = create_rigged_glb(
        vertices=result.vertices, faces=result.faces,
        joints=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32), parents=[None, 0],
        skin_weights=result.skin_weights, normals=result.normals, colors=result.colors,
        animations={"Blink": clip, **exprs}, morph_targets=merged,
    )
    check("a GLB was produced", len(glb) > 0, f"{len(glb)} bytes")
    magic, version, _length = struct.unpack("<III", glb[:12])
    check("GLB header is valid", magic == 0x46546C67 and version == 2)
    json_len, json_type = struct.unpack("<II", glb[12:20])
    check("first chunk is JSON", json_type == 0x4E4F534A)
    doc = json.loads(glb[20:20 + json_len].decode("utf-8"))

    targets = doc["meshes"][0]["primitives"][0].get("targets", [])
    declared = doc["meshes"][0].get("extras", {}).get("targetNames", [])
    check("every morph target survived export", len(targets) == len(names),
          f"{len(targets)} declared: {declared}")
    check("declared target order matches the blink track's column order", declared == names)

    anims = doc.get("animations", [])
    exported = [a["name"] for a in anims]
    check("the blink animation is present", "Blink" in exported)
    check("every expression clip survived export",
          all(label in exported for label in exprs),
          f"{len(anims)} clips exported: Blink + {len(exprs)} expressions")
    check("every exported clip drives the mesh node's weights",
          all(ch["target"]["path"] == "weights" and ch["target"]["node"] == 0
              for a in anims for ch in a["channels"]),
          f"{sum(len(a['channels']) for a in anims)} channels, all on node 0")
    channel = anims[exported.index("Blink")]["channels"][0]
    check("blink channel targets the mesh node's weights",
          channel["target"]["path"] == "weights" and channel["target"]["node"] == 0)
    sampler = anims[0]["samplers"][channel["sampler"]]
    out_acc, in_acc = doc["accessors"][sampler["output"]], doc["accessors"][sampler["input"]]
    check("weight output is SCALAR", out_acc["type"] == "SCALAR")
    check("weight count is frames x targets",
          out_acc["count"] == in_acc["count"] * len(targets),
          f"{out_acc['count']} == {in_acc['count']} x {len(targets)}")
    check("mesh declares a rest weight per target",
          len(doc["meshes"][0].get("weights", [])) == len(targets))

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
