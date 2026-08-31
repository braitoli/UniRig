"""
Eye- and brow-region detection on a character mesh.

Pipeline: take the head region found by `pipeline.mesh_segmentation`, render the head from
many yaw (and, if needed, pitch) angles via `pipeline.head_views`, run semantic face parsing
on every render, and backproject each labelled blob through that view's depth buffer to
recover its vertices on the 3D surface.

The parser is `jonathandinu/face-parsing` (see `pipeline.face_parsing`). It replaced OWLv2
box detection, which found eyes but only to box precision: a box around an eye also holds
cheek and brow, and the eyelid morph built from one deformed whatever else the box caught.
A per-pixel label gives the lid silhouette itself, which is the shape the morph needs. The
same pass also yields the brows, so `browDown*` / `browInnerUp` / `browOuterUp*` can be
placed instead of transferred blind.

An earlier attempt used deepghs/anime_eye_detection. That is a YOLO box detector and a
different model from the parser here, despite both appearing in the same reference repo. It
was measured across ~500 configurations (16 yaws x framings x crops x input sizes x 6 render
styles) on two generated characters and never located a real eye; that finding stands, and
is why no box detector is used.

Four properties of the surrounding design matter, each fixing a way an earlier version went
wrong:

* Detections are pooled across views instead of requiring one view to show both eyes. A
  quadruped carries its eyes on the sides of its skull, so no single camera angle ever sees
  both -- an "exactly 2 boxes in this frame" rule could never fire on one.
* A candidate is judged on the fraction of the azimuths that could SEE it which actually
  called it an eye, not on the raw count that did. Raw counts reward any feature that is
  merely large, because a large feature is visible -- and so detectable -- from many angles
  no matter what it is.
* An eye trusted on only one side is completed by reflecting it across the head's own
  mirror plane. A head with the eyes on opposite sides shows at most one to any camera, so
  finding exactly one is the normal outcome there, not a failure. Nothing is invented: a
  twin is added only where there is real surface and no detection already.
* Every threshold is a fraction of the head size or of the mesh's own edge length, so
  nothing depends on the mesh being at a particular scale (meshes are oriented and grounded
  by the pipeline, but never rescaled).

When the evidence is weak the module returns None. A wrong eye mask is worse than none:
downstream it becomes an eyeBlink morph target that folds the cheek instead of the lid.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .face_parsing import BROW_LABELS, EYE_REGION_LABELS, parse_faces, region_mask
from .head_views import HeadViewRenderer, View, backproject_pixels, spiral_yaws
from .mesh_segmentation import HeadRegion, mesh_edges, connected_region

# --- rendering ----------------------------------------------------------------------
_RENDER_SIZE = 768
_YAW_COUNT = 16
_PITCH_RETRY = (-25.0, 25.0)   # second pass: look down at, and up into, the face
_BATCH = 4

# Half the vertical frame covers this many head radii, so a smaller number is a tighter
# crop. Framing is the single largest factor in whether the parser responds at all, and the
# useful range is narrow. Measured on a generated character, 8 angles each:
#
#     framing   views with eye pixels   eye pixels
#      1.05            0 / 8                    0
#      1.25            2 / 8               10 384
#      1.50            4 / 8               21 001
#      1.80            5 / 8               11 849
#      2.20            5 / 8                4 701
#
# The reference implementation this was ported from reported the opposite -- that a crop
# tight enough for the head to overflow the frame was what made the model fire -- but its
# framing was measured against the head's own bounding *box*, while `head.radius` here is a
# bounding *sphere* over the whole head region, so 1.05 here is a far tighter crop than
# 0.95 was there and lands past the point where the head stops looking like a portrait.
#
# 1.5 is the primary because it maximises pixel mass while still clearing the two-view
# minimum the trust gate needs; 1.8 is the fallback because it sees the eye from one more
# angle, which is what matters when the primary pass comes back with too few views to
# cross-check.
_FRAMING = 1.5
_FRAMING_FALLBACK = 1.8

# --- candidate extraction -----------------------------------------------------------
_MIN_BLOB_PIXELS = 40          # below this a blob is noise, not a candidate
_EYE_HARD_MAX_AREA_FRACTION = 0.06   # of the head's own silhouette; catches only extreme
                                     # outliers, e.g. a whole head-side profile read as one
                                     # giant eye at a grazing angle
_MASK_EDGE_FACTOR = 2.0        # a backprojected point claims vertices this many median
                               # edge lengths away
_MASK_MIN_RADIUS = 0.015       # x head radius -- floor for coarse meshes
_VISIBILITY_EDGE_FACTOR = 3.0  # a vertex counts as visible when its own depth agrees with
                               # the depth buffer to within this many edge lengths

# --- trust gate, all fractions of the head size (= 2 x head radius) -------------------
_GROUP_MERGE_FRACTION = 0.09   # candidates whose 3D centroids land this close are the same
                               # eye seen twice, not two eyes
_GROUP_VISIBLE_FRACTION = 0.5  # a group counts as visible from an azimuth once this
                               # fraction of its vertices are on screen
_MIN_VOTE_RATIO = 0.45         # of the azimuths that could see it, this fraction must have
                               # called it an eye
_MIN_BROW_VOTE_RATIO = 0.35    # a brow is a thin arc and drops out of oblique views more
                               # readily than an eye does
_MIN_GROUP_COHERENCE = 0.25    # guard against a pathologically scattered group whose face
                               # normals point every which way; a real eye sits on one patch
_MIN_VISIBLE_AZIMUTHS = 2      # a patch only one camera could ever see has been
                               # cross-checked by nothing, and its vote ratio is a free pass
_MIN_DETECTING_VIEWS = 2       # and neither has one that only a single camera ever called
                               # an eye. The ratio alone does not catch this: once the
                               # denominator counts only the views the parser answered on,
                               # a lone detection among two responding views scores 0.50
                               # and clears the gate. Measured on a generated character,
                               # the two plausible eyes were seen from 3 and 2 angles while
                               # both false positives beside them were seen from exactly 1.

# --- mirror completion ---------------------------------------------------------------
_MIRROR_MAX_ERROR = 0.12       # a plane whose reflected head misses the real surface by
                               # more than this is not a symmetry plane and is not used
_MIRROR_MIDLINE_FRACTION = 0.04  # a detection this close to the plane is a midline feature
                                 # (snout, brow ridge); it has no twin
_MIRROR_TWIN_FRACTION = 0.15   # if a detection already sits this close to where the mirror
                               # would land, the twin was found directly
_MIRROR_SURFACE_FRACTION = 0.08  # the mirrored position must have real head surface this
                                 # close, otherwise the far side has no such feature

# --- pair selection, fractions of head radius or of the eye separation ----------------
_MIN_SEPARATION = 0.15         # closer than this and it is one eye split in two
_MAX_SEPARATION = 2.20         # wider than this and it is not a pair of eyes
_MAX_EYE_OFFSET = 1.30         # an eye further than this from the head centre is not on it
_MAX_HEIGHT_MISMATCH = 0.35    # x separation -- a pair of eyes sits level
_MAX_DEPTH_MISMATCH = 0.40     # x separation -- and equidistant from the head centre
_MIN_MASK_VERTICES = 8
_MAX_CONTESTED_FRACTION = 0.25  # of the smaller mask. Below this an overlap is region
                                # growth bridging two close eyes and is resolved by
                                # proximity; above it the two "eyes" are the same patch
                                # claimed twice.


@dataclass
class EyeRegions:
    """Per-vertex eye masks plus the head frame they were resolved in."""
    left_mask: np.ndarray      # (N,) bool -- the character's own left eye
    right_mask: np.ndarray     # (N,) bool
    left_center: np.ndarray    # (3,) float32
    right_center: np.ndarray   # (3,) float32
    forward: np.ndarray        # (3,) float32 unit, horizontal, direction the face points
    separation: float          # distance between eye centres
    confidence: float          # score of the weaker of the two accepted eyes
    n_detections: int
    n_views: int
    left_brow_mask: Optional[np.ndarray] = None   # (N,) bool, None when no brow was found
    right_brow_mask: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# 2D parsing
# ---------------------------------------------------------------------------
def detect_eyes_2d(images: Sequence[np.ndarray], threshold: float = 0.0
                   ) -> List[List[Tuple[float, float, float, float, float]]]:
    """
    Locates eyes in a batch of RGB images, as the bounding box of every parsed eye blob.

    Kept box-shaped for the debug overlay in `scripts/debug_head_eye.py`; the 3D pipeline
    below consumes the blobs themselves, not these boxes. `threshold` is a floor on the
    blob's mean parse confidence.

    Returns, per input image, a list of (x1, y1, x2, y2, confidence) in that image's pixel
    coordinates.
    """
    labels, conf = parse_faces(images)
    out: List[List[Tuple[float, float, float, float, float]]] = []
    for i in range(len(images)):
        boxes: List[Tuple[float, float, float, float, float]] = []
        mask = region_mask(labels[i], EYE_REGION_LABELS)
        if mask.any():
            comps, count = ndimage.label(mask)
            for comp_id in range(1, count + 1):
                comp = comps == comp_id
                if comp.sum() < _MIN_BLOB_PIXELS:
                    continue
                score = float(conf[i][comp].mean())
                if score < threshold:
                    continue
                ys, xs = np.nonzero(comp)
                boxes.append((float(xs.min()), float(ys.min()),
                              float(xs.max()), float(ys.max()), score))
        out.append(boxes)
    return out


# ---------------------------------------------------------------------------
# Backprojection helpers
# ---------------------------------------------------------------------------
@dataclass
class _Candidate:
    """One parsed blob in one view, lifted onto the mesh surface."""
    view_key: Tuple[float, float]
    vertex_idx: np.ndarray     # (K,) int64 into the full vertex array
    centroid: np.ndarray       # (3,) float32
    confidence: float          # mean parse confidence over the blob's pixels
    n_pixels: int


def _visible_vertices(view: View, vertices: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Which vertices this view can actually see, by projecting each one into the frame and
    comparing its own camera depth against the rendered depth buffer.

    This is what turns a raw vote count into a vote *ratio*: without knowing which cameras
    could have seen a patch, "3 views called it an eye" says nothing about whether the
    other cameras disagreed or simply had no line of sight.
    """
    inv = np.linalg.inv(view.cam_pose)
    cam = vertices @ inv[:3, :3].T + inv[:3, 3]
    depth = -cam[:, 2]                      # camera looks down -Z

    size = view.image_size
    f = (size / 2.0) / np.tan(view.yfov / 2.0)
    c = size / 2.0
    safe = np.maximum(depth, 1e-6)
    px = np.rint(cam[:, 0] / safe * f + c).astype(np.int32)
    py = np.rint(c - cam[:, 1] / safe * f).astype(np.int32)

    inside = (depth > 1e-6) & (px >= 0) & (px < size) & (py >= 0) & (py < size)
    visible = np.zeros(len(vertices), dtype=bool)
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return visible
    buffered = view.depth[py[idx], px[idx]]
    visible[idx] = (buffered > 0) & (np.abs(depth[idx] - buffered) <= tolerance)
    return visible


def _lift_blob(view: View, comp: np.ndarray, tree: cKDTree, max_dist: float) -> np.ndarray:
    """Backprojects one blob's pixels and returns the vertices they landed on."""
    py, px = np.nonzero(comp)
    points = backproject_pixels(view, px, py)
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    dist, idx = tree.query(points, k=1)
    return np.unique(idx[dist <= max_dist].astype(np.int64))


def _extract_candidates(
    view: View,
    labels: np.ndarray,
    conf: np.ndarray,
    label_ids: Sequence[int],
    vertices: np.ndarray,
    tree: cKDTree,
    max_dist: float,
    max_area_fraction: Optional[float],
) -> List[_Candidate]:
    """
    Splits every connected component of `label_ids` pixels in one view into its own
    candidate, each backprojected to its own vertex set.

    A single winning view is not enough on heads where the eyes sit on opposite sides: no
    azimuth ever frames both, so every view's blobs are kept and cross-checked in 3D
    afterwards rather than collapsing to "the" best view.
    """
    mask = region_mask(labels, label_ids)
    if mask.sum() < _MIN_BLOB_PIXELS:
        return []

    silhouette = int((view.depth > 0).sum())
    comps, count = ndimage.label(mask)
    out: List[_Candidate] = []
    for comp_id in range(1, count + 1):
        comp = comps == comp_id
        area = int(comp.sum())
        if area < _MIN_BLOB_PIXELS:
            continue
        if max_area_fraction is not None and silhouette > 0:
            if area / silhouette > max_area_fraction:
                continue
        idx = _lift_blob(view, comp, tree, max_dist)
        if len(idx) == 0:
            continue
        out.append(_Candidate(
            view_key=(view.yaw_deg, view.pitch_deg),
            vertex_idx=idx,
            centroid=vertices[idx].mean(axis=0).astype(np.float32),
            confidence=float(conf[comp].mean()),
            n_pixels=area,
        ))
    return out


# ---------------------------------------------------------------------------
# Cross-view agreement
# ---------------------------------------------------------------------------
def _normal_coherence(normals: np.ndarray) -> float:
    """
    How much a set of face normals agrees on one direction, ignoring winding.

    Averaging the normals directly would be the obvious measure. It is not usable here
    because face winding is not dependable on generated meshes: where roughly half the
    faces over an eye are wound the other way the normals cancel and a flat, clean patch
    scores near zero. Flipping each normal onto the patch's own dominant axis first --
    the leading eigenvector of the second-moment tensor, which is sign-free -- removes
    that artefact and leaves genuine curvature intact.
    """
    if len(normals) == 0:
        return 0.0
    tensor = (normals[:, :, None] * normals[:, None, :]).mean(axis=0)
    axis = np.linalg.eigh(tensor)[1][:, -1]
    sign = np.sign(normals @ axis)
    sign[sign == 0] = 1.0
    return float(np.linalg.norm((normals * sign[:, None]).mean(axis=0)))


def _cluster_candidates(candidates: List[_Candidate], merge_dist: float) -> List[Dict[str, Any]]:
    """
    Greedily groups candidates whose 3D centroids sit within `merge_dist` -- the same
    physical feature seen from different azimuths.

    Groups that stay apart are reported as distinct features, so a creature with more than
    two eye-like patches yields more than two groups rather than having them forced into a
    pair. Seeded in descending pixel area so the result depends only on the candidates:
    these masks get baked into a GLB, and the same mesh has to produce the same eyes on
    every run.
    """
    groups: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: -c.n_pixels):
        target, best = None, None
        for g in groups:
            d = float(np.linalg.norm(cand.centroid - g["centroid"]))
            if d < merge_dist and (best is None or d < best):
                target, best = g, d
        if target is None:
            groups.append({"members": [cand], "centroid": cand.centroid.copy()})
        else:
            target["members"].append(cand)
            total = sum(m.n_pixels for m in target["members"])
            target["centroid"] = (
                sum(m.centroid * m.n_pixels for m in target["members"]) / total
            ).astype(np.float32)
    return groups


def _evaluate_group(
    group: Dict[str, Any],
    faces: np.ndarray,
    face_normals: np.ndarray,
    n_vertices: int,
    visibility: Dict[Tuple[float, float], np.ndarray],
) -> Dict[str, Any]:
    """
    Measures one group's evidence. The decisive number is `vote_ratio`: of the azimuths
    that could have voted on this patch of surface, what fraction called it an eye?

    Counting only the views that detected it rewards any feature that is merely large and
    stable, because a large feature is visible -- and so detectable -- from many angles no
    matter what it is. Dividing by the views that could have voted separates the two: a
    real eye keeps looking like an eye from every angle that can see it, while a decorative
    feature only does from the angle that flatters it.

    `visibility` must already be narrowed to the views the parser actually responded on.
    Geometric line of sight is not enough: a parser trained on photographs answers only
    from angles where the head still reads as a portrait, and on a generated character it
    returns nothing but background from most of the sweep. Measured on one such character,
    a real eye was geometrically visible from 7 of 8 angles but parsed as an eye from 3,
    for a ratio of 0.43 -- below the gate -- while only 4 of those 8 views produced any eye
    labelling at all. Against that denominator the same eye scores 0.75 and the three
    single-view false positives beside it score 0.25.
    """
    members: List[_Candidate] = group["members"]
    detected = {m.view_key for m in members}
    vertex_idx = np.unique(np.concatenate([m.vertex_idx for m in members]))

    vmask = np.zeros(n_vertices, dtype=bool)
    vmask[vertex_idx] = True
    coherence = _normal_coherence(face_normals[vmask[faces].any(axis=1)])

    visible = {
        key for key, vis in visibility.items()
        if vis[vertex_idx].mean() >= _GROUP_VISIBLE_FRACTION
    }
    # Detection implies visibility, so never let a partly-occluded view make the
    # denominator smaller than the number of views that did detect it.
    n_visible = max(len(visible), len(detected), 1)

    return {
        "vertex_idx": vertex_idx,
        "centroid": group["centroid"],
        "views": detected,
        "n_visible": n_visible,
        "vote_ratio": len(detected) / n_visible,
        "coherence": coherence,
        "confidence": max(m.confidence for m in members),
        "n_detections": len(members),
        "source": "detected",
    }


def _trust(groups: List[Dict[str, Any]], min_vote_ratio: float,
           require_coherence: bool) -> List[Dict[str, Any]]:
    """Applies the gate and scores each surviving group."""
    kept = []
    for g in groups:
        if g["n_visible"] < _MIN_VISIBLE_AZIMUTHS:
            continue
        if len(g["views"]) < _MIN_DETECTING_VIEWS:
            continue
        if g["vote_ratio"] < min_vote_ratio:
            continue
        if require_coherence and g["coherence"] < _MIN_GROUP_COHERENCE:
            continue
        g["score"] = 0.7 * g["vote_ratio"] + 0.3 * g["coherence"]
        kept.append(g)
    return kept


# ---------------------------------------------------------------------------
# Mirror completion
# ---------------------------------------------------------------------------
def _mirror_plane(points: np.ndarray, colors: Optional[np.ndarray], seed: int = 0
                  ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Finds the head's bilateral mirror plane, as (normal, anchor, error).

    The search is over vertical planes only -- the plane a bilaterally symmetric creature
    is mirrored about contains its up axis -- which reduces it to one angle plus one
    offset, both swept exhaustively. `error` is the mean distance from a reflected point to
    the nearest real point, as a fraction of head size.

    Appearance is scored alongside geometry, and that is what makes this work on round
    heads: a cat's or a mascot's skull is nearly a sphere, so on shape alone the
    best-fitting plane cuts front-to-back through the face. The face colouring is not
    symmetric front-to-back at all, so a colour term snaps the plane back. Untextured
    meshes fall back to geometry, which is where they already were.
    """
    rng = np.random.default_rng(seed)
    n = min(2500, len(points))
    idx = rng.choice(len(points), n, replace=False)
    centre = points.mean(axis=0)
    sample = (points[idx] - centre).astype(np.float64)
    scale = float(np.ptp(sample, axis=0).max()) or 1.0
    rgb = colors[idx][:, :3].astype(np.float64) if colors is not None else None
    tree = cKDTree(sample)

    def sweep(thetas: np.ndarray, offsets: np.ndarray):
        best = (np.inf, 0.0, 0.0, np.inf)
        for theta in thetas:
            normal = np.array([np.cos(theta), 0.0, np.sin(theta)])
            projection = sample @ normal
            for offset in offsets:
                reflected = sample - 2.0 * np.outer(projection - offset, normal)
                distance, nearest = tree.query(reflected)
                geometry = float(distance.mean()) / scale
                score = geometry
                if rgb is not None:
                    score += 0.5 * float(np.abs(rgb - rgb[nearest]).mean()) / 255.0
                if score < best[0]:
                    best = (score, theta, offset, geometry)
        return best

    # Coarse sweep of the whole family, then a local refinement: one fine sweep costs
    # several times as much for the same answer.
    _, theta, offset, _ = sweep(
        np.linspace(0.0, np.pi, 45, endpoint=False), np.linspace(-0.15, 0.15, 7) * scale
    )
    step = np.pi / 45
    _, theta, offset, geometry = sweep(
        theta + np.linspace(-step, step, 9),
        offset + np.linspace(-0.05, 0.05, 9) * scale,
    )
    normal = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float32)
    return normal, (centre + offset * normal).astype(np.float32), geometry


def _mirror_missing_eyes(
    eyes: List[Dict[str, Any]],
    vertices: np.ndarray,
    head_vertex_ids: np.ndarray,
    normal: np.ndarray,
    anchor: np.ndarray,
    head_size: float,
) -> List[Dict[str, Any]]:
    """
    Adds the twin of any eye that was only ever detected on one side.

    This is what makes heads with the eyes on opposite sides work. Such a head never shows
    both eyes to one camera, and each eye on its own gets far less corroboration than a
    frontal pair does, so coming back with exactly one of the two is the normal outcome,
    not a failure.

    A twin is added only where the reflection lands on real head surface and nothing was
    already detected there. A detection sitting on the midline is skipped outright: that is
    a snout or a brow ridge, not half of a pair.
    """
    head_points = vertices[head_vertex_ids]
    tree = cKDTree(head_points)
    added: List[Dict[str, Any]] = []
    for eye in eyes:
        signed = float((eye["centroid"] - anchor) @ normal)
        if abs(signed) < _MIRROR_MIDLINE_FRACTION * head_size:
            continue
        twin = (eye["centroid"] - 2.0 * signed * normal).astype(np.float32)
        if any(float(np.linalg.norm(other["centroid"] - twin)) < _MIRROR_TWIN_FRACTION * head_size
               for other in eyes + added):
            continue
        if float(tree.query(twin)[0]) > _MIRROR_SURFACE_FRACTION * head_size:
            continue

        own = vertices[eye["vertex_idx"]]
        radius = max(float(np.linalg.norm(own - eye["centroid"], axis=1).max()),
                     0.02 * head_size)
        local = tree.query_ball_point(twin, radius)
        if not local:
            continue
        added.append({
            **eye,
            "centroid": twin,
            "vertex_idx": head_vertex_ids[np.asarray(local, dtype=np.int64)],
            "source": "mirror",
            "score": eye["score"] * 0.9,
        })
    return added


# ---------------------------------------------------------------------------
# Pair selection and mask building
# ---------------------------------------------------------------------------
def _select_pair(eyes: List[Dict[str, Any]], head: HeadRegion,
                 mirror: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Picks the two groups that best look like a pair of eyes on this head.

    Scoring every pair, rather than taking the top two by score, is what lets a genuine eye
    survive alongside a false positive: on a quadruped the detector also lit up a nostril
    and a gold disc on the cheek, and those pair badly -- wrong height, wrong distance from
    the head centre -- with anything.

    When a mirror plane was found, the pair must also straddle it. Distance and height
    checks alone cannot tell a real pair from one eye that fragmented into two
    detections on the same side of the head: the fragments sit at the same height and a
    plausible distance apart, which is exactly what those checks look for. Measured on a
    turtle, whose eyes face opposite ways and are therefore never seen together, the
    unconstrained version paired two fragments of the same eye and their grown masks then
    overlapped on 66% of the smaller one.
    """
    side = None
    if mirror is not None:
        normal, anchor = mirror
        side = {id(e): float((e["centroid"] - anchor) @ normal) for e in eyes}

    best = None
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            a, b = eyes[i], eyes[j]
            if side is not None and side[id(a)] * side[id(b)] > 0.0:
                continue
            separation = float(np.linalg.norm(a["centroid"] - b["centroid"]))
            if not (_MIN_SEPARATION * head.radius <= separation <= _MAX_SEPARATION * head.radius):
                continue
            if abs(float(a["centroid"][1] - b["centroid"][1])) > _MAX_HEIGHT_MISMATCH * separation:
                continue
            offset_a = float(np.linalg.norm(a["centroid"] - head.center))
            offset_b = float(np.linalg.norm(b["centroid"] - head.center))
            if max(offset_a, offset_b) > _MAX_EYE_OFFSET * head.radius:
                continue
            if abs(offset_a - offset_b) > _MAX_DEPTH_MISMATCH * separation:
                continue

            score = a["score"] + b["score"]
            if best is None or score > best[0]:
                best = (score, a, b)
    return (best[1], best[2]) if best is not None else None


def _median_edge_length(vertices: np.ndarray, faces: np.ndarray, sample: int = 20000) -> float:
    f = faces if len(faces) <= sample else faces[np.linspace(0, len(faces) - 1, sample).astype(np.int64)]
    a = np.linalg.norm(vertices[f[:, 0]] - vertices[f[:, 1]], axis=1)
    b = np.linalg.norm(vertices[f[:, 1]] - vertices[f[:, 2]], axis=1)
    return float(np.median(np.concatenate([a, b])))


def _mask_from_indices(vertex_idx: np.ndarray, n_vertices: int, edges: np.ndarray,
                       vertices: np.ndarray) -> np.ndarray:
    """A per-vertex mask over one connected island of the given vertices."""
    mask = np.zeros(n_vertices, dtype=bool)
    mask[vertex_idx] = True
    if not mask.any():
        return mask
    return connected_region(mask, edges, vertices)


def _collect_views(renderer: HeadViewRenderer, yaws: Sequence[float], pitches: Sequence[float],
                   vertices: np.ndarray, tree: cKDTree, max_dist: float,
                   visibility_tol: float,
                   ) -> Tuple[List[_Candidate], List[_Candidate], Dict[Tuple[float, float], np.ndarray]]:
    """Renders the sweep, parses it in batches, and returns eye and brow candidates."""
    eyes: List[_Candidate] = []
    brows: List[_Candidate] = []
    visibility: Dict[Tuple[float, float], np.ndarray] = {}
    batch: List[View] = []

    def flush():
        if not batch:
            return
        labels, conf = parse_faces([v.color for v in batch])
        for k, view in enumerate(batch):
            visibility[(view.yaw_deg, view.pitch_deg)] = _visible_vertices(
                view, vertices, visibility_tol)
            eyes.extend(_extract_candidates(
                view, labels[k], conf[k], EYE_REGION_LABELS, vertices, tree, max_dist,
                _EYE_HARD_MAX_AREA_FRACTION))
            brows.extend(_extract_candidates(
                view, labels[k], conf[k], BROW_LABELS, vertices, tree, max_dist, None))
        batch.clear()

    for view in renderer.sweep(yaws, pitches):
        batch.append(view)
        if len(batch) == _BATCH:
            flush()
    flush()
    return eyes, brows, visibility


def detect_eye_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    head: HeadRegion,
    vertex_colors: Optional[np.ndarray] = None,
    yaw_count: int = _YAW_COUNT,
    image_size: int = _RENDER_SIZE,
) -> Optional[EyeRegions]:
    """
    Finds the character's two eyes, and where possible its brows, as per-vertex masks.

    Returns None whenever the evidence does not form a credible pair of eyes on this head
    -- callers must skip eye deformation in that case rather than fall back to a guess.
    """
    if head is None or len(vertices) == 0 or len(faces) == 0:
        return None

    import trimesh

    n_vertices = len(vertices)
    head_size = 2.0 * head.radius
    edge = _median_edge_length(vertices, faces)
    max_dist = max(_MASK_EDGE_FACTOR * edge, _MASK_MIN_RADIUS * head.radius)
    visibility_tol = max(_VISIBILITY_EDGE_FACTOR * edge, 0.01 * head.radius)
    tree = cKDTree(vertices)
    yaws = spiral_yaws(yaw_count)

    def sweep(framing: float):
        with HeadViewRenderer(vertices, faces, head.center, head.radius,
                              vertex_colors=vertex_colors, image_size=image_size,
                              framing=framing) as renderer:
            eyes, brows, vis = _collect_views(
                renderer, yaws, (0.0,), vertices, tree, max_dist, visibility_tol)
            if len({c.view_key for c in eyes}) < 2 * _MIN_VISIBLE_AZIMUTHS:
                # Eyes set deep under a brow, or on a head tipped away from the horizon,
                # can be invisible at eye level but obvious from slightly above or below.
                more_eyes, more_brows, more_vis = _collect_views(
                    renderer, yaws, _PITCH_RETRY, vertices, tree, max_dist, visibility_tol)
                eyes += more_eyes
                brows += more_brows
                vis.update(more_vis)
            return eyes, brows, vis

    try:
        eye_cands, brow_cands, visibility = sweep(_FRAMING)
        seen = len({c.view_key for c in eye_cands})
        if seen < _MIN_VISIBLE_AZIMUTHS:
            # Too few angles saw anything for the cross-check to mean much. A wider crop
            # trades pixel mass for coverage (see the framing table above), which is the
            # right trade when coverage is what is missing. The two passes are not pooled:
            # visibility is keyed by camera angle but computed per framing, so mixing
            # candidates from both would test each group against the wrong depth buffers.
            print(f"[EyeDetection] Only {seen} view(s) parsed an eye at framing {_FRAMING}; "
                  f"retrying at {_FRAMING_FALLBACK}.")
            wide = sweep(_FRAMING_FALLBACK)
            if len({c.view_key for c in wide[0]}) > seen:
                eye_cands, brow_cands, visibility = wide
    except Exception as e:
        print(f"[EyeDetection] Rendering/parsing unavailable: {e}")
        return None

    if not eye_cands:
        print(f"[EyeDetection] Rejected: face parser found no eye pixels across "
              f"{len(yaws)} angles at either framing.")
        return None

    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_normals = np.asarray(tmesh.face_normals, dtype=np.float64)

    merge_dist = _GROUP_MERGE_FRACTION * head_size

    def voting_views(candidates: List[_Candidate]) -> Dict[Tuple[float, float], np.ndarray]:
        """Visibility restricted to the views that produced a candidate of this kind."""
        responsive = {c.view_key for c in candidates}
        return {k: v for k, v in visibility.items() if k in responsive}

    groups = [_evaluate_group(g, faces, face_normals, n_vertices, voting_views(eye_cands))
              for g in _cluster_candidates(eye_cands, merge_dist)]
    trusted = _trust(groups, _MIN_VOTE_RATIO, require_coherence=True)

    # --- complete the pair across the head's own mirror plane ---
    head_vertex_ids = np.nonzero(head.mask)[0]
    mirror: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if trusted and len(head_vertex_ids) >= 16:
        try:
            normal, anchor, error = _mirror_plane(
                vertices[head_vertex_ids],
                vertex_colors[head_vertex_ids] if vertex_colors is not None else None,
            )
            if error <= _MIRROR_MAX_ERROR:
                mirror = (normal, anchor)
                twins = _mirror_missing_eyes(trusted, vertices, head_vertex_ids,
                                             normal, anchor, head_size)
                if twins:
                    print(f"[EyeDetection] Mirror plane (error {error:.3f}) added "
                          f"{len(twins)} twin(s) for eyes seen from one side only.")
                trusted = trusted + twins
            else:
                print(f"[EyeDetection] Mirror plane rejected: error {error:.3f} > "
                      f"{_MIRROR_MAX_ERROR}; no twin completion.")
        except Exception as e:
            print(f"[EyeDetection] Mirror completion skipped: {e}")

    pair = _select_pair(trusted, head, mirror)
    if pair is None:
        print(f"[EyeDetection] Rejected: {len(eye_cands)} blobs formed {len(groups)} groups, "
              f"{len(trusted)} passed the trust gate, none of which pair up as eyes on this "
              f"head.")
        return None
    a, b = pair

    # --- resolve the head's own frame from the detections themselves ---
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye_mid = ((a["centroid"] + b["centroid"]) / 2.0).astype(np.float32)
    forward = np.array([eye_mid[0] - head.center[0], 0.0, eye_mid[2] - head.center[2]],
                       dtype=np.float32)
    if float(np.linalg.norm(forward)) < 0.05 * head.radius:
        # Eyes sit directly above the head centre (a flat, forward-tilted face). Fall back
        # to where the cameras that saw them were standing.
        acc = np.zeros(3, dtype=np.float32)
        for eye in (a, b):
            for yaw, _pitch in eye["views"]:
                r = np.radians(yaw)
                acc += np.array([np.sin(r), 0.0, np.cos(r)], dtype=np.float32)
        forward = acc
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        print("[EyeDetection] Rejected: cannot resolve which way the face points.")
        return None
    forward = (forward / norm).astype(np.float32)

    # For a character facing `forward` with +Y up, its own left points along up x forward.
    left_dir = np.cross(up, forward)
    if float(np.linalg.norm(left_dir)) < 1e-6:
        print("[EyeDetection] Rejected: face direction is vertical; left/right undefined.")
        return None
    left_dir = (left_dir / np.linalg.norm(left_dir)).astype(np.float32)

    left_eye, right_eye = (a, b) if float(np.dot(a["centroid"] - eye_mid, left_dir)) > 0 else (b, a)

    # --- lift to per-vertex masks ---
    edges = mesh_edges(faces, vertices)
    left_mask = _mask_from_indices(left_eye["vertex_idx"], n_vertices, edges, vertices)
    right_mask = _mask_from_indices(right_eye["vertex_idx"], n_vertices, edges, vertices)

    contested = left_mask & right_mask
    n_contested = int(contested.sum())
    if n_contested:
        smaller = min(int(left_mask.sum()), int(right_mask.sum()))
        if n_contested > _MAX_CONTESTED_FRACTION * smaller:
            # Two islands that are mostly the same island are one eye split in two, not two
            # eyes found.
            print(f"[EyeDetection] Rejected: left and right masks overlap on {n_contested} "
                  f"of {smaller} vertices -- this is one eye split, not two.")
            return None
        # A narrow head puts the two eyes within a short walk of each other across the
        # surface, and growing each detection into a connected island can bridge them.
        # Hand each contested vertex to the eye it is nearer to instead of discarding a
        # pair that is otherwise sound; a turtle, whose eyes sit on opposite sides of a
        # thin skull, is rejected outright without this.
        idx = np.nonzero(contested)[0]
        nearer_left = (np.linalg.norm(vertices[idx] - left_eye["centroid"], axis=1)
                       <= np.linalg.norm(vertices[idx] - right_eye["centroid"], axis=1))
        left_mask[idx] = nearer_left
        right_mask[idx] = ~nearer_left
        print(f"[EyeDetection] Split {n_contested} contested vertices between the eyes by "
              f"proximity.")

    if left_mask.sum() < _MIN_MASK_VERTICES or right_mask.sum() < _MIN_MASK_VERTICES:
        print(f"[EyeDetection] Rejected: eye masks too small "
              f"({int(left_mask.sum())}/{int(right_mask.sum())} vertices).")
        return None

    # --- brows, assigned to whichever eye they sit above ---
    left_brow_mask, right_brow_mask = None, None
    if brow_cands:
        brow_groups = [_evaluate_group(g, faces, face_normals, n_vertices,
                                       voting_views(brow_cands))
                       for g in _cluster_candidates(brow_cands, merge_dist)]
        brow_trusted = _trust(brow_groups, _MIN_BROW_VOTE_RATIO, require_coherence=False)
        buckets: Dict[str, List[np.ndarray]] = {"Left": [], "Right": []}
        for g in brow_trusted:
            to_left = float(np.linalg.norm(g["centroid"] - left_eye["centroid"]))
            to_right = float(np.linalg.norm(g["centroid"] - right_eye["centroid"]))
            buckets["Left" if to_left <= to_right else "Right"].append(g["vertex_idx"])
        for side, chunks in buckets.items():
            if not chunks:
                continue
            idx = np.unique(np.concatenate(chunks))
            mask = _mask_from_indices(idx, n_vertices, edges, vertices)
            mask &= ~(left_mask | right_mask)   # a brow is not part of the lid it sits over
            if mask.sum() < _MIN_MASK_VERTICES:
                continue
            if side == "Left":
                left_brow_mask = mask
            else:
                right_brow_mask = mask

    separation = float(np.linalg.norm(left_eye["centroid"] - right_eye["centroid"]))
    confidence = min(left_eye["confidence"], right_eye["confidence"])
    n_detections = left_eye["n_detections"] + right_eye["n_detections"]
    n_views = len(left_eye["views"] | right_eye["views"])
    sources = f"{left_eye['source']}/{right_eye['source']}"
    brow_report = (f"{int(left_brow_mask.sum()) if left_brow_mask is not None else 0}/"
                   f"{int(right_brow_mask.sum()) if right_brow_mask is not None else 0}")
    print(f"[EyeDetection] Eyes accepted ({sources}): {n_detections} blobs over {n_views} "
          f"views, parse confidence {confidence:.3f}, vote ratios "
          f"{left_eye['vote_ratio']:.2f}/{right_eye['vote_ratio']:.2f}, coherence "
          f"{left_eye['coherence']:.2f}/{right_eye['coherence']:.2f}, separation "
          f"{separation / head.radius:.2f} head radii, eye masks {int(left_mask.sum())}/"
          f"{int(right_mask.sum())} verts, brow masks {brow_report} verts, "
          f"forward={np.round(forward, 3).tolist()}.")

    return EyeRegions(
        left_mask=left_mask,
        right_mask=right_mask,
        left_center=left_eye["centroid"].astype(np.float32),
        right_center=right_eye["centroid"].astype(np.float32),
        forward=forward,
        separation=separation,
        confidence=confidence,
        n_detections=n_detections,
        n_views=n_views,
        left_brow_mask=left_brow_mask,
        right_brow_mask=right_brow_mask,
    )
