"""
Builds eyelids for characters that do not have any.

The meshes this pipeline produces are a single closed surface with the eyes painted on:
measured on a generated character, one geometry, `body_count == 1`, colour carried in
`ColorVisuals`. There is no eyeball and no lid to rotate, so the obvious rigging answer --
swing a lid over a ball -- has nothing to act on. Deforming the painted eye instead
squashes the pupil rather than covering it, so the character reads as squinting and can
never actually shut its eyes.

What this module does is add the missing lid as geometry: each eye's triangles are
duplicated, recoloured with the skin tone taken from the ring of surface just outside the
eye, and driven by a morph target. Four things separate a lid that reads as real from one
that reads as a sliding shutter, and all four come from how film and game rigs build eyes:

* **An upper and a lower lid, travelling unequal distances.** The upper lid does about 75%
  of the work and the lower lid rises slightly to meet it, so the closure line sits low on
  the eye rather than across its middle.
* **Corners that barely move.** Rigs weight the lid corners to roughly 0.2 of the centre's
  travel. Here that falls out of the geometry for free: each lid vertex starts from the
  eye's own silhouette directly above (or below) it, so where the silhouette pinches to a
  point at the corners there is nothing left to travel.
* **Motion along the eyeball, not across it.** A lid rotates about the eyeball's centre, so
  its path is an arc. A morph target interpolates positions linearly and therefore cuts the
  chord, which passes *under* a convex eye and buries the lid mid-blink. A sphere is fitted
  to the eye patch and each vertex is floated out by that arc's own sagitta, so the chord
  clears the surface it is sliding over.
* **A rest pose that is a crease, not a hiding place.** At weight 0 every lid vertex is
  collapsed onto its own column of the eye's outline, so the lid is a sliver along the
  lash line -- visible in principle, indistinguishable from an eyelid crease in practice --
  and it sweeps the eye from the very first frame. Hiding it *inside* the surface instead
  is where a real lid is, and it does not survive linear morph interpolation: every vertex
  then crosses the surface at the same fraction of its travel, measured at 0.79, so the eye
  looks untouched for three quarters of the blink and the lid appears all at once.

`eyeSquint` is not a half-blink. In FACS it is AU7, the lid tightener, which narrows the
eye opening *primarily by raising the lower lid* -- so it is built here from mostly lower
lid with a little upper, and it emits `cheekSquint` alongside, since AU6 raises the cheek
and squeezes the outer corner whenever a real squint happens.

Gaze is the same problem as the blink, one step further on. A painted pupil cannot be
looked around by deforming the face: a displacement field applied to a circle produces an
ellipse, which is why the eyeLook family used to tear the iris out of shape. The only
motion that leaves a circle circular is a rigid one, so the iris is lifted off the face as
its own cap and ROTATED about the fitted eyeball centre -- the character is given the
eyeball it never had. The sclera under the cap is repainted, which is invisible until the
gaze moves because the cap covers exactly that patch at rest.

`eyeWide` is AU5, the upper lid raiser. On a face whose eye is paint the opening itself
cannot grow, so what moves is the ring of skin around it; the painted eye is left alone.
Everything here obeys that rule -- no morph in this module displaces a vertex carrying the
eye's colour except the iris cap, and that one only rotates.

Only POSITION morph targets are used, so this works in any glTF 2.0 consumer -- including
the Three.js r128 viewer in `playground/`, which does not read morphed vertex colours.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .eye_detection import EyeRegions
from .mesh_segmentation import mesh_edges

# The upper lid covers this fraction of the eye's height and the lower lid takes the rest,
# so the two meet low on the eye. Taken from the ratio rigs are built to: the top lid comes
# down about three quarters of the way and the bottom lid rises slightly to meet it.
_UPPER_SHARE = 0.75

# How far the closed lid floats above the surface it copies, in median edge lengths. Enough
# that the two never z-fight, small enough to stay invisible on a silhouette.
_LID_FLOAT_EDGES = 0.35
# The lid grows past the silhouette it starts from by this fraction of its own span in that
# column, so full closure leaves no rim of pupil showing along the seam. Scaling it by the
# eye's overall height instead adds a constant to every column's travel, which at a corner
# where the column has almost no height IS the travel -- and the corners then move as far as
# the centre.
_LID_OVERSHOOT = 0.10
# The lower lid is built from a band this much taller than the quarter of the eye it ends up
# covering. At the literal 25% the band is one or two rows of vertices deep and no triangle
# fits inside it, so the lower lid silently fails to build and the eye closes with the upper
# lid alone -- which is what the first version of this did. The two lids therefore overlap
# slightly at the seam, which also guarantees no gap there.
_LID_SEAM_OVERLAP = 0.15
# In that overlap the upper lid must be unambiguously in front, so the lower one floats less.
_LOWER_FLOAT_SCALE = 0.60

# --- the iris, as its own piece of geometry -------------------------------------------
# The painted eye cannot be looked around by deforming it. A gaze morph that displaces the
# surface drags the pupil's own vertices with it, and a circle stretched by a displacement
# field is an ellipse -- which is exactly what the eyeLook family did before this. The only
# motion that leaves a circle circular is a rigid one, so the iris is lifted off the face
# as its own cap and ROTATED about the fitted eyeball centre, the way an eye actually moves.
#
# The cap floats less than either lid, so the lids still close over it.
_IRIS_FLOAT_EDGES = 0.15
# A vertex is iris if it is darker than this fraction of the way from the eye's darkest to
# its lightest colour. Generous, because a stylised eye is often almost all pupil.
_IRIS_LUMA_SPLIT = 0.55
# Gaze rotation is capped so the iris never slides out of the opening and onto the cheek.
# A real eye behaves the same way -- the iris stays inside the palpebral fissure -- so the
# limit is derived from how much room the opening actually leaves, then clamped here.
_GAZE_MAX_DEG = 24.0
_MIN_IRIS_VERTS = 6

# AU5 upper lid raiser, on a face whose eye is paint: the opening itself cannot grow, so
# what widens is the skin around it. The ring above the eye lifts and the ring below drops,
# and the painted eye is not touched at all.
_WIDE_UPPER = 0.16
_WIDE_LOWER = 0.08

# AU7 lid tightener: mostly lower lid, a little upper.
_SQUINT_LOWER = 0.55
_SQUINT_UPPER = 0.15
# AU6 cheek raiser, which accompanies it. The band of surface below the eye rises by this
# fraction of the eye's height, over a band this many eye-heights deep.
_CHEEK_RISE = 0.30
_CHEEK_BAND = 0.90

_MIN_LID_FACES = 2
_MIN_LID_VERTS = 6


@dataclass
class EyelidResult:
    """A mesh with lids welded in, and the morph targets that close them."""
    vertices: np.ndarray
    faces: np.ndarray
    colors: Optional[np.ndarray]
    uvs: Optional[np.ndarray]
    normals: Optional[np.ndarray]
    skin_weights: Optional[np.ndarray]
    morph_targets: Dict[str, np.ndarray]
    eye_regions: EyeRegions
    n_added: int
    protected: np.ndarray
    """Vertices no other deformer may touch: the painted eye, the lids, and the iris caps.

    Everything that carries the eye's image has to move rigidly or not at all. A
    transferred ARKit shape landing on any of it applies a displacement field, and a
    displacement field turns a circle into an ellipse -- which is what tore the pupil out of
    shape before the iris was given its own geometry."""


def _dilate(mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """One-ring growth of a vertex mask across the edge graph."""
    out = mask.copy()
    a, b = edges[:, 0], edges[:, 1]
    out[a[mask[b]]] = True
    out[b[mask[a]]] = True
    return out


def eye_frame(points: np.ndarray, normals: np.ndarray, forward: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The eye's own (outward, up, across) axes.

    Up is world up projected onto the eye's surface plane, not world up itself: on a head
    tipped forward, or on a quadruped whose eyes face sideways and slightly down, a lid
    that travels along global -Y slides across the eye rather than down it. Where the eye
    faces almost straight up or down that projection collapses, and the face's forward
    direction supplies the missing axis instead.
    """
    outward = normals.mean(axis=0)
    n = float(np.linalg.norm(outward))
    outward = (outward / n).astype(np.float32) if n > 1e-9 else np.array([0, 0, 1], np.float32)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up = world_up - float(np.dot(world_up, outward)) * outward
    if float(np.linalg.norm(up)) < 1e-3:
        up = np.cross(outward, forward)
    n = float(np.linalg.norm(up))
    if n < 1e-9:
        up = np.cross(outward, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        n = float(np.linalg.norm(up)) or 1.0
    up = (up / n).astype(np.float32)

    across = np.cross(outward, up)
    across = (across / max(float(np.linalg.norm(across)), 1e-9)).astype(np.float32)
    return outward, up, across


def _iris_mask(mask: np.ndarray, colors: Optional[np.ndarray], edges: np.ndarray) -> np.ndarray:
    """
    The dark part of the eye -- iris and pupil -- as its own vertex mask.

    Split on luminance within the eye region and keep only the piece that is connected, so
    a shadow at the eye's corner does not join the iris and drag the cap's centre off. An
    eye with no light part at all, which stylised characters often have, comes back as the
    whole region; that is the right answer for such a face, since the whole painted eye is
    then what moves when the character looks around.
    """
    if colors is None:
        return np.zeros(len(mask), dtype=bool)
    idx = np.nonzero(mask)[0]
    rgb = colors[idx][:, :3].astype(np.float32)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lo, hi = float(luma.min()), float(luma.max())
    if hi - lo < 1e-3:
        dark = np.ones(len(idx), dtype=bool)
    else:
        dark = luma <= lo + _IRIS_LUMA_SPLIT * (hi - lo)
    out = np.zeros(len(mask), dtype=bool)
    out[idx[dark]] = True
    if out.sum() < _MIN_IRIS_VERTS:
        return np.zeros(len(mask), dtype=bool)
    return out


def _rotate_about(points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float
                  ) -> np.ndarray:
    """Rodrigues rotation of `points` about `axis` through `pivot`."""
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    rel = points - pivot
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return (pivot + rel * c + np.cross(a, rel) * s
            + np.outer(rel @ a, a) * (1.0 - c)).astype(np.float32)


def _eyeball_radius(points: np.ndarray, centre: np.ndarray, outward: np.ndarray) -> float:
    """
    Radius of the sphere the eye patch sits on, by algebraic sphere fit.

    This is what turns the lid's travel into an arc rather than a straight slide. The fit is
    the standard linear one -- |p|^2 = 2 p.c + k, solved by least squares -- and its answer
    is clamped, because a nearly flat patch fits an enormous sphere and an almost hemispherical
    one fits a sphere smaller than itself; neither is a plausible eyeball for a patch of this
    size. A lid spanning 60-90 degrees of arc, which is what an eye opening is, sits on a
    sphere between roughly 0.8 and 3 times the patch's own radius.
    """
    spread = float(np.linalg.norm(points - centre, axis=1).max())
    lo, hi, default = 0.8 * spread, 3.0 * spread, 1.2 * spread
    try:
        rel = (points - centre).astype(np.float64)
        A = np.concatenate([2.0 * rel, np.ones((len(rel), 1))], axis=1)
        b = (rel * rel).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = sol[:3]
        r2 = sol[3] + float(c @ c)
        if r2 <= 0:
            return default
        # A centre on the outward side would mean the patch is concave here; the eyeball is
        # behind the surface by definition, so that fit is not usable.
        if float(c @ outward) > 0:
            return default
        return float(np.clip(np.sqrt(r2), lo, hi))
    except Exception:
        return default


def _profile_bins(h: np.ndarray, u: np.ndarray, w: np.ndarray, upper: bool
                  ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The eye's outline as (bin centre, extreme u, its w), ready to interpolate anywhere."""
    n_bins = int(np.clip(np.sqrt(len(h)), 6, 40))
    edges = np.linspace(h.min(), h.max() + 1e-6, n_bins + 1)
    idx = np.clip(np.digitize(h, edges) - 1, 0, n_bins - 1)

    centres = 0.5 * (edges[:-1] + edges[1:])
    prof_u = np.full(n_bins, np.nan)
    prof_w = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = np.nonzero(idx == b)[0]
        if len(sel) == 0:
            continue
        pick = sel[np.argmax(u[sel])] if upper else sel[np.argmin(u[sel])]
        prof_u[b] = u[pick]
        prof_w[b] = w[pick]

    good = ~np.isnan(prof_u)
    if good.sum() < 2:
        return None
    prof_u = np.interp(centres, centres[good], prof_u[good])
    prof_w = np.interp(centres, centres[good], prof_w[good])
    # A gentle one-tap smooth so a single stray vertex does not put a notch in the outline.
    # It is skipped on a coarse profile, where three taps span most of the eye and would
    # flatten the very corner pinch the outline is being read for.
    if n_bins >= 10:
        kernel = np.array([0.2, 0.6, 0.2])
        prof_u = np.convolve(np.pad(prof_u, 1, mode="edge"), kernel, mode="valid")
        prof_w = np.convolve(np.pad(prof_w, 1, mode="edge"), kernel, mode="valid")
    return centres, prof_u, prof_w


def _silhouette_profile(h: np.ndarray, u: np.ndarray, w: np.ndarray, upper: bool
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Where the eye's outline sits directly above (or below) each vertex.

    Binning by the across-axis and taking the extreme vertex in each bin traces the eye's
    own silhouette. Starting every lid vertex from the outline over its own column, rather
    than from one straight line across the whole eye, is what gives the corners their short
    travel: on an almond-shaped opening the outline meets itself at the corners, so a
    corner vertex has almost nowhere to go, while a vertex at the centre travels the full
    height. That is the same falloff rigs paint in by hand.
    """
    bins = _profile_bins(h, u, w, upper)
    if bins is None:
        extreme = float(u.max() if upper else u.min())
        return np.full(len(h), extreme, np.float32), np.full(len(h), float(w.mean()), np.float32)
    centres, prof_u, prof_w = bins
    return (np.interp(h, centres, prof_u).astype(np.float32),
            np.interp(h, centres, prof_w).astype(np.float32))


def _build_lid(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    mask: np.ndarray,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    centre: np.ndarray,
    radius: float,
    edge_length: float,
    upper: bool,
    region: np.ndarray,
    span: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Builds one lid -- the upper or the lower half of one eye.

    `region` is the surface this lid is copied from and `span` is, per vertex, how tall the
    lid is in its own column of the eye. Everything scales off that local span rather than
    off the eye's overall height, which is what keeps the corners still: where the opening
    pinches shut the span goes to zero, and so does every distance derived from it.

    Returns its source vertices, its closed and rest positions, and the triangles joining
    them, or None when that half of the eye has too little geometry to copy.
    """
    outward, up, across = frame

    # Coverage has to be airtight, and which faces each lid claims is what decides it. The
    # upper lid takes every face with ANY vertex above the seam; the lower takes only faces
    # lying wholly below it. Every face of the eye then belongs to at least one lid -- if it
    # is not wholly below the seam it has a vertex above, so the upper lid has it -- and no
    # face can be dropped by both. Requiring all three vertices on each side, which is the
    # obvious rule, drops exactly the faces that straddle the seam and punches a row of
    # holes along the closure line.
    inside = region[faces].sum(axis=1)
    sel = faces[inside >= 1] if upper else faces[inside == 3]
    if len(sel) < _MIN_LID_FACES:
        return None

    src_idx = np.unique(sel.ravel())
    if len(src_idx) < _MIN_LID_VERTS:
        return None

    src = vertices[src_idx]
    h = (src - centre) @ across
    u = (src - centre) @ up
    w = (src - centre) @ outward

    # Where this lid starts from: the eye's outline over each vertex's own column. The
    # overshoot past that outline is a fraction of the lid's own span in that column, not of
    # the eye's overall height -- a constant overshoot is most of the travel at a corner
    # where there is barely any column left, and it was what kept the corners moving.
    rest_u, rest_w = _silhouette_profile(h, u, w, upper=upper)
    rest_u = rest_u + (1.0 if upper else -1.0) * _LID_OVERSHOOT * np.abs(span[src_idx])

    travel = np.abs(rest_u - u)

    # A lid rotating about the eyeball centre sweeps this angle, and a straight morph path
    # cuts under the surface by that arc's sagitta. Float the closed pose out by it so the
    # chord clears the eye it is sliding over.
    theta = np.clip(travel / max(radius, 1e-6), 0.0, np.pi)
    sagitta = radius * (1.0 - np.cos(0.5 * theta))
    float_out = (_LID_FLOAT_EDGES * edge_length + sagitta) * (1.0 if upper else _LOWER_FLOAT_SCALE)

    # Both poses float clear of the surface by the same amount, so the lid is on top of the
    # eye for the whole of its travel.
    #
    # Hiding the rest pose *inside* the surface instead was tried and is wrong, even though
    # it is where a real lid is. A morph interpolates positions linearly, so a vertex that
    # starts inside and ends outside crosses the surface at tuck / (tuck + float) of its
    # travel -- the same fraction for every vertex, because both distances are roughly
    # constant across the lid. Measured, that put the crossing at 0.79: the eye looked
    # untouched through three quarters of the blink and then the whole lid appeared at once.
    # Collapsing each vertex onto its own column's outline instead leaves the rest pose as a
    # sliver along the top of the eye -- an eyelid crease, which is what is there anyway --
    # and the lid sweeps visibly from the first frame.
    # Both poses are lifted along the SAME direction -- each vertex's own normal. Lifting
    # the rest pose along the eye's mean axis instead leaves almost no clearance where the
    # two directions diverge, which is at the eye's rim: measured, the resting lid cleared
    # the surface by 5e-5 there and z-fought with it, punching holes through the lid.
    closed = src + vertex_normals[src_idx] * float_out[:, None]
    rest = (centre
            + across * h[:, None]
            + up * rest_u[:, None]
            + outward * rest_w[:, None]
            + vertex_normals[src_idx] * float_out[:, None])

    return {"src_idx": src_idx, "faces": sel,
            "closed": closed.astype(np.float32), "rest": rest.astype(np.float32)}


def _skin_colors(
    vertices: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    src_idx: np.ndarray,
    colors: Optional[np.ndarray],
    uvs: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Colour and UV for the lid, taken from the ring of surface just outside the eye.

    Per-vertex nearest-neighbour rather than one averaged tone, so a lid on a two-toned face
    picks up the shading it sits in. The UV is carried across for the same reason: on a
    textured export there is no COLOR_0 to fall back on, and a lid keeping the eye's own UV
    would be painted with the pupil it is supposed to cover.
    """
    if colors is None and uvs is None:
        return None, None

    ring = _dilate(_dilate(mask, edges), edges) & ~mask
    ring_idx = np.nonzero(ring)[0]
    if len(ring_idx) == 0:
        return (None if colors is None else colors[src_idx].copy(),
                None if uvs is None else uvs[src_idx].copy())

    _, nearest = cKDTree(vertices[ring_idx]).query(vertices[src_idx], k=1)
    picked = ring_idx[nearest]
    return (None if colors is None else colors[picked].copy(),
            None if uvs is None else uvs[picked].copy())


def _cheek_delta(
    vertices: np.ndarray,
    mask: np.ndarray,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    centre: np.ndarray,
    total: int,
) -> Optional[np.ndarray]:
    """
    AU6, the cheek raiser: the band of surface below the eye lifts and squeezes the outer
    corner.

    A squint without it reads as a droopy eyelid. It is emitted as its own `cheekSquint`
    morph rather than folded into the squint, because that is what ARKit expects and what
    lets a consumer dial the two independently.
    """
    _outward, up, _across = frame
    rel = vertices - centre
    u = rel @ up
    eye_u = u[mask]
    height = float(np.ptp(eye_u)) or 1e-6
    floor = float(eye_u.min())

    band = (u < floor) & (u >= floor - _CHEEK_BAND * height)
    if band.sum() < _MIN_LID_VERTS:
        return None

    # Only the surface near the eye laterally, so the whole jaw does not ride up with it.
    near, _ = cKDTree(vertices[mask]).query(vertices, k=1)
    band &= near <= _CHEEK_BAND * height
    if band.sum() < _MIN_LID_VERTS:
        return None

    idx = np.nonzero(band)[0]
    t = np.clip((u[idx] - (floor - _CHEEK_BAND * height)) / (_CHEEK_BAND * height), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    delta = np.zeros((total, 3), dtype=np.float32)
    delta[idx] = up * (_CHEEK_RISE * height * weight)[:, None]
    return delta


def attach_eyelids(
    vertices: np.ndarray,
    faces: np.ndarray,
    eye_regions: EyeRegions,
    colors: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    skin_weights: Optional[np.ndarray] = None,
    appearance: Optional[np.ndarray] = None,
) -> Optional[EyelidResult]:
    """
    Welds an upper and a lower eyelid over each detected eye and returns the extended mesh
    plus the `eyeBlink*`, `eyeSquint*` and `cheekSquint*` morph targets that drive them.

    Every per-vertex array given is extended in step, so the result can be handed straight
    to `create_rigged_glb`. Lid vertices inherit their source vertex's skin weights, which
    keeps the lid attached to the head without rerunning skinning.

    `appearance` is what the eye LOOKS like, used only to decide where the iris is and what
    the skin around it is coloured. It exists separately from `colors` because a textured
    character has no COLOR_0 at all -- glTF would multiply it with the texture and darken
    the result -- yet the iris still has to be found. The caller passes the texture baked
    down per vertex there while leaving `colors` as None.

    Returns None when neither eye yields enough geometry to copy.
    """
    import trimesh

    n = len(vertices)
    if eye_regions is None or n == 0 or len(faces) == 0:
        return None

    if normals is not None and len(normals) == n:
        vertex_normals = np.asarray(normals, dtype=np.float32)
    else:
        vertex_normals = np.asarray(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals,
            dtype=np.float32,
        )

    if appearance is None:
        appearance = colors

    edges = mesh_edges(faces, vertices)
    tri = faces if len(faces) <= 20000 else faces[np.linspace(0, len(faces) - 1, 20000).astype(np.int64)]
    edge_length = float(np.median(np.concatenate([
        np.linalg.norm(vertices[tri[:, 0]] - vertices[tri[:, 1]], axis=1),
        np.linalg.norm(vertices[tri[:, 1]] - vertices[tri[:, 2]], axis=1),
    ])))

    new_vertices = [vertices.astype(np.float32)]
    new_faces = [faces.astype(np.int64)]
    new_colors = [colors.copy()] if colors is not None else None
    new_uvs = [uvs] if uvs is not None else None
    new_normals = [vertex_normals]
    new_weights = [skin_weights] if skin_weights is not None else None

    offset = n
    built: List[Dict[str, Any]] = []
    cheeks: Dict[str, np.ndarray] = {}
    irises: List[Dict[str, Any]] = []
    widens: Dict[str, np.ndarray] = {}
    sclera_repaint: List[Tuple[np.ndarray, np.ndarray]] = []
    eye_centres = {"Left": None, "Right": None}

    for side, mask in (("Left", eye_regions.left_mask), ("Right", eye_regions.right_mask)):
        idx = np.nonzero(mask)[0]
        if len(idx) < _MIN_LID_VERTS:
            print(f"[EyelidPatch] {side} eye has too few vertices to copy a lid from.")
            continue

        centre = vertices[idx].mean(axis=0).astype(np.float32)
        frame = eye_frame(vertices[idx], vertex_normals[idx], eye_regions.forward)
        outward, up, across_axis = frame
        out_axis = outward
        radius = _eyeball_radius(vertices[idx], centre, outward)

        # Where the two lids meet. This is a CURVE, not a horizontal line: it sits at 75%
        # of each column's own height, so it runs into the eye's outline at the corners
        # where the opening pinches shut. A single height across the whole eye -- which is
        # what this was first written as -- leaves the upper lid at the corner still
        # spanning the full opening there, so the corner travels almost as far as the
        # centre and the lid sweeps down like a shutter. Measured on an almond-shaped eye,
        # a flat split put the corner at 0.81 of the centre's travel; following each
        # column's own height brings it to where rigs paint it by hand.
        h_eye = (vertices[idx] - centre) @ across_axis
        u_eye = (vertices[idx] - centre) @ up
        w_eye = (vertices[idx] - centre) @ out_axis
        h_all = (vertices - centre) @ across_axis
        u_all = (vertices - centre) @ up
        top_bins = _profile_bins(h_eye, u_eye, w_eye, upper=True)
        bot_bins = _profile_bins(h_eye, u_eye, w_eye, upper=False)
        if top_bins is None or bot_bins is None:
            flat = float(u_eye.min() + (1.0 - _UPPER_SHARE) * np.ptp(u_eye))
            top = np.full(len(vertices), float(u_eye.max()), np.float32)
            bot = np.full(len(vertices), float(u_eye.min()), np.float32)
            split_u = np.full(len(vertices), flat, np.float32)
        else:
            top = np.interp(h_all, top_bins[0], top_bins[1])
            bot = np.interp(h_all, bot_bins[0], bot_bins[1])
            split_u = (bot + (1.0 - _UPPER_SHARE) * (top - bot)).astype(np.float32)

        column = np.maximum(top - bot, 0.0)
        # The lower lid reaches a little past the seam so full closure leaves no gap there,
        # and that margin is also a fraction of the column rather than of the whole eye.
        lower_cut = split_u + _LID_SEAM_OVERLAP * column
        regions = {
            True: (mask & (u_all >= split_u), top - split_u),
            False: (mask & (u_all < lower_cut), lower_cut - bot),
        }

        lids = []
        for upper in (True, False):
            region, span = regions[upper]
            lid = _build_lid(vertices, faces, vertex_normals, mask, frame, centre, radius,
                             edge_length, upper, region, span)
            if lid is not None:
                lid["upper"] = upper
                lids.append(lid)
        if not lids:
            print(f"[EyelidPatch] {side} eye has too little geometry to copy a lid from.")
            continue

        eye_centres[side] = centre

        # --- the iris, lifted off the face so it can rotate instead of being stretched ---
        iris = _iris_mask(mask, appearance, edges)
        iris_faces = faces[iris[faces].sum(axis=1) == 3]
        if len(iris_faces) >= _MIN_LID_FACES and appearance is not None:
            iris_idx = np.unique(iris_faces.ravel())
            # The eyeball centre the iris swings about: back along the surface normal by the
            # fitted radius. Every gaze direction is a rotation about this one point, which
            # is what keeps the iris a circle no matter where it is looking.
            pivot = (centre - outward * radius).astype(np.float32)
            iris_rel = vertices[iris_idx] - centre
            half_eye = {"up": float(np.abs(u_eye).max()),
                        "across": float(np.abs(h_eye).max())}
            half_iris = {"up": float(np.abs(iris_rel @ up).max()),
                         "across": float(np.abs(iris_rel @ across_axis).max())}
            irises.append({
                "side": side, "faces": iris_faces, "idx": iris_idx, "pivot": pivot,
                "outward": outward, "up": up, "across": across_axis, "radius": radius,
                "centre": centre, "half_eye": half_eye, "half_iris": half_iris,
            })
            # What shows once the iris has moved off its resting spot. Repainting the eye
            # under the cap is safe because the cap covers exactly that patch at rest, so
            # nothing about the character's appearance changes until it looks somewhere.
            if colors is not None:
                eye_rgb = colors[idx][:, :3].astype(np.float32)
                luma = eye_rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
                sclera = colors[idx][np.argsort(luma)[-max(1, len(idx) // 8):]].mean(axis=0)
                sclera_repaint.append((iris_idx, sclera.astype(colors.dtype)))

        # --- AU5: widen the opening by moving the skin around it, never the eye itself ---
        ring = _dilate(_dilate(mask, edges), edges) & ~mask
        ring_idx = np.nonzero(ring)[0]
        if len(ring_idx) >= _MIN_LID_VERTS:
            eye_h = float(np.ptp(u_eye)) or 1e-6
            ring_u = (vertices[ring_idx] - centre) @ up
            near, _ = cKDTree(vertices[idx]).query(vertices[ring_idx], k=1)
            fade = np.clip(1.0 - near / (0.8 * eye_h), 0.0, 1.0)
            fade = fade * fade * (3.0 - 2.0 * fade)
            amount = np.where(ring_u >= 0.0, _WIDE_UPPER, -_WIDE_LOWER) * eye_h * fade
            wide = np.zeros((n, 3), dtype=np.float32)
            wide[ring_idx] = up * amount[:, None]
            widens[side] = wide

        for lid in lids:
            src_idx = lid["src_idx"]
            count = len(src_idx)
            remap = np.full(n, -1, dtype=np.int64)
            remap[src_idx] = np.arange(offset, offset + count, dtype=np.int64)

            new_vertices.append(lid["rest"])
            new_faces.append(remap[lid["faces"]])
            new_normals.append(vertex_normals[src_idx].copy())
            if new_weights is not None:
                new_weights.append(skin_weights[src_idx].copy())
            lid_colors, lid_uvs = _skin_colors(vertices, edges, mask, src_idx, colors, uvs)
            if new_colors is not None:
                new_colors.append(lid_colors)
            if new_uvs is not None:
                new_uvs.append(lid_uvs)

            built.append({"side": side, "upper": lid["upper"], "start": offset,
                          "stop": offset + count,
                          "travel": (lid["closed"] - lid["rest"]).astype(np.float32)})
            offset += count

        cheek = _cheek_delta(vertices, mask, frame, centre, n)
        if cheek is not None:
            cheeks[side] = cheek

        print(f"[EyelidPatch] {side} eye: eyeball radius {radius:.4f} "
              f"({radius / max(float(np.linalg.norm(vertices[idx] - centre, axis=1).max()), 1e-9):.2f}x "
              f"patch radius), {len(lids)} lid(s).")

    if not built:
        return None

    # --- weld each iris on as its own cap, floating just clear of the sclera ---
    for iris in irises:
        src_idx = iris["idx"]
        count = len(src_idx)
        remap = np.full(n, -1, dtype=np.int64)
        remap[src_idx] = np.arange(offset, offset + count, dtype=np.int64)
        new_vertices.append(
            (vertices[src_idx] + vertex_normals[src_idx] * (_IRIS_FLOAT_EDGES * edge_length))
            .astype(np.float32))
        new_faces.append(remap[iris["faces"]])
        new_normals.append(vertex_normals[src_idx].copy())
        if new_weights is not None:
            new_weights.append(skin_weights[src_idx].copy())
        if new_colors is not None:
            new_colors.append(colors[src_idx].copy())
        if new_uvs is not None:
            new_uvs.append(uvs[src_idx].copy())
        iris["start"], iris["stop"] = offset, offset + count
        offset += count

    # What is under the iris cap. Safe to change because at rest the cap covers exactly this
    # patch, so the character looks identical until its gaze actually moves; without it, a
    # glance drags the cap aside and reveals a second copy of the pupil underneath.
    if new_colors is not None:
        for iris_idx, sclera in sclera_repaint:
            new_colors[0][iris_idx] = sclera

    total = offset
    out_vertices = np.concatenate(new_vertices).astype(np.float32)
    out_faces = np.concatenate(new_faces).astype(np.int64)

    def blank() -> np.ndarray:
        return np.zeros((total, 3), dtype=np.float32)

    morph_targets: Dict[str, np.ndarray] = {}
    for lid in built:
        side, start, stop, travel = lid["side"], lid["start"], lid["stop"], lid["travel"]
        blink = morph_targets.setdefault(f"eyeBlink{side}", blank())
        blink[start:stop] += travel
        # AU7 narrows the opening mostly from below, so the lower lid carries the squint and
        # the upper only tightens. Scaling the blink uniformly instead -- which is what this
        # used to do -- produces a half-shut eye, which reads as drowsy, not narrowed.
        share = _SQUINT_UPPER if lid["upper"] else _SQUINT_LOWER
        squint = morph_targets.setdefault(f"eyeSquint{side}", blank())
        squint[start:stop] += travel * share

    for side, delta in cheeks.items():
        padded = blank()
        padded[:n] = delta
        morph_targets[f"cheekSquint{side}"] = padded

    for side, delta in widens.items():
        padded = blank()
        padded[:n] = delta
        morph_targets[f"eyeWide{side}"] = padded

    # --- gaze: rigid rotation of the iris cap about the eyeball centre ---
    for iris in irises:
        side = iris["side"]
        start, stop = iris["start"], iris["stop"]
        rest = out_vertices[start:stop]
        other = eye_centres["Right" if side == "Left" else "Left"]
        inward = None
        if other is not None:
            d = other - iris["centre"]
            d = d - float(d @ iris["outward"]) * iris["outward"]      # onto the eye's surface
            if float(np.linalg.norm(d)) > 1e-9:
                inward = (d / np.linalg.norm(d)).astype(np.float32)
        if inward is None:
            inward = iris["across"]

        # How far the iris may swing before it would leave the opening and sit on the
        # cheek. A real eye is bounded the same way, so this is the physical limit rather
        # than a fudge factor: the room left over, divided by the eyeball's radius, is an
        # angle.
        def limit(axis_key: str) -> float:
            room = max(iris["half_eye"][axis_key] - iris["half_iris"][axis_key], 0.0)
            return float(min(room / max(iris["radius"], 1e-9), np.radians(_GAZE_MAX_DEG)))

        directions = {
            "Up": (iris["up"], limit("up")),
            "Down": (-iris["up"], limit("up")),
            "In": (inward, limit("across")),
            "Out": (-inward, limit("across")),
        }
        for name, (direction, angle) in directions.items():
            if angle <= 1e-6:
                continue
            # Rotating about `outward x direction` through the pivot carries the cap towards
            # `direction` along the eyeball's own surface.
            axis = np.cross(iris["outward"], direction)
            if float(np.linalg.norm(axis)) < 1e-9:
                continue
            moved = _rotate_about(rest, iris["pivot"], axis, angle)
            delta = blank()
            delta[start:stop] = (moved - rest).astype(np.float32)
            morph_targets[f"eyeLook{name}{side}"] = delta

    def pad(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        out = np.zeros(total, dtype=bool)
        out[:n] = mask
        return out

    padded_regions = EyeRegions(
        left_mask=pad(eye_regions.left_mask),
        right_mask=pad(eye_regions.right_mask),
        left_center=eye_regions.left_center,
        right_center=eye_regions.right_center,
        forward=eye_regions.forward,
        separation=eye_regions.separation,
        confidence=eye_regions.confidence,
        n_detections=eye_regions.n_detections,
        n_views=eye_regions.n_views,
        left_brow_mask=pad(eye_regions.left_brow_mask),
        right_brow_mask=pad(eye_regions.right_brow_mask),
    )

    protected = np.zeros(total, dtype=bool)
    protected[n:] = True                                   # every lid and iris vertex
    protected[:n] = eye_regions.left_mask | eye_regions.right_mask

    added = total - n
    lids_report = ", ".join(f"{l['side']}/{'upper' if l['upper'] else 'lower'}" for l in built)
    print(f"[EyelidPatch] Added {added} lid vertices over {len(built)} lid(s) ({lids_report}); "
          f"mesh {n} -> {total} vertices, {len(faces)} -> {len(out_faces)} faces; "
          f"morphs {sorted(morph_targets)}.")

    return EyelidResult(
        vertices=out_vertices,
        faces=out_faces,
        colors=np.concatenate(new_colors) if new_colors is not None else None,
        uvs=np.concatenate(new_uvs) if new_uvs is not None else None,
        normals=np.concatenate(new_normals).astype(np.float32),
        skin_weights=np.concatenate(new_weights) if new_weights is not None else None,
        morph_targets=morph_targets,
        eye_regions=padded_regions,
        n_added=added,
        protected=protected,
    )


def merge_morph_targets(base: Dict[str, np.ndarray], extra: Dict[str, np.ndarray]
                        ) -> Dict[str, np.ndarray]:
    """
    Combines two sets of morph targets, summing the deltas where a name appears in both.

    Summing rather than overwriting is deliberate. `eyeBlinkLeft` exists in both sets and
    they act on disjoint vertices: the transferred ARKit shape creases the eye opening
    itself, the lid patch draws the new geometry over it. Keeping only one loses either the
    crease or the closure.
    """
    out = {k: v.copy() for k, v in base.items()}
    for name, delta in extra.items():
        if name in out and out[name].shape == delta.shape:
            out[name] = out[name] + delta
        else:
            out[name] = delta.copy()
    return out
