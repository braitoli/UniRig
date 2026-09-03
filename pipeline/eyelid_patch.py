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
from .mesh_segmentation import mesh_edges, weld_groups, weld_mask

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
# Fraction of the measured room the gaze is actually allowed to use.
_GAZE_ROOM_SAFETY = 0.8
_MIN_IRIS_VERTS = 6
# A vertex inside the opening is painted eye rather than skin when its luminance sits this
# many robust standard deviations from the skin ring's own, with a floor for a ring so
# uniform that its deviation is nearly zero.
_EYE_COLOUR_SIGMA = 3.0
_EYE_COLOUR_FLOOR = 0.03
# A colour-segmented iris is believed only when it spans between these fractions of the eye
# opening. Outside them the segmentation has locked onto something that is not an iris, and
# a disc of this proportion is used instead -- an iris is about 12mm across a 30mm opening.
_IRIS_MIN_SHARE = 0.20
_IRIS_MAX_SHARE = 0.80
_IRIS_DISC_SHARE = 0.45
# ...and it must fill this fraction of the disc its own reach implies, or it is a
# thread rather than an iris.
_IRIS_MIN_FILL = 0.5

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
# How far the affine UV fit may miss the ring it was fitted to, as a fraction of the ring's
# own UV extent, before the lid falls back to a single flat UV instead.
_LID_UV_FIT_TOLERANCE = 0.25
# ...and how much faster than the body's own UV-per-world rate the fitted map may run. Above
# this it is not unwrapping the lid, it is racing between atlas islands.
_LID_UV_MAX_GRADIENT = 2.0

# --- the eye opening, as an ellipse rather than as the parser's vertex sample -----------
# Half-axes are taken at this percentile of the mask's extent, not at its maximum: the mask
# is sparse and occasionally holds one vertex up on the brow, and a maximum would stretch
# the opening all the way to it.
_ORBIT_PERCENTILE = 97.0
# ...then grown by this much, so the lid covers the eye rather than stopping level with the
# outermost sample. The lid is skin-coloured, so overshooting onto skin is invisible while
# falling short leaves a rim of pupil showing.
#
# The value is bracketed from both sides by measurement, not chosen. Sweeping it on a
# 4K-textured character, against the fraction of the eye still visible at full blink and
# against whether the opening reached the parser's brow:
#
#     margin   eye still visible   brow vertices swallowed
#      1.15          60.5%                    0
#      1.30          17.2%                    0
#      1.50          11.6%                    0
#      1.75          11.7%                   19
#      2.40          25.8%                  176
#
# Below 1.5 the lid stops short of the sclera; above it the opening starts eating the brow,
# and past 2.0 coverage gets worse again -- an ellipse much larger than the eye puts the
# closure line and the silhouette profile in the wrong place.
_ORBIT_MARGIN = 1.50
# How far from the eye's own plane a vertex may sit and still belong to the opening, as a
# fraction of the larger half-axis. The ellipse is a projection; without this it also claims
# the surface directly behind the eye.
_ORBIT_DEPTH = 1.0
# The ellipse has to land on the surface the parser labelled. Below this fraction of the
# mask recovered, it did not, and the mask is used unchanged.
_ORBIT_MIN_AGREEMENT = 0.5


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


def _dilate(mask: np.ndarray, edges: np.ndarray, groups: Optional[np.ndarray] = None
            ) -> np.ndarray:
    """
    One-ring growth of a vertex mask across the edge graph.

    Welded afterwards when `groups` is given. A single ring cannot cross a UV seam on its
    own: the copies of a point there share no edge, so a copy whose own triangles are all
    outside the mask stays outside while its twin comes in. The ring built from that
    difference drives `eyeWide`, and a delta applied to one copy of a point and not the
    other tore this character's eye open by 1.6 median edges.
    """
    out = mask.copy()
    a, b = edges[:, 0], edges[:, 1]
    out[a[mask[b]]] = True
    out[b[mask[a]]] = True
    return out if groups is None else weld_mask(out, groups=groups)


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


def _orbit_region(vertices: np.ndarray, edges: np.ndarray, mask: np.ndarray,
                  centre: np.ndarray, frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  groups: Optional[np.ndarray] = None,
                  exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The eye opening as a filled ellipse on the surface, recovered from a mask that only
    samples it.

    The parser's mask is a SAMPLE of the eye, not the eye. `_lift_blob` returns one vertex
    per labelled PIXEL, so the mask's size is bounded by the render's resolution and has
    nothing to do with how finely the mesh is tessellated. Measured on a 4K-textured
    character: 76 labelled vertices where the opening actually spans 726, and since a lid
    is copied from the faces of that mask, the lid covered 21% of the eye -- the sawtooth
    patch of grey that showed through the middle of every blink.

    What the mask does get right is WHERE the eye is and HOW BIG it is: the samples are
    spread across the whole opening, merely sparse inside it. Those are the only two things
    asked of it here. Which vertices belong to the eye -- the question it answers badly --
    is answered instead by the ellipse those two quantities define, filled at the mesh's own
    resolution.

    The half-axes come from a percentile of the mask's extent rather than its maximum, so a
    single vertex mislabelled up on the brow cannot stretch the opening into it. The result
    is trimmed to one connected island because the ellipse is a projection and would
    otherwise also claim whatever surface lies behind the eye.
    """
    from .mesh_segmentation import connected_region

    outward, up, across = frame
    rel = vertices - centre
    h, u, w = rel @ across, rel @ up, rel @ outward

    a = float(np.percentile(np.abs(h[mask]), _ORBIT_PERCENTILE)) * _ORBIT_MARGIN
    b = float(np.percentile(np.abs(u[mask]), _ORBIT_PERCENTILE)) * _ORBIT_MARGIN
    if a <= 1e-9 or b <= 1e-9:
        return mask

    inside = (((h / a) ** 2 + (u / b) ** 2) <= 1.0) & (np.abs(w) <= _ORBIT_DEPTH * max(a, b))
    if exclude is not None:
        # The brows come out of the same parsing pass and are the one feature the opening
        # must never swallow: a lid built over the brow erases it for the whole blink.
        # This is the safety catch for a character whose brow sits closer to the eye than
        # the margin above assumes.
        inside &= ~exclude
    if not (inside & mask).any():
        return mask

    region = connected_region(inside, edges, vertices)
    if region[mask].sum() < _ORBIT_MIN_AGREEMENT * mask.sum():
        # The island the ellipse settled on is not the one the parser labelled. Nothing
        # here is trustworthy enough to build a lid on a guess, so fall back to the mask.
        return mask
    return weld_mask(region, groups=groups) if groups is not None else region


def _painted_eye(mask: np.ndarray, ring: np.ndarray,
                 appearance: Optional[np.ndarray]) -> np.ndarray:
    """
    The eye's own paint inside the opening: whatever does not look like the skin around it.

    Three different questions were being answered with one mask, and two got the wrong
    answer. The parser's sample says WHERE the eye is. The orbit ellipse says how far a lid
    must reach. But how far the iris may swing is a question about the PAINTED eye -- the
    sclera it slides over -- and neither of those is that. Taking it from the sample made
    `half_eye` and `half_iris` come out of the same sparse set of vertices, so the room
    between them collapsed: `eyeLookUp`/`eyeLookDown` were not built at all, and In/Out came
    out four times larger on one eye than on the other.

    The skin reference is the ring immediately outside the opening, so it is local. An
    earlier attempt to separate eye from skin by luminance failed because it calibrated
    against the whole head and picked up hair and brows. The threshold is the ring's own
    noise rather than a constant, since how far sclera sits from skin differs per character
    -- on this one it is 0.055 of albedo.
    """
    if appearance is None or not ring.any():
        return mask
    rgb = appearance[:, :3].astype(np.float32)
    if rgb.max() > 1.5:                       # 8-bit colour, as `sample_vertex_colors` bakes
        rgb = rgb / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # Skin and its noise come from the MIDDLE HALF of the ring. The ring is two rings of
    # vertices out, so on a close-set face it catches brow and hair, and those drag a plain
    # median-absolute-deviation up far enough to swallow the signal: measured here, the
    # whole ring gave a deviation of 0.071 where sclera sits only 0.055 from skin, so the
    # sclera was classified as skin and the eye came out as pupil alone.
    ring_lum = lum[ring]
    lo, hi = np.percentile(ring_lum, [25.0, 75.0])
    core = ring_lum[(ring_lum >= lo) & (ring_lum <= hi)]
    if len(core) < 4:
        core = ring_lum
    skin = float(np.median(core))
    mad = float(np.median(np.abs(core - skin))) * 1.4826
    threshold = max(_EYE_COLOUR_SIGMA * mad, _EYE_COLOUR_FLOOR)
    paint = mask & (np.abs(lum - skin) > threshold)
    return paint if paint.sum() >= _MIN_IRIS_VERTS else mask


def _iris_mask(mask: np.ndarray, colors: Optional[np.ndarray], edges: np.ndarray,
               vertices: Optional[np.ndarray] = None,
               centre: Optional[np.ndarray] = None) -> np.ndarray:
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

    # Keep only the connected piece sitting in the MIDDLE of the eye. The docstring
    # promised this and the code never did it, so the dark set also held the eyeliner ring
    # around the outside of the eye. That put iris vertices out at the very rim, `half_iris`
    # came out equal to `half_eye` to four decimal places, the room between them was zero,
    # and the whole eyeLook family was silently skipped -- 0.0 degrees of travel.
    #
    # The piece is chosen by POSITION, not by darkness. On a character with drawn-on
    # eyeliner that liner is darker than a brown iris, so picking the darkest component
    # selects the lash line: measured, a 14-vertex sliver instead of the 400-vertex iris.
    # An iris is the thing in the middle, which is a statement about where it is.
    if vertices is None or centre is None:
        return out

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    sel = np.nonzero(out)[0]
    inner = edges[out[edges[:, 0]] & out[edges[:, 1]]]
    if len(inner) == 0:
        return out
    loc = np.full(len(mask), -1, dtype=np.int64)
    loc[sel] = np.arange(len(sel))
    graph = coo_matrix((np.ones(len(inner), dtype=np.int8),
                        (loc[inner[:, 0]], loc[inner[:, 1]])), shape=(len(sel), len(sel)))
    n_comp, comp = connected_components(graph, directed=False)
    if n_comp <= 1:
        return out

    best, best_d = None, np.inf
    for c in range(n_comp):
        members = sel[comp == c]
        if len(members) < _MIN_IRIS_VERTS:
            continue
        d = float(np.linalg.norm(vertices[members].mean(axis=0) - centre))
        if d < best_d:
            best, best_d = members, d
    if best is None:
        return out
    pupil = np.zeros(len(mask), dtype=bool)
    pupil[best] = True
    return pupil


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


def _silhouette_profile(h: np.ndarray, u: np.ndarray, w: np.ndarray, upper: bool,
                        sel: Optional[np.ndarray] = None
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
    # The outline is traced over `sel` -- the eye itself -- and then evaluated for every
    # vertex, including the ring outside it. A lid claims every face with any vertex in the
    # eye, so its vertices spill one ring past the opening; letting that ring define the
    # outline lifts the profile at exactly the columns where the opening is closing, and the
    # corners then travel as far as the centre. Measured on the synthetic almond: corner
    # travel 0.74 of the centre's against the 0.40 the outline implies.
    hs, us, ws = (h, u, w) if sel is None else (h[sel], u[sel], w[sel])
    bins = _profile_bins(hs, us, ws, upper)
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
    rest_u, rest_w = _silhouette_profile(h, u, w, upper=upper, sel=region[src_idx])
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


def _lid_uvs(vertices: np.ndarray, uvs: np.ndarray, ring_idx: np.ndarray,
             src_idx: np.ndarray, centre: np.ndarray,
             frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
             body_uv_rate: float, appearance: Optional[np.ndarray] = None) -> np.ndarray:
    """
    UV for the lid, as a single affine map from the eye's own plane onto the atlas.

    Per-vertex nearest-neighbour, which this used to do for the UV as well as the colour,
    is NOT continuous. Two adjacent lid vertices can take their UV from two ring vertices
    sitting on opposite sides of a UV seam -- and a seam is precisely where the atlas jumps
    -- so the triangle between them sweeps clear across a 4K atlas and samples hair,
    moustache and eye into what is meant to be a patch of skin. Measured on this character:
    42% of lid edges travelled more than ten times the body's own UV-per-world rate, the
    worst 3400x, and the lid rendered as a brown smear over the closed eye.

    None of the other checks could see it. Colour baked per vertex interpolates smoothly
    however wrong the UVs are, so every image this repo renders looked like clean skin.

    An affine map cannot tear: it is continuous everywhere by construction, and where the
    surface is unwrapped evenly -- which the body is, at a UV stretch of 0.2 with a p99 of
    0.4 -- it also samples the right part of the atlas. Where the ring is itself split
    across atlas islands the fit has no single answer, and rather than return a bad one it
    falls back to one constant UV: a flat patch of skin, which is most of what an eyelid is.
    """
    _outward, up, across = frame
    rel = vertices[ring_idx] - centre
    basis = np.stack([rel @ across, rel @ up, np.ones(len(ring_idx))], axis=1)
    target = uvs[ring_idx].astype(np.float64)
    solution, *_ = np.linalg.lstsq(basis, target, rcond=None)

    # Judge the fit by the quantity that actually matters: how fast it moves across the
    # atlas per unit of surface. A residual test alone passes a map that fits a ring split
    # over several atlas islands by racing between them -- measured, that still stretched
    # the lid 31x the body's rate and painted it with the wrong part of the texture.
    residual = float(np.median(np.linalg.norm(basis @ solution - target, axis=1)))
    spread = max(float(np.ptp(target, axis=0).max()), 1e-9)
    gradient = float(np.linalg.norm(solution[:2], axis=1).max())
    if (residual <= _LID_UV_FIT_TOLERANCE * spread
            and gradient <= _LID_UV_MAX_GRADIENT * body_uv_rate):
        rel_src = vertices[src_idx] - centre
        src_basis = np.stack(
            [rel_src @ across, rel_src @ up, np.ones(len(src_idx))], axis=1)
        return (src_basis @ solution).astype(np.float32)

    # No usable map, so the whole lid takes ONE skin UV and renders flat. The vertex chosen
    # is the ring vertex whose COLOUR is the most ordinary -- an actual point of skin, and
    # the typical one. Picking by UV instead can land on a ring vertex that happens to sit
    # on the eyeliner, and picking the median UV itself can land in empty atlas.
    if appearance is not None:
        lum = (appearance[ring_idx][:, :3].astype(np.float32)
               @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        pick = ring_idx[int(np.argmin(np.abs(lum - float(np.median(lum)))))]
    else:
        median_uv = np.median(target, axis=0)
        pick = ring_idx[int(np.argmin(np.linalg.norm(target - median_uv, axis=1)))]
    return np.tile(uvs[pick], (len(src_idx), 1)).astype(np.float32)


def _skin_colors(
    vertices: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    src_idx: np.ndarray,
    colors: Optional[np.ndarray],
    uvs: Optional[np.ndarray],
    groups: Optional[np.ndarray] = None,
    centre: Optional[np.ndarray] = None,
    frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    body_uv_rate: float = 0.0,
    appearance: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Colour and UV for the lid, taken from the ring of surface just outside the eye.

    The COLOUR is per-vertex nearest-neighbour, so a lid on a two-toned face picks up the
    shading it sits in; a vertex colour is interpolated directly, so a discontinuity in it
    is at worst a visible edge. The UV cannot be treated the same way -- see `_lid_uvs`.
    """
    if colors is None and uvs is None:
        return None, None

    ring = _dilate(_dilate(mask, edges, groups), edges, groups) & ~mask
    ring_idx = np.nonzero(ring)[0]
    if len(ring_idx) == 0:
        return (None if colors is None else colors[src_idx].copy(),
                None if uvs is None else uvs[src_idx].copy())

    _, nearest = cKDTree(vertices[ring_idx]).query(vertices[src_idx], k=1)
    picked = ring_idx[nearest]
    lid_uvs = None
    if uvs is not None:
        lid_uvs = (_lid_uvs(vertices, uvs, ring_idx, src_idx, centre, frame,
                            body_uv_rate, appearance)
                   if centre is not None and frame is not None else uvs[picked].copy())
    return (None if colors is None else colors[picked].copy(), lid_uvs)


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
    # One id per position. Every vertex mask that ends up driving a delta is welded with
    # it, so a UV seam cannot hand the two copies of a point different displacements.
    groups = weld_groups(vertices)
    tri = faces if len(faces) <= 20000 else faces[np.linspace(0, len(faces) - 1, 20000).astype(np.int64)]
    edge_length = float(np.median(np.concatenate([
        np.linalg.norm(vertices[tri[:, 0]] - vertices[tri[:, 1]], axis=1),
        np.linalg.norm(vertices[tri[:, 1]] - vertices[tri[:, 2]], axis=1),
    ])))

    # How fast the character's own unwrap moves across the atlas, so a lid's UV map can be
    # judged against it rather than against an absolute number that depends on the unwrap.
    body_uv_rate = 0.0
    if uvs is not None and len(uvs) == n:
        _d3 = np.linalg.norm(vertices[tri[:, 0]] - vertices[tri[:, 1]], axis=1)
        _d2 = np.linalg.norm(uvs[tri[:, 0]] - uvs[tri[:, 1]], axis=1)
        _ok = _d3 > 1e-12
        if _ok.any():
            body_uv_rate = float(np.median(_d2[_ok] / _d3[_ok]))

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
    eye_masks: Dict[str, np.ndarray] = {}
    brows = None
    for _b in (eye_regions.left_brow_mask, eye_regions.right_brow_mask):
        if _b is not None:
            brows = _b.copy() if brows is None else (brows | _b)

    for side, sample in (("Left", eye_regions.left_mask), ("Right", eye_regions.right_mask)):
        seed = np.nonzero(sample)[0]
        if len(seed) < _MIN_LID_VERTS:
            print(f"[EyelidPatch] {side} eye has too few vertices to copy a lid from.")
            continue

        # The parser's sample gives the eye's centre and its axes -- the two quantities it
        # is reliable about -- and those define the ellipse that gives the opening itself.
        # Everything below is built on the opening, never on the sample.
        mask = _orbit_region(
            vertices, edges, sample,
            vertices[seed].mean(axis=0).astype(np.float32),
            eye_frame(vertices[seed], vertex_normals[seed], eye_regions.forward),
            groups, brows)
        eye_masks[side] = mask
        idx = np.nonzero(mask)[0]
        print(f"[EyelidPatch] {side} eye: parser sampled {len(seed)} vertices, "
              f"orbit ellipse covers {len(idx)}.")

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

        # The ring of skin just outside the opening, taken here rather than further down
        # because it is what calibrates skin colour for `_painted_eye`.
        ring = _dilate(_dilate(mask, edges, groups), edges, groups) & ~mask
        ring_idx = np.nonzero(ring)[0]
        paint = _painted_eye(mask, ring, appearance)

        # --- the iris, lifted off the face so it can rotate instead of being stretched ---
        # Read off the PAINTED eye, not off the ellipse and not off the parser's sample.
        # The ellipse deliberately reaches onto plain skin, so splitting that on luminance
        # puts the darkest skin around the eye into the cap; the sample is too sparse to
        # leave any room between the iris and the eye that bounds its travel.
        iris = _iris_mask(paint, appearance, edges, vertices, centre)

        # How wide the eye opening is: the orbit ellipse's own extent.
        #
        # Not the parser's sample and not a colour segmentation, because neither measures
        # the opening. The sample is a sparse subset of it -- 76 vertices where the eye
        # spans 780 -- and colour cannot find its edge at all on this character: sclera sits
        # 0.055 of albedo from skin, and the painted region recovered by luminance came back
        # the same size as the sample. The ellipse is the only estimate that was checked
        # against what a viewer sees, by sweeping its margin against how much eye survives a
        # blink, so it is the one used here too.
        opening_rel = vertices[idx] - centre
        half_open = {"up": float(np.abs(opening_rel @ up).max()),
                     "across": float(np.abs(opening_rel @ across_axis).max())}

        # An iris segmented out of colour is only believable when it comes back the size an
        # iris is. On this character it did not: the eyeliner is darker than the brown iris,
        # so the dark region is a ring plus a disc, and depending on which of them was
        # picked the "iris" was either the whole eye -- leaving no room to move, so the
        # entire eyeLook family was skipped -- or a 14-vertex sliver of lash line. Where the
        # measurement is not credible, fall back to the anatomical proportion: an iris is
        # about 12mm across an opening of about 30mm.
        iris_rel = vertices[np.nonzero(iris)[0]] - centre if iris.any() else None
        credible = iris_rel is not None and all(
            _IRIS_MIN_SHARE * half_open[k] <= float(np.abs(iris_rel @ ax).max())
            <= _IRIS_MAX_SHARE * half_open[k]
            for k, ax in (("up", up), ("across", across_axis)))
        if credible:
            # ...and it has to be a disc, not a thread. Extent alone passes a sliver of
            # lash line running the width of the eye: measured, 14 vertices where a disc of
            # that reach holds several hundred.
            reach = max(float(np.abs(iris_rel @ up).max()),
                        float(np.abs(iris_rel @ across_axis).max()))
            span = max(half_open["up"], half_open["across"], 1e-9)
            expected = mask.sum() * (reach / span) ** 2
            credible = iris.sum() >= _IRIS_MIN_FILL * max(expected, 1.0)
        if not credible:
            plane = np.sqrt(((vertices - centre) @ across_axis) ** 2
                            + ((vertices - centre) @ up) ** 2)
            iris = mask & (plane <= _IRIS_DISC_SHARE * max(half_open["up"],
                                                           half_open["across"]))
            print(f"[EyelidPatch] {side} eye: iris segmentation not credible, using a disc "
                  f"of {int(iris.sum())} vertices.")
        iris_faces = faces[iris[faces].sum(axis=1) == 3]
        if len(iris_faces) >= _MIN_LID_FACES and appearance is not None:
            iris_idx = np.unique(iris_faces.ravel())
            # The eyeball centre the iris swings about: back along the surface normal by the
            # fitted radius. Every gaze direction is a rotation about this one point, which
            # is what keeps the iris a circle no matter where it is looking.
            pivot = (centre - outward * radius).astype(np.float32)
            iris_rel = vertices[iris_idx] - centre
            # How much room the iris has is set by the EYE OPENING the character was
            # painted with -- the parser's sample -- not by the orbit ellipse. The ellipse
            # reaches out onto plain skin on purpose, and measuring the room against it
            # lets the gaze swing the iris clean off the eye and onto the cheek.
            half_eye = half_open
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
                paint_idx = np.nonzero(paint)[0]
                eye_rgb = colors[paint_idx][:, :3].astype(np.float32)
                luma = eye_rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
                sclera = colors[paint_idx][
                    np.argsort(luma)[-max(1, len(paint_idx) // 8):]].mean(axis=0)
                sclera_repaint.append((iris_idx, sclera.astype(colors.dtype)))

        # --- AU5: widen the opening by moving the skin around it, never the eye itself ---
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
            lid_colors, lid_uvs = _skin_colors(vertices, edges, mask, src_idx, colors, uvs, groups,
                                               centre, frame, body_uv_rate, appearance)
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

    # Every painted eye on the face. AU5 moves the skin AROUND the opening, so it has to
    # avoid all of them, not merely the one whose ring it was built from -- on a face whose
    # eyes sit close together the ring of one reaches into the other.
    painted = np.zeros(n, dtype=bool)
    for _m in eye_masks.values():
        painted |= _m

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
        padded[:n][painted] = 0.0
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
            # The room is taken with a margin because both radii are measured along the
            # eye's own axes while the cap travels on a sphere: at a diagonal the cap's
            # corner reaches further than either axis says, and without the margin it
            # overshot the region the lids cover by 13%.
            room = max(iris["half_eye"][axis_key] - iris["half_iris"][axis_key], 0.0)
            room *= _GAZE_ROOM_SAFETY
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
    protected[:n] = painted if eye_masks else (
        eye_regions.left_mask | eye_regions.right_mask)

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
