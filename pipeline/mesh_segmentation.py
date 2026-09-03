"""
Head-region detection via the QtMeshEditor mesh-segmentation ONNX model
(fernandotonon/QtMeshEditor-mesh-segmentation, CC-BY-4.0): a PointNet++-style network
trained on humanoid, chibi/cartoon, quadruped, and biped-with-tail body plans that
labels each point of a sampled point cloud as one of unknown/head/torso/left_arm/
right_arm/left_leg/right_leg.

This is the only head detector in the pipeline. The heuristics it replaced -- a sphere
around the predicted skeleton "head" joint, and a top-N%-of-Y-height cut -- both failed
on the characters this pipeline actually produces:

* Skeleton-joint sphere: its radius was clamped to an absolute [0.08, 0.16] range while
  meshes are only oriented and grounded, never rescaled. On a chibi character whose head
  is 38% of body height that clamp is off by 2.4x.
* Top-N%-of-Y: assumes the head is the topmost thing on an upright biped. A quadruped
  carries its head at the end of a horizontal neck, and hats/horns/ears sit above it.

The network needs no skeleton and no per-character tuning, so neither failure mode has
an analogue here. Every threshold below is a fraction of a measured quantity (bounding
box diagonal, head radius) -- never an absolute distance.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

LABELS = ["unknown", "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
HEAD_LABEL = LABELS.index("head")

# A head made of fewer vertices than this is noise, not a detection.
_MIN_HEAD_VERTICES = 32
# A "head" spanning more than this fraction of the whole mesh means the network
# collapsed the character into a single part; reject rather than deform everything.
_MAX_HEAD_SPAN = 0.60
# How close another island must be to the main one, as a fraction of the main island's
# radius, to be treated as part of the same body part.
_MERGE_GAP = 0.35
# Rings of mesh neighbours added after component selection, to close the pinholes the
# point-cloud-to-vertex scatter leaves inside an otherwise solid head.
_DILATE_RINGS = 2

_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "mesh_segmentation" / "meshseg.onnx"
_session = None


@dataclass
class HeadRegion:
    """Where the head is, and how big it is, in the mesh's own coordinate space."""
    center: np.ndarray   # (3,) float32, robust centre of the head vertices
    radius: float        # covers ~98% of the head vertices
    mask: np.ndarray     # (N,) bool over the full vertex array

    @property
    def n_vertices(self) -> int:
        return int(self.mask.sum())


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(_MODEL_PATH))
    return _session


def weld_groups(vertices: np.ndarray, tol_ratio: float = 1e-5) -> np.ndarray:
    """
    Group id per vertex, where vertices sharing a position share an id.

    glTF meshes duplicate vertices along every UV and normal seam, so a mesh that is one
    closed surface geometrically can have thousands of index-level islands. Any graph
    walk over raw indices has to stitch those back together first or it will mistake a
    seam for the edge of the model.
    """
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tol = max(scale * tol_ratio, 1e-9)
    quantized = np.round(np.asarray(vertices, dtype=np.float64) / tol).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.ravel()


def weld_mask(mask: np.ndarray, vertices: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> np.ndarray:
    """
    A vertex mask made to agree across coincident vertices: if any copy of a position is
    in the mask, every copy is.

    A mask that comes from a rendered view, a nearest-neighbour lookup or a one-ring walk
    routinely takes one side of a UV seam and not the other, because those copies share no
    edge to carry it across. Whatever drives geometry off the mask afterwards -- freezing a
    region, feathering around it -- then treats the two halves of a single point
    differently, and the surface opens along the seam. Measured on a 4K-textured character:
    an eye mask split over 19 welded groups tore the brow open by 0.73 of a median edge.
    """
    if groups is None:
        groups = weld_groups(vertices)
    hit = np.bincount(groups, weights=np.asarray(mask, dtype=np.float64),
                      minlength=int(groups.max()) + 1) > 0
    return hit[groups]


def mesh_edges(faces: np.ndarray, vertices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    (E, 2) array of undirected connections between vertex indices.

    Triangle edges always; plus, when `vertices` is given, a seam edge from every vertex
    to the first vertex sharing its position, so the graph follows the actual surface
    rather than the index buffer's seams.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    if vertices is None:
        return edges

    groups = weld_groups(vertices)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_groups[1:] != sorted_groups[:-1])))
    representative = np.empty(int(groups.max()) + 1, dtype=np.int64)
    representative[sorted_groups[starts]] = order[starts]
    seams = np.stack([np.arange(len(vertices), dtype=np.int64), representative[groups]], axis=1)

    return np.concatenate([edges, seams], axis=0)


def connected_region(mask: np.ndarray, edges: np.ndarray, vertices: np.ndarray,
                     merge_gap: float = _MERGE_GAP) -> np.ndarray:
    """
    Trims `mask` down to one coherent piece of surface: its largest island, plus every
    other island lying within `merge_gap` (as a fraction of the largest island's radius)
    of it.

    Two separate failures make both halves of that rule necessary, and both were measured
    on generated meshes rather than imagined:

    * Stray vertices mislabelled "head" out on a shoulder used to drag the centre off the
      head and inflate the radius to whole-body scale. Islands far from the main one are
      dropped.
    * A generated head is often not one watertight shell. On a measured chibi character
      the head arrived as two touching halves of 21198 and 13662 vertices; keeping only
      the largest lost half the face and shifted the centre by a third of a head radius.
      Islands that touch the main one are kept.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return mask

    remap = np.full(len(mask), -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))

    inner = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    if len(inner) == 0:
        return mask
    a, b = remap[inner[:, 0]], remap[inner[:, 1]]

    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(len(idx), len(idx)))
    n_comp, comp = connected_components(graph, directed=False)
    if n_comp <= 1:
        return mask

    sizes = np.bincount(comp)
    main = int(sizes.argmax())
    main_points = vertices[idx[comp == main]]
    main_median = np.median(main_points, axis=0)
    main_radius = float(np.percentile(np.linalg.norm(main_points - main_median, axis=1), 98))
    if main_radius < 1e-9:
        keep = comp == main
    else:
        tree = cKDTree(main_points)
        keep = comp == main
        for c in range(n_comp):
            if c == main:
                continue
            members = comp == c
            gap = float(tree.query(vertices[idx[members]], k=1)[0].min())
            if gap <= merge_gap * main_radius:
                keep |= members

    out = np.zeros(len(mask), dtype=bool)
    out[idx[keep]] = True
    return out


def _dilate(mask: np.ndarray, edges: np.ndarray, rings: int) -> np.ndarray:
    """Grows `mask` by `rings` edge hops."""
    out = mask.copy()
    for _ in range(rings):
        grown = out.copy()
        grown[edges[out[edges[:, 0]], 1]] = True
        grown[edges[out[edges[:, 1]], 0]] = True
        out = grown
    return out


def segment_vertices(vertices: np.ndarray, faces: np.ndarray, n_points: int = 4096) -> Optional[np.ndarray]:
    """
    Labels every mesh vertex with a body part (see LABELS).

    Samples `n_points` from the mesh surface (matching the model's training
    distribution), runs the ONNX segmentation network, then scatters each sampled
    point's predicted label back to the nearest mesh vertex.

    Returns:
        (N,) int array of label indices into LABELS, or None if segmentation failed
        (e.g. the ONNX runtime/model is unavailable).
    """
    if len(vertices) == 0 or len(faces) == 0:
        return None
    try:
        session = _get_session()
    except Exception as e:
        print(f"[MeshSegmentation] ONNX model unavailable: {e}")
        return None

    try:
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        scale = float((bbox_max - bbox_min).max())
        if scale < 1e-6:
            return None

        # Fixed seed: downstream code calls this repeatedly on the same mesh and needs a
        # stable head_center/radius, not one that drifts between calls with the sampling.
        sampled_points, _ = tmesh.sample(n_points, return_index=True, seed=0)
        sampled_points = sampled_points.astype(np.float32)
        norm_points = (sampled_points - center) / scale

        logits = session.run(None, {"points": norm_points[None, :, :].astype(np.float32)})[0]  # (1, N, 7)
        point_labels = logits[0].argmax(axis=1)  # (N,)

        tree = cKDTree(sampled_points)
        _, nearest_idx = tree.query(vertices, k=1)
        return point_labels[nearest_idx]
    except Exception as e:
        print(f"[MeshSegmentation] Segmentation failed: {e}")
        return None


def detect_head_region(vertices: np.ndarray, faces: np.ndarray) -> Optional[HeadRegion]:
    """
    Locates the head on an arbitrary character mesh.

    Returns None when the head cannot be identified with confidence -- the caller is
    expected to skip facial work rather than deform a guessed region.
    """
    if len(vertices) == 0 or len(faces) == 0:
        return None

    labels = segment_vertices(vertices, faces)
    if labels is None:
        print("[MeshSegmentation] No head region: segmentation unavailable.")
        return None

    mask = labels == HEAD_LABEL
    if mask.sum() < _MIN_HEAD_VERTICES:
        print(f"[MeshSegmentation] No head region: only {int(mask.sum())} head-labelled vertices.")
        return None

    edges = mesh_edges(faces, vertices)
    mask = connected_region(mask, edges, vertices)
    mask = _dilate(mask, edges, _DILATE_RINGS)
    if mask.sum() < _MIN_HEAD_VERTICES:
        print(f"[MeshSegmentation] No head region: largest head island is only {int(mask.sum())} vertices.")
        return None

    head_verts = vertices[mask]
    # Median rather than mean: a lopsided feature (a long snout, one big ear) should not
    # pull the centre off the skull.
    center = np.median(head_verts, axis=0).astype(np.float32)
    dists = np.linalg.norm(head_verts - center, axis=1)
    radius = float(np.percentile(dists, 98) * 1.05)
    if radius < 1e-6:
        return None

    bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if radius > _MAX_HEAD_SPAN * bbox_diagonal:
        print(f"[MeshSegmentation] No head region: head radius {radius:.3f} spans "
              f"{radius / max(bbox_diagonal, 1e-9):.0%} of the mesh; segmentation is unreliable here.")
        return None

    # The network's per-vertex labels are reliable enough to LOCATE the head but not to
    # delineate it. Measured on a chibi character, the "head" label covered the ear tufts
    # and scattered patches of the cranium while the entire face came back as "torso";
    # on a quadruped it covered the top of the skull but not the snout or jaw. Using that
    # raw mask as the deformation region would leave the face out of the facial
    # blendshapes entirely.
    #
    # So: take the centre and radius from the labels (those are robust -- they are
    # aggregates over tens of thousands of vertices), and take the region itself as the
    # ball they describe, unioned with the labelled vertices so anything tall and thin
    # like an ear tuft is not clipped off. Unlike the fixed [0.08, 0.16] clamp this
    # replaces, the ball is sized from the character in front of us.
    ball = np.linalg.norm(vertices - center, axis=1) <= radius
    mask = mask | ball

    print(f"[MeshSegmentation] Head region: {int(mask.sum())} vertices "
          f"({int(ball.sum())} within the ball), center={np.round(center, 4).tolist()}, "
          f"radius={radius:.4f} ({radius / max(bbox_diagonal, 1e-9):.0%} of bbox diagonal).")
    return HeadRegion(center=center, radius=radius, mask=mask)
