"""
Two-stage template fitting: robust rigid alignment followed by non-rigid ICP.

This is the fitting structure `flame-fitting/fit_lmk3d.py` uses -- step 1 solves only
the global rotation and translation, step 2 opens up the deformation parameters under a
regularizer -- applied to the ARKit blendshape template instead of to FLAME. FLAME itself
is not usable here: it is a model of real human heads, and this pipeline has to rig chibi
characters with enormous eyes, giraffes and birds.

FLAME gets its smoothness from confining the result to a PCA space with the coefficients
penalised towards the mean. There is no PCA basis for an arbitrary character head, so the
regulariser here is the stiffness term of Amberg et al., "Optimal Step Nonrigid ICP
Algorithms for Surface Registration" (CVPR 2007): neighbouring vertices are pushed towards
sharing one affine transform, and the weight on that term is annealed down across the
iterations the same way `fit_scan.py` anneals its objective weights.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


@dataclass
class NonrigidFit:
    """Result of `nonrigid_icp`.

    `vertices` is the wrapped template. `transforms` holds the per-vertex affine linear
    part -- the `A` of `v' = A v + t` -- which is what a displacement defined on the
    original template must be multiplied by to end up pointing the right way on the
    wrapped one.
    """
    vertices: np.ndarray      # (n, 3) deformed template vertices
    transforms: np.ndarray    # (n, 3, 3) per-vertex linear transform
    landmark_rms: float       # RMS landmark residual after fitting
    stiffness: float          # final stiffness weight reached


def robust_similarity(src_pts: np.ndarray, tgt_pts: np.ndarray,
                      iterations: int = 5) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Umeyama similarity transform (R, s, t) minimising sum ||s R src_i + t - tgt_i||^2,
    re-weighted across `iterations` passes by the Geman-McClure function that
    `flame-fitting/sbody/robustifiers.py` wraps its scan-to-mesh term in:
    `w_i = sigma^2 / (sigma^2 + r_i^2)`.

    A landmark detector that puts one point in the wrong place -- a nostril landed on a
    beak, an eye corner on a horn -- otherwise drags the whole alignment towards it,
    because plain least squares gives that one residual a weight proportional to its
    square.
    """
    assert src_pts.shape == tgt_pts.shape and len(src_pts) >= 3
    weights = np.ones(len(src_pts), dtype=np.float64)

    for _ in range(max(1, iterations)):
        w = weights / weights.sum()
        src_mean = (src_pts * w[:, None]).sum(axis=0)
        tgt_mean = (tgt_pts * w[:, None]).sum(axis=0)
        src_c = src_pts - src_mean
        tgt_c = tgt_pts - tgt_mean

        cov = (tgt_c * w[:, None]).T @ src_c
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt

        var_src = float((w[:, None] * src_c ** 2).sum())
        scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
        t = tgt_mean - scale * (R @ src_mean)

        residual = np.linalg.norm(scale * (src_pts @ R.T) + t - tgt_pts, axis=1)
        # Scale-free sigma: the median residual is a robust estimate of the spread the
        # inliers actually have, so the cutoff adapts to how big the head is.
        sigma = max(float(np.median(residual)), 1e-9)
        weights = sigma ** 2 / (sigma ** 2 + residual ** 2)

    return R.astype(np.float32), float(scale), t.astype(np.float32)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                  vertices[faces[:, 2]] - vertices[faces[:, 0]])
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norms, 1e-12)


def _deform(vertices: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Applies the per-vertex affine stack `X` (n, 4, 3) to `vertices` (n, 3)."""
    homo = np.hstack([vertices, np.ones((len(vertices), 1), dtype=vertices.dtype)])
    return np.einsum('nj,njk->nk', homo, X)


def nonrigid_icp(
    src_vertices: np.ndarray,
    src_faces: np.ndarray,
    tgt_vertices: np.ndarray,
    tgt_faces: np.ndarray,
    src_lmk_face_idx: Optional[np.ndarray] = None,
    src_lmk_bary: Optional[np.ndarray] = None,
    tgt_lmk_points: Optional[np.ndarray] = None,
    stiffness_schedule: Tuple[float, ...] = (50.0, 20.0, 8.0, 3.0, 1.0, 0.5),
    inner_iterations: int = 2,
    max_correspondence_ratio: float = 0.15,
    normal_threshold: float = 0.5,
) -> NonrigidFit:
    """
    Wraps `src` onto `tgt` by solving for one affine transform per source vertex.

    Minimises  E = E_data + alpha * E_stiff + beta * E_lmk  as a single sparse linear
    least-squares problem per iteration, with `alpha` annealed down the schedule. The
    unknown is X in R^{4n x 3}, stacking a 4x3 affine per vertex, so a deformed vertex is
    `[x, y, z, 1] @ X_i`.

    Args:
        src_lmk_face_idx, src_lmk_bary: landmark embedding on the source -- a triangle
            index and its barycentric weights per landmark, the same representation
            `flame-fitting` stores in `flame_static_embedding.pkl`. Sub-vertex accurate
            and independent of source tessellation.
        tgt_lmk_points: (L, 3) matching landmark positions on the target.
        max_correspondence_ratio: correspondences longer than this fraction of the target
            bounding-box diagonal are dropped rather than pulled on.
        normal_threshold: minimum normal agreement (cosine) for a correspondence to count,
            which is what stops the template's front face from being pulled onto the back
            of the head through a thin region.
    """
    n = len(src_vertices)
    src_v = np.asarray(src_vertices, dtype=np.float64)
    tgt_v = np.asarray(tgt_vertices, dtype=np.float64)

    edges = _unique_edges(src_faces)
    m = len(edges)

    # Node-arc incidence matrix; kron with G = diag(1,1,1,gamma) gives the stiffness
    # block. gamma trades rotation against translation differences between neighbours;
    # Amberg's paper uses 1 for data in the same units as the geometry.
    rows = np.repeat(np.arange(m), 2)
    cols = edges.reshape(-1)
    vals = np.tile([-1.0, 1.0], m)
    M = sp.coo_matrix((vals, (rows, cols)), shape=(m, n))
    G = sp.diags([1.0, 1.0, 1.0, 1.0])
    MG = sp.kron(M, G).tocsr()          # (4m, 4n)
    stiff_rhs = np.zeros((4 * m, 3))

    # Landmark term: a landmark is a barycentric point, so its deformed position is a
    # weighted sum of three vertices' affines -- still linear in X.
    lmk_block = None
    lmk_rhs = None
    if (src_lmk_face_idx is not None and src_lmk_bary is not None
            and tgt_lmk_points is not None and len(src_lmk_face_idx) > 0):
        tri = src_faces[src_lmk_face_idx]                  # (L, 3)
        L = len(tri)
        homo = np.hstack([src_v, np.ones((n, 1))])         # (n, 4)
        r_idx, c_idx, v_idx = [], [], []
        for k in range(3):
            vert = tri[:, k]                                # (L,)
            b = src_lmk_bary[:, k]                           # (L,)
            for comp in range(4):
                r_idx.append(np.arange(L))
                c_idx.append(vert * 4 + comp)
                v_idx.append(b * homo[vert, comp])
        lmk_block = sp.coo_matrix(
            (np.concatenate(v_idx), (np.concatenate(r_idx), np.concatenate(c_idx))),
            shape=(L, 4 * n)).tocsr()
        lmk_rhs = np.asarray(tgt_lmk_points, dtype=np.float64)

    # X starts as the identity affine at every vertex.
    X = np.tile(np.vstack([np.eye(3), np.zeros((1, 3))]), (n, 1, 1))

    tgt_tree = cKDTree(tgt_v)
    tgt_normals = _vertex_normals(tgt_v, tgt_faces)
    diag = float(np.linalg.norm(tgt_v.max(axis=0) - tgt_v.min(axis=0)))
    max_dist = diag * max_correspondence_ratio

    # The landmark term guides the fit while the surface is still far away and then gets
    # out of the way, mirroring how `fit_scan.py` trades its landmark weight against the
    # scan-to-mesh term as the fit tightens.
    n_steps = len(stiffness_schedule)
    stiffness = stiffness_schedule[-1]

    for step, alpha in enumerate(stiffness_schedule):
        beta = 10.0 * (1.0 - step / max(n_steps - 1, 1)) + 1.0
        stiffness = alpha
        for _ in range(inner_iterations):
            deformed = _deform(src_v, X)
            dist, idx = tgt_tree.query(deformed, k=1)
            U = tgt_v[idx]

            src_normals = _vertex_normals(deformed, src_faces)
            agree = np.einsum('ij,ij->i', src_normals, tgt_normals[idx])
            w = ((dist <= max_dist) & (agree >= normal_threshold)).astype(np.float64)
            if w.sum() < 4:
                # Nothing on the target is close enough or facing the right way; the
                # stiffness term alone would collapse the template, so stop here.
                break

            homo = np.hstack([src_v, np.ones((n, 1))])
            data_rows = np.repeat(np.arange(n), 4)
            data_cols = (np.arange(n)[:, None] * 4 + np.arange(4)[None, :]).reshape(-1)
            data_vals = (homo * w[:, None]).reshape(-1)
            D = sp.coo_matrix((data_vals, (data_rows, data_cols)), shape=(n, 4 * n)).tocsr()

            blocks = [alpha * MG, D]
            rhs = [stiff_rhs, U * w[:, None]]
            if lmk_block is not None:
                blocks.append(beta * lmk_block)
                rhs.append(beta * lmk_rhs)

            A = sp.vstack(blocks).tocsr()
            B = np.vstack(rhs)

            AtA = (A.T @ A).tocsc()
            # Tikhonov floor: a template vertex with no edge and no correspondence would
            # otherwise leave its 4 columns empty and make the system singular.
            AtA = AtA + sp.identity(4 * n, format='csc') * 1e-8
            AtB = A.T @ B
            X = splu(AtA).solve(AtB).reshape(n, 4, 3)

    deformed = _deform(src_v, X)
    transforms = np.transpose(X[:, :3, :], (0, 2, 1))  # X_i[:3] holds A^T; store A

    lmk_rms = float('nan')
    if lmk_block is not None:
        fitted = lmk_block @ X.reshape(4 * n, 3)
        lmk_rms = float(np.sqrt(((fitted - lmk_rhs) ** 2).sum(axis=1).mean()))

    return NonrigidFit(
        vertices=deformed.astype(np.float32),
        transforms=transforms.astype(np.float32),
        landmark_rms=lmk_rms,
        stiffness=stiffness,
    )
