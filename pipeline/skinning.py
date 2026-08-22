import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Tuple, Union

def point_to_segment_distance(points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    """
    Computes shortest Euclidean distance from each point in points (N, 3)
    to a 3D line segment defined by seg_start (3,) and seg_end (3,).
    """
    v = seg_end - seg_start
    seg_len_sq = np.dot(v, v)
    
    if seg_len_sq < 1e-10:
        # Segment is a single point
        return np.linalg.norm(points - seg_start, axis=1)
        
    # Project points onto segment line: t = dot(points - seg_start, v) / |v|^2
    diff = points - seg_start
    t = np.clip(np.sum(diff * v, axis=1) / seg_len_sq, 0.0, 1.0)
    
    # Closest point on segment: seg_start + t[:, None] * v
    projection = seg_start + t[:, np.newaxis] * v
    return np.linalg.norm(points - projection, axis=1)

def compute_skin_weights_geometric(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: List[Optional[int]],
    power: float = 2.5,
    smooth_iters: int = 5,
    smooth_alpha: float = 0.5,
    top_k: int = 4
) -> np.ndarray:
    """
    Computes high-quality, smooth skinning weights using bone-segment distance field
    combined with geodesic Laplacian surface smoothing.
    
    Args:
        vertices: (N, 3) float32
        faces: (F, 3) uint32
        joints: (J, 3) float32
        parents: List of length J, with parent index or None
        power: Distance decay exponent
        smooth_iters: Number of Laplacian diffusion iterations
        smooth_alpha: Laplacian smoothing weight
        top_k: Number of maximum influencing bones per vertex (default 4)
        
    Returns:
        weights: (N, J) float32 skinning weight matrix normalized so sum(weights[i]) == 1.0
    """
    N = len(vertices)
    J = len(joints)
    
    if J == 1:
        return np.ones((N, 1), dtype=np.float32)
        
    # 1. Define bone segments
    # In standard skeletal animation, Joint j controls the limb from joints[j] to its child joints.
    # If j has multiple children, it controls all outward branches.
    # If j is a leaf (no children), it controls the tip extending from parent.
    children = {i: [] for i in range(J)}
    for i in range(J):
        p = parents[i]
        if p is not None and p >= 0:
            children[p].append(i)
            
    # 2. Compute distance matrix D (N, J)
    D = np.zeros((N, J), dtype=np.float32)
    for j in range(J):
        c_list = children[j]
        j_pos = joints[j]
        
        if len(c_list) > 0:
            d_min = np.full(N, np.inf, dtype=np.float32)
            for c in c_list:
                d_c = point_to_segment_distance(vertices, j_pos, joints[c])
                d_min = np.minimum(d_min, d_c)
            D[:, j] = d_min
        else:
            # Leaf joint: extend outward from parent direction
            p = parents[j]
            if p is not None and p >= 0:
                dir_vec = j_pos - joints[p]
                seg_end = j_pos + dir_vec * 0.5
                D[:, j] = point_to_segment_distance(vertices, j_pos, seg_end)
            else:
                D[:, j] = np.linalg.norm(vertices - j_pos, axis=1)
        
    # Scale distances relative to mesh bounding box
    bbox_size = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)) + 1e-6
    D_norm = D / bbox_size
    
    # 3. Compute initial weights using inverse distance
    eps = 1e-3
    raw_weights = 1.0 / (D_norm + eps) ** power
    
    # Normalize initial weights
    row_sums = raw_weights.sum(axis=1, keepdims=True)
    weights = raw_weights / (row_sums + 1e-9)
    
    # 4. Laplacian Surface Smoothing over mesh graph
    if smooth_iters > 0 and len(faces) > 0:
        # Construct sparse adjacency matrix from faces
        i_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
        j_idx = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
        
        # Edge weights (distance inverse)
        diff = vertices[i_idx] - vertices[j_idx]
        edge_dist = np.linalg.norm(diff, axis=1) + 1e-6
        edge_w = 1.0 / edge_dist
        
        adj = sp.coo_matrix((edge_w, (i_idx, j_idx)), shape=(N, N)).tocsr()
        
        # Degree normalization
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        inv_deg = 1.0 / deg
        
        # Iterative diffusion
        for _ in range(smooth_iters):
            smoothed = adj.dot(weights) * inv_deg[:, np.newaxis]
            weights = (1.0 - smooth_alpha) * weights + smooth_alpha * smoothed
            
    # 5. Sparsify to top_k bones per vertex and renormalize
    final_weights = np.zeros_like(weights)
    for i in range(N):
        row = weights[i]
        if J <= top_k:
            final_weights[i] = row
        else:
            top_indices = np.argpartition(-row, top_k)[:top_k]
            final_weights[i, top_indices] = row[top_indices]
            
        s = np.sum(final_weights[i])
        if s > 1e-6:
            final_weights[i] /= s
        else:
            final_weights[i, np.argmin(D[i])] = 1.0
            
    return final_weights.astype(np.float32)

def predict_skin_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: List[Optional[int]],
    method: str = "auto",
    **kwargs
) -> np.ndarray:
    """
    Main skinning entrypoint. Automatically computes skin weights for rigged mesh.
    """
    return compute_skin_weights_geometric(
        vertices=vertices,
        faces=faces,
        joints=joints,
        parents=parents,
        **kwargs
    )
