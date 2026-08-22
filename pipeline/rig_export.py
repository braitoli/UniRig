import numpy as np
import struct
import json
import trimesh
from typing import List, Dict, Tuple, Optional, Union

def build_inverse_bind_matrices(world_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Computes Inverse Bind Matrices (IBM) for each joint given its 4x4 world transform matrix.
    IBM_i = inv(world_matrix_i)
    glTF expects column-major 4x4 matrices in float32.
    """
    ibms = []
    for wm in world_matrices:
        try:
            inv_wm = np.linalg.inv(wm)
        except np.linalg.LinAlgError:
            inv_wm = np.eye(4, dtype=np.float32)
        # Convert to float32 column-major (Fortran order in memory for binary buffer)
        ibms.append(inv_wm.astype(np.float32).T)
    return np.stack(ibms, axis=0) # shape (J, 4, 4)

def calculate_joint_transforms(joints: np.ndarray, parents: List[Optional[int]]):
    """
    Given joint positions (J, 3) in world space and parents list (where parent[i] is index or None),
    compute local translation vectors and world matrices for each joint.
    """
    J = len(joints)
    local_translations = []
    world_matrices = []
    
    for i in range(J):
        p = parents[i]
        pos = joints[i].astype(np.float32)
        if p is None or p < 0:
            local_trans = pos
        else:
            p_pos = joints[p].astype(np.float32)
            local_trans = pos - p_pos
        local_translations.append(local_trans)
        
        # World matrix: translation only (rotation identity in rest pose)
        wm = np.eye(4, dtype=np.float32)
        wm[0:3, 3] = pos
        world_matrices.append(wm)
        
    return local_translations, world_matrices

def create_rigged_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    parents: List[Optional[int]],
    skin_weights: np.ndarray,
    normals: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None,
    animations: Optional[Dict[str, Dict]] = None,
    output_path: Optional[str] = None
) -> bytes:
    """
    Creates a standard, valid binary glTF 2.0 (.glb) file with skinned mesh, bone hierarchy,
    and optional animations without any external heavy dependencies like Blender.
    
    Args:
        vertices: (N, 3) float32
        faces: (F, 3) uint32
        joints: (J, 3) float32
        parents: List of length J, with parent index or None / -1 for root
        skin_weights: (N, J) float32
        normals: (N, 3) float32 (optional)
        uvs: (N, 2) float32 (optional)
        joint_names: List of strings (optional)
        animations: Dict of animation name -> tracks (optional)
        output_path: filepath to save .glb
    """
    N = len(vertices)
    F = len(faces)
    J = len(joints)
    
    if joint_names is None:
        joint_names = [f"Bone_{i:03d}" for i in range(J)]
        
    if normals is None:
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normals = tm.vertex_normals.astype(np.float32)
        
    # Top-4 bone weighting for glTF JOINTS_0 and WEIGHTS_0
    top4_indices = np.zeros((N, 4), dtype=np.uint16)
    top4_weights = np.zeros((N, 4), dtype=np.float32)
    
    for v in range(N):
        row = skin_weights[v]
        if J <= 4:
            sorted_idx = np.argsort(-row)
            for k in range(min(J, 4)):
                top4_indices[v, k] = sorted_idx[k]
                top4_weights[v, k] = row[sorted_idx[k]]
        else:
            part_idx = np.argpartition(-row, 4)[:4]
            sorted_top4 = part_idx[np.argsort(-row[part_idx])]
            top4_indices[v, :] = sorted_top4
            top4_weights[v, :] = row[sorted_top4]
            
        s = np.sum(top4_weights[v])
        if s > 1e-6:
            top4_weights[v] /= s
        else:
            top4_weights[v, 0] = 1.0
            
    local_translations, world_matrices = calculate_joint_transforms(joints, parents)
    ibms = build_inverse_bind_matrices(world_matrices)
    
    # Children lookup
    children_map = {i: [] for i in range(J)}
    root_joints = []
    for i in range(J):
        p = parents[i]
        if p is None or p < 0:
            root_joints.append(i)
        else:
            children_map[p].append(i)
            
    # Nodes in glTF:
    # Node 0: Mesh Node
    # Node 1 to J: Joint Nodes
    nodes = []
    nodes.append({
        "name": "CharacterMesh",
        "mesh": 0,
        "skin": 0
    })
    
    for i in range(J):
        node_def = {
            "name": joint_names[i],
            "translation": [float(c) for c in local_translations[i]],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0]
        }
        if len(children_map[i]) > 0:
            node_def["children"] = [1 + child_idx for child_idx in children_map[i]]
        nodes.append(node_def)
        
    scene_nodes = [0] + [1 + r for r in root_joints]
    
    # Build binary buffers
    buffer_bytes = bytearray()
    buffer_views = []
    accessors = []
    
    def add_buffer_data(data: np.ndarray, target: Optional[int] = None) -> int:
        nonlocal buffer_bytes, buffer_views, accessors
        pad = (4 - (len(buffer_bytes) % 4)) % 4
        if pad > 0:
            buffer_bytes.extend(b'\x00' * pad)
            
        byte_offset = len(buffer_bytes)
        raw = data.tobytes()
        byte_length = len(raw)
        buffer_bytes.extend(raw)
        
        bv_idx = len(buffer_views)
        bv_def = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
        }
        if target is not None:
            bv_def["target"] = target
        buffer_views.append(bv_def)
        return bv_idx
    
    # 1. Position buffer (float32 x 3)
    pos_data = vertices.astype(np.float32)
    pos_bv = add_buffer_data(pos_data, 34962)
    pos_acc = len(accessors)
    accessors.append({
        "bufferView": pos_bv,
        "byteOffset": 0,
        "componentType": 5126, # FLOAT
        "count": N,
        "type": "VEC3",
        "min": [float(x) for x in pos_data.min(axis=0)],
        "max": [float(x) for x in pos_data.max(axis=0)]
    })
    
    # 2. Normal buffer (float32 x 3)
    norm_data = normals.astype(np.float32)
    norm_bv = add_buffer_data(norm_data, 34962)
    norm_acc = len(accessors)
    accessors.append({
        "bufferView": norm_bv,
        "byteOffset": 0,
        "componentType": 5126, # FLOAT
        "count": N,
        "type": "VEC3"
    })
    
    # 3. JOINTS_0 buffer (uint16 x 4)
    joints_data = top4_indices.astype(np.uint16)
    joints_bv = add_buffer_data(joints_data, 34962)
    joints_acc = len(accessors)
    accessors.append({
        "bufferView": joints_bv,
        "byteOffset": 0,
        "componentType": 5123, # UNSIGNED_SHORT
        "count": N,
        "type": "VEC4"
    })
    
    # 4. WEIGHTS_0 buffer (float32 x 4)
    weights_data = top4_weights.astype(np.float32)
    weights_bv = add_buffer_data(weights_data, 34962)
    weights_acc = len(accessors)
    accessors.append({
        "bufferView": weights_bv,
        "byteOffset": 0,
        "componentType": 5126, # FLOAT
        "count": N,
        "type": "VEC4"
    })
    
    # 5. Optional UV buffer
    uv_acc = None
    if uvs is not None:
        uv_data = uvs.astype(np.float32)
        uv_bv = add_buffer_data(uv_data, 34962)
        uv_acc = len(accessors)
        accessors.append({
            "bufferView": uv_bv,
            "byteOffset": 0,
            "componentType": 5126,
            "count": N,
            "type": "VEC2"
        })
        
    # 6. Face indices
    if N < 65535:
        face_data = faces.astype(np.uint16)
        comp_type = 5123
    else:
        face_data = faces.astype(np.uint32)
        comp_type = 5125
    face_bv = add_buffer_data(face_data, 34963)
    face_acc = len(accessors)
    accessors.append({
        "bufferView": face_bv,
        "byteOffset": 0,
        "componentType": comp_type,
        "count": F * 3,
        "type": "SCALAR",
        "min": [int(face_data.min())],
        "max": [int(face_data.max())]
    })
    
    # 7. Inverse Bind Matrices (float32 4x4 x J)
    ibm_data = ibms.astype(np.float32)
    ibm_bv = add_buffer_data(ibm_data)
    ibm_acc = len(accessors)
    accessors.append({
        "bufferView": ibm_bv,
        "byteOffset": 0,
        "componentType": 5126,
        "count": J,
        "type": "MAT4"
    })
    
    attributes = {
        "POSITION": pos_acc,
        "NORMAL": norm_acc,
        "JOINTS_0": joints_acc,
        "WEIGHTS_0": weights_acc
    }
    if uv_acc is not None:
        attributes["TEXCOORD_0"] = uv_acc
        
    meshes = [{
        "name": "SkinnedMesh",
        "primitives": [{
            "attributes": attributes,
            "indices": face_acc,
            "mode": 4 # TRIANGLES
        }]
    }]
    
    skins = [{
        "name": "UniRigSkin",
        "inverseBindMatrices": ibm_acc,
        "joints": [1 + i for i in range(J)],
        "skeleton": 1 + root_joints[0] if len(root_joints) > 0 else 1
    }]
    
    # Process animations if provided
    gltf_animations = []
    if animations:
        for anim_name, anim_data in animations.items():
            channels = []
            samplers = []
            
            for track in anim_data["tracks"]:
                j_idx = track["joint_idx"]
                node_idx = 1 + j_idx
                path = track["path"]
                times = track["times"].astype(np.float32)
                values = track["values"].astype(np.float32)
                
                time_bv = add_buffer_data(times)
                time_acc = len(accessors)
                accessors.append({
                    "bufferView": time_bv,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(times),
                    "type": "SCALAR",
                    "min": [float(times.min())],
                    "max": [float(times.max())]
                })
                
                val_bv = add_buffer_data(values)
                val_acc = len(accessors)
                accessors.append({
                    "bufferView": val_bv,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(values),
                    "type": "VEC4" if path == "rotation" else "VEC3"
                })
                
                sampler_idx = len(samplers)
                samplers.append({
                    "input": time_acc,
                    "output": val_acc,
                    "interpolation": "LINEAR"
                })
                
                channels.append({
                    "sampler": sampler_idx,
                    "target": {
                        "node": node_idx,
                        "path": path
                    }
                })
                
            gltf_animations.append({
                "name": anim_name,
                "channels": channels,
                "samplers": samplers
            })
            
    gltf_dict = {
        "asset": {
            "version": "2.0",
            "generator": "UniRig Pipeline Exporter"
        },
        "scene": 0,
        "scenes": [{
            "nodes": scene_nodes
        }],
        "nodes": nodes,
        "meshes": meshes,
        "skins": skins,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{
            "byteLength": len(buffer_bytes)
        }]
    }
    if len(gltf_animations) > 0:
        gltf_dict["animations"] = gltf_animations
        
    json_bytes = json.dumps(gltf_dict, separators=(',', ':')).encode('utf-8')
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    if json_pad > 0:
        json_bytes += b' ' * json_pad
        
    bin_pad = (4 - (len(buffer_bytes) % 4)) % 4
    if bin_pad > 0:
        buffer_bytes.extend(b'\x00' * bin_pad)
        
    total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_bytes)
    
    header = struct.pack('<4sII', b'glTF', 2, total_length)
    chunk0_header = struct.pack('<I4s', len(json_bytes), b'JSON')
    chunk1_header = struct.pack('<I4s', len(buffer_bytes), b'BIN\x00')
    
    glb_content = header + chunk0_header + json_bytes + chunk1_header + bytes(buffer_bytes)
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(glb_content)
            
    return glb_content
