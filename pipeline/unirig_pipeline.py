import os
import sys
import time
import json
import subprocess
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .rig_export import create_rigged_glb
from .skinning import predict_skin_weights
from .animation import generate_standard_animations
from .pan_retargeting import generate_pan_retargeted_animations

def auto_orient_and_center_mesh(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, trimesh.Trimesh]:
    """
    Centers mesh X and Z around (0, 0) and aligns feet/base at Y = 0,
    preserving coordinate frame alignment between mesh vertices and predicted skeleton joints.
    """
    v = vertices.copy().astype(np.float32)
    j_min = v.min(axis=0)
    j_max = v.max(axis=0)
    
    # Center X and Z around (0, 0)
    center_x = (j_min[0] + j_max[0]) / 2.0
    center_z = (j_min[2] + j_max[2]) / 2.0
    
    v[:, 0] -= center_x
    v[:, 2] -= center_z
    
    # Align feet / bottom base to Y = 0
    v[:, 1] -= j_min[1]
    
    # Recompute vertex and face normals using trimesh
    tm = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    vertex_normals = tm.vertex_normals.astype(np.float32)
    
    return v, vertex_normals, tm

class UniRigPipeline:
    def __init__(self, root_dir: Optional[str] = None):
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent
        else:
            self.root_dir = Path(root_dir)
            
        self.config_skeleton = self.root_dir / "configs/task/quick_inference_skeleton_articulationxl_ar_256_nofbx.yaml"
        self.python_bin = sys.executable
        
    def preprocess_mesh(
        self,
        input_path: str,
        output_dir: str,
        faces_target_count: int = 50000
    ) -> Dict[str, Any]:
        """
        Step 1: Load and preprocess mesh (.glb, .gltf, .obj) using trimesh.
        Automatically auto-orients mesh to Y-Up, centers X/Z, and aligns feet to Y=0.
        Saves raw_data.npz in output_dir matching run.py's get_files path structure.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate relative path matching src.data.extract.get_files
        input_str = str(input_path).removeprefix("./")
        try:
            rel = os.path.relpath(input_str, str(self.root_dir))
            if rel.startswith(".."):
                rel = Path(input_str).stem
            else:
                rel = ".".join(rel.split(".")[:-1])
        except Exception:
            rel = Path(input_str).stem
            
        target_npz_dir = out_dir / rel
        target_npz_dir.mkdir(parents=True, exist_ok=True)
        npz_path = target_npz_dir / "raw_data.npz"
        
        stem = Path(input_path).stem
        
        # Load mesh
        scene_or_mesh = trimesh.load(str(input_path), force='mesh', process=False)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            mesh = scene_or_mesh
            
        raw_vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        
        # Auto-orient to Y-Up, center X/Z around (0,0) and ground feet at Y=0
        vertices, vertex_normals, tm = auto_orient_and_center_mesh(raw_vertices, faces)
        face_normals = tm.face_normals.astype(np.float32)
        
        raw_data = {
            "vertices": vertices.astype(np.float32),
            "vertex_normals": vertex_normals.astype(np.float32),
            "faces": faces.astype(np.int64),
            "face_normals": face_normals.astype(np.float32),
            "joints": None,
            "tails": None,
            "skin": None,
            "no_skin": None,
            "parents": None,
            "names": None,
            "matrix_local": None,
        }
        # Save raw_data.npz in target path and fallback subdirectories
        np.savez(str(npz_path), **raw_data)
        
        # Fallback copies in case run.py resolves path slightly differently
        stem_npz = out_dir / stem / "raw_data.npz"
        if stem_npz != npz_path:
            stem_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(stem_npz), **raw_data)
            
        root_npz = out_dir / "raw_data.npz"
        if root_npz != npz_path:
            np.savez(str(root_npz), **raw_data)
        
        norm_glb_path = out_dir / f"{stem}_input.glb"
        tm.export(str(norm_glb_path))
        
        return {
            "stem": stem,
            "rel_path": rel,
            "npz_path": str(npz_path),
            "norm_glb_path": str(norm_glb_path),
            "num_vertices": len(vertices),
            "num_faces": len(faces),
            "vertices": vertices,
            "faces": faces,
            "normals": vertex_normals
        }

    def predict_skeleton(
        self,
        input_mesh_path: str,
        npz_dir: str,
        output_dir: str,
        seed: int = 12345
    ) -> Dict[str, Any]:
        """
        Step 2: Predict skeleton hierarchy using UniRig AR model.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(input_mesh_path).stem
        
        # Normalize input path relative to root_dir
        input_p_str = str(input_mesh_path)
        try:
            rel_input = os.path.relpath(input_p_str, str(self.root_dir))
            if rel_input.startswith(".."):
                rel_input = input_p_str
        except Exception:
            rel_input = input_p_str
            
        # Run inference via run.py
        cmd = [
            self.python_bin,
            str(self.root_dir / "run.py"),
            f"--task={self.config_skeleton}",
            f"--input={rel_input}",
            f"--output_dir={output_dir}",
            f"--npz_dir={npz_dir}",
            f"--seed={seed}"
        ]
        
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        t0 = time.time()
        res = subprocess.run(
            cmd,
            cwd=str(self.root_dir),
            env=env,
            capture_output=True,
            text=True
        )
        t1 = time.time()
        
        if res.returncode != 0:
            raise RuntimeError(f"Skeleton inference failed (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
            
        # Find output files
        # Output is typically in output_dir/<input_rel_path>/skeleton.obj or predict_skeleton.npz
        # Let's search inside output_dir
        skel_npz_matches = list(out_dir.glob("**/predict_skeleton.npz")) + list(out_dir.glob("**/*_skeleton.npz"))
        skel_obj_matches = list(out_dir.glob("**/skeleton.obj"))
        
        if len(skel_npz_matches) == 0:
            raise FileNotFoundError(f"predict_skeleton.npz not found in {output_dir}")
            
        skel_npz_file = skel_npz_matches[0]
        data = np.load(str(skel_npz_file), allow_pickle=True)
        
        joints = data["joints"].astype(np.float32)
        
        # Unnormalize joints from model's [-1, 1] space back into original mesh space
        raw_matches = list(Path(npz_dir).glob("**/raw_data.npz"))
        if raw_matches:
            raw_mesh = np.load(str(raw_matches[0]), allow_pickle=True)
            v = raw_mesh["vertices"]
            b_min = v.min(axis=0)
            b_max = v.max(axis=0)
            center = (b_max + b_min) / 2.0
            scale = np.max(b_max - b_min) / 2.0
            joints = joints * scale + center
            np.savez(str(skel_npz_file), joints=joints, parents=data["parents"], names=data.get("names", None))

        parents_raw = data["parents"]
        # parents might be ndarray or list with None/-1
        parents = []
        for p in parents_raw:
            if p is None or p == -1 or (isinstance(p, (int, np.integer)) and p < 0):
                parents.append(None)
            else:
                parents.append(int(p))
                
        names = data.get("names", None)
        if names is not None:
            names = [str(n) for n in names]
        else:
            names = [f"Bone_{i:03d}" for i in range(len(joints))]
            
        skel_obj_path = str(skel_obj_matches[0]) if len(skel_obj_matches) > 0 else None
        
        # Build hierarchy tree for frontend
        tree = []
        for i in range(len(joints)):
            tree.append({
                "index": i,
                "name": names[i],
                "position": joints[i].tolist(),
                "parent": parents[i]
            })
            
        return {
            "joints": joints,
            "parents": parents,
            "names": names,
            "tree": tree,
            "num_bones": len(joints),
            "skel_npz_path": str(skel_npz_file),
            "skel_obj_path": skel_obj_path,
            "inference_time_sec": round(t1 - t0, 2)
        }

    def predict_skin(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray,
        parents: List[Optional[int]],
        names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        use_neural: bool = True,
        input_mesh_path: Optional[str] = None,
        skel_stage_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Step 3: Predict / compute skin weights using UniRig Neural Model or Geometric Fallback.
        """
        t0 = time.time()
        weights = None
        method_used = "geometric_laplacian"

        if use_neural and input_mesh_path and skel_stage_dir:
            try:
                skin_task = self.root_dir / "configs/task/quick_inference_unirig_skin.yaml"
                stage3_dir = Path(output_dir) if output_dir else Path(skel_stage_dir).parent / "stage3_neural"
                stage3_dir.mkdir(parents=True, exist_ok=True)
                
                skel_p = Path(skel_stage_dir)
                if skel_p.is_file():
                    skel_dir = skel_p.parent
                else:
                    skel_dir = skel_p
                    
                # Find root directory by stripping relative path components if necessary
                npz_root = skel_dir
                input_p = Path(input_mesh_path)
                if npz_root.name == input_p.stem:
                    npz_root = npz_root.parent
                if input_p.parent.name and npz_root.name == input_p.parent.name:
                    npz_root = npz_root.parent

                cmd = [
                    self.python_bin,
                    str(self.root_dir / "run.py"),
                    f"--task={skin_task}",
                    f"--input={input_mesh_path}",
                    f"--output_dir={stage3_dir}",
                    f"--npz_dir={npz_root}",
                    "--data_name=predict_skeleton.npz",
                    "--seed=12345"
                ]
                
                env = os.environ.copy()
                env["PYTHONNOUSERSITE"] = "1"
                env["CUDA_VISIBLE_DEVICES"] = "0"
                
                res = subprocess.run(
                    cmd,
                    cwd=str(self.root_dir),
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                if res.returncode == 0:
                    skin_matches = list(stage3_dir.glob("**/predict_skin.npz"))
                    if skin_matches:
                        data = np.load(str(skin_matches[0]), allow_pickle=True)
                        sampled_skin = data["skin"]  # (N_sampled, J)
                        sampled_vertices = data["vertices"]
                        
                        from src.system.skin import reskin
                        parents_int = [-1 if p is None else p for p in parents]
                        weights = reskin(
                            sampled_vertices=sampled_vertices,
                            vertices=vertices,
                            parents=parents_int,
                            faces=faces,
                            sampled_skin=sampled_skin,
                            sample_method='median',
                            alpha=2.0,
                            threshold=0.03,
                        )
                        method_used = "unirig_neural_ptv3"
            except Exception as e:
                print(f"[UniRigPipeline] Neural skin prediction fallback to geometric: {e}")
                weights = None

        if weights is None:
            weights = predict_skin_weights(
                vertices=vertices,
                faces=faces,
                joints=joints,
                parents=parents
            )

        t1 = time.time()
        
        N, J = weights.shape
        if names is None:
            names = [f"Bone_{i:03d}" for i in range(J)]
            
        # Compute bone influence statistics
        bone_stats = []
        for j in range(J):
            w_col = weights[:, j]
            bone_stats.append({
                "index": j,
                "name": names[j],
                "max_weight": float(w_col.max()),
                "avg_weight": float(w_col.mean()),
                "affected_vertices": int(np.sum(w_col > 0.05))
            })
            
        skin_npz_path = None
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            skin_npz_path = str(out_p / "skin_weights.npz")
            parents_arr = np.array([-1 if p is None else p for p in parents], dtype=np.int32)
            np.savez(skin_npz_path, weights=weights, names=names, joints=joints, parents=parents_arr)
            
        return {
            "weights": weights,
            "bone_stats": bone_stats,
            "skin_npz_path": skin_npz_path,
            "method": method_used,
            "calc_time_sec": round(t1 - t0, 2)
        }

    def export_rigged_and_animated(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        joints: np.ndarray,
        parents: List[Optional[int]],
        skin_weights: np.ndarray,
        normals: Optional[np.ndarray] = None,
        names: Optional[List[str]] = None,
        output_glb_path: Optional[str] = None,
        use_pan_retargeting: bool = True
    ) -> Dict[str, Any]:
        """
        Step 4: Generate retargeted animations and build complete rigged & animated GLB.
        """
        t0 = time.time()
        if use_pan_retargeting:
            try:
                animations = generate_pan_retargeted_animations(joints, parents)
            except Exception as e:
                print(f"[UniRigPipeline] PAN retargeting fallback: {e}")
                animations = generate_standard_animations(joints, parents)
        else:
            animations = generate_standard_animations(joints, parents)
        
        glb_bytes = create_rigged_glb(
            vertices=vertices,
            faces=faces,
            joints=joints,
            parents=parents,
            skin_weights=skin_weights,
            normals=normals,
            joint_names=names,
            animations=animations,
            output_path=output_glb_path
        )
        t1 = time.time()
        
        return {
            "glb_path": output_glb_path,
            "glb_size_bytes": len(glb_bytes),
            "animations": list(animations.keys()),
            "export_time_sec": round(t1 - t0, 2)
        }

    def run_full_pipeline(
        self,
        input_path: str,
        job_id: str,
        work_dir: str
    ) -> Dict[str, Any]:
        """
        Runs the full 4-stage pipeline end-to-end.
        """
        job_dir = Path(work_dir) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Stage 1
        prep = self.preprocess_mesh(
            input_path=input_path,
            output_dir=str(job_dir / "stage1_prep")
        )
        
        # Stage 2
        skel = self.predict_skeleton(
            input_mesh_path=input_path,
            npz_dir=str(job_dir / "stage1_prep"),
            output_dir=str(job_dir / "stage2_skel")
        )
        
        # Stage 3
        skin = self.predict_skin(
            vertices=prep["vertices"],
            faces=prep["faces"],
            joints=skel["joints"],
            parents=skel["parents"],
            names=skel["names"],
            output_dir=str(job_dir / "stage3_skin"),
            use_neural=True,
            input_mesh_path=input_path,
            skel_stage_dir=skel["skel_npz_path"]
        )
        
        # Stage 4
        rigged_glb = str(job_dir / f"{prep['stem']}_rigged_animated.glb")
        rig = self.export_rigged_and_animated(
            vertices=prep["vertices"],
            faces=prep["faces"],
            joints=skel["joints"],
            parents=skel["parents"],
            skin_weights=skin["weights"],
            normals=prep["normals"],
            names=skel["names"],
            output_glb_path=rigged_glb
        )
        
        return {
            "job_id": job_id,
            "input_file": str(input_path),
            "prep": prep,
            "skel": skel,
            "skin": skin,
            "rig": rig,
            "final_glb": rigged_glb
        }
