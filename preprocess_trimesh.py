#!/usr/bin/env python3
"""
Trimesh-based preprocessor for UniRig skeleton prediction.
Replaces bpy-based src.data.extract for simple formats (.glb, .obj, .gltf).

Output: creates raw_data.npz in the same format as src/data/extract.py
"""
import argparse
import os
import sys
import numpy as np
import trimesh
import fast_simplification


def load_mesh(path: str) -> trimesh.Trimesh:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in ['.glb', '.gltf']:
        mesh = trimesh.load(path, force='mesh')
    elif ext == '.obj':
        mesh = trimesh.load(path, force='mesh', process=False)
    elif ext in ['.fbx', '.dae', '.vrm', '.blend']:
        raise ValueError(
            f"Format {ext} not supported by trimesh preprocessor. "
            f"Use a .glb/.obj file, or run the original bpy-based pipeline."
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if isinstance(mesh, trimesh.Scene):
        # concatenate all geometries
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")

    return mesh


def maybe_simplify(mesh: trimesh.Trimesh, target_count: int) -> trimesh.Trimesh:
    """Simplify the mesh if it has more faces than target_count."""
    if mesh.faces.shape[0] > target_count:
        vertices, faces = fast_simplification.simplify(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int64),
            target_count=target_count,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def build_raw_data(mesh: trimesh.Trimesh) -> dict:
    """Build raw_data dict in the format expected by UniRig skeleton model."""
    return {
        "vertices": mesh.vertices.astype(np.float32),
        "vertex_normals": mesh.vertex_normals.astype(np.float32),
        "faces": mesh.faces.astype(np.int64),
        "face_normals": mesh.face_normals.astype(np.float32),
        "joints": None,
        "tails": None,
        "skin": None,
        "no_skin": None,
        "parents": None,
        "names": None,
        "matrix_local": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input .glb/.obj file (use a path relative to project root so it matches run.py's get_files mapping)")
    parser.add_argument("--output_dir", type=str, required=True, help="npz output directory; same as run.py's --npz_dir")
    parser.add_argument("--faces_target_count", type=int, default=50000)
    args = parser.parse_args()

    # Mirror the path logic in src.data.extract.get_files: strip a leading
    # './' and the file extension, then join with the npz output dir. This
    # keeps the preprocessed file at the exact location that run.py will
    # later look for it in.
    rel = args.input.removeprefix("./")
    rel = ".".join(rel.split(".")[:-1])
    out_dir = os.path.join(args.output_dir, rel)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_data.npz")

    print(f"[preprocess] loading {args.input}...")
    mesh = load_mesh(args.input)
    print(f"[preprocess] mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

    if mesh.faces.shape[0] > args.faces_target_count:
        print(f"[preprocess] simplifying to ~{args.faces_target_count} faces...")
        mesh = maybe_simplify(mesh, args.faces_target_count)
        print(f"[preprocess] simplified: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")

    raw = build_raw_data(mesh)
    np.savez(out_path, **raw)
    print(f"[preprocess] saved -> {out_path}")


if __name__ == "__main__":
    main()
