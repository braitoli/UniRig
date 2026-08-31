"""
Re-export every cached playground job with standardized bone names.

The bone names in a shipped GLB were never predicted -- UniRig emits "bone_0",
"bone_1", ... for the articulationxl class, and the pipeline derives anatomical
names from the skeleton's geometry afterwards. Jobs rigged before that derivation
was fixed carry names no Mixamo clip can bind to ("mixamorig:RightHindShoulder",
"mixamorig:Spine7") and animation curves laid on the wrong joints, so they need
re-exporting -- but not re-predicting: the joints and skin weights are unchanged,
only the naming and the retargeting read off them.

Reuses each job's cached stage1/stage2/stage3 artifacts, so the AR model never
runs. Writes to a temporary file and swaps it in only once the export succeeds,
leaving the existing GLB untouched on failure.

Progress is appended to a ledger as each job lands, and jobs already in the ledger
are skipped, so an interrupted run resumes where it stopped rather than starting over.

Usage:
    python scripts/restandardize_bone_names.py [--limit N] [--force] [--dry-run]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "pan-motion-retargeting"))

from playground import database  # noqa: E402
from pipeline.animation import SkeletonClassifier, assign_anatomical_names  # noqa: E402
from pipeline.unirig_pipeline import UniRigPipeline  # noqa: E402

LEDGER = REPO / "scratch" / "restandardize_ledger.jsonl"
MIXAMO_SPEC = {
    name.split(":", 1)[-1]
    for part in yaml.safe_load(open(REPO / "configs/skeleton/mixamo.yaml"))["parts"].values()
    for name in part
}


def load_ledger() -> Dict[str, dict]:
    if not LEDGER.exists():
        return {}
    done = {}
    for line in LEDGER.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok"):
            done[rec["job"]] = rec
    return done


def append_ledger(rec: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cached_jobs() -> List[Path]:
    out = []
    for jb in sorted((REPO / "playground" / "storage").glob("job_*")):
        if not jb.is_dir():
            continue
        if (list(jb.glob("stage1_prep/**/raw_data.npz"))
                and list(jb.glob("stage2_skel/**/predict_skeleton.npz"))
                and list(jb.glob("stage3_skin/skin_weights.npz"))):
            out.append(jb)
    return out


def pbr_textures(job_dir: Path):
    """The baked PBR maps live only in the preprocessed GLB, not in raw_data.npz."""
    hits = list(job_dir.glob("stage1_prep/*_input.glb"))
    if not hits:
        return None, None
    try:
        import trimesh
        mesh = trimesh.load(str(hits[0]), force="mesh", process=False)
        material = getattr(getattr(mesh, "visual", None), "material", None)
        return (getattr(material, "baseColorTexture", None),
                getattr(material, "metallicRoughnessTexture", None))
    except Exception:
        return None, None


def process(job_dir: Path, pipeline: UniRigPipeline, dry_run: bool) -> dict:
    prep = np.load(str(next(iter(job_dir.glob("stage1_prep/**/raw_data.npz")))), allow_pickle=True)
    skel = np.load(str(next(iter(job_dir.glob("stage2_skel/**/predict_skeleton.npz")))), allow_pickle=True)
    skin = np.load(str(next(iter(job_dir.glob("stage3_skin/skin_weights.npz")))), allow_pickle=True)

    joints = np.asarray(skel["joints"], dtype=np.float32)
    parents = [None if (p is None or int(p) < 0) else int(p) for p in list(skel["parents"])]
    names = assign_anatomical_names(SkeletonClassifier(joints, parents))
    invalid = [n for n in names
               if n.startswith("mixamorig:") and n.split(":", 1)[1] not in MIXAMO_SPEC]

    # Every rigged GLB in the job, not just one: a job run through the image path leaves
    # both "<stem>_rigged_animated.glb" and "<stem>_generated_3d_rigged_animated.glb",
    # built from the same mesh and skeleton. Replacing only one left 6 jobs still serving
    # the old invented names from their other copy.
    rigged = sorted(job_dir.glob("*_rigged_animated.glb"))
    if not rigged:
        return {"job": job_dir.name, "ok": False, "reason": "no rigged glb to replace"}

    old = None
    try:
        import struct
        with open(rigged[0], "rb") as f:
            head = f.read(1 << 22)
        n = struct.unpack("<I", head[12:16])[0]
        gltf = json.loads(head[20:20 + n])
        old = [gltf["nodes"][j].get("name") for j in gltf["skins"][0]["joints"]]
    except Exception:
        pass
    changed = sum(1 for a, b in zip(old or [], names) if a != b) if old else None

    if dry_run:
        return {"job": job_dir.name, "ok": True, "dry_run": True, "bones": len(names),
                "invalid": len(invalid), "changed": changed}

    base_color, metallic_roughness = pbr_textures(job_dir)
    out_glb = rigged[0]
    tmp = out_glb.with_suffix(".glb.tmp")
    res = pipeline.export_rigged_and_animated(
        vertices=prep["vertices"],
        faces=prep["faces"],
        joints=joints,
        parents=parents,
        skin_weights=skin["weights"],
        normals=prep["vertex_normals"],
        names=names,
        colors=prep.get("colors"),
        base_color_texture=base_color,
        metallic_roughness_texture=metallic_roughness,
        output_glb_path=str(tmp),
        use_pan_retargeting=True,
        use_neural_pan=True,          # what playground/server.py ships
        enable_facial_blendshapes=True,
    )
    # Swap in only now that a complete GLB exists on disk, then mirror it to the job's
    # other copies so none of them keeps serving the old names.
    os.replace(tmp, out_glb)
    payload = out_glb.read_bytes()
    for other in rigged[1:]:
        other_tmp = other.with_suffix(".glb.tmp")
        other_tmp.write_bytes(payload)
        os.replace(other_tmp, other)

    try:
        job = database.get_job(job_dir.name)
        if job:
            meta = job.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta["rig"] = {
                "glb_path": str(out_glb),
                "glb_copies": [str(x) for x in rigged],
                "glb_size_bytes": res["glb_size_bytes"],
                "animations": res["animations"],
                "blendshapes": res.get("blendshapes", []),
                "export_time_sec": res["export_time_sec"],
            }
            meta["bone_names"] = names
            database.update_job(job_dir.name, num_bones=len(names), metadata=meta)
    except Exception as e:
        print(f"      (database not updated: {type(e).__name__}: {e})")

    return {"job": job_dir.name, "ok": True, "bones": len(names), "invalid": len(invalid),
            "changed": changed, "animations": res["animations"],
            "glb_mb": round(res["glb_size_bytes"] / 1e6, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo jobs already in the ledger")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    done = {} if args.force else load_ledger()
    jobs = cached_jobs()
    todo = [j for j in jobs if j.name not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[restandardize] {len(jobs)} jobs with complete cached stages, "
          f"{len(done)} already done, {len(todo)} to process")
    if not todo:
        return 0

    pipeline = None if args.dry_run else UniRigPipeline()
    t0 = time.time()
    failures = 0
    total_changed = 0
    for i, job_dir in enumerate(todo, 1):
        elapsed = time.time() - t0
        eta = (elapsed / max(i - 1, 1)) * (len(todo) - i + 1) if i > 1 else 0.0
        print(f"[{i}/{len(todo)}] {job_dir.name[:46]:48s} "
              f"({elapsed:5.0f}s elapsed, ~{eta:4.0f}s left)", flush=True)
        try:
            rec = process(job_dir, pipeline, args.dry_run)
        except Exception as e:
            rec = {"job": job_dir.name, "ok": False, "reason": f"{type(e).__name__}: {e}"}
        rec["t"] = round(time.time() - t0, 1)
        if not args.dry_run:
            # A dry run must not mark anything done, or the real run skips everything.
            append_ledger(rec)
        if rec.get("ok"):
            total_changed += rec.get("changed") or 0
            print(f"      bones={rec.get('bones')} renamed={rec.get('changed')} "
                  f"invalid={rec.get('invalid')}", flush=True)
        else:
            failures += 1
            print(f"      FAILED: {rec.get('reason')}", flush=True)

    print(f"\n[restandardize] {len(todo) - failures}/{len(todo)} jobs re-exported, "
          f"{total_changed} bones renamed, {failures} failed  ({time.time() - t0:.0f}s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
