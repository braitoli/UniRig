"""
Verification harness for predicted-skeleton bone naming.

UniRig itself never names bones: the articulationxl class has no entry in
Order.parts, so src/data/order.py falls through to "bone_0", "bone_1", ... for
every joint. The pipeline replaces those with anatomical names, and this script
checks the replacements are actually usable downstream:

  1. Anything carrying the `mixamorig:` prefix is a real Mixamo bone. The
     reference set is UniRig's own configs/skeleton/mixamo.yaml, so a name that
     passes here is one a Mixamo clip or a game engine can bind to.
  2. Limb segment names follow the parent chain. The branch lists are consumed
     positionally by neural_pan_adapter, so a branch ordered any other way hands
     the source shoulder's rotation to whatever joint sits highest.
  3. Names are unique, since they become glTF node names.

Runs over every cached skeleton under playground/storage plus synthetic rigs
covering the poses that broke the geometric ordering. Exits non-zero on failure.

Usage:
    python scripts/verify_bone_names.py [--limit N]
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.animation import SkeletonClassifier, assign_anatomical_names  # noqa: E402

# UniRig's own Mixamo skeleton spec is the reference set, not a list retyped here.
MIXAMO_SPEC = {
    name.split(":", 1)[-1]
    for part in yaml.safe_load(open(REPO / "configs/skeleton/mixamo.yaml"))["parts"].values()
    for name in part
}


def hierarchy_order(branch: List[int], parents: List[Optional[int]]) -> List[int]:
    """The branch's joints sorted by depth in the parent chain."""
    depth = {}
    for node in branch:
        d, cur, guard = 0, parents[node], 0
        while cur is not None and guard < len(parents):
            d, cur, guard = d + 1, parents[cur], guard + 1
        depth[node] = d
    return sorted(branch, key=lambda n: depth[n])


def check(joints: np.ndarray, parents: List[Optional[int]], label: str) -> Dict:
    cls = SkeletonClassifier(joints, parents)
    names = assign_anatomical_names(cls)

    fake = sorted({
        n.split(":", 1)[1] for n in names
        if n.startswith("mixamorig:") and n.split(":", 1)[1] not in MIXAMO_SPEC
    })
    dupes = sorted({n for n in names if names.count(n) > 1})
    scrambled = []
    for branch in (cls.left_leg_branches + cls.right_leg_branches
                   + cls.left_arm_branches + cls.right_arm_branches):
        if branch != hierarchy_order(branch, parents):
            scrambled.append([names[j].split(":", 1)[-1] for j in branch])

    return {"label": label, "J": len(joints), "fake": fake,
            "dupes": dupes, "scrambled": scrambled, "names": names}


# ------------------------------------------------------------------ synthetic

def _rig(spec: List[Tuple[Optional[int], Tuple[float, float, float]]]):
    parents = [p for p, _ in spec]
    joints = np.array([xyz for _, xyz in spec], dtype=np.float32)
    return joints, parents


def synthetic_rigs() -> List[Tuple[str, np.ndarray, List[Optional[int]]]]:
    """Rigs whose limbs are not monotonically descending -- the case that breaks
    ordering a branch by joint height."""
    rigs = []

    # Biped with both arms raised to about 45 degrees -- hands above their own shoulders
    # but still below the head, the usual pose an image-to-3D generator produces. Each arm
    # branch climbs from shoulder to hand, so ordering it by joint height runs it
    # tip-to-shoulder, exactly backwards, while the spine walker still finds the real neck.
    rigs.append(("biped_arms_raised", *_rig([
        (None, (0.0, 1.00, 0.0)),   # 0 hips
        (0,    (0.0, 1.25, 0.0)),   # 1 spine
        (1,    (0.0, 1.50, 0.0)),   # 2 chest
        (2,    (0.0, 1.65, 0.0)),   # 3 neck
        (3,    (0.0, 1.80, 0.0)),   # 4 head
        (2,    (0.18, 1.55, 0.0)),  # 5 L shoulder
        (5,    (0.34, 1.62, 0.0)),  # 6 L arm      (up)
        (6,    (0.48, 1.68, 0.0)),  # 7 L forearm  (up)
        (7,    (0.60, 1.72, 0.0)),  # 8 L hand     (highest of the branch)
        (2,    (-0.18, 1.55, 0.0)),  # 9  R shoulder
        (9,    (-0.34, 1.62, 0.0)),  # 10 R arm
        (10,   (-0.48, 1.68, 0.0)),  # 11 R forearm
        (11,   (-0.60, 1.72, 0.0)),  # 12 R hand
        (0,    (0.10, 0.65, 0.0)),  # 13 L upleg
        (13,   (0.10, 0.35, 0.0)),  # 14 L leg
        (14,   (0.10, 0.05, 0.0)),  # 15 L foot
        (15,   (0.10, 0.07, 0.12)),  # 16 L toe    (tips back up off the floor)
        (0,    (-0.10, 0.65, 0.0)),  # 17 R upleg
        (17,   (-0.10, 0.35, 0.0)),  # 18 R leg
        (18,   (-0.10, 0.05, 0.0)),  # 19 R foot
        (19,   (-0.10, 0.07, 0.12)),  # 20 R toe
    ])))

    # Quadruped: horizontal spine, four legs, a tail. Front and hind legs sit at
    # the same heights, so nothing about height distinguishes their segments.
    rigs.append(("quadruped", *_rig([
        (None, (0.0, 0.60, -0.30)),  # 0 hips
        (0,    (0.0, 0.62, 0.00)),   # 1 spine
        (1,    (0.0, 0.64, 0.30)),   # 2 chest
        (2,    (0.0, 0.70, 0.50)),   # 3 neck
        (3,    (0.0, 0.75, 0.65)),   # 4 head
        (2,    (0.12, 0.55, 0.32)),  # 5  L front upper
        (5,    (0.12, 0.30, 0.32)),  # 6  L front lower
        (6,    (0.12, 0.03, 0.34)),  # 7  L front paw
        (2,    (-0.12, 0.55, 0.32)),  # 8  R front upper
        (8,    (-0.12, 0.30, 0.32)),  # 9  R front lower
        (9,    (-0.12, 0.03, 0.34)),  # 10 R front paw
        (0,    (0.12, 0.55, -0.32)),  # 11 L hind upper
        (11,   (0.12, 0.30, -0.32)),  # 12 L hind lower
        (12,   (0.12, 0.03, -0.30)),  # 13 L hind paw
        (0,    (-0.12, 0.55, -0.32)),  # 14 R hind upper
        (14,   (-0.12, 0.30, -0.32)),  # 15 R hind lower
        (15,   (-0.12, 0.03, -0.30)),  # 16 R hind paw
        (0,    (0.0, 0.65, -0.50)),  # 17 tail base
        (17,   (0.0, 0.70, -0.68)),  # 18 tail tip
    ])))
    return rigs


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    results = [check(j, p, name) for name, j, p in synthetic_rigs()]

    jobs = sorted(glob.glob(str(REPO / "playground/storage/job_*")),
                  key=os.path.getmtime, reverse=True)[:args.limit]
    for jb in jobs:
        hits = glob.glob(os.path.join(jb, "stage2_skel", "**", "predict_skeleton.npz"),
                         recursive=True)
        if not hits:
            continue
        d = np.load(hits[0], allow_pickle=True)
        joints = d["joints"].astype(np.float32)
        if len(joints) < 2:
            continue
        parents = [None if (p is None or p < 0) else int(p) for p in d["parents"]]
        results.append(check(joints, parents, os.path.basename(jb)[:38]))

    n_fake = sum(len(r["fake"]) for r in results)
    n_scram = sum(len(r["scrambled"]) for r in results)
    n_dupes = sum(len(r["dupes"]) for r in results)

    print(f"[verify] {len(results)} skeletons "
          f"({len(results) - len(synthetic_rigs())} cached, {len(synthetic_rigs())} synthetic)\n")
    for r in results:
        if not (r["fake"] or r["scrambled"] or r["dupes"]):
            continue
        print(f"  {r['label']:40s} J={r['J']}")
        if r["fake"]:
            print(f"      non-Mixamo names under mixamorig: {r['fake'][:8]}")
        if r["scrambled"]:
            print(f"      branch not in hierarchy order:    {r['scrambled'][:2]}")
        if r["dupes"]:
            print(f"      duplicate names:                  {r['dupes'][:6]}")

    checks = [
        ("Every mixamorig: name is a real Mixamo bone", n_fake),
        ("Every limb branch follows the parent chain", n_scram),
        ("Bone names are unique", n_dupes),
    ]
    print()
    failed = 0
    for label, bad in checks:
        mark = "PASS" if bad == 0 else "FAIL"
        failed += bad != 0
        print(f"  [{mark}] {label:44s} {bad} violation(s)")
    print(f"\n[verify] {len(checks) - failed}/{len(checks)} criteria met")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
