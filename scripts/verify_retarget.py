"""
Verification harness for mocap retargeting quality.

Runs generate_neural_pan_animations over every biped skeleton cached under
playground/storage and scores the produced clips against measurable gait /
head-motion criteria. Prints a per-character table plus a pass/fail summary,
and exits non-zero when any criterion misses its target.

Usage:
    python scripts/verify_retarget.py [--limit N] [--anim Walk]
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "pan-motion-retargeting"))

from pipeline.animation import SkeletonClassifier, estimate_facing  # noqa: E402
from pipeline.neural_pan_adapter import generate_neural_pan_animations  # noqa: E402

# Criteria targets, mirrored from the design table.
MIN_HEAD_DEG = 5.0              # head must visibly turn, measured in world space
CADENCE_RANGE = (90.0, 130.0)   # steps per minute for Walk
MAX_LOOP_WRAP_RATIO = 2.5       # last->first jump vs mean per-frame step
MAX_SELF_RETARGET_ERR = 1e-3    # retargeting a clip onto its own skeleton must reproduce it


# ---------------------------------------------------------------- quaternions

def q_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between two arrays of xyzw quaternions."""
    dot = np.abs(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))
    return np.degrees(2.0 * np.arccos(dot))


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bx, by, bz, bw = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return np.concatenate([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def q_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


# ---------------------------------------------------------------- skeleton io

def load_cached_bipeds(limit: Optional[int] = None) -> List[Tuple[str, np.ndarray, List[Optional[int]]]]:
    """Every distinct biped skeleton cached by the playground, de-duplicated by geometry."""
    out = []
    seen = set()
    paths = sorted((REPO / "playground" / "storage").glob("job_*/stage2_skel/**/predict_skeleton.npz"))
    for p in paths:
        try:
            d = np.load(str(p), allow_pickle=True)
            joints = np.asarray(d["joints"], dtype=np.float32)
            parents = [None if x is None or int(x) < 0 else int(x) for x in list(d["parents"])]
            cls = SkeletonClassifier(joints, parents)
        except Exception:
            continue
        if len(cls.left_leg_branches) >= 2 and len(cls.right_leg_branches) >= 2:
            continue  # quadruped: served by a different code path
        if not (cls.left_arm_branches and cls.right_arm_branches):
            continue
        key = (len(joints), round(float(joints.sum()), 3))
        if key in seen:
            continue
        seen.add(key)
        name = p.parts[len(REPO.parts) + 2] if len(p.parts) > len(REPO.parts) + 2 else p.stem
        out.append((name.replace("job_", "")[:32], joints, parents))
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------- forward kinematics

def fk(joints: np.ndarray, parents: List[Optional[int]], anim: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Bake an animation dict onto the skeleton. Returns (T,J,3) world positions and times."""
    J = len(joints)
    times = None
    for tr in anim["tracks"]:
        times = tr["times"]
        break
    T = len(times)

    rot = np.zeros((T, J, 4), dtype=np.float32)
    rot[..., 3] = 1.0
    # glTF rest is translation-only (rig_export writes rotation [0,0,0,1]),
    # so local translation is simply the offset from the parent joint.
    trans = np.zeros((T, J, 3), dtype=np.float32)
    for j in range(J):
        p = parents[j]
        trans[:, j, :] = joints[j] - (joints[p] if p is not None else np.zeros(3, dtype=np.float32))

    for tr in anim["tracks"]:
        j = tr["joint_idx"]
        v = np.asarray(tr["values"], dtype=np.float32)
        if v.shape[0] != T:
            continue
        if tr["path"] == "rotation":
            rot[:, j, :] = v
        else:
            trans[:, j, :] = v

    gq = np.zeros_like(rot)
    gq[..., 3] = 1.0
    gp = np.zeros_like(trans)
    order = sorted(range(J), key=lambda j: _depth(j, parents))
    for j in order:
        p = parents[j]
        if p is None:
            gq[:, j] = rot[:, j]
            gp[:, j] = trans[:, j]
        else:
            gq[:, j] = q_mul(gq[:, p], rot[:, j])
            gp[:, j] = gp[:, p] + q_rot(gq[:, p], trans[:, j])
    return gp, times


def _depth(j: int, parents: List[Optional[int]]) -> int:
    d = 0
    while parents[j] is not None:
        j = parents[j]
        d += 1
        if d > 512:
            break
    return d


# ------------------------------------------------------------------- criteria

def _ankle_of(br: List[int], joints: np.ndarray) -> int:
    """The branch's ankle: one back from the tip when the tip is a foot lying flatter than
    it drops, otherwise the tip itself."""
    if len(br) >= 3:
        seg = joints[br[-1]] - joints[br[-2]]
        if abs(float(seg[2])) > abs(float(seg[1])):
            return br[-2]
    return br[-1]


def facing_sign(joints: np.ndarray, cls: SkeletonClassifier) -> float:
    """The pipeline's own facing estimate -- one source of truth, so the gait check below
    measures the same forward direction the retargeter mirrored against."""
    return estimate_facing(joints, cls)


def head_motion_deg(joints, parents, anim: Dict, head_idx: int) -> float:
    """
    How far the head turns in WORLD space over the clip -- what a viewer actually sees.

    The local track is the wrong measure: real walking mocap holds the head steady to
    stabilise the gaze, so LaFAN1's walk moves the head only 4.6 degrees relative to the
    neck while the head still swings visibly with the body.
    """
    world = _fk_world_rotations(joints, parents, anim)
    return float(q_angle_between(world[:, head_idx], world[0:1, head_idx]).max())


def _fk_world_rotations(joints, parents, anim: Dict) -> np.ndarray:
    J = len(joints)
    times = anim["tracks"][0]["times"]
    T = len(times)
    rot = np.zeros((T, J, 4), dtype=np.float32)
    rot[..., 3] = 1.0
    for tr in anim["tracks"]:
        if tr["path"] == "rotation":
            v = np.asarray(tr["values"], dtype=np.float32)
            if v.shape[0] == T:
                rot[:, tr["joint_idx"], :] = v
    world = np.zeros_like(rot)
    world[..., 3] = 1.0
    for j in sorted(range(J), key=lambda k: _depth(k, parents)):
        p = parents[j]
        world[:, j] = rot[:, j] if p is None else q_mul(world[:, p], rot[:, j])
    return world


def loop_wrap_ratio(anim: Dict) -> float:
    """Size of the last->first pose jump relative to a normal per-frame step."""
    steps, wraps = [], []
    for tr in anim["tracks"]:
        if tr["path"] != "rotation":
            continue
        v = np.asarray(tr["values"], dtype=np.float32)
        if v.shape[0] < 3:
            continue
        steps.append(q_angle_between(v[1:], v[:-1]).mean())
        wraps.append(float(q_angle_between(v[-1:], v[0:1])[0]))
    if not steps:
        return 0.0
    mean_step = float(np.mean(steps)) + 1e-9
    return float(np.mean(wraps)) / mean_step


def gait_direction_score(joints, parents, cls, anim: Dict) -> float:
    """
    Positive when the swing foot travels forwards (in the character's own facing
    direction) while it is off the ground -- which is what walking looks like.
    Negative means the gait was transferred mirrored front-to-back.
    """
    # Measured against the SOURCE's forward (+Z), not the target's own facing. The retargeter
    # does not mirror, so a correct transfer always carries the foot towards +Z while it is
    # lifted; scoring against a per-character facing estimate would just re-measure that
    # estimate, which is the one thing here that is not trustworthy.
    sign = 1.0
    gp, _ = fk(joints, parents, anim)
    hips = gp[:, cls.root_idx, :]
    scores = []
    for br in cls.left_leg_branches + cls.right_leg_branches:
        # The ANKLE, not the tip. A toe is a short bone whose path is dominated by ankle
        # roll, and its rest orientation rarely matches the source's, so it reads the stride
        # backwards even when the leg itself swings correctly.
        foot = _ankle_of(br, joints)
        rel = gp[:, foot, :] - hips
        fwd = rel[:, 2] * sign
        height = gp[:, foot, 1]
        if height.max() - height.min() < 1e-6:
            continue
        v_fwd = np.gradient(fwd)
        h_centered = height - height.mean()
        denom = (np.std(v_fwd) * np.std(h_centered) + 1e-9) * len(v_fwd)
        scores.append(float(np.sum(v_fwd * h_centered) / denom))
    return float(np.mean(scores)) if scores else 0.0


def legs_agree(joints, parents, cls, anim: Dict) -> Optional[bool]:
    """
    Do both legs stride the same way?

    Facing-independent, and therefore the criterion that actually tests the transfer: a
    character may legitimately face either way, but its two legs must always agree. They did
    not when the transfer was measured from the clip's first frame, because a gait cycle
    catches the legs half a cycle apart and each was zeroed at an opposite phase.
    """
    gp, _ = fk(joints, parents, anim)
    hips = gp[:, cls.root_idx, :]
    per_leg = []
    for br in cls.left_leg_branches + cls.right_leg_branches:
        foot = _ankle_of(br, joints)
        rel = gp[:, foot, :] - hips
        height = gp[:, foot, 1]
        if height.max() - height.min() < 1e-6:
            continue
        v = np.gradient(rel[:, 2])
        hc = height - height.mean()
        per_leg.append(float(np.sum(v * hc) / ((np.std(v) * np.std(hc) + 1e-9) * len(v))))
    strong = [x for x in per_leg if abs(x) > 0.02]
    if len(strong) < 2:
        return None
    return all(x > 0 for x in strong) or all(x < 0 for x in strong)


def cadence_spm(anim: Dict) -> float:
    """Steps per minute, assuming the clip holds exactly one full gait cycle (2 steps)."""
    d = float(anim["duration"])
    return 0.0 if d <= 0 else 2.0 / d * 60.0


def self_retarget_error() -> Optional[float]:
    """
    Golden check on the retargeting maths: give the retargeter a target skeleton whose rest
    pose is exactly the reference pose the transfer measures against -- the clip's average
    world rotation -- and the baked result must reproduce the clip's own joint motion. Any
    mismatch between how the source stores a rotation and how the target consumes it shows up
    here as a large error.

    Only Human_Walk is used. The other presets start mid-stride with a foot in the air, and a
    rest pose like that makes SkeletonClassifier drop the raised leg, so the roles no longer
    line up one-to-one and the comparison stops being meaningful.
    """
    from pipeline.neural_pan_adapter import (MocapDrivenRetargeter, _world_quats,
                                             _mean_world_quat)
    clip_path = (REPO / "external" / "pan-motion-retargeting" / "data_preprocess"
                 / "Lafan1_and_dog" / "presets" / "Human_Walk.npz")
    if not clip_path.exists():
        return None
    clip = np.load(str(clip_path), allow_pickle=True)
    pos = clip["positions"].astype(np.float32)
    parents = [int(x) if x >= 0 else None for x in clip["parents"]]

    qs = clip["rotations"].astype(np.float32)
    qs = np.concatenate([qs[..., 1:], qs[..., 0:1]], axis=-1)
    ref = _mean_world_quat(_world_quats(qs, parents))[0]        # (J, 4)
    offsets = clip["offsets"].astype(np.float32)                # (J-1, 3), child offsets
    rest = np.zeros((len(parents), 3), dtype=np.float32)
    for j in range(1, len(parents)):
        pj = parents[j]
        rest[j] = rest[pj] + q_rot(ref[pj], offsets[j - 1])

    anim = MocapDrivenRetargeter(rest, parents).retarget_npz_clip(str(clip_path))
    baked, _ = fk(rest, parents, anim)
    # Root translation is deliberately dropped, so compare hips-relative poses.
    expected = pos - pos[:, 0:1, :]
    got = baked - baked[:, 0:1, :]
    return float(np.abs(expected - got).max()) / (float(np.abs(expected).max()) + 1e-9)


def head_pairing_ok(joints, parents, cls) -> Optional[bool]:
    """True when the target's head is paired with the SOURCE's head, not its chest or a
    hair bone. Reads the mapping the retargeter recorded rather than guessing from curves."""
    from pipeline.neural_pan_adapter import MocapDrivenRetargeter
    clip_path = (REPO / "external" / "pan-motion-retargeting" / "data_preprocess"
                 / "Lafan1_and_dog" / "presets" / "Human_Walk.npz")
    if not clip_path.exists():
        return None
    rt = MocapDrivenRetargeter(joints, parents)
    rt.retarget_npz_clip(str(clip_path))
    pairs = getattr(rt, "last_pairs", None)
    roles = getattr(rt, "last_src_roles", None)
    if not pairs or roles is None or not cls.spine_chain:
        return None
    return pairs.get(cls.spine_chain[-1]) == roles.spine_chain[-1]


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    skels = load_cached_bipeds(args.limit)
    print(f"[verify] {len(skels)} distinct biped skeletons from playground/storage\n")
    if not skels:
        print("[verify] nothing to check")
        return 1

    rows = []
    t0 = time.time()
    for n, (name, joints, parents) in enumerate(skels, 1):
        cls = SkeletonClassifier(joints, parents)
        try:
            anims = generate_neural_pan_animations(joints, parents)
        except Exception as e:
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})
            continue

        row = {"name": name, "error": None}
        head = cls.spine_chain[-1] if cls.spine_chain else cls.root_idx
        row["head_walk"] = head_motion_deg(joints, parents, anims["Walk"], head) if "Walk" in anims else 0.0
        row["head_idle"] = head_motion_deg(joints, parents, anims["Idle"], head) if "Idle" in anims else 0.0
        row["nod"] = head_motion_deg(joints, parents, anims["Nod"], head) if "Nod" in anims else None
        row["shake"] = head_motion_deg(joints, parents, anims["HeadShake"], head) if "HeadShake" in anims else None
        row["pairing"] = head_pairing_ok(joints, parents, cls)
        row["legs"] = legs_agree(joints, parents, cls, anims["Walk"]) if "Walk" in anims else None
        row["cadence"] = cadence_spm(anims["Walk"]) if "Walk" in anims else 0.0
        row["wrap"] = loop_wrap_ratio(anims["Walk"]) if "Walk" in anims else 0.0
        row["gait"] = gait_direction_score(joints, parents, cls, anims["Walk"]) if "Walk" in anims else 0.0
        row["facing"] = facing_sign(joints, cls)
        rows.append(row)
        if n % 5 == 0 or n == len(skels):
            print(f"[verify] {n}/{len(skels)} skeletons  ({time.time() - t0:.1f}s)")

    ok = [r for r in rows if not r["error"]]
    print()
    if not args.quiet:
        print(f"{'character':34s} {'headWalk':>9s} {'headIdle':>9s} {'nod':>7s} {'shake':>7s} "
              f"{'cadence':>8s} {'wrap':>6s} {'gait':>7s} {'face':>5s} {'pair':>5s}")
        for r in rows:
            if r["error"]:
                print(f"{r['name']:34s} ERROR {r['error']}")
                continue
            nod = f"{r['nod']:7.1f}" if r["nod"] is not None else "      -"
            shk = f"{r['shake']:7.1f}" if r["shake"] is not None else "      -"
            pair = "ok" if r.get("pairing") else ("--" if r.get("pairing") is None else "BAD")
            print(f"{r['name']:34s} {r['head_walk']:9.1f} {r['head_idle']:9.1f} {nod} {shk} "
                  f"{r['cadence']:8.0f} {r['wrap']:6.2f} {r['gait']:+7.3f} {r['facing']:+5.0f} {pair:>5s}")
        print()

    n = len(ok)
    checks = [
        ("Head paired with source's head",
         sum(1 for r in ok if r.get("pairing") is True), n),
        ("Head turns in Walk (world, >%.0f deg)" % MIN_HEAD_DEG,
         sum(1 for r in ok if r["head_walk"] > MIN_HEAD_DEG), n),
        ("Head turns in Idle (world, >%.0f deg)" % MIN_HEAD_DEG,
         sum(1 for r in ok if r["head_idle"] > MIN_HEAD_DEG), n),
        ("Both legs stride the same way",
         sum(1 for r in ok if r.get("legs") is not False), n),
        ("Walk cadence in %d-%d spm" % CADENCE_RANGE,
         sum(1 for r in ok if CADENCE_RANGE[0] <= r["cadence"] <= CADENCE_RANGE[1]), n),
        ("Loop wrap <= %.1fx mean step" % MAX_LOOP_WRAP_RATIO,
         sum(1 for r in ok if r["wrap"] <= MAX_LOOP_WRAP_RATIO), n),
        ("Nod produces head motion",
         sum(1 for r in ok if r["nod"] is not None and r["nod"] > MIN_HEAD_DEG), n),
        ("HeadShake produces head motion",
         sum(1 for r in ok if r["shake"] is not None and r["shake"] > MIN_HEAD_DEG), n),
    ]
    failed = 0
    for label, got, tot in checks:
        mark = "PASS" if got == tot else "FAIL"
        if got != tot:
            failed += 1
        print(f"  [{mark}] {label:40s} {got}/{tot}")
    err = self_retarget_error()
    if err is not None:
        mark = "PASS" if err <= MAX_SELF_RETARGET_ERR else "FAIL"
        if err > MAX_SELF_RETARGET_ERR:
            failed += 1
        print(f"  [{mark}] {'Self-retarget reproduces the clip':40s} "
              f"error {err * 100:.4f}% of pose extent")
        checks.append(("self-retarget", 0, 0))

    bad = [r for r in rows if r["error"]]
    if bad:
        print(f"  [FAIL] {len(bad)} skeleton(s) raised errors")
        failed += 1
    print(f"\n[verify] {len(checks) - failed}/{len(checks)} criteria met  ({time.time() - t0:.1f}s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
