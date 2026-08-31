import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Ensure external/pan-motion-retargeting is on sys.path
PAN_DIR = Path(__file__).resolve().parent.parent / "external" / "pan-motion-retargeting"
if str(PAN_DIR) not in sys.path:
    sys.path.insert(0, str(PAN_DIR))

from pipeline.animation import (SkeletonClassifier, euler_to_quat, quat_multiply,
                                generate_head_gesture_animations,
                                resolve_named_roles, hierarchy_depths)
from outer_utils import BVH
from outer_utils.Animation import positions_global

def quat_inv_batch(q: np.ndarray) -> np.ndarray:
    """Invert array of quaternions [x, y, z, w] of shape (T, 4)."""
    inv = q.copy()
    inv[..., :3] = -inv[..., :3]
    norms = np.sum(inv**2, axis=-1, keepdims=True)
    return inv / (norms + 1e-9)

def quat_mult_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two arrays of quaternions [x, y, z, w] of shape (T, 4)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    
    res = np.stack([x, y, z, w], axis=-1)
    norms = np.linalg.norm(res, axis=-1, keepdims=True)
    return res / (norms + 1e-9)

def _hierarchy_order(parents: List[Optional[int]]) -> List[int]:
    """Joint indices ordered so that a parent always comes before its children."""
    depths = hierarchy_depths(parents)
    return sorted(range(len(parents)), key=lambda j: depths[j])


def _world_quats(local_q: np.ndarray, parents: List[Optional[int]]) -> np.ndarray:
    """Compose parent-local rotations (T, J, 4) xyzw into world rotations."""
    world = np.zeros_like(local_q)
    world[..., 3] = 1.0
    for j in _hierarchy_order(parents):
        p = parents[j]
        world[:, j] = local_q[:, j] if p is None else quat_mult_batch(world[:, p], local_q[:, j])
    return world


def _mean_world_quat(world: np.ndarray) -> np.ndarray:
    """
    Average world rotation of each joint over the clip, shape (T, J, 4) -> (1, J, 4).

    This is the reference pose the transfer is measured against, and it has to be a neutral
    one. Frame 0 is not: a gait cycle catches the two legs half a cycle apart, so zeroing
    each joint there offsets the left leg's swing backwards and the right leg's forwards, and
    the character walks with one leg leading and the other trailing. Averaging a whole cycle
    cancels that. The clip's own bind pose is no use either -- LaFAN1's lays the entire
    skeleton along X rather than standing it up.

    Signs are aligned to frame 0 first, since q and -q are the same rotation and averaging
    them raw cancels to nothing.
    """
    ref = world[0:1]
    aligned = np.where(np.sum(world * ref, axis=-1, keepdims=True) < 0.0, -world, world)
    mean = aligned.mean(axis=0, keepdims=True)
    return mean / (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-9)


def _order_sides_by_x(left: List[List[int]], right: List[List[int]],
                      bind_pos: np.ndarray, lr_axis: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Relabel a left/right pair of limb branch sets by which side sits at the smaller X.

    SkeletonClassifier names whichever limb is at smaller X the "left" one regardless of
    anatomy, and the target is always classified that way. Bone names carry real anatomy, so
    named source roles are relabelled to the same convention before pairing -- otherwise a
    +Z-facing source (whose anatomical left is at +X) hands its left arm to the target's
    right. This only reconciles the two naming conventions; it does not turn anyone round.
    """
    def mean_x(branches: List[List[int]]) -> float:
        vals = [float(np.mean(bind_pos[b, lr_axis])) for b in branches if b]
        return float(np.mean(vals)) if vals else 0.0

    if left and right and mean_x(left) > mean_x(right):
        return right, left
    return left, right


def _scale_quat_angle(q: np.ndarray, scale: float) -> np.ndarray:
    """Scale the rotation angle of each quaternion in (T, 4), keeping its axis."""
    # q and -q are the same rotation; force w >= 0 so `angle` is the short arc (0..pi).
    # Without this a quaternion stored with w < 0 reads as angle > pi and scaling it spins
    # the joint the wrong way.
    q = np.where(q[:, 3:4] < 0.0, -q, q)
    w = np.clip(q[:, 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(0.0, 1.0 - w * w)) + 1e-9
    axis = q[:, :3] / sin_half[:, None]

    # Clamp instead of letting the angle run past pi: a quaternion cannot represent more
    # than a half turn, so an unclamped scale wraps around and reverses the joint
    # (120deg x 2.5 = 300deg comes back as 60deg the other way).
    scaled = np.minimum(angle * scale, np.pi)
    res = np.column_stack([axis * np.sin(scaled * 0.5)[:, None], np.cos(scaled * 0.5)])
    return res / (np.linalg.norm(res, axis=-1, keepdims=True) + 1e-9)


class MocapDrivenRetargeter:
    """
    Retargets real Mixamo/Lafan1 mocap BVH clips onto UniRig predicted skeletons.

    The two skeletons are matched role to role -- root, spine, limb branches -- because
    UniRig joint names (bone_0, bone_1, ...) never match a mocap rig's. Source roles come
    from its bone names when it has real ones, and from geometry otherwise; the target is
    always classified geometrically.
    """
    def __init__(self, joints: np.ndarray, parents: List[Optional[int]]):
        self.joints = joints.astype(np.float32)
        self.parents = parents
        self.J = len(joints)
        self.classifier = SkeletonClassifier(self.joints, self.parents)
        self.pan_dir = PAN_DIR
        self.mixamo_dir = PAN_DIR / "data_preprocess" / "Mixamo" / "Mixamo" / "Aj"
        
    def retarget_mixamo_bvh(
        self,
        bvh_file_path: str,
        fps: int = 30,
        max_duration: float = 4.0,
        leg_amp: float = 1.0,
        arm_amp: float = 1.0,
        cycle_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retargets a real Mixamo Mocap BVH clip onto the target UniRig predicted skeleton
        topology. Amplification defaults to 1.0, transferring the captured performance as
        recorded.
        """
        bvh_path = Path(bvh_file_path)
        if not bvh_path.exists():
            raise FileNotFoundError(f"Mixamo BVH file not found: {bvh_path}")

        anim, names, frametime = BVH.load(str(bvh_path))
        # Classify the source off its average pose, not frame 0: frame 0 of a locomotion clip
        # catches the actor mid-stride with a foot in the air, and a raised foot fails the
        # "reaches the floor" test, so that leg gets read as a tail and never drives anything.
        bvh_bind_pos = positions_global(anim).mean(axis=0)  # (J_bvh, 3) world-space rest pose
        bvh_parents = [int(p) if p >= 0 else None for p in anim.parents]

        total_frames = anim.rotations.qs.shape[0]
        step = max(1, int((1.0 / fps) / frametime))
        frame_indices = list(range(0, total_frames, step))
        if max_duration:
            max_f = int(max_duration * fps)
            frame_indices = frame_indices[:max_f]

        # Extract BVH rotations (w, x, y, z) and convert to glTF (x, y, z, w)
        qs_bvh = anim.rotations.qs[frame_indices] # (T, J_bvh, 4)
        qs_gltf = np.concatenate([qs_bvh[..., 1:], qs_bvh[..., 0:1]], axis=-1)
        pos_bvh = anim.positions[frame_indices] # (T, J_bvh, 3)

        return self._retarget_from_source(
            bvh_bind_pos, bvh_parents, qs_gltf, pos_bvh,
            fps=fps, leg_amp=leg_amp, arm_amp=arm_amp,
            src_names=list(names), cycle_duration=cycle_duration
        )

    def retarget_npz_clip(
        self,
        npz_file_path: str,
        fps: int = 30,
        leg_amp: float = 1.0,
        arm_amp: float = 1.0,
        cycle_duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retargets a pre-extracted mocap clip (Lafan1-and-dog dataset, saved as a small
        standalone .npz — see data_preprocess/Lafan1_and_dog/presets/) onto the target
        UniRig predicted skeleton, using the same structural role mapping as Mixamo BVH clips.
        Serves both the quadruped Dog_* clips and the biped Human_* locomotion cycles.
        """
        npz_path = Path(npz_file_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Mocap clip not found: {npz_path}")

        clip = np.load(str(npz_path), allow_pickle=True)
        # Clip rotations are (w, x, y, z) per the dataset's quat_fk convention; convert to glTF (x, y, z, w)
        qs_wxyz = clip["rotations"].astype(np.float32)  # (T, J, 4)
        qs_gltf = np.concatenate([qs_wxyz[..., 1:], qs_wxyz[..., 0:1]], axis=-1)
        pos = clip["positions"].astype(np.float32)  # (T, J, 3)
        parents_raw = clip["parents"]
        clip_parents = [int(p) if p >= 0 else None for p in parents_raw]
        clip_bind_pos = pos.mean(axis=0)  # (J, 3) world-space rest pose, averaged over the clip
        names = [str(n) for n in clip["joint_names"]] if "joint_names" in clip else None

        return self._retarget_from_source(
            clip_bind_pos, clip_parents, qs_gltf, pos,
            fps=fps, leg_amp=leg_amp, arm_amp=arm_amp,
            src_names=names, cycle_duration=cycle_duration
        )

    def _retarget_from_source(
        self,
        src_bind_pos: np.ndarray,
        src_parents: List[Optional[int]],
        qs_gltf: np.ndarray,
        pos: np.ndarray,
        fps: int = 30,
        leg_amp: float = 1.0,
        arm_amp: float = 1.0,
        src_names: Optional[List[str]] = None,
        cycle_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Shared retargeting core, used by both retarget_mixamo_bvh and retarget_npz_clip.

        Motion crosses over in WORLD space: the source joint's world-space rotation change
        since its own first frame is applied to the target's rest orientation, then converted
        back into the target's parent-local frame for the glTF track. Copying the source's
        parent-local delta straight across (what this did before) assumes both skeletons
        express a joint's rotation in the same frame, and they do not -- LaFAN1's hip sits
        173.8 degrees from identity at its clip's first frame, so "swing the leg forward"
        arrived at the target as very nearly "swing it backward". The giveaway was that
        retargeting a clip onto its own skeleton failed to reproduce the clip.
        """
        src_roles = resolve_named_roles(src_names, src_parents) \
            or SkeletonClassifier(src_bind_pos, src_parents)
        # Named roles are anatomical; the target's are geometric. Align the two conventions
        # before anything is paired.
        src_roles.left_leg_branches, src_roles.right_leg_branches = _order_sides_by_x(
            src_roles.left_leg_branches, src_roles.right_leg_branches,
            src_bind_pos, src_roles.lr_axis)
        src_roles.left_arm_branches, src_roles.right_arm_branches = _order_sides_by_x(
            src_roles.left_arm_branches, src_roles.right_arm_branches,
            src_bind_pos, src_roles.lr_axis)
        tgt = self.classifier

        num_frames = qs_gltf.shape[0]
        duration = float(cycle_duration) if cycle_duration else num_frames / fps
        # Frames are spaced duration/num_frames, not duration/(num_frames - 1). These clips
        # are cycles: the frame after the last one is the first one again, so the last
        # keyframe belongs one interval short of the end. Spreading the samples to land on
        # `duration` instead stretched every clip by a frame, and left the loop no interval
        # to wrap across.
        frame_dt = duration / num_frames
        times = (np.arange(num_frames, dtype=np.float32) * frame_dt).astype(np.float32)

        # World rotation of every source joint, and how far it has turned away from the
        # clip's own neutral pose. The same average pose the roles are classified against.
        world_src = _world_quats(qs_gltf, src_parents)
        src_delta = quat_mult_batch(world_src, quat_inv_batch(_mean_world_quat(world_src)))

        # A target that faces away from the source ought to have this motion turned a half
        # circle about the vertical (and its limb sides swapped) or it strides backwards.
        # That is deliberately NOT done, because nothing available here can say reliably WHICH
        # characters need it: across the cached corpus the three usable facing cues -- toe
        # ahead of ankle, foot mesh extent, head mesh extent -- disagree with each other more
        # often than they agree, and on the one character whose facing the repo's own eye
        # detector could establish, the skeleton cue had it backwards. Turning characters
        # round on a signal that unreliable is worse than never turning one. Feeding
        # detect_eye_regions().forward in here would settle it.

        def branch_joint_idx(branches: List[List[int]], b_idx: int, seg_idx: int) -> Optional[int]:
            """Looks up the source joint for the b_idx-th limb branch's seg_idx-th segment
            (0=thigh/shoulder, 1=knee/elbow, 2+=ankle/hand), clamped to what the source
            skeleton actually has."""
            if not branches:
                return None
            branch = branches[min(b_idx, len(branches) - 1)]
            if not branch:
                return None
            return branch[min(seg_idx, len(branch) - 1)]

        # ---- target joint -> source joint ------------------------------------------------
        pairs: Dict[int, int] = {}

        # Sample the source spine by normalised position so BOTH ends stay anchored: the
        # target's last spine joint always receives the source's head. Indexing from the root
        # and clamping (the previous rule) handed the head whichever source joint sat at the
        # same depth -- the chest on a four-joint spine, and on a longer one Mixamo's hair
        # bone, which does not rotate at all.
        src_spine, tgt_spine = src_roles.spine_chain, tgt.spine_chain
        if src_spine and tgt_spine:
            span = len(tgt_spine) - 1
            for i, joint in enumerate(tgt_spine):
                frac = 0.0 if span == 0 else i / span
                pairs[joint] = src_spine[int(round(frac * (len(src_spine) - 1)))]

        # Leg branches are ordered along the forward axis so a quadruped's front legs meet
        # front legs rather than hind ones.
        tgt_left_legs = sorted(tgt.left_leg_branches, key=lambda b: self.joints[b[0]][tgt.fw_axis])
        tgt_right_legs = sorted(tgt.right_leg_branches, key=lambda b: self.joints[b[0]][tgt.fw_axis])
        src_left_legs = sorted(src_roles.left_leg_branches,
                               key=lambda b: src_bind_pos[b[0]][src_roles.fw_axis])
        src_right_legs = sorted(src_roles.right_leg_branches,
                                key=lambda b: src_bind_pos[b[0]][src_roles.fw_axis])
        src_left_arms = list(src_roles.left_arm_branches)
        src_right_arms = list(src_roles.right_arm_branches)

        limb_pairs = (
            (tgt_left_legs, src_left_legs),
            (tgt_right_legs, src_right_legs),
            (list(tgt.left_arm_branches), src_left_arms),
            (list(tgt.right_arm_branches), src_right_arms),
        )
        for tgt_branches, src_branches in limb_pairs:
            for b_idx, branch in enumerate(tgt_branches):
                for seg_idx, joint in enumerate(branch):
                    src_idx = branch_joint_idx(src_branches, b_idx, seg_idx)
                    if src_idx is not None:
                        pairs[joint] = src_idx

        # Optional per-limb exaggeration. Left at 1.0 the source mocap transfers untouched,
        # which is the point of using mocap; the old hard-coded 1.2x on knees and spine was
        # compensating for the frame mismatch fixed above.
        amp: Dict[int, float] = {}
        for branches, value in ((tgt_left_legs + tgt_right_legs, leg_amp),
                                (list(tgt.left_arm_branches) + list(tgt.right_arm_branches), arm_amp)):
            if abs(value - 1.0) < 1e-3:
                continue
            for branch in branches:
                for joint in branch:
                    amp[joint] = value

        # ---- world rotations on the target, then back to parent-local --------------------
        # The target's rest orientation is identity (rig_export writes every node with
        # rotation [0,0,0,1]), so its world rotation IS the source's world delta.
        world_tgt = np.zeros((num_frames, self.J, 4), dtype=np.float32)
        world_tgt[..., 3] = 1.0
        for joint in _hierarchy_order(self.parents):
            parent = self.parents[joint]
            src_idx = pairs.get(joint)
            if src_idx is None:
                # Unmapped joints ride along with their parent, i.e. local rotation identity.
                if parent is not None:
                    world_tgt[:, joint] = world_tgt[:, parent]
                continue
            delta = src_delta[:, src_idx, :]
            scale = amp.get(joint)
            if scale:
                delta = _scale_quat_angle(delta, scale)
            world_tgt[:, joint] = delta

        # Kept so the role mapping can be asserted directly rather than inferred from the
        # baked curves -- scripts/verify_retarget.py checks that the target's head really is
        # paired with the source's head.
        self.last_pairs = dict(pairs)
        self.last_src_roles = src_roles

        tracks = []

        # Root bounce from the mocap stride, expressed as a fraction of the source's own
        # height so it transfers at the target's scale. Only the vertical component is kept:
        # these are looping cycles, and the source's horizontal path is per-frame
        # yaw-normalised rather than a straight line, so carrying it over just drags the
        # character sideways and pops on every repeat.
        root_idx = tgt.root_idx
        j_span = self.joints.max(axis=0) - self.joints.min(axis=0)
        skel_height = j_span[tgt.up_axis] + 1e-6
        src_span = src_bind_pos.max(axis=0) - src_bind_pos.min(axis=0)
        src_height = src_span[1] + 1e-6
        src_disp = pos[:, 0, :] - pos[0:1, 0, :]
        root_trans = np.tile(self.joints[root_idx], (num_frames, 1)).astype(np.float32)
        root_trans[:, tgt.up_axis] += (skel_height / src_height) * src_disp[:, 1]
        tracks.append({"joint_idx": root_idx, "path": "translation",
                       "times": times, "values": root_trans})

        for joint in sorted(pairs):
            parent = self.parents[joint]
            if parent is None:
                local = world_tgt[:, joint]
            else:
                local = quat_mult_batch(quat_inv_batch(world_tgt[:, parent]), world_tgt[:, joint])
            tracks.append({"joint_idx": joint, "path": "rotation",
                           "times": times, "values": local.astype(np.float32)})

        return {"duration": duration, "tracks": tracks}

def generate_neural_pan_animations(
    joints: np.ndarray,
    parents: List[Optional[int]],
    bvh_file_path: Optional[str] = None,
    fps: int = 30
) -> Dict[str, Dict]:
    """
    Generates realistic Mocap animations by retargeting real Mixamo BVH clips onto the
    target skeleton (structural role mapping) with leg swing amplification.
    """
    retargeter = MocapDrivenRetargeter(joints, parents)
    mixamo_dir = retargeter.mixamo_dir
    dog_dir = PAN_DIR / "data_preprocess" / "Lafan1_and_dog" / "presets"
    animations = {}

    # Quadrupeds keep the procedural animations. Copying rotation deltas role-to-role assumes
    # source and target limbs are proportioned alike, which holds well enough between two
    # bipeds but breaks between the dog clip and, say, a giraffe: the neck is several times
    # longer and the forelegs straight rather than folded, so the transferred angles splay the
    # legs and pitch the neck down. Retargeting across body plans that far apart is what a
    # trained model (the actual PAN network) is for.
    is_quadruped = len(retargeter.classifier.left_leg_branches) >= 2 and len(retargeter.classifier.right_leg_branches) >= 2
    if is_quadruped:
        from pipeline.pan_retargeting import generate_pan_retargeted_animations
        animations = generate_pan_retargeted_animations(joints, parents, fps=fps)
        animations.update(generate_head_gesture_animations(joints, parents, fps=fps))
        return animations

    # Walk/Run come from continuous locomotion mocap (LaFAN1), cut to a single gait cycle so
    # the clip loops without popping -- see scratch/extract_human_locomotion_presets.py. The
    # Mixamo clips are one-shot performances: "Catwalk Walk.bvh" is a runway walk that turns
    # around mid-clip, so it neither loops nor travels in a straight line.
    #
    # Walk is retimed. The captured cycle runs 1.667s, a 72 steps/min amble at roughly
    # 0.67 m/s; people walk at 100-120 steps/min. Only the playback clock changes, so the
    # stride keeps its captured shape and proportion (0.71 body heights per cycle) and lands
    # at a natural speed. Run was captured at 157 steps/min and is already right.
    locomotion = [("Walk", "Human_Walk.npz", 1.1), ("Run", "Human_Run.npz", None)]
    for anim_name, npz_filename, cycle in locomotion:
        npz_path = dog_dir / npz_filename
        if npz_path.exists():
            animations[anim_name] = retargeter.retarget_npz_clip(
                str(npz_path), fps=fps, cycle_duration=cycle)

    # (animation key, source BVH filename, leg amplification, arm amplification, fallback preset)
    # Amplification stays at 1.0: the source clips are real human mocap, so a 1:1 transfer
    # already is the realistic motion. Scaling it up was compensating for limb branches that
    # used to be misclassified, and it pushes joints past the half turn a quaternion can hold.
    # `Idle.bvh` (0.4deg of travel) and `Male Sitting Pose.bvh` (0.0deg -- a static T-pose,
    # not an animation) were replaced by clips that actually move.
    presets = [
        ("Walk", "Catwalk Walk.bvh", 1.0, 1.0, "Walk_Retargeted"),
        ("Run", "Running.bvh", 1.0, 1.0, "Run_Retargeted"),
        ("Dance", "Dancing Running Man.bvh", 1.0, 1.0, "Dance_Retargeted"),
        ("Idle", "Idle (1).bvh", 1.0, 1.0, "Idle_Retargeted"),
        ("Wave", "Taunt Gesture.bvh", 1.0, 1.0, "Dance_Retargeted"),
        ("Jump", "Jumping.bvh", 1.0, 1.0, "Idle_Retargeted"),
        ("Turn", "Standing Turn Right 90.bvh", 1.0, 1.0, "Idle_Retargeted"),
        ("Sit", "Sitting Yell.bvh", 1.0, 1.0, "Idle_Retargeted"),
        ("Punch", "Combo Punch.bvh", 1.0, 1.0, "Dance_Retargeted"),
    ]

    for anim_name, bvh_filename, leg_amp, arm_amp, fallback_preset in presets:
        if anim_name in animations:
            continue  # already covered by the locomotion cycle above
        bvh_path = mixamo_dir / bvh_filename
        if bvh_path.exists():
            animations[anim_name] = retargeter.retarget_mixamo_bvh(
                str(bvh_path), fps=fps, leg_amp=leg_amp, arm_amp=arm_amp
            )
        else:
            from pipeline.pan_retargeting import PANMotionRetargeter
            animations[anim_name] = PANMotionRetargeter(joints, parents).retarget_motion(fallback_preset)

    # Head gestures are generated rather than retargeted -- see
    # generate_head_gesture_animations for why mocap is the wrong source for these two.
    animations.update(generate_head_gesture_animations(joints, parents, fps=fps))

    return animations
