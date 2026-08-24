import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Ensure external/pan-motion-retargeting is on sys.path
PAN_DIR = Path(__file__).resolve().parent.parent / "external" / "pan-motion-retargeting"
if str(PAN_DIR) not in sys.path:
    sys.path.insert(0, str(PAN_DIR))

from pipeline.animation import SkeletonClassifier, euler_to_quat, quat_multiply
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

class MocapDrivenRetargeter:
    """
    Retargets real Mixamo/Lafan1 mocap BVH clips onto UniRig predicted skeletons.
    Both skeletons are structurally classified (root/spine/limb roles) via SkeletonClassifier
    and mapped role-to-role, since UniRig joint names (bone_0, bone_1, ...) never match
    Mixamo bone names. Rotation deltas are amplified for dynamic walking/running strides.
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
        leg_amp: float = 2.2,
        arm_amp: float = 1.8
    ) -> Dict[str, Any]:
        """
        Retargets a real Mixamo Mocap BVH clip onto the target UniRig predicted skeleton topology
        with amplified leg stride and arm swing for natural, high-amplitude walking and running.
        """
        bvh_path = Path(bvh_file_path)
        if not bvh_path.exists():
            raise FileNotFoundError(f"Mixamo BVH file not found: {bvh_path}")

        anim, _names, frametime = BVH.load(str(bvh_path))
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
            fps=fps, leg_amp=leg_amp, arm_amp=arm_amp
        )

    def retarget_npz_clip(
        self,
        npz_file_path: str,
        fps: int = 30,
        leg_amp: float = 1.0,
        arm_amp: float = 1.0
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

        return self._retarget_from_source(
            clip_bind_pos, clip_parents, qs_gltf, pos,
            fps=fps, leg_amp=leg_amp, arm_amp=arm_amp
        )

    def _retarget_from_source(
        self,
        src_bind_pos: np.ndarray,
        src_parents: List[Optional[int]],
        qs_gltf: np.ndarray,
        pos: np.ndarray,
        fps: int = 30,
        leg_amp: float = 2.2,
        arm_amp: float = 1.8
    ) -> Dict[str, Any]:
        """
        Shared retargeting core: classifies the source skeleton's structural roles the same
        way the UniRig target skeleton is classified (source bone names never match UniRig's
        generic joint names), maps role-to-role, and produces amplified rotation/translation
        tracks. Used by both retarget_mixamo_bvh and retarget_npz_clip.
        """
        src_classifier = SkeletonClassifier(src_bind_pos, src_parents)

        num_frames = qs_gltf.shape[0]
        duration = num_frames / fps
        times = np.linspace(0.0, duration, num_frames, dtype=np.float32)

        root_pos = pos[:, 0, :]  # (T, 3)

        # Calculate UniRig target skeleton height
        j_span = self.joints.max(axis=0) - self.joints.min(axis=0)
        skel_height = j_span[self.classifier.up_axis] + 1e-6

        tracks = []
        root_idx = self.classifier.root_idx
        root_orig_trans = self.joints[root_idx]

        # Root bounce from the mocap stride, expressed as a fraction of the source's own
        # height so it transfers at the target's scale. The previous fixed 0.02 factor
        # multiplied LaFAN1's centimetres directly and slid the root several body heights.
        # Only the vertical component is kept: these are looping cycles, and the source's
        # horizontal path is per-frame yaw-normalised rather than a straight line, so
        # carrying it over just drags the character sideways and pops on every repeat.
        src_span = src_bind_pos.max(axis=0) - src_bind_pos.min(axis=0)
        src_height = src_span[1] + 1e-6
        src_disp = root_pos - root_pos[0:1, :]
        root_trans = np.tile(root_orig_trans, (num_frames, 1))
        root_trans[:, self.classifier.up_axis] += (skel_height / src_height) * src_disp[:, 1]

        tracks.append({"joint_idx": root_idx, "path": "translation", "times": times, "values": root_trans})

        # Computes amplified local delta quaternion: q_delta(t) = q(0)^(-1) * q(t)
        # Scaled along axis by scale_amp
        def get_bvh_scaled_delta_quat(idx: int, scale_amp: float = 1.0) -> np.ndarray:
            q_t = qs_gltf[:, idx, :] # (T, 4)
            q_0 = q_t[0:1, :]       # (1, 4)
            q_0_inv = quat_inv_batch(q_0)
            q_delta = quat_mult_batch(q_0_inv, q_t) # Local delta from frame 0
            
            if abs(scale_amp - 1.0) < 1e-3:
                return q_delta

            # q and -q are the same rotation; force w >= 0 so `angle` is the short arc
            # (0..pi). Without this a delta stored with w < 0 reads as angle > pi and
            # scaling it spins the joint the wrong way.
            q_delta = np.where(q_delta[:, 3:4] < 0.0, -q_delta, q_delta)

            w = np.clip(q_delta[:, 3], -1.0, 1.0)
            angle = 2.0 * np.arccos(w) # Angle in radians
            sin_half = np.sqrt(np.maximum(0.0, 1.0 - w*w)) + 1e-9
            axis = q_delta[:, :3] / sin_half[:, None]

            # Clamp instead of letting the angle run past pi: a quaternion cannot represent
            # more than a half turn, so an unclamped scale wraps around and reverses the
            # joint (120deg x 2.5 = 300deg comes back as 60deg the other way).
            scaled_angle = np.minimum(angle * scale_amp, np.pi)
            scaled_w = np.cos(scaled_angle * 0.5)
            scaled_sin = np.sin(scaled_angle * 0.5)
            scaled_xyz = axis * scaled_sin[:, None]
            
            res = np.column_stack([scaled_xyz, scaled_w])
            norms = np.linalg.norm(res, axis=-1, keepdims=True)
            return res / (norms + 1e-9)

        def branch_joint_idx(bvh_branches: List[List[int]], b_idx: int, seg_idx: int) -> Optional[int]:
            """Looks up the BVH joint index for the b_idx-th limb branch's seg_idx-th
            segment (0=thigh/shoulder, 1=knee/elbow, 2+=ankle/hand), clamped to what the
            BVH source skeleton actually has."""
            if not bvh_branches:
                return None
            branch = bvh_branches[min(b_idx, len(bvh_branches) - 1)]
            if not branch:
                return None
            return branch[min(seg_idx, len(branch) - 1)]

        # Spine mapping
        src_spine = src_classifier.spine_chain
        for i, s_idx in enumerate(self.classifier.spine_chain):
            if not src_spine:
                continue
            src_idx = src_spine[min(i, len(src_spine) - 1)]
            rot_vals = get_bvh_scaled_delta_quat(src_idx, scale_amp=1.2)
            tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})

        # Left Legs mapping (Amplified leg swing & deep knee flex)
        left_branches_sorted = sorted(self.classifier.left_leg_branches, key=lambda b: self.joints[b[0]][self.classifier.fw_axis])
        src_left_legs = sorted(src_classifier.left_leg_branches, key=lambda b: src_bind_pos[b[0]][src_classifier.fw_axis])
        for b_idx, branch in enumerate(left_branches_sorted):
            for idx, leg_j in enumerate(branch):
                seg_amp = leg_amp * 1.2 if idx == 1 else leg_amp  # deeper knee flex
                src_idx = branch_joint_idx(src_left_legs, b_idx, idx)
                if src_idx is None:
                    continue
                rot = get_bvh_scaled_delta_quat(src_idx, scale_amp=seg_amp)
                tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot})

        # Right Legs mapping (Amplified leg swing & deep knee flex)
        right_branches_sorted = sorted(self.classifier.right_leg_branches, key=lambda b: self.joints[b[0]][self.classifier.fw_axis])
        src_right_legs = sorted(src_classifier.right_leg_branches, key=lambda b: src_bind_pos[b[0]][src_classifier.fw_axis])
        for b_idx, branch in enumerate(right_branches_sorted):
            for idx, leg_j in enumerate(branch):
                seg_amp = leg_amp * 1.2 if idx == 1 else leg_amp
                src_idx = branch_joint_idx(src_right_legs, b_idx, idx)
                if src_idx is None:
                    continue
                rot = get_bvh_scaled_delta_quat(src_idx, scale_amp=seg_amp)
                tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot})

        # Left Arm mapping
        src_left_arms = src_classifier.left_arm_branches
        for b_idx, branch in enumerate(self.classifier.left_arm_branches):
            for idx, arm_j in enumerate(branch):
                src_idx = branch_joint_idx(src_left_arms, b_idx, idx)
                if src_idx is None:
                    continue
                rot = get_bvh_scaled_delta_quat(src_idx, scale_amp=arm_amp)
                tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot})

        # Right Arm mapping
        src_right_arms = src_classifier.right_arm_branches
        for b_idx, branch in enumerate(self.classifier.right_arm_branches):
            for idx, arm_j in enumerate(branch):
                src_idx = branch_joint_idx(src_right_arms, b_idx, idx)
                if src_idx is None:
                    continue
                rot = get_bvh_scaled_delta_quat(src_idx, scale_amp=arm_amp)
                tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot})

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
        return generate_pan_retargeted_animations(joints, parents, fps=fps)

    # Walk/Run come from continuous locomotion mocap (LaFAN1), cut to a single gait cycle so
    # the clip loops without popping -- see scratch/extract_human_locomotion_presets.py. The
    # Mixamo clips are one-shot performances: "Catwalk Walk.bvh" is a runway walk that turns
    # around mid-clip, so it neither loops nor travels in a straight line.
    for anim_name, npz_filename in [("Walk", "Human_Walk.npz"), ("Run", "Human_Run.npz")]:
        npz_path = dog_dir / npz_filename
        if npz_path.exists():
            animations[anim_name] = retargeter.retarget_npz_clip(str(npz_path), fps=fps)

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

    return animations
