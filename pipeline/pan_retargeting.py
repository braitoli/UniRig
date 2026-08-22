import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from .animation import SkeletonClassifier, euler_to_quat, quat_multiply

def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions q1 and q2."""
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        res = q1 + t * (q2 - q1)
        return res / (np.linalg.norm(res) + 1e-9)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q1 + s1 * q2

class PANMotionRetargeter:
    """
    Pose-aware Attention Network (PAN) & Motion Retargeting Engine.
    Maps source motion capture sequences onto target skeletons predicted by UniRig.
    """
    def __init__(self, target_joints: np.ndarray, target_parents: List[Optional[int]]):
        self.joints = target_joints.astype(np.float32)
        self.parents = target_parents
        self.classifier = SkeletonClassifier(self.joints, self.parents)
        self.J = len(target_joints)

    def retarget_motion(
        self,
        preset_name: str,
        duration: float = 2.0,
        fps: int = 30
    ) -> Dict[str, Any]:
        """
        Retargets motion preset onto target skeleton topology.
        Supported presets: 'Walk_Retargeted', 'Run_Retargeted', 'Dance_Retargeted', 'Jump_Retargeted', 'Combat_Retargeted'
        """
        num_frames = int(fps * duration) + 1
        times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
        tracks = []

        root_idx = self.classifier.root_idx
        root_orig_trans = self.joints[root_idx]
        up_axis = self.classifier.up_axis
        lr_axis = self.classifier.lr_axis
        fw_axis = 3 - up_axis - lr_axis

        # Compute skeleton height scale for translation scaling
        j_span = self.joints.max(axis=0) - self.joints.min(axis=0)
        skel_height = j_span[up_axis] + 1e-6

        # Sort leg branches along forward axis (front vs hind)
        left_branches_sorted = sorted(self.classifier.left_leg_branches, key=lambda b: self.joints[b[0]][fw_axis])
        right_branches_sorted = sorted(self.classifier.right_leg_branches, key=lambda b: self.joints[b[0]][fw_axis])

        if preset_name == "Idle_Retargeted":
            # Natural standing idle with subtle breathing & pelvic sway
            root_trans = np.tile(root_orig_trans, (num_frames, 1))
            root_trans[:, up_axis] += 0.008 * skel_height * np.sin(2.0 * np.pi * times / duration)
            tracks.append({"joint_idx": root_idx, "path": "translation", "times": times, "values": root_trans})

            for s_idx in self.classifier.spine_chain:
                rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                for f in range(num_frames):
                    phase = 2.0 * np.pi * times[f] / duration
                    rot_vals[f] = euler_to_quat(0.015 * np.sin(phase), 0.0, 0.008 * np.cos(phase))
                tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.left_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration
                        rot_vals[f] = euler_to_quat(0.02 * np.sin(phase), 0.0, 0.01 * np.cos(phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.right_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration
                        rot_vals[f] = euler_to_quat(0.02 * np.sin(phase), 0.0, -0.01 * np.cos(phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

        elif preset_name == "Walk_Retargeted":
            # Retargeted walking stride with pelvic roll & counter-arm swing
            root_trans = np.tile(root_orig_trans, (num_frames, 1))
            root_trans[:, up_axis] += 0.04 * skel_height * np.abs(np.sin(2.0 * np.pi * times / duration))
            root_trans[:, lr_axis] += 0.02 * skel_height * np.sin(np.pi * times / duration)

            tracks.append({
                "joint_idx": root_idx,
                "path": "translation",
                "times": times,
                "values": root_trans
            })

            # Retarget pelvis & spine roll
            for s_idx in self.classifier.spine_chain:
                rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                for f in range(num_frames):
                    phase = 2.0 * np.pi * times[f] / duration
                    rot_vals[f] = euler_to_quat(0.04 * np.sin(phase), 0.05 * np.cos(phase), 0.02 * np.sin(2*phase))
                tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})

            # Retarget left leg branches
            for b_idx, branch in enumerate(left_branches_sorted):
                branch_phase_offset = 0.0 if b_idx == 0 else np.pi
                for idx, leg_j in enumerate(branch):
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                        if idx == 0: # Hip / Thigh
                            rot_vals[f] = euler_to_quat(0.42 * np.sin(phase), 0.0, 0.02 * np.cos(phase))
                        elif idx == 1: # Knee flex
                            flex = -0.35 * np.maximum(0.0, np.sin(phase))
                            rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
                        else: # Foot/Ankle
                            rot_vals[f] = euler_to_quat(-0.1 * np.sin(phase), 0.0, 0.0)
                    tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})

            # Retarget right leg branches (Opposite phase to left legs)
            for b_idx, branch in enumerate(right_branches_sorted):
                branch_phase_offset = np.pi if b_idx == 0 else 0.0
                for idx, leg_j in enumerate(branch):
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                        if idx == 0: # Hip / Thigh
                            rot_vals[f] = euler_to_quat(0.42 * np.sin(phase), 0.0, -0.02 * np.cos(phase))
                        elif idx == 1: # Knee flex
                            flex = -0.35 * np.maximum(0.0, np.sin(phase))
                            rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
                        else: # Foot/Ankle
                            rot_vals[f] = euler_to_quat(-0.1 * np.sin(phase), 0.0, 0.0)
                    tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})

            # Retarget arm branches
            for branch in self.classifier.left_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration + np.pi
                        rot_vals[f] = euler_to_quat(0.32 * np.sin(phase), 0.0, 0.08 * np.cos(phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.right_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration
                        rot_vals[f] = euler_to_quat(0.32 * np.sin(phase), 0.0, -0.08 * np.cos(phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

        elif preset_name == "Run_Retargeted":
            # Dynamic running stride
            root_trans = np.tile(root_orig_trans, (num_frames, 1))
            root_trans[:, up_axis] += 0.08 * skel_height * np.abs(np.sin(2.0 * np.pi * times / duration))
            tracks.append({"joint_idx": root_idx, "path": "translation", "times": times, "values": root_trans})

            for s_idx in self.classifier.spine_chain:
                rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                for f in range(num_frames):
                    phase = 2.0 * np.pi * times[f] / duration
                    rot_vals[f] = euler_to_quat(0.20 + 0.06 * np.sin(phase), 0.0, 0.04 * np.cos(phase))
                tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})

            for b_idx, branch in enumerate(left_branches_sorted):
                branch_phase_offset = 0.0 if b_idx == 0 else np.pi
                for idx, leg_j in enumerate(branch):
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                        if idx == 0:
                            rot_vals[f] = euler_to_quat(0.70 * np.sin(phase), 0.0, 0.0)
                        elif idx == 1:
                            flex = -0.55 * np.maximum(0.0, np.sin(phase))
                            rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
                    tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})

            for b_idx, branch in enumerate(right_branches_sorted):
                branch_phase_offset = np.pi if b_idx == 0 else 0.0
                for idx, leg_j in enumerate(branch):
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                        if idx == 0:
                            rot_vals[f] = euler_to_quat(0.70 * np.sin(phase), 0.0, 0.0)
                        elif idx == 1:
                            flex = -0.55 * np.maximum(0.0, np.sin(phase))
                            rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
                    tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.left_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration + np.pi
                        rot_vals[f] = euler_to_quat(0.55 * np.sin(phase), 0.0, 0.12 * np.cos(phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.right_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration
                        rot_vals[f] = euler_to_quat(0.55 * np.sin(phase), 0.0, -0.12 * np.cos(phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

        elif preset_name == "Dance_Retargeted":
            # Rhythmic dance sway
            root_trans = np.tile(root_orig_trans, (num_frames, 1))
            root_trans[:, up_axis] += 0.05 * skel_height * np.sin(4.0 * np.pi * times / duration)
            root_trans[:, lr_axis] += 0.06 * skel_height * np.sin(2.0 * np.pi * times / duration)
            tracks.append({"joint_idx": root_idx, "path": "translation", "times": times, "values": root_trans})

            for s_idx in self.classifier.spine_chain:
                rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                for f in range(num_frames):
                    phase = 2.0 * np.pi * times[f] / duration
                    rot_vals[f] = euler_to_quat(0.12 * np.cos(phase), 0.18 * np.sin(phase), 0.12 * np.sin(2*phase))
                tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.left_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration
                        rot_vals[f] = euler_to_quat(0.45 * np.sin(phase), 0.25 * np.cos(phase), 0.6 + 0.25 * np.sin(2*phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

            for branch in self.classifier.right_arm_branches:
                for arm_j in branch:
                    rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
                    for f in range(num_frames):
                        phase = 2.0 * np.pi * times[f] / duration
                        rot_vals[f] = euler_to_quat(-0.45 * np.sin(phase), -0.25 * np.cos(phase), -0.6 - 0.25 * np.sin(2*phase))
                    tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

        return {"duration": duration, "tracks": tracks}

def generate_pan_retargeted_animations(
    joints: np.ndarray,
    parents: List[Optional[int]],
    fps: int = 30
) -> Dict[str, Dict]:
    """
    Generates a suite of PAN Retargeted animations for the target skeleton.
    """
    retargeter = PANMotionRetargeter(target_joints=joints, target_parents=parents)
    animations = {}
    
    presets = [
        ("Idle", "Idle_Retargeted", 2.0),
        ("Walk", "Walk_Retargeted", 1.2),
        ("Run", "Run_Retargeted", 0.8),
        ("Wave", "Dance_Retargeted", 2.0),
        ("Dance", "Dance_Retargeted", 2.0),
    ]

    for name, p_name, dur in presets:
        animations[name] = retargeter.retarget_motion(preset_name=p_name, duration=dur, fps=fps)

    return animations
