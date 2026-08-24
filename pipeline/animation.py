import numpy as np
from typing import List, Dict, Optional, Tuple

def euler_to_quat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Convert Euler angles (radians: pitch/X, yaw/Y, roll/Z) to glTF quaternion [x, y, z, w].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.array([x, y, z, w], dtype=np.float32)
    norm = np.linalg.norm(q)
    return q / (norm + 1e-9)

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float32)

class SkeletonClassifier:
    """
    Analyzes skeleton joint positions and hierarchy to identify anatomical roles:
    root, spine, head, left/right arms/wings, left/right legs (front & hind for quadrupeds).
    """
    def __init__(self, joints: np.ndarray, parents: List[Optional[int]]):
        self.joints = joints.astype(np.float32)
        self.parents = parents
        self.J = len(joints)
        
        self.children = {i: [] for i in range(self.J)}
        self.root_idx = 0
        for i in range(self.J):
            p = parents[i]
            if p is None or p < 0:
                self.root_idx = i
            else:
                self.children[p].append(i)
                
        self._classify()
        
    def _classify(self):
        j_min = self.joints.min(axis=0)
        j_max = self.joints.max(axis=0)
        span = j_max - j_min
        
        # Up axis is Y (1), Left-Right is X (0), Forward-Backward is Z (2) in Y-Up mesh space
        self.up_axis = 1
        self.lr_axis = 0
        self.fw_axis = 2
        
        # 1. Identify main spine chain, following the body's longest axis away from the root.
        # An upright biped is tallest along `up`, but a quadruped is longest along `forward`:
        # walking "upwards" on a horizontal body climbs a front leg instead of the backbone,
        # which then feeds a foreleg's rotation to whatever the target's spine is.
        body_axis = self.fw_axis if span[self.fw_axis] > span[self.up_axis] else self.up_axis

        # Judge a branch by how far it eventually reaches along the body axis, not by where its
        # first joint sits: a shoulder can sit closer to the mid-line than the neck does, and
        # scoring on the immediate joint alone then picks the whole front leg as the backbone.
        centre_lr = float(np.median(self.joints[:, self.lr_axis]))

        def walk_spine(sign: int) -> List[int]:
            chain = []
            curr = self.root_idx
            while curr is not None:
                chain.append(curr)
                cand_children = self.children[curr]
                if len(cand_children) == 0:
                    break
                best_c = None
                best_score = -1e9
                for c in cand_children:
                    reach = max(sign * self.joints[j][body_axis] for j in self._get_sub_tree(c))
                    dist_lr = abs(self.joints[c][self.lr_axis] - centre_lr)
                    score = reach - 0.25 * dist_lr
                    if score > best_score:
                        best_score = score
                        best_c = c
                curr = best_c
            return chain

        if body_axis == self.up_axis:
            self.spine_chain = walk_spine(1)
        else:
            # Along the forward axis the head could be at either end, so follow whichever
            # direction traces the longer backbone (the other one runs down the tail).
            forward, backward = walk_spine(1), walk_spine(-1)
            self.spine_chain = forward if len(forward) >= len(backward) else backward

        self.head_idx = self.spine_chain[-1] if self.spine_chain else self.root_idx
        spine_x = np.median(self.joints[self.spine_chain, self.lr_axis])
        
        # 2. Collect limb branches attached to spine joints
        self.left_leg_branches = []
        self.right_leg_branches = []
        self.left_arm_branches = []
        self.right_arm_branches = []
        self.tail_branches = []
        
        skel_height = span[self.up_axis] + 1e-6
        ground = j_min[self.up_axis]

        for idx, spine_j in enumerate(self.spine_chain):
            siblings = [c for c in self.children[spine_j] if c not in self.spine_chain]
            if not siblings:
                continue
            # A left/right limb pair always hangs off the same spine joint, so split the
            # siblings against each other. An absolute mid-line (the spine's median, or the
            # spine joint itself) puts both legs on one side whenever the skeleton sits
            # off-centre, and a side with no branches leaves those joints unanimated.
            # The key averages the whole branch because the attachment joint alone can be
            # identical for both sides -- a giraffe's two shoulders sit right on the mid-line.
            sib_branches = {c: self._get_sub_tree(c) for c in siblings}
            sib_lr = {c: float(np.mean(self.joints[b, self.lr_axis]))
                      for c, b in sib_branches.items() if b}
            mid_lr = (min(sib_lr.values()) + max(sib_lr.values())) / 2.0 if len(sib_lr) >= 2 \
                else float(self.joints[spine_j][self.lr_axis])

            for child in siblings:
                branch = sib_branches[child]
                if not branch:
                    continue
                    
                # Order branch joints from top (closest to spine) to tip
                branch_sorted = sorted(branch, key=lambda j: self.joints[j][self.up_axis], reverse=True)
                root_j = branch_sorted[0]
                tip_j = branch_sorted[-1]
                
                dir_vec = self.joints[tip_j] - self.joints[root_j]
                is_left = (sib_lr[child] < mid_lr)
                
                # A limb is a leg when its tip reaches the floor, not merely when it points
                # downward -- a human's arms hang down too, and classifying them as legs
                # makes a biped read as a quadruped (and get a dog's gait retargeted onto it).
                tip_height = (self.joints[tip_j][self.up_axis] - ground) / skel_height
                if tip_height < 0.15:
                    if is_left:
                        self.left_leg_branches.append(branch_sorted)
                    else:
                        self.right_leg_branches.append(branch_sorted)
                # Backward/Horizontal pointing = Tail or Arm
                else:
                    fw_len = abs(dir_vec[self.fw_axis])
                    lr_len = abs(dir_vec[self.lr_axis])
                    # A tail runs along the mid-line, while a limb is displaced to one side.
                    # Direction alone is not enough: an arm swung forward in a running clip
                    # points the same way a tail does, and calling it a tail leaves that whole
                    # side of the target with no rotation tracks at all.
                    spread = max(sib_lr.values()) - min(sib_lr.values()) if len(sib_lr) >= 2 else 0.0
                    on_midline = abs(sib_lr[child] - mid_lr) <= 0.25 * spread
                    if fw_len > 1.5 * lr_len and on_midline:
                        self.tail_branches.append(branch_sorted)
                    else:
                        if is_left:
                            self.left_arm_branches.append(branch_sorted)
                        else:
                            self.right_arm_branches.append(branch_sorted)

        # Fallback collections
        self.left_legs = [j for b in self.left_leg_branches for j in b]
        self.right_legs = [j for b in self.right_leg_branches for j in b]
        self.left_arms = [j for b in self.left_arm_branches for j in b]
        self.right_arms = [j for b in self.right_arm_branches for j in b]

    def _get_sub_tree(self, root: int) -> List[int]:
        res = []
        stack = [root]
        while stack:
            curr = stack.pop()
            res.append(curr)
            stack.extend(self.children[curr])
        return res

# Anatomical name segments for a single limb branch, ordered top (closest to spine) to tip.
_LEG_SEGMENT_NAMES = ["UpLeg", "Leg", "Foot", "ToeBase"]
_ARM_SEGMENT_NAMES = ["Shoulder", "Arm", "ForeArm", "Hand"]
# Spine chain names for common chain lengths, ordered root -> head.
_SPINE_NAMES_BY_LENGTH = {
    1: ["Hips"],
    2: ["Hips", "Head"],
    3: ["Hips", "Spine", "Head"],
    4: ["Hips", "Spine", "Neck", "Head"],
    5: ["Hips", "Spine", "Spine1", "Neck", "Head"],
    6: ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"],
}


def _segment_names(count: int, base_names: List[str]) -> List[str]:
    """Returns `count` segment names, extending the last name with numeric suffixes
    (e.g. Foot, Foot1, Foot2...) if the branch is longer than the base list."""
    if count <= len(base_names):
        return base_names[:count]
    names = list(base_names)
    last = base_names[-1]
    extra = count - len(base_names)
    names.extend(f"{last}{i}" for i in range(1, extra + 1))
    return names


def _spine_names(count: int) -> List[str]:
    if count in _SPINE_NAMES_BY_LENGTH:
        return _SPINE_NAMES_BY_LENGTH[count]
    if count < 1:
        return []
    if count > 6:
        # Insert extra numbered Spine segments between Spine1 and Neck.
        base = _SPINE_NAMES_BY_LENGTH[6]
        extra = count - 6
        insert_at = base.index("Spine2") + 1
        extra_names = [f"Spine{2 + i}" for i in range(1, extra + 1)]
        return base[:insert_at] + extra_names + base[insert_at:]
    # Fewer than 6: drop from the middle-most optional names first (Spine2, Spine1, Neck).
    base = _SPINE_NAMES_BY_LENGTH[6]
    drop_order = ["Spine2", "Spine1", "Neck"]
    names = list(base)
    for name in drop_order:
        if len(names) <= count:
            break
        names.remove(name)
    return names


def assign_anatomical_names(classifier: "SkeletonClassifier") -> List[str]:
    """
    Assigns Mixamo-style anatomical joint names (prefixed `mixamorig:`) to every joint
    in a classified skeleton, based purely on structural role (spine chain, limb branch
    position) since UniRig's predicted joints carry no meaningful names of their own.

    Quadruped/multi-limb skeletons get Front/Hind leg-branch disambiguation (sorted by
    the forward axis), and any detected tail branches are named Tail, Tail1, ...
    Joints not covered by any recognized role keep a generic `mixamorig:Bone_<idx>` name.
    """
    names = [f"mixamorig:Bone_{i}" for i in range(classifier.J)]

    spine_names = _spine_names(len(classifier.spine_chain))
    for idx, joint in enumerate(classifier.spine_chain):
        if idx < len(spine_names):
            names[joint] = f"mixamorig:{spine_names[idx]}"

    def _branch_prefix(b_idx: int, total: int, side: str) -> str:
        """Two branches per side => Front/Hind (quadruped legs). More than two =>
        numbered suffix to keep every branch's names unique."""
        if total <= 1:
            return side
        if total == 2:
            return f"{side}Front" if b_idx == 0 else f"{side}Hind"
        suffix = "" if b_idx == 0 else str(b_idx)
        return f"{side}Limb{suffix}"

    def _name_leg_branches(branches: List[List[int]], side: str):
        branches_sorted = sorted(branches, key=lambda b: classifier.joints[b[0]][classifier.fw_axis], reverse=True)
        for b_idx, branch in enumerate(branches_sorted):
            prefix = _branch_prefix(b_idx, len(branches_sorted), side)
            seg_names = _segment_names(len(branch), _LEG_SEGMENT_NAMES)
            for seg_idx, joint in enumerate(branch):
                names[joint] = f"mixamorig:{prefix}{seg_names[seg_idx]}"

    def _name_arm_branches(branches: List[List[int]], side: str):
        for b_idx, branch in enumerate(branches):
            prefix = _branch_prefix(b_idx, len(branches), side)
            seg_names = _segment_names(len(branch), _ARM_SEGMENT_NAMES)
            for seg_idx, joint in enumerate(branch):
                names[joint] = f"mixamorig:{prefix}{seg_names[seg_idx]}"

    _name_leg_branches(classifier.left_leg_branches, "Left")
    _name_leg_branches(classifier.right_leg_branches, "Right")
    _name_arm_branches(classifier.left_arm_branches, "Left")
    _name_arm_branches(classifier.right_arm_branches, "Right")

    for b_idx, branch in enumerate(classifier.tail_branches):
        suffix = "" if b_idx == 0 else str(b_idx)
        for seg_idx, joint in enumerate(branch):
            tail_num = seg_idx if seg_idx > 0 else ""
            names[joint] = f"mixamorig:Tail{suffix}{tail_num}"

    return names

def generate_standard_animations(
    joints: np.ndarray,
    parents: List[Optional[int]],
    fps: int = 30
) -> Dict[str, Dict]:
    """
    Generates a full suite of procedural animations with alternating step gait:
    - Idle (breathing)
    - Walk (alternating leg swing with knee flexion)
    - Run (dynamic stride)
    - Wave (raising arm)
    - Dance (groove sway)
    """
    classifier = SkeletonClassifier(joints, parents)
    animations = {}
    
    # IDLE
    duration = 2.0
    num_frames = int(fps * duration) + 1
    times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    idle_tracks = []
    
    root_orig_trans = joints[classifier.root_idx]
    root_trans_vals = np.tile(root_orig_trans, (num_frames, 1))
    root_trans_vals[:, classifier.up_axis] += 0.015 * np.sin(2.0 * np.pi * times / duration)
    idle_tracks.append({
        "joint_idx": classifier.root_idx,
        "path": "translation",
        "times": times,
        "values": root_trans_vals
    })
    
    for s_idx in classifier.spine_chain:
        rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
        for f in range(num_frames):
            phase = 2.0 * np.pi * times[f] / duration
            rot_vals[f] = euler_to_quat(0.02 * np.sin(phase), 0.0, 0.01 * np.cos(phase))
        idle_tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})
        
    animations["Idle"] = {"duration": duration, "tracks": idle_tracks}
    
    # WALK ANIMATION (1.2s loop) - Alternating & Quadruped Diagonal Gait
    duration = 1.2
    num_frames = int(fps * duration) + 1
    times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    walk_tracks = []
    
    root_trans_vals = np.tile(root_orig_trans, (num_frames, 1))
    root_trans_vals[:, classifier.up_axis] += 0.03 * np.abs(np.sin(2.0 * np.pi * times / duration))
    walk_tracks.append({"joint_idx": classifier.root_idx, "path": "translation", "times": times, "values": root_trans_vals})
    
    # Sort leg branches along forward axis (front vs hind)
    left_branches_sorted = sorted(classifier.left_leg_branches, key=lambda b: joints[b[0]][classifier.fw_axis])
    right_branches_sorted = sorted(classifier.right_leg_branches, key=lambda b: joints[b[0]][classifier.fw_axis])

    # Left Legs
    for b_idx, branch in enumerate(left_branches_sorted):
        branch_phase_offset = 0.0 if b_idx == 0 else np.pi
        for idx, leg_j in enumerate(branch):
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                if idx == 0: # Hip/Thigh: main swing
                    rot_vals[f] = euler_to_quat(0.35 * np.sin(phase), 0.0, 0.0)
                elif idx == 1: # Knee: flex backwards when lifting
                    flex = -0.3 * np.maximum(0.0, np.sin(phase))
                    rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
                else: # Ankle/Foot
                    rot_vals[f] = euler_to_quat(-0.1 * np.sin(phase), 0.0, 0.0)
            walk_tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})
            
    # Right Legs (Opposite phase to left legs)
    for b_idx, branch in enumerate(right_branches_sorted):
        branch_phase_offset = np.pi if b_idx == 0 else 0.0
        for idx, leg_j in enumerate(branch):
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                if idx == 0: # Hip/Thigh
                    rot_vals[f] = euler_to_quat(0.35 * np.sin(phase), 0.0, 0.0)
                elif idx == 1: # Knee flex
                    flex = -0.3 * np.maximum(0.0, np.sin(phase))
                    rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
                else: # Ankle/Foot
                    rot_vals[f] = euler_to_quat(-0.1 * np.sin(phase), 0.0, 0.0)
            walk_tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})

    # Arms (Counter-phase to legs if bipedal)
    for branch in classifier.left_arm_branches:
        for idx, arm_j in enumerate(branch):
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration + np.pi
                rot_vals[f] = euler_to_quat(0.25 * np.sin(phase), 0.0, 0.05 * np.cos(phase))
            walk_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

    for branch in classifier.right_arm_branches:
        for idx, arm_j in enumerate(branch):
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration
                rot_vals[f] = euler_to_quat(0.25 * np.sin(phase), 0.0, -0.05 * np.cos(phase))
            walk_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})

    animations["Walk"] = {"duration": duration, "tracks": walk_tracks}
    
    # RUN ANIMATION (0.8s loop)
    duration = 0.8
    num_frames = int(fps * duration) + 1
    times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    run_tracks = []
    
    root_trans_vals = np.tile(root_orig_trans, (num_frames, 1))
    root_trans_vals[:, classifier.up_axis] += 0.06 * np.abs(np.sin(2.0 * np.pi * times / duration))
    run_tracks.append({"joint_idx": classifier.root_idx, "path": "translation", "times": times, "values": root_trans_vals})
    
    for s_idx in classifier.spine_chain:
        rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
        for f in range(num_frames):
            phase = 2.0 * np.pi * times[f] / duration
            rot_vals[f] = euler_to_quat(0.15 + 0.05 * np.sin(phase), 0.0, 0.03 * np.cos(phase))
        run_tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})
        
    for b_idx, branch in enumerate(left_branches_sorted):
        branch_phase_offset = 0.0 if b_idx == 0 else np.pi
        for idx, leg_j in enumerate(branch):
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                if idx == 0:
                    rot_vals[f] = euler_to_quat(0.6 * np.sin(phase), 0.0, 0.0)
                elif idx == 1:
                    flex = -0.5 * np.maximum(0.0, np.sin(phase))
                    rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
            run_tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})
            
    for b_idx, branch in enumerate(right_branches_sorted):
        branch_phase_offset = np.pi if b_idx == 0 else 0.0
        for idx, leg_j in enumerate(branch):
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration + branch_phase_offset
                if idx == 0:
                    rot_vals[f] = euler_to_quat(0.6 * np.sin(phase), 0.0, 0.0)
                elif idx == 1:
                    flex = -0.5 * np.maximum(0.0, np.sin(phase))
                    rot_vals[f] = euler_to_quat(flex, 0.0, 0.0)
            run_tracks.append({"joint_idx": leg_j, "path": "rotation", "times": times, "values": rot_vals})
            
    for branch in classifier.left_arm_branches:
        for arm_j in branch:
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration + np.pi
                rot_vals[f] = euler_to_quat(0.45 * np.sin(phase), 0.0, 0.1 * np.cos(phase))
            run_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})
            
    for branch in classifier.right_arm_branches:
        for arm_j in branch:
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration
                rot_vals[f] = euler_to_quat(0.45 * np.sin(phase), 0.0, -0.1 * np.cos(phase))
            run_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})
            
    animations["Run"] = {"duration": duration, "tracks": run_tracks}
    
    # WAVE
    duration = 2.0
    num_frames = int(fps * duration) + 1
    times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    wave_tracks = []
    target_arm_branches = classifier.right_arm_branches if classifier.right_arm_branches else classifier.left_arm_branches
    for branch in target_arm_branches:
        for arm_j in branch:
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                t = times[f]
                wave_angle = 0.4 * np.sin(4.0 * np.pi * t)
                rot_vals[f] = euler_to_quat(0.2, wave_angle, 0.9)
            wave_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})
            
    for s_idx in classifier.spine_chain:
        rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
        for f in range(num_frames):
            rot_vals[f] = euler_to_quat(0.05 * np.sin(2.0 * np.pi * times[f] / duration), 0.0, -0.05)
        wave_tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})
        
    animations["Wave"] = {"duration": duration, "tracks": wave_tracks}
    
    # DANCE
    duration = 2.0
    num_frames = int(fps * duration) + 1
    times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    dance_tracks = []
    
    root_trans_vals = np.tile(root_orig_trans, (num_frames, 1))
    root_trans_vals[:, classifier.up_axis] += 0.04 * np.sin(4.0 * np.pi * times / duration)
    root_trans_vals[:, classifier.lr_axis] += 0.05 * np.sin(2.0 * np.pi * times / duration)
    dance_tracks.append({"joint_idx": classifier.root_idx, "path": "translation", "times": times, "values": root_trans_vals})
    
    for s_idx in classifier.spine_chain:
        rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
        for f in range(num_frames):
            phase = 2.0 * np.pi * times[f] / duration
            rot_vals[f] = euler_to_quat(0.1 * np.cos(phase), 0.15 * np.sin(phase), 0.1 * np.sin(2*phase))
        dance_tracks.append({"joint_idx": s_idx, "path": "rotation", "times": times, "values": rot_vals})
        
    for branch in classifier.left_arm_branches:
        for arm_j in branch:
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration
                rot_vals[f] = euler_to_quat(0.4 * np.sin(phase), 0.2 * np.cos(phase), 0.5 + 0.2 * np.sin(2*phase))
            dance_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})
            
    for branch in classifier.right_arm_branches:
        for arm_j in branch:
            rot_vals = np.zeros((num_frames, 4), dtype=np.float32)
            for f in range(num_frames):
                phase = 2.0 * np.pi * times[f] / duration
                rot_vals[f] = euler_to_quat(-0.4 * np.sin(phase), -0.2 * np.cos(phase), -0.5 - 0.2 * np.sin(2*phase))
            dance_tracks.append({"joint_idx": arm_j, "path": "rotation", "times": times, "values": rot_vals})
            
    animations["Dance"] = {"duration": duration, "tracks": dance_tracks}
    return animations
    
    return animations
