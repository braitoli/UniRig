import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        #
        # All three axes are candidates, not just up/forward. A near-flat model (span barely
        # a hundredth of a unit along Z) can still run its spine along X, the axis nominally
        # reserved for left-right -- choosing only between fw and up left walk_spine with no
        # axis the body actually extends along, so every branch scored close to zero and the
        # tie went to whichever kept the smallest left-right spread rather than the real spine.
        body_axis = int(np.argmax(span))

        # Judge a branch by how far it eventually reaches along the body axis, not by where its
        # first joint sits: a shoulder can sit closer to the mid-line than the neck does, and
        # scoring on the immediate joint alone then picks the whole front leg as the backbone.
        #
        # The penalty axis must differ from body_axis -- walking the spine along X and then
        # penalising branches by their spread along X as well double-counts the same axis and
        # cancels the score gap that was supposed to separate the spine from its limbs.
        penalty_axis = self.lr_axis if self.lr_axis != body_axis else self.fw_axis
        centre_lr = float(np.median(self.joints[:, penalty_axis]))

        def straight_reach(node: int, sign: int) -> float:
            """
            How far a branch extends along body_axis before it first forks -- i.e. while it
            is still a single joint chain, not yet a limb splaying into fingers or into a
            paired left/right continuation.

            Scoring on the branch's full subtree (its deepest point along body_axis, however
            it got there) let an arm with fingers outrank the real spine: a shoulder joint
            with a 7-segment clawed hand hanging off it can reach further than the actual neck
            does, purely because the hand happens to curl up near head height. Once a branch
            forks it stops being read as spine continuing -- everything past that fork is a
            limb's own business, so it does not extend the reach that decides here.
            """
            best = sign * self.joints[node][body_axis]
            curr = node
            while len(self.children[curr]) == 1:
                curr = self.children[curr][0]
                best = max(best, sign * self.joints[curr][body_axis])
            return best

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
                    reach = straight_reach(c, sign)
                    dist_lr = abs(self.joints[c][penalty_axis] - centre_lr)
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

        # Which axis is front-to-back and which is left-right is a per-character fact, not
        # the X/Z constant it started as. The spine chain just walked IS the body's own
        # forward-ish axis (that is what "reaches furthest along body_axis" means), so once
        # body_axis is settled, forward is body_axis and left-right is whichever of the
        # remaining two horizontal axes has real width. That covers not just a turtle facing
        # +X (forward is X, promoted from the old lr_axis) but a near-flat model whose spine
        # itself runs along X while up stays Y (forward is X, demoted from up_axis, with lr
        # staying whichever of {lr_axis, fw_axis} still has width) -- one case the old
        # fw/lr-only swap could not reach at all, because neither of its two candidates was
        # ever the axis the spine had actually chosen.
        #
        # Guarded the same way as before: the chain must be long enough to trust (a two-joint
        # stub says nothing about the body), and the axis being promoted to left-right must
        # have real width (several inputs are near-flat cutouts, a hundredth of a unit deep).
        # Up is NOT up for grabs. preprocess_mesh runs auto_orient_and_center_mesh over every
        # input, which stands the character along +Y and drops its feet onto Y=0, so Y is up
        # by construction. Letting this block hand "up" to whichever axis was left over
        # instead pointed it sideways on 4 of 12 cached characters -- and up is what
        # `ground`, `skel_height` and the whole leg-vs-arm test are measured against, so a
        # sideways up quietly turned real legs into arms and left those characters striding
        # with nothing. Only forward and left-right are in question here.
        if len(self.spine_chain) >= 3 and body_axis != self.up_axis:
            new_lr = next(a for a in (0, 1, 2) if a not in (body_axis, self.up_axis))
            wide_enough = span[new_lr] > 0.20 * span[body_axis]
            reach = self.joints[self.spine_chain[-1]] - self.joints[self.spine_chain[0]]
            spine_is_long = abs(float(reach[body_axis])) > 0.15 * (span[self.up_axis] + 1e-9)
            if wide_enough and spine_is_long:
                self.fw_axis, self.lr_axis = body_axis, new_lr

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
            # A sibling attached to the spine is not necessarily one limb. A hip with three
            # children used to hand a whole 19-joint subtree to _get_sub_tree as "one branch"
            # when it forked partway along into an arm, a hand, and a stray joint that
            # happened to reach up near head height -- and everything past that fork got
            # named as if it were still the same leg (a wrist joint titled "LeftUpLeg").
            # Splitting at every fork first, so each resulting chain is a single unbranched
            # path from the spine to one tip, is what makes "one branch = one limb" true.
            sib_chains = [chain for c in siblings for chain in self._limb_chains(c)]
            sib_lr = [float(np.mean(self.joints[chain, self.lr_axis])) for chain in sib_chains]
            mid_lr = (min(sib_lr) + max(sib_lr)) / 2.0 if len(sib_lr) >= 2 \
                else float(self.joints[spine_j][self.lr_axis])

            for branch, branch_lr in zip(sib_chains, sib_lr):
                if not branch:
                    continue

                # _limb_chains already hands back the branch in parent-chain order, spine
                # end first. Re-sorting it by joint height threw that away, and the order is
                # consumed positionally twice over: assign_anatomical_names walks it to lay
                # down Shoulder/Arm/ForeArm/Hand, and neural_pan_adapter pairs segment k of
                # this branch with segment k of the mocap source. An arm raised overhead
                # sorts tip-to-shoulder, so the hand got called the shoulder AND received
                # the shoulder's rotation curve.
                root_j = branch[0]
                tip_j = branch[-1]

                dir_vec = self.joints[tip_j] - self.joints[root_j]
                is_left = (branch_lr < mid_lr)
                
                # A limb is a leg when it reaches the floor, not merely when it points
                # downward -- a human's arms hang down too, and classifying them as legs
                # makes a biped read as a quadruped (and get a dog's gait retargeted onto it).
                #
                # Measured over the whole branch, not at its last joint: ordering the branch
                # by hierarchy means the chain can end on a toe that curls back up, and
                # reading only that joint's height called several real legs arms, leaving
                # those characters striding with nothing.
                tip_height = (float(self.joints[branch, self.up_axis].min()) - ground) / skel_height
                if tip_height < 0.15:
                    if is_left:
                        self.left_leg_branches.append(branch)
                    else:
                        self.right_leg_branches.append(branch)
                # Backward/Horizontal pointing = Tail or Arm
                else:
                    fw_len = abs(dir_vec[self.fw_axis])
                    lr_len = abs(dir_vec[self.lr_axis])
                    # A tail runs along the mid-line, while a limb is displaced to one side.
                    # Direction alone is not enough: an arm swung forward in a running clip
                    # points the same way a tail does, and calling it a tail leaves that whole
                    # side of the target with no rotation tracks at all.
                    spread = max(sib_lr) - min(sib_lr) if len(sib_lr) >= 2 else 0.0
                    on_midline = abs(branch_lr - mid_lr) <= 0.25 * spread
                    if fw_len > 1.5 * lr_len and on_midline:
                        self.tail_branches.append(branch)
                    else:
                        if is_left:
                            self.left_arm_branches.append(branch)
                        else:
                            self.right_arm_branches.append(branch)

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

    def _limb_chains(self, root: int) -> List[List[int]]:
        """
        Splits a spine-attached subtree into one chain per tip, cut at every fork.

        `root` up to its first fork is a shared stem -- one shoulder, say -- and belongs to
        every limb that grows out of it. Without splitting, a hand's fingers made the whole
        arm one "branch" reaching from the shoulder to a fingertip, and a shoulder with two
        children (a short stub limb and a full clawed hand) merged into a single 19-joint
        tangle whose far end -- a wrist -- ended up misread as a hip.

        The shared stem is attributed to only the LONGEST resulting chain -- the limb whose
        own fork is what the stem is really leading to -- and dropped from the others, which
        keep just their post-fork segment plus the single fork joint itself (so their role
        can still be judged by direction from a real starting point). A joint kept in every
        chain would otherwise get its anatomical name overwritten once per limb that shares
        it, landing on whichever limb happened to be classified last.
        """
        stem = [root]
        curr = root
        while len(self.children[curr]) == 1:
            curr = self.children[curr][0]
            stem.append(curr)
        if len(self.children[curr]) == 0:
            return [stem]
        forks = [self._limb_chains(c) for c in self.children[curr]]
        tails = [tail for group in forks for tail in group]
        longest = max(range(len(tails)), key=lambda i: len(tails[i]))
        return [
            (stem + tail) if i == longest else ([curr] + tail)
            for i, tail in enumerate(tails)
        ]

# The Mixamo skeleton is a closed set: one spine, one arm and one leg per side, plus
# fingers. A name outside it is not a Mixamo bone however plausible it reads, and stamping
# `mixamorig:` on an invented one ("RightLimb1Hand3", "Spine4") tells every downstream
# consumer to bind to a bone that does not exist. Whatever the predicted skeleton has spare
# is named under `aux:` instead, so a clip or engine matching by name skips it cleanly.
_MIXAMO_PREFIX = "mixamorig:"
_AUX_PREFIX = "aux:"
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
    """
    `count` slots along one limb: the Mixamo segments in order, then None for every joint
    past the last one. A limb longer than four joints used to keep counting (Hand1, Hand2,
    ToeBase3), which named bones Mixamo does not have.
    """
    return (list(base_names) + [None] * count)[:count]


def _spine_names(count: int) -> List[str]:
    """
    `count` slots along the spine, each a Mixamo name or None where Mixamo has no bone.

    A chain longer than Mixamo's six keeps the canonical names anchored at both ends --
    Hips/Spine/Spine1/Spine2 at the base and Neck/Head at the tip, so the head is still the
    head -- and leaves the surplus joints in the middle unnamed rather than numbering them
    onward into Spine3, Spine4, ... which no Mixamo clip can bind to.
    """
    if count < 1:
        return []
    if count in _SPINE_NAMES_BY_LENGTH:
        return list(_SPINE_NAMES_BY_LENGTH[count])
    base = _SPINE_NAMES_BY_LENGTH[6]
    return base[:4] + [None] * (count - 6) + base[4:]


def assign_anatomical_names(classifier: "SkeletonClassifier") -> List[str]:
    """
    Names every joint of a classified skeleton, since UniRig's predicted joints carry none
    of their own: the articulationxl class has no skeleton template in UniRig's Order, so
    src/data/order.py falls through to "bone_0", "bone_1", ... for all of them.

    Joints filling a Mixamo role get that bone's name under `mixamorig:`; everything else --
    tails, wings, surplus limbs and spine segments -- gets an `aux:` name carrying its joint
    index, which keeps it out of the way of anything binding by Mixamo name.

    A quadruped predicts four ground-reaching limbs and no arms. Its front pair takes the
    Mixamo arm bones and its hind pair the leg bones, which is what lets the existing clips
    drive it; front and hind are told apart by the direction the character faces, not by a
    raw coordinate, so a character modelled facing either way names them the same.
    """
    names = [f"{_AUX_PREFIX}Bone{i}" for i in range(classifier.J)]
    joints = classifier.joints

    spine = _spine_names(len(classifier.spine_chain))
    for slot, joint in enumerate(classifier.spine_chain):
        names[joint] = (f"{_MIXAMO_PREFIX}{spine[slot]}" if spine[slot]
                        else f"{_AUX_PREFIX}Spine{joint}")

    # SkeletonClassifier splits limbs by which side of the mid-line they sit on and calls the
    # smaller coordinate "left". That is a geometric convention, not anatomy -- and one
    # neural_pan_adapter depends on and reconciles its source against, so it is left alone.
    # Exported names have to be real anatomy, so the side is resolved here instead, from the
    # body's own left (up x forward). Without it a character modelled facing the other way
    # wears its left and right labels mirrored.
    forward = estimate_forward(joints, classifier)
    up = np.zeros(3, dtype=np.float32)
    up[classifier.up_axis] = 1.0
    left_dir = np.cross(up, forward) if forward is not None else None
    centre = 0.5 * (joints.max(axis=0) + joints.min(axis=0))

    def side_of(branch: List[int], fallback: str) -> str:
        if left_dir is None:
            return fallback
        offset = joints[branch].mean(axis=0) - centre
        return "Left" if float(np.dot(offset, left_dir)) > 0 else "Right"

    def forwardness(branch: List[int]) -> float:
        if forward is None:
            return float(joints[branch[0]][classifier.fw_axis])
        return float(np.dot(joints[branch[0]] - centre, forward))

    legs: Dict[str, List[List[int]]] = {"Left": [], "Right": []}
    arms: Dict[str, List[List[int]]] = {"Left": [], "Right": []}
    for bucket, source, fallback in (
        (legs, classifier.left_leg_branches, "Left"),
        (legs, classifier.right_leg_branches, "Right"),
        (arms, classifier.left_arm_branches, "Left"),
        (arms, classifier.right_arm_branches, "Right"),
    ):
        for branch in source:
            if branch:
                bucket[side_of(branch, fallback)].append(branch)

    def name_limb(branch: List[int], side: str, segments: List[str]):
        slots = _segment_names(len(branch), segments)
        for slot, joint in enumerate(branch):
            names[joint] = (f"{_MIXAMO_PREFIX}{side}{slots[slot]}" if slots[slot]
                            else f"{_AUX_PREFIX}{side}Limb{joint}")

    # Named from least to most authoritative. _limb_chains hands the joint at a fork to
    # every chain growing out of it, so whichever branch is named last owns that joint --
    # and a tail or a surplus limb winning one punched a hole through a real Mixamo chain
    # (LeftArm and LeftHand present, LeftShoulder overwritten by an aux name) on 12 of the
    # 51 cached skeletons, leaving a clip bound by name driving only part of the limb.
    for branch in classifier.tail_branches:
        for joint in branch:
            names[joint] = f"{_AUX_PREFIX}Tail{joint}"

    allocation = []
    for side in ("Left", "Right"):
        side_legs = sorted(legs[side], key=forwardness, reverse=True)  # front of body first
        side_arms = list(arms[side])
        if side_arms:
            arm = side_arms[0]
            leg = side_legs[-1] if side_legs else None      # the hindmost is the real leg
            spare = side_arms[1:] + side_legs[:-1]
        elif len(side_legs) >= 2:
            arm, leg = side_legs[0], side_legs[-1]          # quadruped: front pair -> arms
            spare = side_legs[1:-1]
        else:
            arm, leg = None, (side_legs[0] if side_legs else None)
            spare = []
        for branch in spare:
            for joint in branch:
                names[joint] = f"{_AUX_PREFIX}{side}Limb{joint}"
        allocation.append((side, arm, leg))

    for side, _, leg in allocation:
        if leg is not None:
            name_limb(leg, side, _LEG_SEGMENT_NAMES)
    for side, arm, _ in allocation:
        if arm is not None:
            name_limb(arm, side, _ARM_SEGMENT_NAMES)

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


# --------------------------------------------------------------------- named role resolution

# Mocap sources ship real bone names (Mixamo BVH, LaFAN1 preset clips), so their anatomical
# roles can be read straight off the names instead of being guessed from geometry. Names are
# strictly better wherever they exist: the geometric classifier reports both LaFAN1 presets
# with left and right swapped, and walks every Mixamo spine one joint too far, into a hair
# bone that never rotates at all.
_ROLE_KEYWORDS = (
    ("leg", ("upleg", "thigh", "shin", "calf", "knee", "ankle", "foot", "toe", "leg")),
    ("arm", ("shoulder", "clavicle", "forearm", "elbow", "wrist", "hand", "arm")),
    ("spine", ("hips", "pelvis", "spine", "chest", "torso", "neck", "head")),
)

# Accessory joints that borrow an anatomical word but carry no skeletal motion. Mixamo rigs
# really do contain "LeftPantLeg" (not a leg) and "LFootTongue" (not a foot).
_ROLE_EXCLUDE = (
    "pant", "sleeve", "tongue", "hair", "hat", "hood", "backpack", "strap", "cape",
    "skirt", "cloth", "prop", "weapon", "facial", "eye", "jaw", "teeth", "breast",
)
# Finger words only disqualify a joint when it also sits on a hand: "LeftHandMiddle1" is a
# finger, but a spine joint called "SpineMiddle" is not.
_FINGER_WORDS = ("thumb", "index", "middle", "ring", "pinky", "finger")


def _canonical_bone_name(name) -> str:
    """Lowercase alphanumerics of a bone name, minus any exporter namespace prefix."""
    raw = str(name).split(":")[-1]
    return "".join(ch for ch in raw.lower() if ch.isalnum())


def _joint_role(name) -> Optional[str]:
    """'leg', 'arm', 'spine', or None when the joint carries no locomotion role."""
    n = _canonical_bone_name(name)
    if any(bad in n for bad in _ROLE_EXCLUDE):
        return None
    if "hand" in n and any(f in n for f in _FINGER_WORDS):
        return None
    for role, keywords in _ROLE_KEYWORDS:
        if any(k in n for k in keywords):
            return role
    return None


def _joint_side(name) -> Optional[str]:
    """'left', 'right', or None. Checks the spelled-out word before the _L/_R shorthand."""
    low = str(name).split(":")[-1].lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    if low.endswith(("_l", ".l", "-l")):
        return "left"
    if low.endswith(("_r", ".r", "-r")):
        return "right"
    return None


def hierarchy_depths(parents: List[Optional[int]]) -> List[int]:
    depths = [0] * len(parents)
    for i in range(len(parents)):
        d, j, guard = 0, parents[i], 0
        while j is not None and guard < len(parents):
            d += 1
            j = parents[j]
            guard += 1
        depths[i] = d
    return depths


class NamedSkeletonRoles:
    """
    Anatomical roles resolved from bone names, exposing the same attribute surface as
    SkeletonClassifier so retargeting code can consume either interchangeably.
    """

    def __init__(
        self,
        root_idx: int,
        spine_chain: List[int],
        left_arm_branches: List[List[int]],
        right_arm_branches: List[List[int]],
        left_leg_branches: List[List[int]],
        right_leg_branches: List[List[int]],
    ):
        self.root_idx = root_idx
        self.spine_chain = spine_chain
        self.left_arm_branches = left_arm_branches
        self.right_arm_branches = right_arm_branches
        self.left_leg_branches = left_leg_branches
        self.right_leg_branches = right_leg_branches
        self.head_idx = spine_chain[-1] if spine_chain else root_idx
        self.tail_branches = []
        self.up_axis, self.lr_axis, self.fw_axis = 1, 0, 2


def resolve_named_roles(names, parents: List[Optional[int]]) -> Optional[NamedSkeletonRoles]:
    """
    Reads anatomical roles off bone names. Returns None when the names are generic
    (UniRig predicts "bone_0", "bone_1", ...), leaving geometric classification to
    SkeletonClassifier.
    """
    if names is None or len(names) != len(parents):
        return None

    depths = hierarchy_depths(parents)
    groups: Dict[str, List[int]] = {}
    for i, raw in enumerate(names):
        role = _joint_role(raw)
        if role is None:
            continue
        if role == "spine":
            key = "spine"
        else:
            side = _joint_side(raw)
            if side is None:
                continue
            key = side + "_" + role
        groups.setdefault(key, []).append(i)

    def chain(key: str) -> List[int]:
        """
        The group's deepest joint walked back up through its own members. Following the
        hierarchy rather than sorting by depth is what drops Mixamo's "Pelvis", a childless
        stub that hangs off Hips at the same depth as the real first spine joint.
        """
        members = set(groups.get(key, ()))
        if not members:
            return []
        j = max(members, key=lambda m: depths[m])
        out = []
        while j is not None and j in members:
            out.append(j)
            j = parents[j]
        out.reverse()
        return out

    spine = chain("spine")
    left_arm, right_arm = chain("left_arm"), chain("right_arm")
    left_leg, right_leg = chain("left_leg"), chain("right_leg")
    if len(spine) < 2 or not (left_arm and right_arm) or not (left_leg and right_leg):
        return None

    root_idx = next((i for i, p in enumerate(parents) if p is None), 0)
    return NamedSkeletonRoles(
        root_idx, spine, [left_arm], [right_arm], [left_leg], [right_leg]
    )


def estimate_forward(joints: np.ndarray, roles) -> Optional[np.ndarray]:
    """
    The horizontal direction the character faces, as a unit vector, or None when no cue
    is strong enough to tell.

    Three cues, strongest first:

    1. The spine reaching out horizontally. On a quadruped the head sticks out in front, so
       root-to-head is the body's own forward. An upright biped's spine is nearly vertical,
       so this cue stays quiet for one.
    2. The feet pointing ahead of the ankles -- only counted for a limb whose last segment
       lies flatter than it drops, since a leg predicted without a toe ends on the shin and
       its lean says nothing about facing.
    3. The knee bowing ahead of the straight hip-to-ankle line.

    Returning a vector rather than a +Z/-Z sign matters: nothing guarantees a character faces
    along Z at all. The one turtle in the test corpus faces +X, and treating world X as its
    left-right axis turned its nod into a sideways roll.
    """
    up = 1
    size = float(joints.max(axis=0)[up] - joints.min(axis=0)[up]) + 1e-9

    def horizontal(v: np.ndarray) -> np.ndarray:
        out = np.array([float(v[0]), 0.0, float(v[2])], dtype=np.float32)
        return out

    spine = list(getattr(roles, "spine_chain", []) or [])
    if len(spine) >= 2:
        reach = horizontal(joints[spine[-1]] - joints[spine[0]])
        if float(np.linalg.norm(reach)) > 0.15 * size:
            return reach / float(np.linalg.norm(reach))

    legs = list(roles.left_leg_branches) + list(roles.right_leg_branches)
    foot_cue = []
    for br in legs:
        if len(br) < 2:
            continue
        seg = joints[br[-1]] - joints[br[-2]]
        flat = horizontal(seg)
        if float(np.linalg.norm(flat)) > abs(float(seg[up])):
            foot_cue.append(flat)
    if foot_cue:
        mean = np.mean(foot_cue, axis=0)
        if float(np.linalg.norm(mean)) > 0.01 * size:
            return mean / float(np.linalg.norm(mean))

    knee_cue = [horizontal(joints[br[1]] - 0.5 * (joints[br[0]] + joints[br[-1]]))
                for br in legs if len(br) >= 3]
    if knee_cue:
        mean = np.mean(knee_cue, axis=0)
        if float(np.linalg.norm(mean)) > 0.01 * size:
            return mean / float(np.linalg.norm(mean))
    return None


def estimate_facing(joints: np.ndarray, roles) -> float:
    """
    Which way the skeleton faces along Z: +1.0 towards +Z, -1.0 towards -Z, 0.0 when
    undecidable or when it faces sideways along X. A thin reading of estimate_forward, kept
    for callers that only care about the Z sign.
    """
    forward = estimate_forward(joints, roles)
    if forward is None or abs(float(forward[2])) < 0.5:
        return 0.0
    return 1.0 if float(forward[2]) > 0 else -1.0
    height = float(joints.max(axis=0)[1] - joints.min(axis=0)[1]) + 1e-9
    eps = 0.01 * height

    # A limb's last segment only reports facing when it is a foot -- something that lies
    # flatter than it drops. Legs predicted with no toe joint end on the shin instead, and
    # reading a shin as a foot mistakes the leg's lean for the direction the character faces.
    foot_cue = []
    for br in legs:
        if len(br) < 2:
            continue
        seg = joints[br[-1]] - joints[br[-2]]
        if abs(float(seg[2])) > abs(float(seg[1])):
            foot_cue.append(float(seg[2]))
    if foot_cue and abs(float(np.mean(foot_cue))) > eps:
        return 1.0 if float(np.mean(foot_cue)) > 0 else -1.0

    knee_cue = [float(joints[br[1]][2] - 0.5 * (joints[br[0]][2] + joints[br[-1]][2]))
                for br in legs if len(br) >= 3]
    if knee_cue and abs(float(np.mean(knee_cue))) > eps:
        return 1.0 if float(np.mean(knee_cue)) > 0 else -1.0
    return 0.0


def _axis_quat(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Turn (T,) angles in radians about a unit axis into (T, 4) glTF [x, y, z, w] quats."""
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-9)
    q = np.zeros((len(angles), 4), dtype=np.float32)
    q[:, :3] = axis[None, :] * np.sin(angles * 0.5)[:, None]
    q[:, 3] = np.cos(angles * 0.5)
    return q


def generate_head_gesture_animations(
    joints: np.ndarray,
    parents: List[Optional[int]],
    fps: int = 30
) -> Dict[str, Dict]:
    """
    Nod and HeadShake, driven procedurally from the neck and head joints.

    Deliberately not mocap. Both are single-axis oscillations of one or two joints, where a
    generated curve is indistinguishable from a captured one and the amplitude stays
    adjustable. The Mixamo library has no head shake at all, and its only nodding clip
    ("Stroke Nodding.bvh") turns the head by 8.6 degrees -- too little to read on screen.
    """
    classifier = SkeletonClassifier(joints, parents)
    chain = classifier.spine_chain
    if not chain:
        return {}
    head = chain[-1]
    neck = chain[-2] if len(chain) >= 2 else None

    # A nod turns about the body's own left-right axis, which is up x forward -- NOT about
    # world X. Only a character that happens to face along Z has those coincide; the test
    # corpus contains a turtle facing +X, where rotating about world X rolls its head from
    # side to side instead of nodding it. Rotating by a positive angle about up x forward
    # always tips the head towards the front, so the dip needs no separate sign guess.
    forward = estimate_forward(joints, classifier)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if forward is None:
        # No cue for which way it faces. Nodding about world X is then a coin flip between a
        # nod and a roll, so pitch about the axis the body is widest across instead, which is
        # at least a plausible left-right axis for a character modelled facing the camera.
        span = joints.max(axis=0) - joints.min(axis=0)
        lateral = np.array([1.0, 0.0, 0.0], np.float32) if span[0] >= span[2] \
            else np.array([0.0, 0.0, 1.0], np.float32)
    else:
        lateral = np.cross(up, forward)

    duration = 1.6
    num_frames = int(fps * duration) + 1
    times = np.linspace(0.0, duration, num_frames, dtype=np.float32)
    phase = 2.0 * np.pi * times / duration

    # A nod dips and returns rather than swinging symmetrically either side of rest, so it is
    # a raised cosine; it also starts and ends at zero, so the clip loops without a pop.
    nod = np.radians(24.0) * 0.5 * (1.0 - np.cos(2.0 * phase))
    shake = np.radians(30.0) * np.sin(3.0 * phase)

    share_head = 0.6 if neck is not None else 1.0
    animations = {}
    for name, angles, axis in (("Nod", nod, lateral), ("HeadShake", shake, up)):
        tracks = []
        if neck is not None:
            tracks.append({"joint_idx": neck, "path": "rotation", "times": times,
                           "values": _axis_quat(axis, angles * 0.4)})
        tracks.append({"joint_idx": head, "path": "rotation", "times": times,
                       "values": _axis_quat(axis, angles * share_head)})
        animations[name] = {"duration": duration, "tracks": tracks}
    return animations


# --- Auto-blink ---------------------------------------------------------------------
# A blink is not symmetric. The closing phase is a ballistic snap of the orbicularis oculi
# lasting 30-50 ms; the opening phase is driven by the levator palpebrae working against
# gravity and the viscosity of the tear film, and takes 150-300 ms, decelerating into a
# creep over the last stretch. Closing three to six times faster than opening is the single
# clearest cue that separates a blink from a shutter, and an equal-time close and open --
# which is what this used to emit -- reads as mechanical no matter how good the geometry is.
_BLINK_CLOSE = 0.040
_BLINK_HOLD = 0.020
_BLINK_OPEN = 0.180

# The opening samples of (1 - s)^2, which drops fast and then tails off. Keyframes are
# LINEAR in glTF, so the curve has to be carried by where the keys are placed.
_OPEN_SAMPLES = (0.15, 0.30, 0.50, 0.75, 1.00)

# Both eyes blink together, but not to the millisecond. A small fixed lead on one eye is
# enough to break the symmetry that makes a synchronised pair look animatronic.
_EYE_LEAD = 0.015

# A resting person blinks 15-20 times a minute. The gaps are not uniform -- short gaps are
# common and long ones rare -- so they are drawn from an exponential with a floor rather
# than from a flat range, and blinks occasionally arrive in pairs.
_INTERVAL_MEAN = 3.0
_INTERVAL_FLOOR = 1.5
_DOUBLE_BLINK_CHANCE = 0.12
_DOUBLE_BLINK_GAP = 0.10


def _blink_keys() -> List[Tuple[float, float]]:
    """One blink as (time offset, weight) control points, starting and ending fully open."""
    keys = [(0.0, 0.0), (_BLINK_CLOSE, 1.0), (_BLINK_CLOSE + _BLINK_HOLD, 1.0)]
    base = _BLINK_CLOSE + _BLINK_HOLD
    for s in _OPEN_SAMPLES:
        keys.append((base + s * _BLINK_OPEN, float((1.0 - s) ** 2)))
    return keys


def _blink_starts(duration: float, rng: np.random.Generator) -> List[float]:
    """When each blink begins, over one loop of the clip."""
    span = _BLINK_CLOSE + _BLINK_HOLD + _BLINK_OPEN
    starts: List[float] = []
    t = _INTERVAL_FLOOR + float(rng.exponential(_INTERVAL_MEAN))
    # Stop a full blink short of the end so the clip loops without a lid caught half-shut
    # across the seam.
    while t + span < duration:
        starts.append(t)
        if rng.random() < _DOUBLE_BLINK_CHANCE and t + 2 * span + _DOUBLE_BLINK_GAP < duration:
            t += span + _DOUBLE_BLINK_GAP
            starts.append(t)
        t += span + _INTERVAL_FLOOR + float(rng.exponential(_INTERVAL_MEAN))
    if not starts:
        starts = [max((duration - span) / 2.0, 0.0)]
    return starts


def generate_blink_animation(
    morph_names: Sequence[str],
    duration: float = 12.0,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    An idle clip that blinks the character's eyes at irregular intervals.

    Emitted as a glTF `weights` track, so the blinking is baked into the file and plays in
    any viewer that supports morph-target animation -- no runtime code on the consuming
    side. Keyframes are placed only where the curve bends, so a clip costs a few dozen
    scalars rather than a frame rate's worth.

    The two eyes are offset by a few milliseconds and therefore bend at different times, so
    the track carries the union of both eyes' keyframes and samples each eye at all of
    them. Intervals come from a seeded generator, because baked assets that differ run to
    run cannot be diffed or cached.

    Returns None when the mesh has no blink morph targets, which is the normal outcome for
    a character whose eyes could not be located.
    """
    names = list(morph_names)
    columns = {side: [i for i, n in enumerate(names) if n == f"eyeBlink{side}"]
               for side in ("Left", "Right")}
    if not any(columns.values()):
        return None

    rng = np.random.default_rng(seed)
    starts = _blink_starts(duration, rng)
    keys = _blink_keys()
    leads = {"Left": 0.0, "Right": _EYE_LEAD}

    # Every bend of every eye's curve becomes a keyframe, and both eyes are sampled at all
    # of them -- a shared time array is all a glTF sampler has.
    times = {0.0, float(duration)}
    for side, lead in leads.items():
        if not columns[side]:
            continue
        for start in starts:
            for offset, _w in keys:
                t = start + lead + offset
                if 0.0 < t < duration:
                    times.add(float(t))
    times = np.array(sorted(times), dtype=np.float32)

    values = np.zeros((len(times), len(names)), dtype=np.float32)
    key_t = np.array([k[0] for k in keys], dtype=np.float64)
    key_w = np.array([k[1] for k in keys], dtype=np.float64)
    span = float(key_t[-1])
    for side, cols in columns.items():
        if not cols:
            continue
        curve = np.zeros(len(times), dtype=np.float32)
        for start in starts:
            rel = times - (start + leads[side])
            live = (rel >= 0.0) & (rel <= span)
            if not live.any():
                continue
            curve[live] = np.maximum(curve[live], np.interp(rel[live], key_t, key_w))
        for col in cols:
            values[:, col] = curve

    return {
        "duration": float(duration),
        "tracks": [{
            "path": "weights",
            "times": times,
            "values": values,
        }],
    }


# --- Eye expressions as playable clips ------------------------------------------------
# A morph target is a pose, not a performance. Handing an application `eyeSquintLeft = 0.85`
# tells it what the face should look like but nothing about how the face gets there, and a
# weight snapped straight to its value reads as a jump cut. Each expression is therefore
# also exported as its own clip, with an attack, a hold and a release.
#
# The timings are not one envelope reused. What separates these expressions from each other
# is largely how fast they arrive:
#
# * A gaze shift is a saccade -- ballistic, 30-100 ms, among the fastest movements the body
#   makes. Easing a glance in over a quarter of a second reads as a head turn, not a look.
# * A startle (eyes wide) is nearly as fast on the way in and slow on the way out.
# * A deliberate squint or frown is muscular and gradual at both ends.
#
# (attack, hold, release), all seconds.
_EXPRESSION_TIMING = {
    "gaze":    (0.07, 1.10, 0.09),
    "wink":    (0.06, 0.20, 0.14),
    "blink":   (0.05, 0.10, 0.18),
    "startle": (0.09, 0.70, 0.42),
    "muscle":  (0.20, 0.90, 0.36),
}
_EXPRESSION_KIND = {
    "wink_right": "wink", "wink_left": "wink", "blink": "blink",
    "squint": "muscle", "frown": "muscle", "wide": "startle",
    "look_up": "gaze", "look_left": "gaze", "look_right": "gaze",
}
# A pause after the release so a looping clip breathes instead of firing on repeat.
_EXPRESSION_TAIL = 0.6
# Attack eases out and release eases in, both sampled because glTF interpolates LINEARLY --
# the curve has to be carried by where the keys sit, not by an interpolation mode.
_ATTACK_SAMPLES = (0.35, 0.70, 1.00)
_RELEASE_SAMPLES = (0.30, 0.60, 1.00)


def _weights_clip(morph_names: Sequence[str],
                  keys: Sequence[Tuple[float, Dict[str, float]]]) -> Dict[str, Any]:
    """Packs (time, {morph: weight}) keyframes into a glTF `weights` track."""
    names = list(morph_names)
    index = {n: i for i, n in enumerate(names)}
    times = np.array([t for t, _ in keys], dtype=np.float32)
    values = np.zeros((len(keys), len(names)), dtype=np.float32)
    for row, (_t, weights) in enumerate(keys):
        for name, weight in weights.items():
            col = index.get(name)
            if col is not None:
                values[row, col] = float(weight)
    return {"duration": float(times[-1]),
            "tracks": [{"path": "weights", "times": times, "values": values}]}


def generate_expression_animations(morph_names: Sequence[str],
                                   presets: Optional[Dict[str, Dict[str, Any]]] = None,
                                   ) -> Dict[str, Dict[str, Any]]:
    """
    One playable clip per eye expression, named as the UI names them.

    A preset whose morph targets are all missing from this mesh produces no clip, rather
    than a clip that plays and does nothing: a character whose eyes were never located has
    no eye shapes to drive, and an empty entry in the animation list is worse than an
    absent one. A preset that is only partly available still gets a clip driving whatever
    it does have.
    """
    if presets is None:
        from .facial_blendshapes import EXPRESSION_PRESETS as presets

    names = list(morph_names)
    available = set(names)
    clips: Dict[str, Dict[str, Any]] = {}

    for key, preset in presets.items():
        target = {n: w for n, w in preset.get("weights", {}).items() if n in available}
        if not target:
            continue

        attack, hold, release = _EXPRESSION_TIMING[_EXPRESSION_KIND.get(key, "muscle")]
        keys: List[Tuple[float, Dict[str, float]]] = [(0.0, {})]
        for s in _ATTACK_SAMPLES:
            level = 1.0 - (1.0 - s) ** 2
            keys.append((s * attack, {n: w * level for n, w in target.items()}))
        keys.append((attack + hold, dict(target)))
        for s in _RELEASE_SAMPLES:
            level = (1.0 - s) ** 2
            keys.append((attack + hold + s * release,
                         {n: w * level for n, w in target.items()}))
        keys.append((attack + hold + release + _EXPRESSION_TAIL, {}))

        clips[preset.get("name", key)] = _weights_clip(names, keys)

    return clips
