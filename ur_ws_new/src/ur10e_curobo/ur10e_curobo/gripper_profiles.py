import math
import os

OLD_GRIPPER_OPEN_POSITION = [
    -0.0942, -0.1500, 2.0660, -0.5062,
    -1.6318, 0.1309, 1.6753, -0.4887,
    0.3333, 0.2234, 2.0673, -0.4311,
]

OLD_GRIPPER_CLOSED_POSITION = OLD_GRIPPER_OPEN_POSITION.copy()
OLD_GRIPPER_CLOSED_POSITION[2] = 2.5660
OLD_GRIPPER_CLOSED_POSITION[6] = 2.3753
OLD_GRIPPER_CLOSED_POSITION[10] = 2.5673

NEW_GRIPPER_OPEN_POSITION = [
    -0.0942, -0.1500, 2.1260, -0.5062,
    -1.6318, 0.1309, 1.6953, -0.4887,
    0.3333, 0.2234, 2.1260, -0.4311,
]
NEW_GRIPPER_CLOSE_DELTA = 2.5673 - 2.1260
NEW_GRIPPER_OPEN_EXTRA_DEG = 5.0

OLD_FINGER_JOINT_INDICES = {
    0: (2,),
    1: (6,),
    2: (10,),
}

NEW_FINGER_JOINT_INDICES = {
    0: (2, 3),
    1: (6, 7),
    2: (10, 11),
}

# Backwards-compatible default used by DG-3F-M diagnostics.
CALIBRATED_OPEN_POSITION = NEW_GRIPPER_OPEN_POSITION
CALIBRATED_CLOSE_DELTA = NEW_GRIPPER_CLOSE_DELTA
FINGER_JOINT_INDICES = NEW_FINGER_JOINT_INDICES


def new_gripper_open_extra_deg() -> float:
    value = os.environ.get("UR10E_NEW_GRIPPER_OPEN_EXTRA_DEG")
    if value is None:
        return NEW_GRIPPER_OPEN_EXTRA_DEG
    try:
        return float(value)
    except ValueError:
        return NEW_GRIPPER_OPEN_EXTRA_DEG


def normalize_gripper_profile(gripper_profile: str = "old") -> str:
    profile = str(gripper_profile or "old").strip().lower()
    return profile if profile in ("old", "new") else "old"


def finger_joint_indices_for_profile(gripper_profile: str = "old"):
    if normalize_gripper_profile(gripper_profile) == "new":
        return NEW_FINGER_JOINT_INDICES
    return OLD_FINGER_JOINT_INDICES


def make_open_position(gripper_profile: str = "old"):
    profile = normalize_gripper_profile(gripper_profile)
    if profile == "old":
        return OLD_GRIPPER_OPEN_POSITION.copy()

    position = NEW_GRIPPER_OPEN_POSITION.copy()
    if profile == "new":
        open_extra = math.radians(new_gripper_open_extra_deg())
        for joints in NEW_FINGER_JOINT_INDICES.values():
            for j_idx in joints:
                position[j_idx] -= open_extra
    return position


def make_closed_position(gripper_profile: str = "new"):
    profile = normalize_gripper_profile(gripper_profile)
    if profile == "old":
        return OLD_GRIPPER_CLOSED_POSITION.copy()

    position = NEW_GRIPPER_OPEN_POSITION.copy()
    for joints in NEW_FINGER_JOINT_INDICES.values():
        for j_idx in joints:
            position[j_idx] = NEW_GRIPPER_OPEN_POSITION[j_idx] + NEW_GRIPPER_CLOSE_DELTA
    return position
