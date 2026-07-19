import math
import os

CALIBRATED_OPEN_POSITION = [
    -0.0942, -0.1500, 2.1260, -0.5062,
    -1.6318, 0.1309, 1.6953, -0.4887,
    0.3333, 0.2234, 2.1260, -0.4311,
]
CALIBRATED_CLOSE_DELTA = 2.5673 - 2.1260
NEW_GRIPPER_OPEN_EXTRA_DEG = 5.0

FINGER_JOINT_INDICES = {
    0: (2, 3),
    1: (6, 7),
    2: (10, 11),
}


def new_gripper_open_extra_deg() -> float:
    value = os.environ.get("UR10E_NEW_GRIPPER_OPEN_EXTRA_DEG")
    if value is None:
        return NEW_GRIPPER_OPEN_EXTRA_DEG
    try:
        return float(value)
    except ValueError:
        return NEW_GRIPPER_OPEN_EXTRA_DEG


def make_open_position(gripper_profile: str = "old"):
    position = CALIBRATED_OPEN_POSITION.copy()
    if gripper_profile == "new":
        open_extra = math.radians(new_gripper_open_extra_deg())
        for joints in FINGER_JOINT_INDICES.values():
            for j_idx in joints:
                position[j_idx] -= open_extra
    return position


def make_closed_position():
    position = CALIBRATED_OPEN_POSITION.copy()
    for joints in FINGER_JOINT_INDICES.values():
        for j_idx in joints:
            position[j_idx] = CALIBRATED_OPEN_POSITION[j_idx] + CALIBRATED_CLOSE_DELTA
    return position
