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
    0.2510, -0.1270, 2.152266, -0.235734,
    -1.1900, 0.0100, 1.707266, -0.362734,
    0.3300, 0.2020, 1.986266, -0.278734,
]
NEW_GRIPPER_CLOSE_DELTA = 2.5673 - 2.1260
NEW_GRIPPER_OPEN_EXTRA_DEG = 7.0

# Measured physical postures for the enveloping three-finger grasp. These are
# full M1..M12 targets captured from /gripper/joint_states, rather than an
# alpha extrapolation of the normal curl posture.
ENVELOP_GRIPPER_OPEN_POSITION = [
    -1.2950343050, 0.1692969374, 2.2636920398, -0.8639379797,
    -1.2007865254, -0.0872664626, 1.8849555922, -0.3630284844,
    1.5742869853, 0.1308996939, 2.3561944902, -0.8918632478,
]

ENVELOP_GRIPPER_CLOSED_POSITION = [
    -1.3037609512, 0.0226892803, 2.4068090385, -0.5009094953,
    -1.2758356832, 0.1745329252, 1.5044738152, 0.6195918845,
    1.5830136316, -0.0785398163, 2.6145032195, -0.5829399702,
]

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


def normalize_gripper_profile(gripper_profile: str = "new") -> str:
    profile = str(gripper_profile or "new").strip().lower()
    return profile if profile in ("old", "new") else "new"


def finger_joint_indices_for_profile(gripper_profile: str = "new"):
    if normalize_gripper_profile(gripper_profile) == "new":
        return NEW_FINGER_JOINT_INDICES
    return OLD_FINGER_JOINT_INDICES


def make_open_position(gripper_profile: str = "new"):
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


def make_envelop_open_position():
    return ENVELOP_GRIPPER_OPEN_POSITION.copy()


def make_envelop_closed_position():
    return ENVELOP_GRIPPER_CLOSED_POSITION.copy()
