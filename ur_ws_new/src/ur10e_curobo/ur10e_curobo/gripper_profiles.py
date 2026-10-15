import math
import os
import json
from pathlib import Path

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

# Postures measured from /gripper/joint_states on the second physical DG-3F-M
# unit, which has slightly different zero/alignment values from the other
# unit, so keep its complete 12-joint targets rather than applying a delta.
# Applied to the "lab" environment -- see default_environment_calibrations().
LAB_NORMAL_OPEN_POSITION = [
    0.4350, -0.1130, 2.0250, -0.3700,
    -1.1330, 0.0000, 1.6580, -0.0930,
    0.3770, 0.1710, 2.0320, -0.4400,
]

# Measured from /gripper/joint_states on 2026-09-08 with the hand confirmed
# fully and accurately closed (median of 997 samples over 5s, 0.10deg jitter).
# The previous values were captured before the fingers were fully shut, so the
# hand routinely closed 5-6deg PAST its own "closed" target -- which is what
# produced the "calibrated close not fully reached" warning on every grasp and
# made the position-based contact test meaningless.
LAB_NORMAL_CLOSED_POSITION = [
    0.3473, -0.0367, 1.9897, -0.0663,
    -1.0978, 0.0401, 2.0769, -0.0663,
    0.3944, 0.1047, 2.1101, -0.0419,
]

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

GRIPPER_CALIBRATION_PATH = Path(os.environ.get(
    "UR10E_GRIPPER_CALIBRATION_FILE",
    "~/.config/datepalm/gripper_calibration.json",
)).expanduser()


def _current_normal_open_default():
    position = NEW_GRIPPER_OPEN_POSITION.copy()
    open_extra = math.radians(NEW_GRIPPER_OPEN_EXTRA_DEG)
    for joints in NEW_FINGER_JOINT_INDICES.values():
        for j_idx in joints:
            position[j_idx] -= open_extra
    return position


def _current_normal_closed_default():
    position = NEW_GRIPPER_OPEN_POSITION.copy()
    for joints in NEW_FINGER_JOINT_INDICES.values():
        for j_idx in joints:
            position[j_idx] = NEW_GRIPPER_OPEN_POSITION[j_idx] + NEW_GRIPPER_CLOSE_DELTA
    return position


def default_environment_calibrations():
    """Return independent Lab/Outdoor copies of today's four calibrated poses."""
    poses = {
        "normal_open": _current_normal_open_default(),
        "normal_closed": _current_normal_closed_default(),
        "envelop_open": ENVELOP_GRIPPER_OPEN_POSITION.copy(),
        "envelop_closed": ENVELOP_GRIPPER_CLOSED_POSITION.copy(),
    }
    calibrations = {
        environment: {name: values.copy() for name, values in poses.items()}
        for environment in ("lab", "outdoor")
    }
    # The measured second-unit postures (LAB_NORMAL_*) apply to "lab", and
    # "outdoor" keeps the computed NEW_GRIPPER-derived default.
    calibrations["lab"]["normal_open"] = (
        LAB_NORMAL_OPEN_POSITION.copy())
    calibrations["lab"]["normal_closed"] = (
        LAB_NORMAL_CLOSED_POSITION.copy())
    return calibrations


def load_environment_calibration(environment=None):
    environment = str(environment or os.environ.get(
        "UR10E_ENVIRONMENT", "outdoor")).strip().lower()
    if environment not in ("lab", "outdoor"):
        environment = "outdoor"
    defaults = default_environment_calibrations()[environment]
    try:
        data = json.loads(GRIPPER_CALIBRATION_PATH.read_text())
        selected = data.get(environment, {})
    except (OSError, ValueError, TypeError):
        return defaults
    result = {}
    for name, fallback in defaults.items():
        values = selected.get(name)
        if (isinstance(values, list) and len(values) == 12
                and all(isinstance(value, (int, float))
                        and math.isfinite(float(value)) for value in values)):
            result[name] = [float(value) for value in values]
        else:
            result[name] = fallback
    return result


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

    # Preserve the legacy environment-variable trim when explicitly supplied;
    # otherwise use the environment-specific calibrated 12-joint posture.
    if "UR10E_NEW_GRIPPER_OPEN_EXTRA_DEG" in os.environ:
        position = NEW_GRIPPER_OPEN_POSITION.copy()
        open_extra = math.radians(new_gripper_open_extra_deg())
        for joints in NEW_FINGER_JOINT_INDICES.values():
            for j_idx in joints:
                position[j_idx] -= open_extra
        return position
    return load_environment_calibration()["normal_open"]


def make_closed_position(gripper_profile: str = "new"):
    profile = normalize_gripper_profile(gripper_profile)
    if profile == "old":
        return OLD_GRIPPER_CLOSED_POSITION.copy()

    return load_environment_calibration()["normal_closed"]


def make_envelop_open_position():
    return load_environment_calibration()["envelop_open"]


def make_envelop_closed_position():
    return load_environment_calibration()["envelop_closed"]
