"""The two live URDFs carry the gripper tip separately; keep them in step.

cuRobo reads robot_assets/ur_description/ur10e_curobo.urdf and plans
gripper_tip onto the goal. robot_state_publisher reads ur_macro.xacro and
publishes the TF that pollination and the markers measure against. Nothing
links the two files, so a tip change applied to one and not the other leaves
the planner and TF disagreeing -- the same class of silent drift that let the
cuRobo URDF hold a 257mm-wrong camera mount (feeb1eee8).
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
CUROBO_URDF = SRC / "ur10e_curobo/ur10e_curobo/robot_assets/ur_description/ur10e_curobo.urdf"
XACRO = SRC / "universal_robot/urdf/ur_macro.xacro"
CONFIG = SRC / "ur10e_curobo/ur10e_curobo/config.py"

# delto_base_link sits here in tool0, in both files.
TOOL0_TO_DELTO = 0.106
# The xacro references the start of the fingertip pads, cuRobo the modelled tip
# end. The gap predates the longer fingertips and is not yet reconciled against
# a tape measure; pin it so a change to one file alone fails here.
KNOWN_GAP_M = 0.015


def _tip_z(path: str, joint: str) -> float:
    text = Path(path).read_text()
    block = text[text.index(joint):]
    block = block[:block.index("</joint>")]
    xyz = re.search(r'xyz="([^"]+)"', block).group(1).split()
    return float(xyz[2])


def test_both_urdfs_moved_together():
    curobo = _tip_z(CUROBO_URDF, 'name="gripper_link_to_tip"')
    xacro = _tip_z(XACRO, 'gripper_link_to_tip"')
    assert abs((xacro - curobo) - KNOWN_GAP_M) < 1e-6, (
        f"tip offsets drifted: cuRobo {curobo} vs xacro {xacro}, "
        f"gap {xacro - curobo} != {KNOWN_GAP_M}")


def test_pollination_fallback_matches_the_xacro():
    """gripper_tip_offset_m substitutes for the tool0->gripper_tip TF lookup."""
    fallback = float(re.search(
        r"gripper_tip_offset_m: float = ([0-9.]+)", CONFIG.read_text()).group(1))
    expected = TOOL0_TO_DELTO + _tip_z(XACRO, 'gripper_link_to_tip"')
    assert abs(fallback - expected) < 1e-6, (
        f"gripper_tip_offset_m {fallback} should be {expected:.3f}")


if __name__ == "__main__":
    test_both_urdfs_moved_together()
    test_pollination_fallback_matches_the_xacro()
    print("ok")
