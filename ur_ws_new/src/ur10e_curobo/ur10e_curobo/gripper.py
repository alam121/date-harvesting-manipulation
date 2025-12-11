# ruff: noqa
from .grasp_outcome_classifier import GraspOutcomeClassifier
from .delto_gripper_controller import DeltoGripperController
from . import grasp_visualizer as grasp_viz_mod

from ur_msgs.srv import SetIO
import time


# ============================================================
# INITIALIZATION
# ============================================================
def init_gripper(node, suction: bool = None):
    """
    Initialize the gripper and classifier.
    suction = True  → use suction + suction-closing finger profile
    suction = False → finger-only mode
    suction = None  → use config value (default)
    """
    # Use config value if not explicitly specified
    if suction is None:
        suction = node.cfg.gripper.use_suction

    node.gripper_controller = DeltoGripperController(
        node,
        suction=suction,
        force_thresholds={
            0: node.cfg.gripper.force_threshold_left,
            1: node.cfg.gripper.force_threshold_center,
            2: node.cfg.gripper.force_threshold_right,
        },
        min_fingers_for_stop=node.cfg.gripper.min_fingers_for_stop,
        steps=node.cfg.gripper.closing_steps,
        step_delay=node.cfg.gripper.step_delay_s,
    )

    node.gripper_closed = False
    node.slip_detection = False
    node.grab_miss = False
    node.weak_grab = False
    node.last_grasp_end_template = None

    node.classifier = GraspOutcomeClassifier(
        on_outcome=lambda o, e: on_grasp_outcome(node, o, e),
        dead_time_thresh_s=1.40,
        hold_time_s=0.5
    )


# ============================================================
# MAIN CONTROL ENTRY
# ============================================================
def control_gripper(node, action: str):
    """
    Unified logic for OPEN / CLOSE based on selected mode.
    """

    act = action.upper()

    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------
    if act == "OPEN":

        # Only disable suction if suction-mode is active
        if node.gripper_controller.suction:
            activate_suction(node, False)

        node.gripper_controller.open_gripper()
        node.gripper_closed = False
        node.slip_detection = False
        node.grab_miss = False
        node.classifier.start_opening()
        return

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------
    elif act == "CLOSE":

        # Only enable suction if suction-mode is active
        if node.gripper_controller.suction:
            activate_suction(node, True)

        node.classifier.start_closing()
        node.gripper_controller.run_closure_loop()
        node.classifier.mark_close_done()
        node.gripper_closed = True
        return


# ============================================================
# CLASSIFIER OUTCOME CALLBACK
# ============================================================
def on_grasp_outcome(node, outcome: str, end: str):
    node.slip_detection = outcome == "SLIPPED"
    node.grab_miss = outcome == "NO_GRAB"
    node.weak_grab = (outcome == "GRABBED") and (end == "WEAK")
    node.last_grasp_end_template = end

    node.get_logger().info(
        f"[grasp] outcome={outcome} end={end} slip={node.slip_detection} "
        f"miss={node.grab_miss} weak={node.weak_grab}"
    )
    if hasattr(node, "visualizer"):
        node.visualizer.update_outcome(outcome)


# ============================================================
# SUCTION VALVE CONTROL (UR IO)
# ============================================================
def activate_suction(node, state: bool):
    """
    Turn suction valves ON/OFF using UR digital outputs.
    Only used when suction=True.
    """

    req = SetIO.Request()
    req.fun = 1   # digital output
    req.state = 1.0 if state else 0.0

    # Multiple solenoid pins (0,1,3)
    for pin in [0, 1, 3]:
        req.pin = pin
        _ = node.io_client.call_async(req)
