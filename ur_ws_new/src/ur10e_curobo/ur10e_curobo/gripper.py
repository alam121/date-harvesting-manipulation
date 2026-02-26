# ruff: noqa
from .grasp_outcome_classifier import GraspOutcomeClassifier
from .delto_gripper_controller import DeltoGripperController
from . import grasp_visualizer as grasp_viz_mod

from ur_msgs.srv import SetIO
import time

# Gripper aperture constants (meters)
GRIPPER_MAX_APERTURE = 0.080   # 80mm total opening at full open
APERTURE_MARGIN = 0.025        # 15mm extra clearance beyond fruit diameter


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
def control_gripper(node, action: str, fruit_radius: float = None):
    """
    Unified logic for OPEN / CLOSE based on selected mode.
    fruit_radius: optional fruit radius in meters (from vision) for adaptive aperture.
    """

    act = action.upper()

    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------
    if act == "OPEN":

        # Only disable suction if suction-mode is active
        if node.gripper_controller.suction:
            activate_suction(node, False)

        if fruit_radius is not None and fruit_radius > 0:
            # Adaptive aperture: open only enough for this fruit
            desired_aperture = (fruit_radius * 2) + APERTURE_MARGIN
            open_alpha = max(0.0, 1.0 - desired_aperture / GRIPPER_MAX_APERTURE)
            node.get_logger().info(
                f"Adaptive gripper: radius={fruit_radius*1000:.0f}mm, "
                f"aperture={desired_aperture*1000:.0f}mm, alpha={open_alpha:.2f}")
            node.gripper_controller.open_gripper_to(open_alpha)
        else:
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
