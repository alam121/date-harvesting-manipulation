# ruff: noqa
from .grasp_outcome_classifier import GraspOutcomeClassifier
from .delto_gripper_controller import DeltoGripperController
from . import grasp_visualizer as grasp_viz_mod

from ur_msgs.srv import SetIO
import time


# ============================================================
# INITIALIZATION
# ============================================================
def init_gripper(node, suction: bool = True):
    """
    Initialize the gripper and classifier.
    suction = True  → use suction + suction-closing finger profile
    suction = False → finger-only mode
    """

    node.gripper_controller = DeltoGripperController(node, suction=suction)

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
    node.visualizer = grasp_viz_mod.GraspVisualizer()
    node.visual_timer = node.create_timer(
        0.03,  # ~30 fps
        lambda: node.visualizer.draw()
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
