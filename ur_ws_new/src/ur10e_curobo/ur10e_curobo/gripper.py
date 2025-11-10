# ruff: noqa
from .grasp_outcome_classifier import GraspOutcomeClassifier
from .delto_gripper_controller import DeltoGripperController
from ur_msgs.srv import SetIO
import time

def init_gripper(node):
    node.gripper_controller = DeltoGripperController(node)
    node.gripper_closed = False
    node.slip_detection = False
    node.grab_miss = False
    node.weak_grab = False
    node.last_grasp_end_template = None
    node.classifier = GraspOutcomeClassifier(on_outcome=lambda o,e: on_grasp_outcome(node, o, e),
                                             dead_time_thresh_s=1.40, hold_time_s=0.5)


def control_gripper(node, action: str):
    act = action.upper()
    if act == "OPEN":
        #node.get_logger().info("🟡 Releasing suction before opening gripper...")
        activate_suction(node, False)  # Turn off suction before opening
        node.gripper_controller.open_gripper()
        node.gripper_closed = False
        node.slip_detection = False
        node.grab_miss = False
        node.classifier.start_opening()
        #node.get_logger().info("Gripper OPEN → slip detection paused")
    elif act == "CLOSE":
        #node.get_logger().info("🟢 Activating suction before grip...")
        #activate_suction(node, True)
        node.classifier.start_closing()
        node.gripper_controller.run_closure_loop()
        node.classifier.mark_close_done()
        node.gripper_closed = True
        #node.get_logger().info("Gripper CLOSED → slip detection active")
        
        #TO-DO Gripper close confirmation


def on_grasp_outcome(node, outcome: str, end: str):
    node.slip_detection = outcome == "SLIPPED"
    node.grab_miss = outcome == "NO_GRAB"
    node.weak_grab = (outcome == "GRABBED") and (end == "WEAK")
    node.last_grasp_end_template = end
    node.get_logger().info(f"[grasp] outcome={outcome} end={end} slip={node.slip_detection} miss={node.grab_miss} weak={node.weak_grab}")

def activate_suction(node, state: bool):
    """Turn suction valves ON/OFF using UR I/O."""
    req = SetIO.Request()
    req.fun = 1  # Digital output
    req.state = 1.0 if state else 0.0

    # Control multiple pins (0, 1, 3)
    for pin in [0, 1, 3]:
        req.pin = pin
        future = node.io_client.call_async(req)
        #node.get_logger().info(f"{'🟢 Activated' if state else '⚪ Deactivated'} suction pin {pin}")
