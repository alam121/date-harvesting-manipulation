# ruff: noqa
from grasp_outcome_classifier import GraspOutcomeClassifier
from delto_gripper_controller import DeltoGripperController


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
        node.gripper_controller.open_gripper()
        node.gripper_closed = False
        node.slip_detection = False
        node.grab_miss = False
        node.classifier.start_opening()
        node.get_logger().info("Gripper OPEN → slip detection paused")
    elif act == "CLOSE":
        node.classifier.start_closing()
        node.gripper_controller.run_closure_loop()
        node.classifier.mark_close_done()
        node.gripper_closed = True
        node.get_logger().info("Gripper CLOSED → slip detection active")


def on_grasp_outcome(node, outcome: str, end: str):
    node.slip_detection = outcome == "SLIPPED"
    node.grab_miss = outcome == "NO_GRAB"
    node.weak_grab = (outcome == "GRABBED") and (end == "WEAK")
    node.last_grasp_end_template = end
    node.get_logger().info(f"[grasp] outcome={outcome} end={end} slip={node.slip_detection} miss={node.grab_miss} weak={node.weak_grab}")
