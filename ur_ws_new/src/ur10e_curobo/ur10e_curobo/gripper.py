# ruff: noqa
from .grasp_outcome_classifier import GraspOutcomeClassifier
from .delto_gripper_controller import DeltoGripperController

from ur_msgs.srv import SetIO
from sensor_msgs.msg import JointState
import os
from .gripper_profiles import (
    finger_joint_indices_for_profile,
    make_closed_position,
    make_open_position,
    new_gripper_open_extra_deg,
)

# Gripper aperture constants (meters)
GRIPPER_MAX_APERTURE = 0.070   # 80mm total opening at full open
APERTURE_MARGIN = 0.010        # 15mm extra clearance beyond fruit diameter


class FakeGripperController:
    """No-hardware stand-in used when the robot is running fake ros2_control."""

    joint_names = [
        "F1M1", "F1M2", "F1M3", "F1M4",
        "F2M1", "F2M2", "F2M3", "F2M4",
        "F3M1", "F3M2", "F3M3", "F3M4",
    ]

    def __init__(self, node, suction: bool = False, steps: int = 10, step_delay: float = 0.0):
        self.node = node
        self.fake = True
        # Keep suction off in fake mode so OPEN/CLOSE never touches UR IO.
        self.suction = False
        self.configured_suction = bool(suction)
        self.steps = int(steps)
        self.step_delay = float(step_delay)

        self.force_data = [2.25, 2.55, 3.15]
        self.baseline_force = self.force_data.copy()
        self.frozen_fingers = set()
        self.current_step = 0
        self.state = "IDLE"

        self.closure_stopped_early = False
        self.closure_step_stopped = self.steps
        self.closure_first_contact_step = self.steps
        self.closure_force_profile = []
        self.closure_deltas = [0.0, 0.0, 0.0]
        self.closed = False
        self.current_open_alpha = 0.0

        self.gripper_profile = os.environ.get("UR10E_GRIPPER_PROFILE", "new").strip().lower()
        if self.gripper_profile not in ("old", "new"):
            self.node.get_logger().warn(
                f"Unknown UR10E_GRIPPER_PROFILE={self.gripper_profile!r}; using new"
            )
            self.gripper_profile = "new"

        # Mirror DeltoGripperController's selected profile values so fake
        # mode/RViz and hardware mode show the same open/close pose.
        self.finger_joint_idx = {0: 2, 1: 6, 2: 10}
        self.finger_joint_indices = finger_joint_indices_for_profile(self.gripper_profile)
        self.open_position = make_open_position(self.gripper_profile)
        self.closed_position = make_closed_position(self.gripper_profile)
        self.current_position = self.open_position.copy()
        self.current_open_alpha = 0.0
        self.node.get_logger().info(
            f"Fake gripper profile: {self.gripper_profile} "
            f"(new open extra={new_gripper_open_extra_deg():.1f}deg)"
        )
        self.joint_state_pub = node.create_publisher(JointState, "/joint_states", 10)
        self.joint_state_timer = node.create_timer(0.2, self.publish_joint_state)
        self.publish_joint_state()

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(v) for v in self.current_position]
        self.joint_state_pub.publish(msg)

    def _feed_classifier(self):
        if hasattr(self.node, "classifier"):
            self.node.classifier.on_force(self.force_data[:3])

    def open_gripper(self):
        self.state = "OPENING"
        self.closed = False
        self.current_step = 0
        self.current_position = self.open_position.copy()
        self.force_data = [2.25, 2.55, 3.15]
        self.baseline_force = self.force_data.copy()
        self.frozen_fingers.clear()
        self.publish_joint_state()
        self._feed_classifier()
        self.state = "IDLE"
        self.node.get_logger().info("Fake gripper: OPEN")

    def open_gripper_to(self, alpha: float = 0.0):
        alpha = max(-0.10, min(1.0, float(alpha)))
        self.state = "OPENING"
        self.closed = alpha >= 0.95
        self.current_open_alpha = alpha
        self.current_step = int(round(alpha * self.steps))
        position = self.open_position.copy()
        for ch in (0, 1, 2):
            for j_idx in self.finger_joint_indices.get(ch, (self.finger_joint_idx[ch],)):
                position[j_idx] = (
                    (1.0 - alpha) * self.open_position[j_idx]
                    + alpha * self.closed_position[j_idx]
                )
        self.current_position = position
        self.force_data = [2.25, 2.55, 3.15]
        self.baseline_force = self.force_data.copy()
        self.frozen_fingers.clear()
        self.publish_joint_state()
        self._feed_classifier()
        self.state = "IDLE"
        self.node.get_logger().info(f"Fake gripper: OPEN alpha={alpha:.2f}")

    def run_closure_loop(self):
        self.state = "CLOSING"
        self.closed = True
        self.current_step = self.steps
        self.current_position = self.closed_position.copy()
        self.closure_stopped_early = False
        self.closure_step_stopped = self.steps
        self.closure_first_contact_step = self.steps
        self.closure_force_profile = [[0.3, 0.3, 0.0], [2.75, 2.95, 1.85]]
        self.closure_deltas = self.closure_force_profile[-1]
        self.force_data = [5.0, 5.5, 5.0]
        self.publish_joint_state()
        self._feed_classifier()
        self.state = "IDLE"
        self.node.get_logger().info("Fake gripper: CLOSE")


class DisabledGripperController:
    """No-op controller used when the launcher disables gripper hardware."""

    def __init__(self, node):
        self.node = node
        self.fake = True
        self.disabled = True
        self.suction = False
        self.steps = 0
        self.closure_stopped_early = False
        self.closure_step_stopped = -1
        self.closure_first_contact_step = -1
        self.closure_force_profile = []
        self.closure_deltas = [0.0, 0.0, 0.0]
        self.closed = False
        self.current_open_alpha = 0.0
        self.gripper_profile = os.environ.get("UR10E_GRIPPER_PROFILE", "new").strip().lower()
        self.state = "DISABLED"

    def open_gripper(self):
        self.closed = False
        self.current_open_alpha = 0.0
        self.node.get_logger().info("Gripper disabled: OPEN ignored")

    def open_gripper_to(self, alpha: float = 0.0):
        self.closed = False
        self.current_open_alpha = max(-0.10, min(1.0, float(alpha)))
        self.node.get_logger().info(f"Gripper disabled: OPEN alpha={alpha:.2f} ignored")

    def run_closure_loop(self):
        self.closed = True
        self.node.get_logger().info("Gripper disabled: CLOSE ignored")


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

    gripper_enabled = os.environ.get("UR10E_GRIPPER_ENABLED", "true").strip().lower()
    if gripper_enabled in ("0", "false", "no", "off"):
        node.gripper_controller = DisabledGripperController(node)
        node.get_logger().info(
            "Gripper disabled by UR10E_GRIPPER_ENABLED=false: no Delto topics or UR IO.")
    elif getattr(node.cfg.planner, "use_fake_hardware", False):
        node.gripper_controller = FakeGripperController(
            node,
            suction=suction,
            steps=node.cfg.gripper.closing_steps,
            step_delay=node.cfg.gripper.step_delay_s,
        )
        node.get_logger().info(
            "Fake hardware enabled: using fake gripper controller (no Delto topics or UR IO)."
        )
    else:
        node.gripper_controller = DeltoGripperController(
            node,
            suction=suction,
            min_fingers_for_stop=node.cfg.gripper.min_fingers_for_stop,
            steps=node.cfg.gripper.closing_steps,
            step_delay=node.cfg.gripper.step_delay_s,
        )
        node.gripper_controller.fake = False

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
    if getattr(node.gripper_controller, "disabled", False):
        if act == "OPEN":
            node.gripper_controller.open_gripper()
            node.gripper_closed = False
        elif act == "CLOSE":
            node.gripper_controller.run_closure_loop()
            node.gripper_closed = True
        return

    # --------------------------------------------------------
    # OPEN
    # --------------------------------------------------------
    if act == "OPEN":

        # Only disable suction if suction-mode is active
        if node.gripper_controller.suction and not getattr(node.gripper_controller, "fake", False):
            activate_suction(node, False)

        adaptive_aperture = bool(getattr(
            node.cfg.gripper, "adaptive_aperture_enabled", False))
        if adaptive_aperture and fruit_radius is not None and fruit_radius > 0:
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
        if node.gripper_controller.suction and not getattr(node.gripper_controller, "fake", False):
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
    if hasattr(node, "grasp_history"):
        node.grasp_history.append({"outcome": outcome, "end": end})
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
