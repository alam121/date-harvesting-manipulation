# ur10e_curobo/managers/state_manager.py
"""Robot state management for UR10e cuRobo node."""

import math
import time
import threading
from typing import Optional, List, Tuple, TYPE_CHECKING
from rclpy.node import Node
from sensor_msgs.msg import JointState as ROSJointState
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Vector3Stamped, PointStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import InteractiveMarkerFeedback
from tf2_ros import Buffer, TransformListener

if TYPE_CHECKING:
    from .config_manager import ConfigManager


class StateManager:
    """Manages robot state tracking, TF, and synchronization flags."""

    def __init__(self, node: Node, config: "ConfigManager"):
        self._node = node
        self._config = config

        # Joint state (thread-safe access)
        self._state_lock = threading.Lock()
        # (names, position indices, velocity indices) cache for the 500Hz
        # /joint_states callback; rebuilt only if the name list changes.
        self._js_index_cache = None
        self._current_joint_positions: Optional[List[float]] = None
        self._current_joint_velocities: Optional[List[float]] = None


        # TF
        self.tf_buffer: Buffer = Buffer()
        self.tf_listener: Optional[TransformListener] = None

        # Control flags
        self._running: bool = True
        self._stop_requested: bool = False
        # With fake hardware, the GPIO "program_running" state interface stays at its
        # initial value (0) forever, so the io_and_status_controller never publishes
        # /io_and_status_controller/robot_program_running. Default to True in that case
        # so wait_until_xyz() doesn't sit forever in the "Robot program OFF" branch.
        self._robot_running: bool = bool(
            self._config.cfg.planner.use_fake_hardware
        )

        # TF status flags (for logging suppression)
        self.tf_warning_printed: bool = False
        self.tf_printed: bool = False

        # Path tracking for visualization
        self.path_points: List = []

        # Marker state (from RViz interactive marker)
        self.latest_marker_pose = None

        # Direction tracking (for pre-grasp bias)
        self.fruit_direction: Optional[Tuple[float, float, float]] = None
        self.date_tip_point: Optional[Tuple[float, float, float]] = None
        self.date_tip_time: float = 0.0
        self.date_tip_stable: bool = False
        self._date_tip_history: List[Tuple[float, float, float]] = []

        # Branch gap detection (for 2-finger mode)
        self.fruit_between_branches: bool = False
        self.fruit_gap_angle: float = 0.0
        self.fruit_contact_roll_valid: bool = False
        self.fruit_contact_roll: float = 0.0
        self.fruit_contact_roll_score: float = 0.0
        self.fruit_contact_roll_stable: bool = False
        self.fruit_contact_roll_samples: int = 0
        self.fruit_contact_roll_spread_deg: float = float("inf")
        self._fruit_contact_roll_history: List[float] = []
        self.safe_grasp_candidate = None
        self.fruit_major_axis_angle: float = 0.0
        self.fruit_major_axis_confidence: float = 0.0
        self.fruit_major_axis_stable: bool = False
        self.fruit_major_axis_samples: int = 0
        self.fruit_major_axis_spread_deg: float = float("inf")
        self._fruit_major_axis_history: List[float] = []

        # Normalised image-space bounding-box centre of best fruit [cx_norm, cy_norm]
        # cx_norm: 0=left, 1=right  |  cy_norm: 0=top, 1=bottom
        self.fruit_image_norm: Optional[Tuple[float, float]] = None
        self.fruit_bunch_rel_x: Optional[float] = None
        self.fruit_bunch_rel_y: Optional[float] = None

        # Trunk position from vision (updated continuously)
        self._trunk_x: Optional[float] = None
        self._trunk_xyz: Optional[Tuple[float, float, float]] = None  # full XYZ in base_link

        # Plan preview confirmation (set by keyboard thread or GUI)
        self.plan_waiting: bool = False  # True when plan preview is awaiting user input
        self.plan_confirm_event = threading.Event()
        self.plan_confirmed: Optional[bool] = None  # True=execute, False=cancel

        # Timer reference for cleanup
        self._timer_wait_js = None

    def initialize(self) -> None:
        """Initialize TF listener and subscriptions."""
        # TF setup
        self.tf_listener = TransformListener(self.tf_buffer, self._node)

        # Joint state subscription
        self._node.create_subscription(
            ROSJointState,
            "/joint_states",
            self._joint_state_cb,
            10
        )

        # Robot running status
        self._node.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            self._robot_running_cb,
            10
        )

        # Emergency stop
        self._node.create_subscription(
            Bool,
            "/emergency_stop",
            self._stop_cb,
            10
        )

        # RViz interactive marker feedback
        self._node.create_subscription(
            InteractiveMarkerFeedback,
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback",
            self._marker_cb,
            10
        )

        # Direction for pre-grasp bias
        self._node.create_subscription(
            Vector3Stamped,
            "/datefruit_direction",
            self._direction_cb,
            10
        )
        self._node.create_subscription(
            PointStamped,
            "/datefruit_tip_point",
            self._date_tip_cb,
            10
        )

        # Branch gap info for 2-finger mode
        self._node.create_subscription(
            Float32MultiArray,
            "/datefruit_gap_info",
            self._gap_info_cb,
            10
        )

        # Evaluation-only ranked three-finger rotations from the vision node.
        # Forward these through the main motion-node logger so field runs need
        # only one captured log. This callback never changes motion state.
        self._node.create_subscription(
            String,
            "/vision/grasp_candidates",
            self._grasp_candidates_cb,
            10
        )
        self._node.create_subscription(
            Float32MultiArray,
            "/vision/grasp_candidate_selection",
            self._grasp_candidate_selection_cb,
            10
        )

        # Normalised image bbox centre from vision (for image-based classification)
        self._node.create_subscription(
            Float32MultiArray,
            "/fruit_image_bbox_norm",
            self._fruit_image_norm_cb,
            10
        )

        # Trunk position from vision (continuous updates)
        self._node.create_subscription(
            PointStamped,
            "/trunk_position",
            self._trunk_position_cb,
            10
        )

        # Timer to check for initial joint states
        self._timer_wait_js = self._node.create_timer(0.5, self._check_joint_states)

        self._node.get_logger().info("StateManager initialized")

    # ============ Properties (thread-safe) ============

    @property
    def current_joint_positions(self) -> Optional[List[float]]:
        with self._state_lock:
            return list(self._current_joint_positions) if self._current_joint_positions else None

    @current_joint_positions.setter
    def current_joint_positions(self, value: Optional[List[float]]):
        with self._state_lock:
            self._current_joint_positions = value

    @property
    def current_joint_velocities(self) -> Optional[List[float]]:
        with self._state_lock:
            return list(self._current_joint_velocities) if self._current_joint_velocities else None

    @current_joint_velocities.setter
    def current_joint_velocities(self, value: Optional[List[float]]):
        with self._state_lock:
            self._current_joint_velocities = value

    @property
    def running(self) -> bool:
        return self._running

    @running.setter
    def running(self, value: bool):
        self._running = value

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @stop_requested.setter
    def stop_requested(self, value: bool):
        self._stop_requested = value

    @property
    def robot_running(self) -> bool:
        return self._robot_running

    @robot_running.setter
    def robot_running(self, value: bool):
        self._robot_running = value

    @property
    def trunk_x(self) -> Optional[float]:
        return self._trunk_x

    @property
    def trunk_xyz(self) -> Optional[Tuple[float, float, float]]:
        return self._trunk_xyz

    # ============ Callbacks ============

    def _joint_state_cb(self, msg: ROSJointState) -> None:
        """Update joint positions and velocities from /joint_states.

        This is the hottest Python path in the node: the UR driver publishes at
        500Hz and the gripper publishes to this topic too. It used to build two
        dicts and two comprehensions per message, and all of that needs the GIL
        -- which is what starves the planning thread (measured: 25ms of solver
        CPU stretched over 230ms of wall clock). The joint name list is
        identical every message, so resolve the index mapping once and reuse it.

        Semantics are unchanged: positions cover only the joints actually
        present, in joint_order order; velocities always span joint_order,
        with 0.0 where the message does not supply one.
        """
        names = tuple(msg.name)
        cached = self._js_index_cache
        if cached is None or cached[0] != names:
            order = self._config.joint_order
            cached = (
                names,
                [names.index(j) for j in order if j in names],
                [names.index(j) if j in names else -1 for j in order],
            )
            self._js_index_cache = cached
        _, pos_idx, vel_idx = cached

        pos = msg.position
        vel = msg.velocity
        n_vel = len(vel)
        with self._state_lock:
            self._current_joint_positions = [pos[i] for i in pos_idx]
            self._current_joint_velocities = [
                vel[i] if 0 <= i < n_vel else 0.0 for i in vel_idx
            ]

    def _robot_running_cb(self, msg: Bool) -> None:
        """Track robot program running state."""
        if self._config.cfg.planner.use_fake_hardware:
            # Mock hardware has no UR program, so its GPIO controller reports
            # false even while the trajectory controller is active.
            self._robot_running = True
            return
        self._robot_running = msg.data
        if msg.data:
            self._node.get_logger().info("Robot program is running.")
        else:
            self._node.get_logger().info("Robot program is NOT running.")

    def _stop_cb(self, msg: Bool) -> None:
        """Handle emergency stop signal."""
        if not msg.data:
            return
        self._node.get_logger().warn("Emergency stop!")
        self._stop_requested = True
        # Latch a separate abort flag for batch runs. stop_requested is CONSUMED
        # by the first handler that sees it (plan_and_execute's _check_stop
        # clears both flags after halting), so by the time control returns to the
        # AUTO-HARVEST loop the flag reads False again and the loop would happily
        # discover the next goal and carry on -- an emergency stop would abort
        # only the current goal, not the run. This latch is cleared solely when a
        # new batch is deliberately started.
        self._node._auto_harvest_abort = True
        # Immediately publish stop trajectory to halt the robot
        try:
            from ..motions import publish_stop_trajectory
            publish_stop_trajectory(self._node)
        except Exception:
            pass

    def _marker_cb(self, msg: InteractiveMarkerFeedback) -> None:
        """Store last clicked RViz interactive marker pose."""
        self.latest_marker_pose = msg.pose

    def _direction_cb(self, msg: Vector3Stamped) -> None:
        """Store fruit direction for pre-grasp bias."""
        self.fruit_direction = (msg.vector.x, msg.vector.y, msg.vector.z)

    def _date_tip_cb(self, msg: PointStamped) -> None:
        point = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
        history = self._date_tip_history
        history.append(point)
        del history[:-5]
        mean = tuple(sum(p[i] for p in history) / len(history) for i in range(3))
        spread = max(
            math.dist(point_i, mean) for point_i in history) if history else float("inf")
        self.date_tip_point = mean
        self.date_tip_time = time.time()
        self.date_tip_stable = len(history) >= 3 and spread <= 0.012

    def _gap_info_cb(self, msg: Float32MultiArray) -> None:
        """Store branch-gap and vision-selected three-finger roll results."""
        if len(msg.data) >= 2:
            self.fruit_between_branches = msg.data[0] > 0.5
            self.fruit_gap_angle = float(msg.data[1])
        if len(msg.data) >= 5:
            self.fruit_contact_roll_valid = msg.data[2] > 0.5
            self.fruit_contact_roll_score = float(msg.data[4])
            if self.fruit_contact_roll_valid:
                raw_roll = float(msg.data[3])
                period = 2.0 * math.pi / 3.0  # three-finger symmetry
                history = self._fruit_contact_roll_history
                if history:
                    reference = sum(history) / len(history)
                    unwrapped = reference + (
                        (raw_roll - reference + period / 2.0) % period
                        - period / 2.0)
                    reset_jump = math.radians(float(getattr(
                        self._config.cfg.planner,
                        "low_center_tool_roll_reset_jump_deg", 25.0)))
                    if abs(unwrapped - reference) > reset_jump:
                        history.clear()
                        unwrapped = raw_roll
                else:
                    unwrapped = raw_roll
                history.append(unwrapped)
                stable_frames = max(2, int(getattr(
                    self._config.cfg.planner,
                    "low_center_tool_roll_stable_frames", 4)))
                del history[:-max(stable_frames + 3, 7)]
                recent = history[-stable_frames:]
                spread = max(recent) - min(recent) if len(recent) > 1 else float("inf")
                max_spread = math.radians(float(getattr(
                    self._config.cfg.planner,
                    "low_center_tool_roll_max_spread_deg", 8.0)))
                self.fruit_contact_roll_stable = (
                    len(recent) >= stable_frames and spread <= max_spread)
                self.fruit_contact_roll_samples = len(recent)
                self.fruit_contact_roll_spread_deg = math.degrees(spread)
                self.fruit_contact_roll = sum(recent) / len(recent)
            else:
                self._fruit_contact_roll_history.clear()
                self.fruit_contact_roll_stable = False
                self.fruit_contact_roll_samples = 0
                self.fruit_contact_roll_spread_deg = float("inf")
        else:
            self.fruit_contact_roll_valid = False
            self.fruit_contact_roll_score = 0.0
            self._fruit_contact_roll_history.clear()
            self.fruit_contact_roll_stable = False
            self.fruit_contact_roll_samples = 0
            self.fruit_contact_roll_spread_deg = float("inf")
        if len(msg.data) >= 7:
            self.fruit_major_axis_confidence = float(msg.data[6])
            raw_axis = float(msg.data[5])
            # Axial data repeats every pi. Unwrap each observation about the
            # recent estimate so +89/-89 degrees remain only 2 degrees apart.
            history = self._fruit_major_axis_history
            if history:
                reference = sum(history) / len(history)
                unwrapped = reference + (
                    (raw_axis - reference + math.pi / 2.0) % math.pi
                    - math.pi / 2.0)
            else:
                unwrapped = raw_axis
            history.append(unwrapped)
            stable_frames = max(2, int(getattr(
                self._config.cfg.planner,
                "approach_date_axis_stable_frames", 5)))
            del history[:-max(stable_frames + 3, 8)]
            recent = history[-stable_frames:]
            spread = max(recent) - min(recent) if len(recent) > 1 else float("inf")
            max_spread = math.radians(float(getattr(
                self._config.cfg.planner,
                "approach_date_axis_max_spread_deg", 8.0)))
            # Unwrapping exists so the MEAN is computed correctly across the
            # +/-90deg seam, but the unwrapped values were exported as-is and
            # nothing ever wrapped them back. They ratchet: each new sample is
            # unwrapped about the drifting mean, so the mean walks out of the
            # valid axial range and never returns. Observed in the field as
            # "axis=+209.7deg" -- meaningless for an undirected major axis --
            # which np.clip(angle, -45, +45) in the corridor code then saturated
            # to exactly +45deg, pinning the approach to the extreme corridor.
            # It read as stable because the drifted samples agree with EACH
            # OTHER, so the spread gate never fired.
            _mean_unwrapped = sum(recent) / len(recent)
            _mean_wrapped = (
                (_mean_unwrapped + math.pi / 2.0) % math.pi - math.pi / 2.0)
            # Re-anchor the stored history by the same whole-pi shift. Spread is
            # a difference so it is unchanged, but the values stay bounded
            # instead of growing without limit.
            _shift = _mean_wrapped - _mean_unwrapped
            if _shift:
                for _i in range(len(history)):
                    history[_i] += _shift
            self.fruit_major_axis_angle = _mean_wrapped
            self.fruit_major_axis_samples = len(recent)
            self.fruit_major_axis_spread_deg = math.degrees(spread)
            self.fruit_major_axis_stable = (
                len(recent) >= stable_frames and spread <= max_spread)
        else:
            self._fruit_major_axis_history.clear()
            self.fruit_major_axis_angle = 0.0
            self.fruit_major_axis_confidence = 0.0
            self.fruit_major_axis_stable = False
            self.fruit_major_axis_samples = 0
            self.fruit_major_axis_spread_deg = float("inf")

    def _grasp_candidates_cb(self, msg: String) -> None:
        """Mirror Phase-3 evaluation results into the main application log."""
        if msg.data:
            self._node.get_logger().debug(msg.data)
            mapping = list(getattr(
                self._config.cfg.planner, "grasp_visual_to_force_map", [0, 1, 2]))
            if len(mapping) == 3:
                self._node.get_logger().debug(
                    "[GRASP_CHANNEL_MAP] "
                    f"visual_F1->force{int(mapping[0]) + 1} "
                    f"visual_F2->force{int(mapping[1]) + 1} "
                    f"visual_F3->force{int(mapping[2]) + 1} "
                    "status=UNVALIDATED")

    def _grasp_candidate_selection_cb(self, msg: Float32MultiArray) -> None:
        """Store the frozen, validated Phase-4 candidate handoff."""
        if len(msg.data) < 9:
            return
        self.safe_grasp_candidate = {
            "available": bool(msg.data[0] > 0.5),
            "delta_deg": float(msg.data[1]),
            "psi_deg": float(msg.data[2]),
            "score": float(msg.data[3]),
            "minimum_score": float(msg.data[4]),
            "score_loss": float(msg.data[5]),
            "target_xyz": [float(v) for v in msg.data[6:9]],
        }

    def _fruit_image_norm_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            self.fruit_image_norm = (float(msg.data[0]), float(msg.data[1]))
        if len(msg.data) >= 3:
            rel_x = float(msg.data[2])
            self.fruit_bunch_rel_x = rel_x if 0.0 <= rel_x <= 1.0 else None
        if len(msg.data) >= 4:
            rel_y = float(msg.data[3])
            self.fruit_bunch_rel_y = rel_y if 0.0 <= rel_y <= 1.0 else None

    def _trunk_position_cb(self, msg: PointStamped) -> None:
        """Update trunk position from vision."""
        self._trunk_x = msg.point.x
        self._trunk_xyz = (msg.point.x, msg.point.y, msg.point.z)

    def _check_joint_states(self) -> None:
        """Check if initial joint states have been received."""
        if self._current_joint_positions is not None:
            self._node.get_logger().info("Initial joints received.")
            if self._timer_wait_js:
                self._node.destroy_timer(self._timer_wait_js)
                self._timer_wait_js = None

    # ============ Public Methods ============

    def request_stop(self) -> None:
        """Request motion stop (thread-safe)."""
        self._stop_requested = True

    def clear_stop(self) -> None:
        """Clear stop request (thread-safe)."""
        self._stop_requested = False

    def shutdown(self) -> None:
        """Clean shutdown of state manager."""
        self._running = False
        self._stop_requested = True
