# ur10e_curobo/managers/state_manager.py
"""Robot state management for UR10e cuRobo node."""

import threading
from typing import Optional, List, Tuple, TYPE_CHECKING
from rclpy.node import Node
from sensor_msgs.msg import JointState as ROSJointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3Stamped
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
        self._current_joint_positions: Optional[List[float]] = None
        self._current_joint_velocities: Optional[List[float]] = None

        # TF
        self.tf_buffer: Buffer = Buffer()
        self.tf_listener: Optional[TransformListener] = None

        # Control flags
        self._running: bool = True
        self._stop_requested: bool = False
        self._robot_running: bool = False

        # TF status flags (for logging suppression)
        self.tf_warning_printed: bool = False
        self.tf_printed: bool = False

        # Path tracking for visualization
        self.path_points: List = []

        # Marker state (from RViz interactive marker)
        self.latest_marker_pose = None

        # Direction tracking (for pre-grasp bias)
        self.fruit_direction: Optional[Tuple[float, float, float]] = None

        # Branch gap detection (for 2-finger mode)
        self.fruit_between_branches: bool = False
        self.fruit_gap_angle: float = 0.0

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

        # Branch gap info for 2-finger mode
        self._node.create_subscription(
            Float32MultiArray,
            "/datefruit_gap_info",
            self._gap_info_cb,
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

    # ============ Callbacks ============

    def _joint_state_cb(self, msg: ROSJointState) -> None:
        """Update joint positions and velocities from /joint_states."""
        jm = dict(zip(msg.name, msg.position))
        vm = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}

        with self._state_lock:
            self._current_joint_positions = [
                jm[j] for j in self._config.joint_order if j in jm
            ]
            self._current_joint_velocities = [
                vm.get(j, 0.0) for j in self._config.joint_order
            ]

    def _robot_running_cb(self, msg: Bool) -> None:
        """Track robot program running state."""
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

    def _gap_info_cb(self, msg: Float32MultiArray) -> None:
        """Store branch gap detection results for 2-finger mode."""
        if len(msg.data) >= 2:
            self.fruit_between_branches = msg.data[0] > 0.5
            self.fruit_gap_angle = float(msg.data[1])

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
