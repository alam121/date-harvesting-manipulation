# ur10e_curobo/managers/config_manager.py
"""Configuration management for UR10e cuRobo node."""

from typing import List, TYPE_CHECKING
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from ..config import AppConfig, DEFAULT_QOS, JOINT_ORDER

if TYPE_CHECKING:
    pass


class ConfigManager:
    """Handles all configuration loading, ROS parameter declaration, and validation."""

    def __init__(self, node: Node):
        self._node = node

        # Master config object
        self.cfg: AppConfig = AppConfig()

        # QoS profiles
        self.qos: QoSProfile = DEFAULT_QOS
        self.goal_qos: QoSProfile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Static robot constants
        self.joint_order: List[str] = list(JOINT_ORDER)

        # Shortcut attributes (exposed for backward compatibility)
        self.speed_scale: float = 0.5
        self.home_joints: List[float] = []
        self.home_left_joints: List[float] = []
        self.home_left_low_joints: List[float] = []
        self.home_right_low_joints: List[float] = []
        self.home_right_joints: List[float] = []
        self.dropoff_joints: List[float] = []
        self.predropoff_joints: List[float] = []
        self.yoffset: float = 0.0
        self.zoffset: float = 0.0
        self.cam_frame: str = ""

        # Topic names
        self.traj_cmd_topic: str = ""
        self.joint_states_topic: str = ""
        self.goal_marker_topic: str = ""
        self.path_marker_topic: str = ""

    def initialize(self) -> None:
        """Declare ROS parameters, load values, and apply env overrides."""
        self._declare_parameters()
        self._load_parameters()
        self._apply_env_overrides()
        self._expose_shortcuts()
        self._node.add_on_set_parameters_callback(self._on_parameter_change)
        self._node.get_logger().info("ConfigManager initialized")

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters with defaults from AppConfig."""
        # Planner params
        self._node.declare_parameter("planner.speed_scale", self.cfg.planner.speed_scale)
        self._node.declare_parameter("planner.urdf_config", self.cfg.planner.urdf_config)
        self._node.declare_parameter("planner.interpolation_dt", self.cfg.planner.interpolation_dt)
        self._node.declare_parameter("planner.pre_dropoff_z_offset", self.cfg.planner.pre_dropoff_z_offset)
        self._node.declare_parameter("planner.pre_dropoff_y_offset", self.cfg.planner.pre_dropoff_y_offset)

        # Planner speed & smoothness params
        self._node.declare_parameter("planner.base_dt", self.cfg.planner.base_dt)
        self._node.declare_parameter("planner.global_speed_multiplier", self.cfg.planner.global_speed_multiplier)
        self._node.declare_parameter("planner.speed_home", self.cfg.planner.speed_home)
        self._node.declare_parameter("planner.speed_dropoff", self.cfg.planner.speed_dropoff)
        self._node.declare_parameter("planner.speed_predropoff", self.cfg.planner.speed_predropoff)
        self._node.declare_parameter("planner.speed_approach", self.cfg.planner.speed_approach)
        self._node.declare_parameter("planner.speed_final", self.cfg.planner.speed_final)
        self._node.declare_parameter("planner.max_joint_velocity", self.cfg.planner.max_joint_velocity)
        self._node.declare_parameter("planner.max_joint_acceleration", self.cfg.planner.max_joint_acceleration)
        self._node.declare_parameter("planner.ramp_points", self.cfg.planner.ramp_points)
        self._node.declare_parameter("planner.min_dt", self.cfg.planner.min_dt)
        self._node.declare_parameter("planner.max_dt", self.cfg.planner.max_dt)
        self._node.declare_parameter("planner.max_traj_velocity", self.cfg.planner.max_traj_velocity)
        self._node.declare_parameter("planner.direct_branch_retry_min_dist", self.cfg.planner.direct_branch_retry_min_dist)
        self._node.declare_parameter("planner.direct_branch_retry_seeds", self.cfg.planner.direct_branch_retry_seeds)
        self._node.declare_parameter("planner.direct_final_cart_waypoints", self.cfg.planner.direct_final_cart_waypoints)
        self._node.declare_parameter("planner.direct_final_joint_fallback_max_delta_deg", self.cfg.planner.direct_final_joint_fallback_max_delta_deg)
        self._node.declare_parameter("planner.low_left_standoff_x", self.cfg.planner.low_left_standoff_x)
        self._node.declare_parameter("planner.low_left_standoff_y", self.cfg.planner.low_left_standoff_y)
        self._node.declare_parameter("planner.low_left_standoff_z", self.cfg.planner.low_left_standoff_z)
        self._node.declare_parameter("planner.low_right_standoff_x", self.cfg.planner.low_right_standoff_x)
        self._node.declare_parameter("planner.low_right_standoff_y", self.cfg.planner.low_right_standoff_y)
        self._node.declare_parameter("planner.low_right_standoff_z", self.cfg.planner.low_right_standoff_z)
        self._node.declare_parameter("planner.low_side_final_y_offset", self.cfg.planner.low_side_final_y_offset)
        self._node.declare_parameter("planner.low_left_final_z_offset", self.cfg.planner.low_left_final_z_offset)
        self._node.declare_parameter("planner.low_right_final_z_offset", self.cfg.planner.low_right_final_z_offset)
        self._node.declare_parameter("planner.low_side_final_front_tilt_deg", self.cfg.planner.low_side_final_front_tilt_deg)
        self._node.declare_parameter("planner.bunch_lower_center_band", self.cfg.planner.bunch_lower_center_band)
        self._node.declare_parameter("planner.final_overshoot_threshold", self.cfg.planner.final_overshoot_threshold)
        self._node.declare_parameter("planner.final_overshoot_max_backoff", self.cfg.planner.final_overshoot_max_backoff)
        self._node.declare_parameter("planner.debug_plan_preview", self.cfg.planner.debug_plan_preview)
        self._node.declare_parameter("planner.log_cycle_start", self.cfg.planner.log_cycle_start)
        self._node.declare_parameter("planner.log_phase_timings", self.cfg.planner.log_phase_timings)
        self._node.declare_parameter("planner.log_path_publish", self.cfg.planner.log_path_publish)
        self._node.declare_parameter("planner.log_gripper_force_profile", self.cfg.planner.log_gripper_force_profile)
        self._node.declare_parameter("planner.subscribe_goal_min_settle_s", self.cfg.planner.subscribe_goal_min_settle_s)
        self._node.declare_parameter("planner.subscribe_goal_max_wait_s", self.cfg.planner.subscribe_goal_max_wait_s)
        self._node.declare_parameter("planner.subscribe_goal_stable_tol", self.cfg.planner.subscribe_goal_stable_tol)
        self._node.declare_parameter("planner.subscribe_goal_median_window", self.cfg.planner.subscribe_goal_median_window)

        # Grasp learning params
        self._node.declare_parameter("grasp.learning_enabled", self.cfg.grasp.learning_enabled)
        self._node.declare_parameter("grasp.contact_delta_threshold", self.cfg.grasp.contact_delta_threshold)
        self._node.declare_parameter("grasp.late_step_margin", self.cfg.grasp.late_step_margin)
        self._node.declare_parameter("grasp.min_samples_to_learn", self.cfg.grasp.min_samples_to_learn)
        self._node.declare_parameter("grasp.learning_rate", self.cfg.grasp.learning_rate)
        self._node.declare_parameter("grasp.default_contact_step_threshold", self.cfg.grasp.default_contact_step_threshold)

        # Perception params
        self._node.declare_parameter("perception.enabled", self.cfg.perception.enabled)
        self._node.declare_parameter("perception.weights", self.cfg.perception.weights)
        self._node.declare_parameter("perception.img_size", self.cfg.perception.img_size)
        self._node.declare_parameter("perception.conf_thres", self.cfg.perception.conf_thres)
        self._node.declare_parameter("perception.cam_frame", self.cfg.perception.cam_frame)
        self._node.declare_parameter("perception.show_view", self.cfg.perception.show_view)
        self._node.declare_parameter("perception.use_gpu", self.cfg.perception.use_gpu)

        # Topic params
        self._node.declare_parameter("topics.joint_traj", self.cfg.topics.traj_cmd)
        self._node.declare_parameter("topics.goal_marker", self.cfg.topics.goal_marker)
        self._node.declare_parameter("topics.path_marker", self.cfg.topics.path_marker)
        self._node.declare_parameter("topics.joint_states", self.cfg.topics.joint_states)

        # Joint preset params
        self._node.declare_parameter("joints.home", self.cfg.joints.home)
        self._node.declare_parameter("joints.home_left", self.cfg.joints.home_left)
        self._node.declare_parameter("joints.home_left_low", self.cfg.joints.home_left_low)
        self._node.declare_parameter("joints.home_right", self.cfg.joints.home_right)
        self._node.declare_parameter("joints.home_right_low", self.cfg.joints.home_right_low)
        self._node.declare_parameter("joints.dropoff", self.cfg.joints.dropoff)
        self._node.declare_parameter("joints.predropoff", self.cfg.joints.predropoff)

    def _load_parameters(self) -> None:
        """Read ROS parameters into cfg object."""
        # Planner
        self.cfg.planner.speed_scale = self._node.get_parameter("planner.speed_scale").value
        self.cfg.planner.urdf_config = self._node.get_parameter("planner.urdf_config").value
        self.cfg.planner.interpolation_dt = float(self._node.get_parameter("planner.interpolation_dt").value)
        self.cfg.planner.pre_dropoff_z_offset = float(self._node.get_parameter("planner.pre_dropoff_z_offset").value)
        self.cfg.planner.pre_dropoff_y_offset = float(self._node.get_parameter("planner.pre_dropoff_y_offset").value)

        # Planner speed & smoothness
        self.cfg.planner.base_dt = float(self._node.get_parameter("planner.base_dt").value)
        self.cfg.planner.global_speed_multiplier = float(self._node.get_parameter("planner.global_speed_multiplier").value)
        self.cfg.planner.speed_home = float(self._node.get_parameter("planner.speed_home").value)
        self.cfg.planner.speed_dropoff = float(self._node.get_parameter("planner.speed_dropoff").value)
        self.cfg.planner.speed_predropoff = float(self._node.get_parameter("planner.speed_predropoff").value)
        self.cfg.planner.speed_approach = float(self._node.get_parameter("planner.speed_approach").value)
        self.cfg.planner.speed_final = float(self._node.get_parameter("planner.speed_final").value)
        self.cfg.planner.max_joint_velocity = float(self._node.get_parameter("planner.max_joint_velocity").value)
        self.cfg.planner.max_joint_acceleration = float(self._node.get_parameter("planner.max_joint_acceleration").value)
        self.cfg.planner.ramp_points = int(self._node.get_parameter("planner.ramp_points").value)
        self.cfg.planner.min_dt = float(self._node.get_parameter("planner.min_dt").value)
        self.cfg.planner.max_dt = float(self._node.get_parameter("planner.max_dt").value)
        self.cfg.planner.max_traj_velocity = float(self._node.get_parameter("planner.max_traj_velocity").value)
        self.cfg.planner.direct_branch_retry_min_dist = float(self._node.get_parameter("planner.direct_branch_retry_min_dist").value)
        self.cfg.planner.direct_branch_retry_seeds = int(self._node.get_parameter("planner.direct_branch_retry_seeds").value)
        self.cfg.planner.direct_final_cart_waypoints = int(self._node.get_parameter("planner.direct_final_cart_waypoints").value)
        self.cfg.planner.direct_final_joint_fallback_max_delta_deg = float(self._node.get_parameter("planner.direct_final_joint_fallback_max_delta_deg").value)
        self.cfg.planner.low_left_standoff_x = float(self._node.get_parameter("planner.low_left_standoff_x").value)
        self.cfg.planner.low_left_standoff_y = float(self._node.get_parameter("planner.low_left_standoff_y").value)
        self.cfg.planner.low_left_standoff_z = float(self._node.get_parameter("planner.low_left_standoff_z").value)
        self.cfg.planner.low_right_standoff_x = float(self._node.get_parameter("planner.low_right_standoff_x").value)
        self.cfg.planner.low_right_standoff_y = float(self._node.get_parameter("planner.low_right_standoff_y").value)
        self.cfg.planner.low_right_standoff_z = float(self._node.get_parameter("planner.low_right_standoff_z").value)
        self.cfg.planner.low_side_final_y_offset = float(self._node.get_parameter("planner.low_side_final_y_offset").value)
        self.cfg.planner.low_left_final_z_offset = float(self._node.get_parameter("planner.low_left_final_z_offset").value)
        self.cfg.planner.low_right_final_z_offset = float(self._node.get_parameter("planner.low_right_final_z_offset").value)
        self.cfg.planner.low_side_final_front_tilt_deg = float(self._node.get_parameter("planner.low_side_final_front_tilt_deg").value)
        self.cfg.planner.bunch_lower_center_band = float(self._node.get_parameter("planner.bunch_lower_center_band").value)
        self.cfg.planner.final_overshoot_threshold = float(self._node.get_parameter("planner.final_overshoot_threshold").value)
        self.cfg.planner.final_overshoot_max_backoff = float(self._node.get_parameter("planner.final_overshoot_max_backoff").value)
        self.cfg.planner.debug_plan_preview = bool(self._node.get_parameter("planner.debug_plan_preview").value)
        self.cfg.planner.log_cycle_start = bool(self._node.get_parameter("planner.log_cycle_start").value)
        self.cfg.planner.log_phase_timings = bool(self._node.get_parameter("planner.log_phase_timings").value)
        self.cfg.planner.log_path_publish = bool(self._node.get_parameter("planner.log_path_publish").value)
        self.cfg.planner.log_gripper_force_profile = bool(self._node.get_parameter("planner.log_gripper_force_profile").value)
        self.cfg.planner.subscribe_goal_min_settle_s = float(self._node.get_parameter("planner.subscribe_goal_min_settle_s").value)
        self.cfg.planner.subscribe_goal_max_wait_s = float(self._node.get_parameter("planner.subscribe_goal_max_wait_s").value)
        self.cfg.planner.subscribe_goal_stable_tol = float(self._node.get_parameter("planner.subscribe_goal_stable_tol").value)
        self.cfg.planner.subscribe_goal_median_window = int(self._node.get_parameter("planner.subscribe_goal_median_window").value)

        # Grasp learning
        self.cfg.grasp.learning_enabled = bool(self._node.get_parameter("grasp.learning_enabled").value)
        self.cfg.grasp.contact_delta_threshold = float(self._node.get_parameter("grasp.contact_delta_threshold").value)
        self.cfg.grasp.late_step_margin = int(self._node.get_parameter("grasp.late_step_margin").value)
        self.cfg.grasp.min_samples_to_learn = int(self._node.get_parameter("grasp.min_samples_to_learn").value)
        self.cfg.grasp.learning_rate = float(self._node.get_parameter("grasp.learning_rate").value)
        self.cfg.grasp.default_contact_step_threshold = int(self._node.get_parameter("grasp.default_contact_step_threshold").value)

        # Perception
        self.cfg.perception.enabled = bool(self._node.get_parameter("perception.enabled").value)
        self.cfg.perception.weights = self._node.get_parameter("perception.weights").value
        self.cfg.perception.img_size = int(self._node.get_parameter("perception.img_size").value)
        self.cfg.perception.conf_thres = float(self._node.get_parameter("perception.conf_thres").value)
        self.cfg.perception.cam_frame = self._node.get_parameter("perception.cam_frame").value
        self.cfg.perception.show_view = bool(self._node.get_parameter("perception.show_view").value)
        self.cfg.perception.use_gpu = bool(self._node.get_parameter("perception.use_gpu").value)

        # Topics
        self.cfg.topics.traj_cmd = self._node.get_parameter("topics.joint_traj").value
        self.cfg.topics.goal_marker = self._node.get_parameter("topics.goal_marker").value
        self.cfg.topics.path_marker = self._node.get_parameter("topics.path_marker").value
        self.cfg.topics.joint_states = self._node.get_parameter("topics.joint_states").value

        # Joints (arrays come back as tuples in some ROS versions)
        self.cfg.joints.home = list(self._node.get_parameter("joints.home").value)
        self.cfg.joints.home_left = list(self._node.get_parameter("joints.home_left").value)
        self.cfg.joints.home_left_low = list(self._node.get_parameter("joints.home_left_low").value)
        self.cfg.joints.home_right = list(self._node.get_parameter("joints.home_right").value)
        self.cfg.joints.home_right_low = list(self._node.get_parameter("joints.home_right_low").value)
        self.cfg.joints.dropoff = list(self._node.get_parameter("joints.dropoff").value)
        self.cfg.joints.predropoff = list(self._node.get_parameter("joints.predropoff").value)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (UR10E_*)."""
        self.cfg = AppConfig.from_env(self.cfg)

    def _expose_shortcuts(self) -> None:
        """Expose commonly-used config values as direct attributes for convenience."""
        self.speed_scale = self.cfg.planner.speed_scale
        self.home_joints = self.cfg.joints.home
        self.home_left_joints = self.cfg.joints.home_left
        self.home_left_low_joints = self.cfg.joints.home_left_low
        self.home_right_low_joints = self.cfg.joints.home_right_low
        self.home_right_joints = self.cfg.joints.home_right
        self.dropoff_joints = self.cfg.joints.dropoff
        self.predropoff_joints = self.cfg.joints.predropoff
        self.yoffset = self.cfg.planner.pre_dropoff_y_offset
        self.zoffset = self.cfg.planner.pre_dropoff_z_offset
        self.cam_frame = self.cfg.perception.cam_frame

        self.traj_cmd_topic = self.cfg.topics.traj_cmd
        self.joint_states_topic = self.cfg.topics.joint_states
        self.goal_marker_topic = self.cfg.topics.goal_marker
        self.path_marker_topic = self.cfg.topics.path_marker

    def set_home_joints(self, joints: List[float]) -> None:
        """Update active HOME joint preset for the running node."""
        home = [float(v) for v in joints]
        self.cfg.joints.home = home
        self.home_joints = self.cfg.joints.home
        try:
            self._node.set_parameters([
                Parameter("joints.home", Parameter.Type.DOUBLE_ARRAY, home)
            ])
        except Exception as exc:
            self._node.get_logger().warn(
                f"Updated HOME in memory, but failed to update ROS parameter joints.home: {exc}")

    def _on_parameter_change(self, params) -> SetParametersResult:
        """Handle runtime parameter changes via ros2 param set."""
        # Map ROS param names to (config_obj, attr_name, type_cast)
        param_map = {
            # Planner speed & smoothness
            "planner.base_dt": (self.cfg.planner, "base_dt", float),
            "planner.global_speed_multiplier": (self.cfg.planner, "global_speed_multiplier", float),
            "planner.speed_home": (self.cfg.planner, "speed_home", float),
            "planner.speed_dropoff": (self.cfg.planner, "speed_dropoff", float),
            "planner.speed_predropoff": (self.cfg.planner, "speed_predropoff", float),
            "planner.speed_approach": (self.cfg.planner, "speed_approach", float),
            "planner.speed_final": (self.cfg.planner, "speed_final", float),
            "planner.max_joint_velocity": (self.cfg.planner, "max_joint_velocity", float),
            "planner.max_joint_acceleration": (self.cfg.planner, "max_joint_acceleration", float),
            "planner.ramp_points": (self.cfg.planner, "ramp_points", int),
            "planner.min_dt": (self.cfg.planner, "min_dt", float),
            "planner.max_dt": (self.cfg.planner, "max_dt", float),
            "planner.max_traj_velocity": (self.cfg.planner, "max_traj_velocity", float),
            "planner.direct_branch_retry_min_dist": (self.cfg.planner, "direct_branch_retry_min_dist", float),
            "planner.direct_branch_retry_seeds": (self.cfg.planner, "direct_branch_retry_seeds", int),
            "planner.direct_final_cart_waypoints": (self.cfg.planner, "direct_final_cart_waypoints", int),
            "planner.direct_final_joint_fallback_max_delta_deg": (self.cfg.planner, "direct_final_joint_fallback_max_delta_deg", float),
            "planner.low_left_standoff_x": (self.cfg.planner, "low_left_standoff_x", float),
            "planner.low_left_standoff_y": (self.cfg.planner, "low_left_standoff_y", float),
            "planner.low_left_standoff_z": (self.cfg.planner, "low_left_standoff_z", float),
            "planner.low_right_standoff_x": (self.cfg.planner, "low_right_standoff_x", float),
            "planner.low_right_standoff_y": (self.cfg.planner, "low_right_standoff_y", float),
            "planner.low_right_standoff_z": (self.cfg.planner, "low_right_standoff_z", float),
            "planner.low_side_final_y_offset": (self.cfg.planner, "low_side_final_y_offset", float),
            "planner.low_left_final_z_offset": (self.cfg.planner, "low_left_final_z_offset", float),
            "planner.low_right_final_z_offset": (self.cfg.planner, "low_right_final_z_offset", float),
            "planner.low_side_final_front_tilt_deg": (self.cfg.planner, "low_side_final_front_tilt_deg", float),
            "planner.bunch_lower_center_band": (self.cfg.planner, "bunch_lower_center_band", float),
            "planner.final_overshoot_threshold": (self.cfg.planner, "final_overshoot_threshold", float),
            "planner.final_overshoot_max_backoff": (self.cfg.planner, "final_overshoot_max_backoff", float),
            "planner.debug_plan_preview": (self.cfg.planner, "debug_plan_preview", bool),
            "planner.log_cycle_start": (self.cfg.planner, "log_cycle_start", bool),
            "planner.log_phase_timings": (self.cfg.planner, "log_phase_timings", bool),
            "planner.log_path_publish": (self.cfg.planner, "log_path_publish", bool),
            "planner.log_gripper_force_profile": (self.cfg.planner, "log_gripper_force_profile", bool),
            "planner.subscribe_goal_min_settle_s": (self.cfg.planner, "subscribe_goal_min_settle_s", float),
            "planner.subscribe_goal_max_wait_s": (self.cfg.planner, "subscribe_goal_max_wait_s", float),
            "planner.subscribe_goal_stable_tol": (self.cfg.planner, "subscribe_goal_stable_tol", float),
            "planner.subscribe_goal_median_window": (self.cfg.planner, "subscribe_goal_median_window", int),
            "planner.speed_scale": (self.cfg.planner, "speed_scale", float),
            "planner.pre_dropoff_z_offset": (self.cfg.planner, "pre_dropoff_z_offset", float),
            "planner.pre_dropoff_y_offset": (self.cfg.planner, "pre_dropoff_y_offset", float),
            # Grasp learning
            "grasp.learning_enabled": (self.cfg.grasp, "learning_enabled", bool),
            "grasp.contact_delta_threshold": (self.cfg.grasp, "contact_delta_threshold", float),
            "grasp.late_step_margin": (self.cfg.grasp, "late_step_margin", int),
            "grasp.min_samples_to_learn": (self.cfg.grasp, "min_samples_to_learn", int),
            "grasp.learning_rate": (self.cfg.grasp, "learning_rate", float),
            "grasp.default_contact_step_threshold": (self.cfg.grasp, "default_contact_step_threshold", int),
        }
        for p in params:
            if p.name in param_map:
                obj, attr, cast = param_map[p.name]
                setattr(obj, attr, cast(p.value))
                self._node.get_logger().info(f"Parameter updated: {p.name} = {p.value}")
        return SetParametersResult(successful=True)
