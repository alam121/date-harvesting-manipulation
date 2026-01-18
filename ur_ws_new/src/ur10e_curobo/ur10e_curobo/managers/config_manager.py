# ur10e_curobo/managers/config_manager.py
"""Configuration management for UR10e cuRobo node."""

from typing import List, TYPE_CHECKING
from rclpy.node import Node
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
        self._node.get_logger().info("ConfigManager initialized")

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters with defaults from AppConfig."""
        # Planner params
        self._node.declare_parameter("planner.speed_scale", self.cfg.planner.speed_scale)
        self._node.declare_parameter("planner.urdf_config", self.cfg.planner.urdf_config)
        self._node.declare_parameter("planner.interpolation_dt", self.cfg.planner.interpolation_dt)
        self._node.declare_parameter("planner.pre_dropoff_z_offset", self.cfg.planner.pre_dropoff_z_offset)
        self._node.declare_parameter("planner.pre_dropoff_y_offset", self.cfg.planner.pre_dropoff_y_offset)

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
        self.cfg.joints.dropoff = list(self._node.get_parameter("joints.dropoff").value)
        self.cfg.joints.predropoff = list(self._node.get_parameter("joints.predropoff").value)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (UR10E_*)."""
        self.cfg = AppConfig.from_env(self.cfg)

    def _expose_shortcuts(self) -> None:
        """Expose commonly-used config values as direct attributes for convenience."""
        self.speed_scale = self.cfg.planner.speed_scale
        self.home_joints = self.cfg.joints.home
        self.dropoff_joints = self.cfg.joints.dropoff
        self.predropoff_joints = self.cfg.joints.predropoff
        self.yoffset = self.cfg.planner.pre_dropoff_y_offset
        self.zoffset = self.cfg.planner.pre_dropoff_z_offset
        self.cam_frame = self.cfg.perception.cam_frame

        self.traj_cmd_topic = self.cfg.topics.traj_cmd
        self.joint_states_topic = self.cfg.topics.joint_states
        self.goal_marker_topic = self.cfg.topics.goal_marker
        self.path_marker_topic = self.cfg.topics.path_marker
