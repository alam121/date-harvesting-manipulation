# ur10e_curobo/config.py
from dataclasses import dataclass, field
from typing import List
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

# ---------- QoS ----------
DEFAULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

# ---------- Static robot constants ----------
JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]

WORLD_CONFIG = {
    "cuboid": {
        "table": {"dims": [5.0, 5.0, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0]},
        "pole":  {"dims": [0.02, 0.02, 1.0], "pose": [0.0, -0.95, 0.5, 1, 0, 0, 0]},
    }
}

PLAN_CFG_DEFAULT = MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)

# ---------- Runtime parameters (override via ROS params / env) ----------
@dataclass
class Topics:
    traj_cmd: str = "/scaled_joint_trajectory_controller/joint_trajectory"
    goal_marker: str = "/goal_positions_marker"
    path_marker: str = "/robot_path_marker"
    joint_states: str = "/joint_states"
    marker_feedback: str = "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback"
    gripper_force: str = "/gripper/force"
    e_stop: str = "/emergency_stop"
    prog_running: str = "/io_and_status_controller/robot_program_running"
    external_goal: str = "/external_goal_pose"
    ee_point_out: str = "/datefruit_3d_point"

@dataclass
class JointsPreset:
    home: List[float] = field(default_factory=lambda: [-1.5034225622760218, -1.5057066318443795, 2.2289393583880823, 
                                                       -4.545150896111959, 4.573554992675781, 0.052810169756412506]

)
    dropoff: List[float] = field(default_factory=lambda: [-2.16858417192568, -1.3347657362567347,
        2.0885677337646484, -2.6394265333758753, 4.78283166885376, 0.013545919209718704])
    
    
    predropoff: List[float] = field(default_factory=lambda: [-1.5063117186175745, -1.512526014154293, 2.141698662434713, -4.593678136865133, 4.582080364227295, 0.047529518604278564]

)

@dataclass
class Planner:
    urdf_config: str = "ur10e.yml"
    interpolation_dt: float = 0.004
    speed_scale: float = 0.5
    base_dt: float = 0.02          # common base timestep (s)
    speed_home: float = 4.0        # for move_to_home_position
    speed_dropoff: float = 4.5     # for move_to_dropoff_position
    speed_predropoff: float = 2.5  # for pre-dropoff
    speed_approach: float = 0.5    # for approach motion
    speed_final: float = 0.5       # for precise grasp
    
    pre_droffoff_z_offset: float = -1.0  # m above dropoff
    pre_droffoff_y_offset: float = 1.5  # m back from dropoff

@dataclass
class Perception:
    enabled: bool = True
    weights: str = "yolov8n-seg.pt"
    img_size: int = 512
    conf_thres: float = 0.45
    cam_frame: str = "zed2_left_camera_frame"
    show_view: bool = False
    use_gpu: bool = False

@dataclass
class AppConfig:
    topics: Topics = field(default_factory=Topics)
    joints: JointsPreset = field(default_factory=JointsPreset)
    planner: Planner = field(default_factory=Planner)
    perception: Perception = field(default_factory=Perception)

    @staticmethod
    def from_env(cfg: "AppConfig") -> "AppConfig":
        """Optional env overrides (kept short—add what you need)."""
        p = cfg.perception
        p.weights   = os.getenv("UR10E_YOLO_WEIGHTS", p.weights)
        p.img_size  = int(os.getenv("UR10E_YOLO_IMGSZ", p.img_size))
        p.conf_thres= float(os.getenv("UR10E_YOLO_CONF", p.conf_thres))
        p.cam_frame = os.getenv("UR10E_CAM_FRAME", p.cam_frame)
        p.show_view = bool(int(os.getenv("UR10E_SHOW_VIEW", "1" if p.show_view else "0")))
        p.use_gpu   = os.getenv("UR10E_PERCEPTION_USE_GPU", "0") == "1"
        return cfg
