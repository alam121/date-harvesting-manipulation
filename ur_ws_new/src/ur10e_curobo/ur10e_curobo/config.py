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
        "pole":  {"dims": [0.02, 0.02, 1.0], "pose": [0.25, -1.0, 0.5, 1, 0, 0, 0]},
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
    home: List[float] = field(default_factory=lambda: [-1.5000560919391077, -1.6181756458678187, 2.1864479223834437, -4.4248088798918666, 4.572711944580078, 0.04942631721496582]


)
    dropoff: List[float] = field(default_factory=lambda: [-2.16858417192568, -1.3347657362567347,
        2.0885677337646484, -2.6394265333758753, 4.78283166885376, 0.013545919209718704])
    
    
    predropoff: List[float] = field(default_factory=lambda: [-1.5009062925921839, -1.7099877796568812, 2.0609028975116175, -4.172773023644918, 4.572351932525635, 0.05295269936323166]

)

@dataclass
class Planner:
    urdf_config: str = "ur10e.yml"
    interpolation_dt: float = 0.004
    speed_scale: float = 0.5
    base_dt: float = 0.02          # common base timestep (s)

    # === GLOBAL SPEED CONTROL ===
    # Increase this to make ALL motions faster (0.5 = half speed, 2.0 = double speed)
    global_speed_multiplier: float = 5.0

    # Motion-specific speed factors (multiplied by global_speed_multiplier)
    speed_home: float = 1.0        # for move_to_home_position
    speed_dropoff: float = 3.0    # for move_to_dropoff_position
    speed_predropoff: float = 0.1  # for pre-dropoff (slower)
    speed_approach: float = 0.5    # for approach motion
    speed_final: float = 0.5       # for precise grasp

    # === SMOOTHNESS PARAMETERS ===
    # Lower values = smoother but slower transitions
    max_joint_velocity: float = 1.5      # rad/s max velocity per joint
    max_joint_acceleration: float = 2.0  # rad/s^2 max acceleration
    ramp_points: int = 8                 # number of points for accel/decel ramps

    # === TRAJECTORY LIMITS ===
    # These control how fast trajectories can actually execute
    min_dt: float = 0.005                # minimum timestep (lower = faster, but may cause instability)
    max_dt: float = 0.05                 # maximum timestep
    max_traj_velocity: float = 0.8       # max velocity sent to UR controller (was hardcoded to 0.25)

    pre_dropoff_z_offset: float = -0.1  # m above dropoff
    pre_dropoff_y_offset: float = 0.25  # m back from dropoff

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
class Gripper:
    use_suction: bool = False
    min_fingers_for_stop: int = 2
    closing_steps: int = 10
    step_delay_s: float = 0.05

@dataclass
class AppConfig:
    topics: Topics = field(default_factory=Topics)
    joints: JointsPreset = field(default_factory=JointsPreset)
    planner: Planner = field(default_factory=Planner)
    perception: Perception = field(default_factory=Perception)
    gripper: Gripper = field(default_factory=Gripper)

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
