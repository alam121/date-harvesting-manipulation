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

# Height threshold (m) that classifies targets as LOW vs MID/HIGH.
LOW_Z_THRESH = 0.94

# Lateral threshold (m) from trunk center to classify LEFT/RIGHT vs CENTER.
LATERAL_THRESH = 0.03

# ---------- Static Obstacles (single source of truth) ----------
# Define obstacles once here, used for both cuRobo planning and RViz visualization
STATIC_OBSTACLES = [
    {
        "name": "table",
        "type": "cuboid",
        "dims": [5.0, 5.0, 0.2],
        "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
        "color": (1.0, 0.0, 0.0, 1.0),  # Red
    },
    {
        "name": "trunk",
        "type": "cylinder",
        "radius": 0.02,    # 2cm radius = 4cm diameter trunk
        "height": 1.2,     # 1.2m visible trunk section
        "pose": [0.16, -1.00, 0.6, 1, 0, 0, 0],  # center at 0.6m height
        "color": (0.55, 0.27, 0.07, 1.0),  # Brown
    },
]

# Auto-generate WORLD_CONFIG for cuRobo from STATIC_OBSTACLES.
# cuRobo's OBB collision checker only reads "cuboid" — cylinders are not loaded.
# Cylinders are converted to their bounding cuboid (2r × 2r × h) for collision avoidance.
# RViz visualization uses the original cylinder type separately.
def _to_curobo_cuboid(obs: dict) -> dict:
    if obs.get("type") == "cylinder":
        d = obs["radius"] * 2
        return {"dims": [d, d, obs["height"]], "pose": obs["pose"]}
    return {"dims": obs["dims"], "pose": obs["pose"]}

WORLD_CONFIG = {
    "cuboid": {obs["name"]: _to_curobo_cuboid(obs) for obs in STATIC_OBSTACLES}
}

# Voxel grid configuration for depth-based obstacle avoidance
VOXEL_CONFIG = {
    "dims": [1.5, 1.5, 1.5],           # 1.5m cube workspace
    "pose": [0.3, -0.5, 0.8, 1, 0, 0, 0],  # Workspace center (x, y, z, qw, qx, qy, qz)
    "voxel_size": 0.02,                # 2cm resolution
    "max_esdf_distance": 0.3,          # Max distance to compute ESDF
    # Collision verification parameters
    "collision_safety_margin": 0.03,   # 3cm safety buffer around robot
    "collision_check_interval": 5,     # Check every Nth waypoint for speed
    "max_replan_attempts": 2,          # Max replans if collision detected
    "verify_before_execute": False,     # Enable/disable pre-execution verification
    "target_exclusion_radius": 0.08,   # 8cm radius around target to skip collision check
}

PLAN_CFG_DEFAULT = MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)

# For joint-space planning (plan_single_js): no graph search needed since start/goal joints
# are already known. Graph search resizes internal buffers which corrupts plan_single_js state.
PLAN_CFG_JS = MotionGenPlanConfig(
    max_attempts=20,
    enable_finetune_trajopt=True,
    enable_graph=False,
    enable_graph_attempt=None,
)

# Fallback for when finetune trajopt fails (FINETUNE_TRAJOPT_FAIL).
# Used as a second attempt in plan_execute_js — skips fine-tune smoothing
# but still produces a valid collision-free trajectory.
PLAN_CFG_JS_NO_FINETUNE = MotionGenPlanConfig(
    max_attempts=20,
    enable_finetune_trajopt=False,
    enable_graph=False,
    enable_graph_attempt=None,
)

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
    home: List[float] = field(default_factory=lambda: [4.941936492919922, -1.6112495861449183, 2.1890562216388147, 1.7179061609455566, 4.491253852844238, -0.09996301332582647]


)
    dropoff: List[float] = field(default_factory=lambda: [-2.362258497868673, -1.5946093998351039, 2.3843892256366175, -2.6575151882567347, 4.8416242599487305, -0.17230397859682256]
)
    
    
    predropoff: List[float] = field(default_factory=lambda: [-1.5009062925921839, -1.7099877796568812, 2.0609028975116175, -4.172773023644918, 4.572351932525635, 0.05295269936323166]
)
    # Side home positions for laterally distant fruits
    home_left: List[float] = field(default_factory=lambda:  [-0.7986648718463343, -1.370279149418213, 1.5869911352740687, -3.958400150338644, 3.832206964492798, -0.4182942549334925]


)
    home_right: List[float] = field(default_factory=lambda:  [-2.1769216696368616, -1.6072222195067347, 1.8494580427752894, -4.613555570641989, 5.513333320617676, 1.2812821865081787]


)
    home_right_low: List[float] = field(default_factory=lambda:  [-2.0225699583636683, -1.5956303081908167, 2.210489575062887, -4.641205211678976, 5.276980876922607, 0.710979700088501]

)
    home_left_low: List[float] = field(default_factory=lambda: [-0.9872930685626429, -1.384446458225586, 2.0082204977618616, -4.594993253747457, 4.218744277954102, -0.3455312887774866]

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
    speed_home: float = 0.08        # for move_to_home_position
    speed_dropoff: float = 0.15   # for move_to_dropoff_position
    speed_predropoff: float = 0.2  # for pre-dropoff reverse
    speed_approach: float = 1.0    # for approach motion
    speed_final: float = 0.15      # for precise grasp (slow & gentle)

    # === SMOOTHNESS PARAMETERS ===
    # Lower values = smoother but slower transitions
    max_joint_velocity: float = 1.5      # rad/s max velocity per joint
    max_joint_acceleration: float = 1.5  # rad/s^2 max acceleration (lower = less jerk at end)
    ramp_points: int = 15                # number of points for accel/decel ramps

    # === TRAJECTORY LIMITS ===
    # These control how fast trajectories can actually execute
    min_dt: float = 0.010                # minimum timestep — keep ≥0.010 to avoid UR joint velocity limit faults
    max_dt: float = 0.05                 # maximum timestep
    max_traj_velocity: float = 0.8       # max velocity sent to UR controller (was hardcoded to 0.25)

    # === DIRECT IK TUNING ===
    direct_branch_retry_min_dist: float = 0.15  # m; skip expensive branch search for close moves
    direct_branch_retry_seeds: int = 8          # extra perturbed IK seeds when branch retry is needed
    direct_final_cart_waypoints: int = 2        # intermediate Cartesian IK waypoints for FINAL only

    very_low_center_cy_thresh: float = 0.88     # image cy; force center-home handling for very low fruit


    low_center_approach_y_offset: float = 0.07
    low_center_approach_z_offset: float = -0.07


    very_low_center_approach_y_offset: float = 0.07
    very_low_center_approach_z_offset: float = -0.035

    mid_center_approach_y_offset: float = 0.10
    mid_center_approach_z_offset: float = -0.05
    bunch_edge_side_band: float = 0.20        # rel-x within outer 20% of bunch is forced LEFT/RIGHT
    bunch_lower_center_band: float = 0.80     # rel-y >= this is forced VERY LOW/CENTER

    low_left_standoff_x: float = 0.12           # m; left side-low lateral standoff
    low_left_standoff_y: float = 0.065          # m; left side-low back standoff
    low_left_standoff_z: float = -0.055         # m; left side-low vertical standoff

    low_right_standoff_x: float = 0.08          # m; right side-low lateral standoff
    low_right_standoff_y: float = 0.050         # m; right side-low back standoff
    low_right_standoff_z: float = -0.025        # m; avoid large upward push on right-side final

    low_side_final_y_offset: float = 0.0        # m; keep side-low final motion lateral
    low_left_final_z_offset: float = 0.020      # m; left side-low gripper center offset
    low_right_final_z_offset: float = 0.010     # m; right side-low gripper center offset
    low_side_final_front_tilt_deg: float = 10.0 # max final +Z/front tilt toward fruit

    mid_center_approach_pitch_deg: float = 0.0 # local tool X pitch for MID/HIGH center; keep 0.0 to preserve approach→final orientation continuity

    low_center_final_y_offset: float = -0.02    # m; small pull toward camera for low-center final
    low_center_final_z_offset: float = 0.025    # m; gripper center offset above low-center fruit

    mid_center_final_y_offset: float = 0.008    # m; small pull toward camera/robot for MID/HIGH center final
    mid_center_final_z_offset: float = 0.030    # m; gripper center offset above MID/HIGH center fruit
    mid_center_slip_final_z_offset: float = 0.020 # m; slightly lower final target during slip retry

    final_overshoot_threshold: float = 0.004    # m; correct only if TCP passes target by >4mm
    final_overshoot_max_backoff: float = 0.012  # m; max one-shot pullback before closing
    reverse_initial_wait: float = 0.15          # s; minimum wait after publishing partial reverse
    reverse_final_settle: float = 0.05          # s; settle after reverse stops before hold check
    hold_check_settle_s: float = 0.05           # s; force settle before post-reverse hold samples
    hold_check_window_s: float = 0.30           # s; median force sample window after reverse
    hold_check_sample_dt: float = 0.04          # s; post-reverse force sample period

    pre_dropoff_z_offset: float = -0.1  # m above dropoff
    pre_dropoff_y_offset: float = 0.25  # m back from dropoff

    # === REACQUIRE ===
    reacquire_after_approach: bool = True   # re-detect fruit position after reaching approach standoff
    slip_check_reacquire: bool = False      # query depth after grasp to detect fruit slip
    regrip_after_slip: bool = False         # attempt regrip correction when grip is weak after slip

    # === DEBUG ===
    debug_plan_preview: bool = True    # show full plan and wait for confirmation before executing
    log_cycle_start: bool = False      # verbose per-goal start/decision metadata
    log_phase_timings: bool = False    # per-phase timing lines; cycle summary always includes timings
    log_path_publish: bool = False     # RViz path marker publish messages
    log_gripper_force_profile: bool = False  # full per-step closure force profile

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
class Grasp:
    learning_enabled: bool = False
    contact_delta_threshold: float = 0.5
    late_step_margin: int = 2
    min_samples_to_learn: int = 3
    learning_rate: float = 0.3
    default_contact_step_threshold: int = 7

@dataclass
class LidarScan:
    # Directory where bag files are saved (~ is expanded)
    bag_dir: str = "~/lidar_scans"
    # Speed factor multiplied into plan_execute_js (lower = slower = denser scan)
    speed_factor: float = 0.08
    # Joint-space waypoints defining the half-circle arc around the tree.
    # Default: home_left → home → home_right  (calibrate to your setup)
    scan_waypoints: List[List[float]] = field(default_factory=lambda: [
        # home_left — left side of tree
        [-0.7986648718463343, -1.370279149418213, 1.5869911352740687,
         -3.958400150338644, 3.832206964492798, -0.4182942549334925],
        # home — front center of tree
        [4.985577583312988, -1.6374036274352015, 2.2965741793261927,
         1.622551603908203, 4.465203285217285, -0.12867910066713506],
        # home_right — right side of tree
        [-2.1769216696368616, -1.6072222195067347, 1.8494580427752894,
         -4.613555570641989, 5.513333320617676, 1.2812821865081787],
    ])

@dataclass
class AppConfig:
    topics: Topics = field(default_factory=Topics)
    joints: JointsPreset = field(default_factory=JointsPreset)
    planner: Planner = field(default_factory=Planner)
    perception: Perception = field(default_factory=Perception)
    gripper: Gripper = field(default_factory=Gripper)
    grasp: Grasp = field(default_factory=Grasp)
    lidar_scan: LidarScan = field(default_factory=LidarScan)

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
