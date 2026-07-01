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

# Change these when moving the software between robots / locations.
ROBOT_PROFILE = "old"  # "new" or "old"  — robot mounting / kinematics convention
ENVIRONMENT   = "outdoor"  # "lab" or "outdoor" — workspace layout (home/dropoff positions)

# Joint configs depend on BOTH the robot (mounting/kinematics) and the environment (where
# the tree and dropoff bin are), so they are defined per (robot, environment).
# x_forward_y_lateral is a robot-only property and does not change with environment.
_NEW_LAB_JOINTS = {
    "home": [
        0.2966473698616028, -1.6924630604186, 2.319343153630392,
        -4.599916835824484, -1.8223055044757288, 2.912656307220459,
    ],
    "dropoff": [
        -1.085060343146324, -1.570564409295553, 2.355928007756368,
        -2.613685270349020, -1.386652294789450, 2.914407730102539,
    ],
    "predropoff": [
        -0.223708137869835, -1.685942789117330, 2.032441679631368,
        2.154242201442383, -1.655924622212545, 3.139664408062593,
    ],
    "home_left": [
        0.643364429473877, -1.5703463061984912, 1.9470141569720667,
        -4.550284524957174, -2.1596739927874964, 2.4302515983581543,
    ],
    "home_right": [
        -0.9581039587603968, -1.9419809780516566, 2.136008087788717,
        -4.9219705067076625, -0.6445449034320276, 4.893132209777832,
    ],
    "home_right_low": [
        -0.579287354146139, -1.8774057827391566, 2.3498411814319056,
        -4.595168252984518, -1.0441439787494105, 3.820992946624756,
    ],
    "home_left_low": [
        0.529047966003418, -1.5133289259723206, 2.0492852369891565,
        -4.616746803323263, -2.054364029561178, 2.6127915382385254,
    ],
}

_OLD_LAB_JOINTS = {
    "home": [
        5.108725070953369, -1.794300218621725, 2.4091363588916224,
        1.6457602220722656, 4.382841110229492, -0.23138553300966436,
    ],
    "dropoff": [
        -2.362258497868673, -1.5946093998351039, 2.3843892256366175,
        -2.6575151882567347, 4.8416242599487305, -0.17230397859682256,
    ],
    "predropoff": [
        -1.5009062925921839, -1.7099877796568812, 2.0609028975116175,
        -4.172773023644918, 4.572351932525635, 0.05295269936323166,
    ],
    "home_left": [
        -0.7986648718463343, -1.370279149418213, 1.5869911352740687,
        -3.958400150338644, 3.832206964492798, -0.4182942549334925,
    ],
    "home_right": [
        -2.1769216696368616, -1.6072222195067347, 1.8494580427752894,
        -4.613555570641989, 5.513333320617676, 1.2812821865081787,
    ],
    "home_right_low": [
        -2.0225699583636683, -1.5956303081908167, 2.210489575062887,
        -4.641205211678976, 5.276980876922607, 0.710979700088501,
    ],
    "home_left_low": [
        -0.9872930685626429, -1.384446458225586, 2.0082204977618616,
        -4.594993253747457, 4.218744277954102, -0.3455312887774866,
    ],
}

# ===== OUTDOOR joint sets — EDIT THESE for the field/orchard =====
# Outdoor homes are LOW (below the base plane, near the ground). Per-joint order:
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3], radians.
# Seeded from the lab values for now — jog the arm outdoors, read /joint_states, and
# replace each list with the recorded pose. Only the keys you change take effect.
_NEW_OUTDOOR_JOINTS = {
    "home": [
        0.2966473698616028, -1.6924630604186, 2.319343153630392,
        -4.599916835824484, -1.8223055044757288, 2.912656307220459,
    ],
    "dropoff": [
        -1.085060343146324, -1.570564409295553, 2.355928007756368,
        -2.613685270349020, -1.386652294789450, 2.914407730102539,
    ],
    "predropoff": [
        -0.223708137869835, -1.685942789117330, 2.032441679631368,
        2.154242201442383, -1.655924622212545, 3.139664408062593,
    ],
    "home_left": [
        0.643364429473877, -1.5703463061984912, 1.9470141569720667,
        -4.550284524957174, -2.1596739927874964, 2.4302515983581543,
    ],
    "home_right": [
        -0.9581039587603968, -1.9419809780516566, 2.136008087788717,
        -4.9219705067076625, -0.6445449034320276, 4.893132209777832,
    ],
    "home_right_low": [
        -0.579287354146139, -1.8774057827391566, 2.3498411814319056,
        -4.595168252984518, -1.0441439787494105, 3.820992946624756,
    ],
    "home_left_low": [
        0.529047966003418, -1.5133289259723206, 2.0492852369891565,
        -4.616746803323263, -2.054364029561178, 2.6127915382385254,
    ],
}

_OLD_OUTDOOR_JOINTS = {
    "home": [
        5.108725070953369, -1.794300218621725, 2.4091363588916224,
        1.6457602220722656, 4.382841110229492, -0.23138553300966436,
    ],
    "dropoff": [
        -2.362258497868673, -1.5946093998351039, 2.3843892256366175,
        -2.6575151882567347, 4.8416242599487305, -0.17230397859682256,
    ],
    "predropoff": [
        -1.5009062925921839, -1.7099877796568812, 2.0609028975116175,
        -4.172773023644918, 4.572351932525635, 0.05295269936323166,
    ],
    "home_left": [
        -0.7986648718463343, -1.370279149418213, 1.5869911352740687,
        -3.958400150338644, 3.832206964492798, -0.4182942549334925,
    ],
    "home_right": [
        -2.1769216696368616, -1.6072222195067347, 1.8494580427752894,
        -4.613555570641989, 5.513333320617676, 1.2812821865081787,
    ],
    "home_right_low": [
        -2.0225699583636683, -1.5956303081908167, 2.210489575062887,
        -4.641205211678976, 5.276980876922607, 0.710979700088501,
    ],
    "home_left_low": [
        -0.9872930685626429, -1.384446458225586, 2.0082204977618616,
        -4.594993253747457, 4.218744277954102, -0.3455312887774866,
    ],
}

ROBOT_PROFILES = {
    "new": {
        "x_forward_y_lateral": True,
        "environments": {
            "lab": {"joints": _NEW_LAB_JOINTS},
            "outdoor": {"joints": _NEW_OUTDOOR_JOINTS},
        },
    },
    "old": {
        "x_forward_y_lateral": False,
        "environments": {
            "lab": {"joints": _OLD_LAB_JOINTS},
            "outdoor": {"joints": _OLD_OUTDOOR_JOINTS},
        },
    },
}

if ROBOT_PROFILE not in ROBOT_PROFILES:
    raise ValueError(
        f"Unknown ROBOT_PROFILE {ROBOT_PROFILE!r}; "
        f"choose one of {sorted(ROBOT_PROFILES)}")

ACTIVE_ROBOT_PROFILE = ROBOT_PROFILES[ROBOT_PROFILE]
X_FORWARD_Y_LATERAL = ACTIVE_ROBOT_PROFILE["x_forward_y_lateral"]

_ENVIRONMENTS = ACTIVE_ROBOT_PROFILE["environments"]
if ENVIRONMENT not in _ENVIRONMENTS:
    raise ValueError(
        f"Unknown ENVIRONMENT {ENVIRONMENT!r}; "
        f"choose one of {sorted(_ENVIRONMENTS)}")

ACTIVE_ENVIRONMENT = _ENVIRONMENTS[ENVIRONMENT]


def _profile_joints(name: str) -> List[float]:
    return list(ACTIVE_ENVIRONMENT["joints"][name])

# ---------- Static Obstacles (single source of truth) ----------
# Define obstacles once here, used for both cuRobo planning and RViz visualization
_TABLE_OBSTACLE = {
    "name": "table",
    "type": "cuboid",
    "dims": [5.0, 5.0, 0.2],
    "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
    "color": (1.0, 0.0, 0.0, 1.0),  # Red
}
_TRUNK_OBSTACLE = {
    "name": "trunk",
    "type": "cylinder",
    "radius": 0.02,    # 2cm radius = 4cm diameter trunk
    "height": 1.2,     # 1.2m visible trunk section
    "pose": (
        [1.00, 0.16, 0.6, 1, 0, 0, 0]
        if X_FORWARD_Y_LATERAL
        else [0.16, -1.00, 0.6, 1, 0, 0, 0]
    ),
    "color": (0.55, 0.27, 0.07, 1.0),  # Brown
}

# The lab has a physical table under the arm; outdoor (field/orchard) does not — drop the
# table obstacle outdoors so it doesn't block low approaches. Applies to both cuRobo
# planning (WORLD_CONFIG) and RViz visualization, which both derive from STATIC_OBSTACLES.
STATIC_OBSTACLES = ([] if ENVIRONMENT == "outdoor" else [_TABLE_OBSTACLE]) + [_TRUNK_OBSTACLE]

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
    "pose": (
        [0.5, 0.3, 0.8, 1, 0, 0, 0]
        if X_FORWARD_Y_LATERAL
        else [0.3, -0.5, 0.8, 1, 0, 0, 0]
    ),
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

# Last-resort recovery for large/contorted reconfigurations (e.g. HOME ~170deg away from a
# flipped in-zone pose) where the straight-line trajopt seed sweeps through obstacles and the
# optimizer can't converge from any local seed. The graph planner finds a collision-free
# GLOBAL seed that trajopt then smooths. Graph search resizes internal buffers that otherwise
# corrupt subsequent plan_single_js calls, so the caller MUST motion_gen.reset() afterward.
PLAN_CFG_JS_GRAPH = MotionGenPlanConfig(
    max_attempts=8,
    enable_finetune_trajopt=True,
    enable_graph=True,
    enable_graph_attempt=1,
)

# Cheap config for reachability PREFLIGHT (e.g. lidar-scan candidate probing). A preflight
# only needs a yes/no "can the arm get here" answer, not a polished trajectory. The big cost
# was the strict finetune smoothing pass (~12s per failure, and FINETUNE_TRAJOPT_FAIL on
# reachable poses) — dropping it gives most of the speedup. max_attempts is kept moderately
# high because it controls IK-seed diversity: too low (e.g. 4) makes borderline-reachable
# semicircle poses spuriously IK_FAIL and collapses the sweep to a tiny fallback arc.
PLAN_CFG_SCAN_PREFLIGHT = MotionGenPlanConfig(
    max_attempts=20,
    enable_finetune_trajopt=False,
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
    home: List[float] = field(default_factory=lambda: _profile_joints("home"))
    dropoff: List[float] = field(default_factory=lambda: _profile_joints("dropoff"))
    predropoff: List[float] = field(default_factory=lambda: _profile_joints("predropoff"))
    home_left: List[float] = field(default_factory=lambda: _profile_joints("home_left"))
    home_right: List[float] = field(default_factory=lambda: _profile_joints("home_right"))
    home_right_low: List[float] = field(
        default_factory=lambda: _profile_joints("home_right_low"))
    home_left_low: List[float] = field(
        default_factory=lambda: _profile_joints("home_left_low"))

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
    speed_home: float = 0.11        # faster return; effective scale=0.55 with global=5
    speed_dropoff: float = 0.18     # faster carrying move; effective scale=0.90
    speed_predropoff: float = 0.2  # for pre-dropoff reverse
    speed_approach: float = 1.0    # for approach motion
    speed_final: float = 0.18       # precise grasp; moderate increase from 0.15

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
    approach_max_joint_step_deg: float = 12.0  # reject APPROACH plans with abrupt per-waypoint joint jumps
    approach_max_path_ratio: float = 2.5       # reject roundabout approach paths
    home_reached_tolerance_deg: float = 3.0    # skip planning when already at HOME
    home_direct_fallback_max_delta_deg: float = 40.0  # guarded fallback after HOME trajopt failure (still forearm/flange clearance-checked)
    home_verified_interp_max_delta_deg: float = 185.0  # HOME-only recovery: dense cuRobo validity + forearm/flange checked
    home_verified_interp_step_deg: float = 1.0         # max joint step for HOME verified interpolation samples
    home_verified_route_max_candidates: int = 120      # bounded deterministic route search after direct HOME path is invalid
    home_cartesian_pose_recovery: bool = False         # TCP-only HOME can land on the wrong joint branch; keep exact-HOME strict
    home_cart_finish_max_delta_deg: float = 45.0       # after HOME_CART, only chase exact joint branch when close
    goal_failure_recover_to_posture: bool = True       # after queued goal failure, move to nearest good posture and retry once
    goal_recovery_cartesian_fallback: bool = False     # after posture retry fails, try collision-aware Cartesian plan once
    goal_recovery_repeat_staging_after_posture: bool = False  # avoid repeating same expensive staging sweep after posture retry
    goal_recovery_local_staging: bool = True           # try target-local standoff before moving to stored postures
    goal_recovery_stage_offsets_m: List[float] = field(
        default_factory=lambda: [0.20, 0.12, 0.30])
    goal_recovery_stage_max_attempts: int = 3          # cap expensive standoff/descent planner attempts per recovery pass
    goal_recovery_stage_marker_first: bool = True      # when marker/current orientations differ, try requested marker-orientation standoff first
    safe_zone_path_margin_m: float = 0.01              # keep manual Cartesian TCP samples this far inside the safe-zone box
    dynamic_goal_ordering: bool = False                # False: run the queue strictly in insertion order; True: choose next queued goal by current IK reachability
    goal_reachability_skip_delta_deg: float = 100.0    # postpone/skip queued goals above this nearest-IK delta
    goal_recovery_max_ik_delta_deg: float = 100.0      # do not grind staging/posture recovery above this nearest-IK delta
    shortest_ik_plan_max_delta_deg: float = 80.0       # fail fast when nearest IK branch is too far for short-goal planning
    shortest_ik_max_tries: int = 2                     # number of near IK branches to try before recovery/fallback
    safe_zone_verified_interp_first: bool = True       # for nearby IK goals, validate/execute direct joint interpolation before slow Cartesian trajopt
    safe_zone_verified_interp_max_delta_deg: float = 80.0
    safe_zone_verified_interp_step_deg: float = 1.0
    reachability_cloud_enabled: bool = True            # RViz cloud of TCP positions reachable from the current posture
    reachability_cloud_period_s: float = 1.0
    reachability_cloud_samples: int = 1000
    reachability_cloud_point_size_m: float = 0.025
    reachability_cloud_max_delta_deg: float = 100.0
    reachability_cloud_validate_green: bool = True
    reachability_direction_enabled: bool = True        # draw TCP local-axis direction rays on green samples
    reachability_direction_axis: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0])       # TCP local +Z matches approach-axis helpers
    reachability_direction_length_m: float = 0.08
    reachability_direction_stride: int = 1             # draw every Nth green direction ray; 1 shows every green point
    reachability_click_max_distance_m: float = 0.06  # RViz /clicked_point must be this close to a green sample
    reachability_click_green_only: bool = True        # only queue clicks in the fast-success green band
    reachability_goal_speed_factor: float = 0.20      # very slow execution for reachability-click goals

    # === DIRECT IK TUNING ===
    direct_branch_retry_min_dist: float = 0.15  # m; skip expensive branch search for close moves
    direct_branch_retry_seeds: int = 8          # extra perturbed IK seeds when branch retry is needed
    direct_final_cart_waypoints: int = 2        # intermediate Cartesian IK waypoints for FINAL only
    direct_final_joint_fallback_max_delta_deg: float = 25.0  # allow FINAL joint fallback only for small endpoint moves
    very_low_ik_return_seeds: int = 8           # IK branches scored before moving to a very-low goal
    very_low_ik_max_joint_delta_deg: float = 75.0
    # Sphere-surface clearance, not physical caliper distance. A modeled 41.8mm
    # has produced a real UR clamping stop, so all trajectories must stay above 45mm.
    clamp_safety_threshold_mm: float = 45.0
    very_low_preflight_min_clearance_mm: float = 50.0
    very_low_preflight_early_accept_mm: float = 50.0
    very_low_preflight_preferred_pitch_deg: float = -5.0  # negative pitch tried first on approach (start at -5)
    very_low_preflight_second_pitch_deg: float = -30.0
    very_low_preflight_preferred_wrist_deg: float = 0.0
    very_low_preflight_pitch_step_deg: float = 10.0
    very_low_preflight_pitch_steps: int = 3
    very_low_clearance_window_mm: float = 8.0   # choose path cost only among near-safest IK branches
    very_low_center_cy_thresh: float = 0.88     # image cy; force center-home handling for very low fruit
    # Offset names below are semantic. They are mapped through ROBOT_PROFILE:
    # - new robot: depth -> X, lateral -> Y
    # - old robot: depth -> Y, lateral -> X
    side_home_lateral_offset: float = 0.16      # m; side-home lateral distance from trunk
    side_home_x_offset: float = 0.16            # legacy alias for side_home_lateral_offset
    side_home_partial_reverse_m: float = 0.55   # m; partial reverse clearance for side-approach fruits (vs 0.35m center)


    mid_center_approach_depth_offset: float = 0.10
    mid_center_approach_y_offset: float = 0.10  # legacy alias for mid_center_approach_depth_offset
    mid_center_approach_z_offset: float = -0.05
    mid_high_left_thresh: float = 0.20      # bunch/image rel-x below this is MID/HIGH LEFT
    mid_high_right_thresh: float = 0.80     # bunch/image rel-x above this is MID/HIGH RIGHT
    low_left_thresh: float = 0.32           # bunch/image rel-x below this is LOW LEFT
    low_right_thresh: float = 0.68          # bunch/image rel-x above this is LOW RIGHT
    bunch_edge_side_band: float = 0.20        # rel-x within outer 20% of bunch is forced LEFT/RIGHT
    bunch_lower_center_band: float = 0.80     # rel-y >= this is forced VERY LOW/CENTER



    low_center_approach_depth_offset: float = 0.03
    low_center_approach_y_offset: float = 0.03  # legacy alias for low_center_approach_depth_offset
    low_center_approach_z_offset: float = -0.07


    very_low_center_approach_depth_offset: float = 0.08
    very_low_center_approach_y_offset: float = 0.08  # legacy alias for very_low_center_approach_depth_offset
    very_low_center_approach_z_offset: float = -0.025
    very_low_center_approach_pitch_deg: float = 5.0  # local tool X pitch to open forearm-flange clearance


    low_left_standoff_lateral: float = 0.12
    low_left_standoff_depth: float = 0.035
    low_left_standoff_x: float = 0.12           # legacy alias for low_left_standoff_lateral
    low_left_standoff_y: float = 0.035          # legacy alias for low_left_standoff_depth
    low_left_standoff_z: float = -0.055         # m; left side-low vertical standoff



    low_right_standoff_lateral: float = 0.08
    low_right_standoff_depth: float = 0.050
    low_right_standoff_x: float = 0.08          # legacy alias for low_right_standoff_lateral
    low_right_standoff_y: float = 0.050         # legacy alias for low_right_standoff_depth
    low_right_standoff_z: float = -0.025        # m; avoid large upward push on right-side final




    low_side_final_depth_offset: float = 0.0
    low_side_final_y_offset: float = 0.0        # legacy alias for low_side_final_depth_offset
    low_left_final_z_offset: float = 0.020      # m; left side-low gripper center offset
    low_right_final_z_offset: float = 0.010     # m; right side-low gripper center offset

    low_side_final_front_tilt_deg: float = 10.0 # max final +Z/front tilt toward fruit
    mid_center_approach_pitch_deg: float = 0.0 # local tool X pitch for MID/HIGH center; keep 0.0 to preserve approach→final orientation continuity

    low_center_final_depth_offset: float = -0.01
    low_center_final_y_offset: float = -0.01    # legacy alias for low_center_final_depth_offset
    low_center_final_z_offset: float = 0.033    # m; gripper center offset above low-center fruit

    mid_center_final_depth_offset: float = -0.02
    mid_center_final_y_offset: float = -0.02    # legacy alias for mid_center_final_depth_offset
    mid_center_final_z_offset: float = 0.030    # m; gripper center offset above MID/HIGH center fruit
    mid_center_slip_final_z_offset: float = 0.020 # m; slightly lower final target during slip retry

#---------------------------------------------------------------------------------------------------------
    final_overshoot_threshold: float = 0.004    # m; correct only if TCP passes target by >4mm
    final_overshoot_max_backoff: float = 0.012  # m; max one-shot pullback before closing
    reverse_initial_wait: float = 0.15          # s; minimum wait after publishing partial reverse
    reverse_final_settle: float = 0.05          # s; settle after reverse stops before hold check
    reverse_dt_multiplier: float = 1.7          # reverse dt=min_dt*multiplier (was fixed at 2.0)
    grasp_post_close_settle_s: float = 0.10      # closure loop is synchronous; only sensor settle remains


    hold_check_settle_s: float = 0.05           # s; force settle before post-reverse hold samples
    hold_check_window_s: float = 0.30           # s; median force sample window after reverse
    hold_check_sample_dt: float = 0.04          # s; post-reverse force sample period
    dropoff_preplan_wait_s: float = 15.0        # wait for background plan before starting another planner


    pre_dropoff_z_offset: float = -0.1  # m above dropoff
    pre_dropoff_depth_offset: float = 0.25
    pre_dropoff_y_offset: float = 0.25  # legacy alias for pre_dropoff_depth_offset

    # === REACQUIRE ===
    reacquire_after_approach: bool = False  # re-detect fruit position after reaching approach standoff
    slip_check_reacquire: bool = False      # query depth after grasp to detect fruit slip
    regrip_after_slip: bool = False         # attempt regrip correction when grip is weak after slip
    subscribe_goal_min_settle_s: float = 0.40  # ignore first depth samples after Subscribe
    subscribe_goal_max_wait_s: float = 1.20    # fast median fallback if XYZ does not stabilize
    subscribe_goal_stable_tol: float = 0.020   # m; last 2 XYZ readings must be this close
    subscribe_goal_median_window: int = 5      # recent XYZ samples for fast fallback median

    # === DEBUG ===
    use_fake_hardware: bool = False    # ros2_control mock hardware never publishes robot_program_running;
                                        # treat the program as always "running" so wait_until_xyz doesn't stall
    debug_plan_preview: bool = True    # show full plan and wait for confirmation before executing
    log_cycle_start: bool = False      # verbose per-goal start/decision metadata
    log_phase_timings: bool = False    # per-phase timing lines; cycle summary always includes timings
    log_path_publish: bool = False     # RViz path marker publish messages
    log_gripper_force_profile: bool = False  # full per-step closure force profile

@dataclass
class Perception:
    enabled: bool = False
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
    open_settle_s: float = 0.15

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
    use_semicircle: bool = True
    semicircle_points: int = 7
    semicircle_radius_m: float = 0.22
    semicircle_arc_deg: float = 180.0
    semicircle_use_base_angles: bool = True           # use fixed base-frame scan angles instead of centering arc on current TCP
    semicircle_start_deg: float = 0.0                 # front half in base frame; 0=+X, 90=+Y
    semicircle_end_deg: float = 180.0                 # opposite of the previous back-side sweep
    semicircle_z_offset_m: float = 0.0
    semicircle_face_target: bool = False              # keep current tool orientation; arc shape matters more than exact center-facing
    semicircle_local_axis: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0])
    semicircle_adaptive_scan: bool = True             # allow skipped angles/variable radius instead of forcing a perfect arc
    semicircle_radius_candidates_m: List[float] = field(
        default_factory=lambda: [0.12, 0.22])
    semicircle_min_points: int = 3
    local_close_fallback_enabled: bool = True          # if arc fails, scan a small local reachable sweep
    local_close_first: bool = False                    # keep the scan arc-shaped; use local fallback only if the arc cannot run
    local_close_offsets_m: List[float] = field(
        default_factory=lambda: [0.0, 0.04, 0.08])
    local_close_axis: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 0.0])       # base-frame direction for local fallback sweep
    semicircle_use_reachability: bool = False          # generated center-facing scan poses keep LiDAR aimed at center
    semicircle_allow_geometric_fallback: bool = True   # use generated horizontal arc when green samples are sparse
    semicircle_reachability_max_radius_error_m: float = 0.12
    semicircle_z_tolerance_m: float = 0.02             # keep reachability-picked scan points near one horizontal plane
    semicircle_path_z_tolerance_m: float = 0.035       # reject joint shortcuts that visibly climb out of the scan plane
    semicircle_max_joint_delta_deg: float = 75.0       # reject scan poses that require a large joint-branch jump
    semicircle_max_joint_step_deg: float = 8.0         # reject Cartesian plans with abrupt joint jumps between samples
    semicircle_max_failed_candidates: int = 3          # stop scan preflight early when this arc side is clearly unreachable
    semicircle_preview_enabled: bool = True            # show candidate scan arc before pressing lidar_scan
    arc_highlight_enabled: bool = True                 # paint reachability-cloud points that sit on/near the scan arc a darker green
    arc_highlight_tol_m: float = 0.04                  # radial band (m) for "near or on the arc"
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
