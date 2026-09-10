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

# Only the old physical robot is supported. Environment remains selectable.
ROBOT_PROFILE = "old"
ENVIRONMENT   = os.environ.get("UR10E_ENVIRONMENT", "outdoor").strip().lower()

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
        -1.4994238058673304, -0.7151940625957032, 1.7512252966510218,
        -4.656813760797018, -1.7659905592547815, 1.1700091361999512,
    ],
    "dropoff": [
        -1.90818959871401, -1.079150215988495, 2.008786980305807,
        -3.1874824963011683, -1.4845169226275843, 0.9511620998382568,
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
        -0.10284835497011358, -0.7675350469401856, 1.8947790304767054,
        -4.78630056003713, -1.4767573515521448, 1.9139177799224854,
    ],
    "dropoff": [
        -0.3758776823626917, -1.1837181013873597, 2.0136449972735804,
        -2.9486800632872523, -1.5171645323382776, 1.5264148712158203,
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

# Final-grasp depth/Z offsets and TCP-to-closure corrections depend on the
# environment (fruit size/position calibration differs lab vs outdoor), same
# as the joint presets above. Seeded identically for both environments for
# now -- tune one side independently as real per-environment grasp data
# comes in; only the keys you change take effect.
_LAB_GRASP_OFFSETS = {
    "low_side_final_depth_offset": 0.0,
    "low_left_final_z_offset": 0.020,
    "low_right_final_z_offset": 0.010,
    "low_center_final_depth_offset": -0.010,
    "low_center_final_z_offset": 0.0,
    "mid_center_final_depth_offset": -0.010,
    "mid_center_final_z_offset": 0.0,
    "mid_center_slip_final_z_offset": 0.020,
    "closure_center_offset_tcp_m": [0.0, 0.005, 0.0],
    "envelop_closure_center_offset_tcp_m": [0.0, 0.005, 0.0],
}

_OUTDOOR_GRASP_OFFSETS = {
    "low_side_final_depth_offset": 0.0,
    "low_left_final_z_offset": 0.020,
    "low_right_final_z_offset": 0.010,
    "low_center_final_depth_offset": -0.010,
    "low_center_final_z_offset": 0.0,
    "mid_center_final_depth_offset": -0.010,
    "mid_center_final_z_offset": 0.0,
    "mid_center_slip_final_z_offset": 0.020,
    "closure_center_offset_tcp_m": [0.0, 0.005, 0.0],
    "envelop_closure_center_offset_tcp_m": [0.0, 0.005, 0.0],
}

ROBOT_PROFILES = {
    "old": {
        "x_forward_y_lateral": False,
        "environments": {
            "lab": {"joints": _OLD_LAB_JOINTS, "grasp_offsets": _LAB_GRASP_OFFSETS},
            "outdoor": {"joints": _OLD_OUTDOOR_JOINTS, "grasp_offsets": _OUTDOOR_GRASP_OFFSETS},
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


def _profile_offset(name: str):
    value = ACTIVE_ENVIRONMENT["grasp_offsets"][name]
    return list(value) if isinstance(value, list) else value

# ---------- Static Obstacles (single source of truth) ----------
# Define obstacles once here, used for both cuRobo planning and RViz visualization
_TABLE_OBSTACLE = {
    "name": "table",
    "type": "cuboid",
    "dims": [5.0, 5.0, 0.2],
    "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
    "color": (1.0, 0.0, 0.0, 1.0),  # Red
}
_GOLFCART_BASE_PLATE_OBSTACLE = {
    # Outdoor-only robot mounting plate extracted from
    # ~/Downloads/Golfcart_pallet.step. The four STEP screw circles are centered
    # at CAD (670, 1920) mm, so base_link is treated as that mounting center.
    # Local plate bbox in the CAD: x/y +/-150 mm, z roughly -50..0 mm below the
    # mounting surface.
    "name": "golfcart_robot_base_plate",
    "type": "cuboid",
    "dims": [0.30, 0.30, 0.06],
    "pose": [0.0, 0.0, -0.03, 1, 0, 0, 0],
    "color": (0.18, 0.32, 0.42, 0.25),
}

_GOLFCART_BASE_PLATE_VISUAL = {
    # RViz-only top overlay for the robot mounting plate. The collision plate
    # above stays below z=0 where the CAD places it; this overlay is raised so
    # it remains visible instead of disappearing under the RViz grid/UR base.
    "name": "golfcart_robot_base_plate_visual",
    "type": "cuboid",
    "dims": [0.34, 0.34, 0.040],
    "pose": [0.0, 0.0, 0.035, 1, 0, 0, 0],
    "color": (0.12, 0.26, 0.34, 0.95),
    "collision": False,
}

_GOLFCART_BASE_PLATE_OUTLINE = {
    "name": "golfcart_robot_base_plate_outline",
    "type": "cuboid_outline",
    "dims": [0.36, 0.36, 0.08],
    "pose": [0.0, 0.0, 0.085, 1, 0, 0, 0],
    "line_width": 0.018,
    "color": (0.0, 0.95, 1.0, 1.0),
    "collision": False,
}

_GOLFCART_PALLET_VISUAL = {
    # Full visible pallet envelope from the STEP bbox, relative to the screw
    # pattern center. Visual-only: this shows the golf-cart/pallet footprint in
    # RViz without adding a huge planning obstacle.
    "name": "golfcart_pallet_visual",
    "type": "cuboid",
    "dims": [0.93, 2.47, 0.035],
    "pose": [-0.265, -0.715, 0.035, 1, 0, 0, 0],
    "color": (0.05, 0.55, 1.0, 0.82),
    "collision": False,
}

_GOLFCART_PALLET_OUTLINE = {
    "name": "golfcart_pallet_outline",
    "type": "cuboid_outline",
    "dims": [0.93, 2.47, 0.05],
    "pose": [-0.265, -0.715, 0.095, 1, 0, 0, 0],
    "line_width": 0.020,
    "color": (0.0, 0.95, 1.0, 1.0),
    "collision": False,
}

_GOLFCART_BASE_SCREW_MARKERS = [
    {
        "name": f"golfcart_base_screw_{i + 1}",
        "type": "cylinder",
        # STEP CIRCLE radius = 4.25 mm. Draw larger in RViz so the mounting
        # pattern is visible under/around the UR base. Visual-only.
        "radius": 0.025,
        "height": 0.018,
        "pose": [x, y, 0.070, 1, 0, 0, 0],
        "color": (1.0, 0.82, 0.12, 1.0),
        "collision": False,
    }
    for i, (x, y) in enumerate(
        [
            (0.0601040764, -0.0601040764),
            (-0.0601040764, 0.0601040764),
            (-0.0601040764, -0.0601040764),
            (0.0601040764, 0.0601040764),
        ]
    )
]

_GOLFCART_PALLET_OBSTACLE = {
    # Outdoor pallet collision approximation for cuRobo. This follows the
    # RViz pallet after the requested rotations about base_link Z. Keep the top
    # slightly below z=0 so the UR base can sit on the pallet without
    # start-state collision.
    "name": "golfcart_pallet",
    "type": "cuboid",
    # Rotated -90° about base_link Z to match the pallet STL: dims x/y swapped,
    # center (x,y)->(y,-x).
    "dims": [1.12, 0.93, 0.12],
    "pose": [0.03, 0.265, -0.08, 1, 0, 0, 0],
    "color": (0.20, 0.45, 0.75, 0.28),
}

# Do not model the large table in either environment. The golf-cart pallet
# support/collision approximation is included in BOTH lab and outdoor. This
# applies to both cuRobo planning and RViz, which derive from STATIC_OBSTACLES.
STATIC_OBSTACLES = [_GOLFCART_PALLET_OBSTACLE]

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
    "cuboid": {
        obs["name"]: _to_curobo_cuboid(obs)
        for obs in STATIC_OBSTACLES
        if obs.get("collision", True)
    }
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
    # cuRobo default is 4: runs 4 parallel full trajectory optimizations per
    # plan and keeps the best, on every approach/final/dropoff/return_home
    # move. That's a real, repeated GPU cost competing with YOLO. Testing
    # a lower value here (this motion is short, repetitive, small-workspace
    # reaches, not open-ended novel scenes) -- watch plan success rate and
    # trajectory smoothness on real grasps before trusting a value long-term.
    num_trajopt_seeds: int = int(os.environ.get("UR10E_NUM_TRAJOPT_SEEDS", "2"))
    speed_scale: float = 0.5
    base_dt: float = 0.02          # common base timestep (s)

    # === GLOBAL SPEED CONTROL ===
    # Increase this to make ALL motions faster (0.5 = half speed, 2.0 = double speed)
    global_speed_multiplier: float = 5.0

    # Motion-specific speed factors (multiplied by global_speed_multiplier)
    speed_home: float = 0.11        # faster return; effective scale=0.55 with global=5
    speed_dropoff: float = 0.18     # faster carrying move; effective scale=0.90
    speed_predropoff: float = 0.2  # for pre-dropoff reverse
    # Only FINAL is genuinely near the fruit and needs to be slow. Everything
    # before it is gross motion well outside the fruit corridor.
    #
    # dt = base_dt / (speed_scale(0.5) * this), clamped to [curobo_dt, max_dt]:
    #   4.0 -> 10ms, the same rate HOME/DROPOFF already run at safely
    #   2.0 -> 20ms,  0.8 -> 50ms,  0.06 -> clamped to max_dt = 80ms
    #
    # speed_alignment now sets the dt for the WHOLE staged approach, since the
    # straight entry leg is appended into the same trajectory (see plan_and_send
    # extend_fn). The entry leg does not inherit this speed: its own duration is
    # held by approach_entry_duration_s below, via sample density.
    speed_alignment: float = 4.0  # staged approach (dt 10ms, HOME rate)
    speed_approach: float = 2.0   # standalone approach path (dt 20ms)
    speed_final: float = 0.06     # precise, deliberately slow near the fruit
    # Wall-clock duration held for the straight near-fruit entry leg regardless
    # of how fast the staging leg runs. Previously this leg was 41 samples at
    # 50ms = ~2.05s; keeping it close to that preserves the careful entry while
    # the approach to it got faster.
    approach_entry_duration_s: float = 1.5

    # Final base-frame guard for vision goals. A stable but physically impossible
    # background-depth estimate must never enter the motion queue.
    goal_workspace_forward_min_m: float = 0.35
    goal_workspace_forward_max_m: float = 1.60
    goal_workspace_lateral_min_m: float = -0.80
    goal_workspace_lateral_max_m: float = 0.90
    goal_workspace_z_min_m: float = 0.02
    goal_workspace_z_max_m: float = 1.40

    # === SMOOTHNESS PARAMETERS ===
    # Lower values = smoother but slower transitions
    max_joint_velocity: float = 1.5      # rad/s max velocity per joint
    max_joint_acceleration: float = 1.5  # rad/s^2 max acceleration (lower = less jerk at end)
    ramp_points: int = 15                # number of points for accel/decel ramps

    # === TRAJECTORY LIMITS ===
    # These control how fast trajectories can actually execute
    min_dt: float = 0.010                # minimum timestep — keep ≥0.010 to avoid UR joint velocity limit faults
    max_dt: float = 0.08                 # allow slow approach/final trajectories
    approach_endpoint_settle_s: float = 0.25  # let zero-velocity endpoint finish; avoid hold preemption
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
    # AUTO-HARVEST batching. The loop used to discover ONE goal, execute it, and
    # re-discover from scratch every cycle -- paying the discovery/settle wait each
    # time, and leaving the goal queue empty so the "more goals queued" smart-return
    # path (which skips the trip HOME when the next approach is directly reachable)
    # could never trigger. Batching queues several targets from one detection pass
    # instead. The per-goal reacquire still runs before each approach, so targets
    # are refreshed as the bunch shifts after a pick -- the batch fixes the ORDER
    # of work, not the accuracy of any individual grasp.
    auto_harvest_batch_goals: bool = True
    auto_harvest_max_batch: int = 3                    # targets queued per discovery pass
    goal_reachability_skip_delta_deg: float = 100.0    # postpone/skip queued goals above this nearest-IK delta
    goal_recovery_max_ik_delta_deg: float = 100.0      # do not grind staging/posture recovery above this nearest-IK delta
    shortest_ik_plan_max_delta_deg: float = 80.0       # fail fast when nearest IK branch is too far for short-goal planning
    shortest_ik_max_tries: int = 2                     # number of near IK branches to try before recovery/fallback
    safe_zone_verified_interp_first: bool = True       # for nearby IK goals, validate/execute direct joint interpolation before slow Cartesian trajopt
    safe_zone_verified_interp_max_delta_deg: float = 80.0
    safe_zone_verified_interp_step_deg: float = 1.0
    reachability_cloud_enabled: bool = False           # RViz cloud of TCP positions reachable from the current posture
    reachability_cloud_period_s: float = 1.5
    reachability_cloud_samples: int = 320
    reachability_cloud_point_size_m: float = 0.025
    reachability_cloud_max_delta_deg: float = 100.0
    reachability_cloud_validate_green: bool = True
    reachability_direction_enabled: bool = True        # draw TCP local-axis direction rays on green samples
    reachability_direction_axis: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0])       # TCP local +Z matches approach-axis helpers
    reachability_direction_length_m: float = 0.08
    reachability_direction_stride: int = 2             # draw every Nth green direction ray; 1 shows every green point
    reachability_click_max_distance_m: float = 0.06  # RViz /clicked_point must be this close to a green sample
    reachability_click_green_only: bool = True        # only queue clicks in the fast-success green band
    reachability_goal_speed_factor: float = 0.20      # very slow execution for reachability-click goals
    current_joint_goal_preflight_enabled: bool = False # cuRobo replay check for "Add Current Pos"; off keeps vision FPS stable

    # === DIRECT IK TUNING ===
    direct_branch_retry_min_dist: float = 0.15  # m; skip expensive branch search for close moves
    direct_branch_retry_seeds: int = 8          # extra perturbed IK seeds when branch retry is needed
    direct_final_cart_waypoints: int = 2        # intermediate Cartesian IK waypoints for FINAL only
    approach_alignment_enabled: bool = True     # stage outside fruit before entering approach standoff
    approach_alignment_extra_m: float = 0.08    # extra distance behind the existing approach pose
    direct_approach_cart_waypoints: int = 3     # enforce straight alignment -> approach motion
    direct_final_joint_fallback_max_delta_deg: float = 25.0  # allow FINAL joint fallback only for small endpoint moves
    strict_final_cartesian_only: bool = True  # never replace straight FINAL insertion with a joint/planner curve
    very_low_ik_return_seeds: int = 8           # IK branches scored before moving to a very-low goal
    very_low_ik_max_joint_delta_deg: float = 75.0
    # Sphere-surface clearance, not physical caliper distance. A modeled 41.8mm
    # has produced a real UR clamping stop, so all trajectories must stay above 45mm.
    clamp_safety_threshold_mm: float = 45.0
    # Lowered 50.0 -> 45.0 (2026-09-09). A VERY_LOW goal was aborted outright at
    # best=48.3mm of 33 branches: above the 45mm clamp floor, but under the old
    # 50mm preflight bar, so no grasp was attempted at all.
    # This is now equal to clamp_safety_threshold_mm, i.e. no preflight margin
    # above the trajectory-level guard -- a branch admitted at 45-46mm here can
    # still be rejected later by the per-waypoint clamp check in motions.py.
    # very_low_preflight_early_accept_mm stays at 50.0 on purpose: the search
    # still stops early on a >=50mm branch and still prefers the safest branch
    # within very_low_clearance_window_mm, so this relaxes the FLOOR, not the
    # preference.
    very_low_preflight_min_clearance_mm: float = 45.0
    # Short-circuit the branch search at this clearance. Effective value is
    # max(min_clearance, this). At 50.0 with a 45.0 floor, a goal whose best
    # branch is e.g. 48.3mm never early-accepts, so the search runs to
    # exhaustion over every pitch x wrist x seed before picking a winner --
    # measured as a 12s stall between plan confirmation and APPROACH_ALIGNMENT.
    # 48.0 stops on the first branch comfortably above the floor while still
    # preferring high clearance and never going below very_low_preflight_min.
    very_low_preflight_early_accept_mm: float = 48.0
    very_low_preflight_preferred_pitch_deg: float = 0.0   # straight-on approach tried first
    very_low_preflight_second_pitch_deg: float = -30.0
    very_low_preflight_preferred_wrist_deg: float = 0.0
    very_low_preflight_pitch_step_deg: float = 10.0
    very_low_preflight_pitch_steps: int = 3
    very_low_clearance_window_mm: float = 8.0   # choose path cost only among near-safest IK branches
    very_low_center_cy_thresh: float = 0.88     # image cy; force center-home handling for very low fruit
    side_approach_enabled: bool = False         # enable LEFT/RIGHT side-home and side-low approach routing
    low_side_home_min_goal_z_m: float = 0.55    # below this, skip fixed side-low HOME; use target-local very-low approach


    # Offset names below are semantic for the old robot:
    # depth -> Y, lateral -> X
    side_home_lateral_offset: float = 0.16      # m; side-home lateral distance from trunk
    side_home_x_offset: float = 0.16            # legacy alias for side_home_lateral_offset
    center_partial_reverse_m: float = 0.20      # m; controlled retreat after a center grasp
    side_home_partial_reverse_m: float = 0.30   # m; extra clearance for side-approach fruits


    mid_center_approach_depth_offset: float = 0.10
    mid_center_approach_y_offset: float = 0.10  # legacy alias for mid_center_approach_depth_offset
    mid_center_approach_z_offset: float = 0.0  # legacy; mid/high APPROACH now uses FINAL grasp Z
    mid_high_left_thresh: float = 0.20      # bunch/image rel-x below this is MID/HIGH LEFT
    mid_high_right_thresh: float = 0.80     # bunch/image rel-x above this is MID/HIGH RIGHT
    low_left_thresh: float = 0.32           # bunch/image rel-x below this is LOW LEFT
    low_right_thresh: float = 0.68          # bunch/image rel-x above this is LOW RIGHT
    bunch_edge_side_band: float = 0.20        # rel-x within outer 20% of bunch is forced LEFT/RIGHT
    bunch_lower_center_band: float = 0.80     # rel-y >= this is forced VERY LOW/CENTER



    low_center_approach_depth_offset: float = 0.07
    low_center_approach_y_offset: float = 0.07  # legacy alias for low_center_approach_depth_offset
    low_center_approach_z_offset: float = 0.001  # legacy; center approach is derived from insertion angle


    very_low_center_approach_depth_offset: float = 0.07
    very_low_center_approach_y_offset: float = 0.07  # legacy alias for very_low_center_approach_depth_offset
    very_low_center_approach_z_offset: float = 0.001  # legacy; center approach is derived from insertion angle


    low_center_insertion_pitch_deg: float = 12.0     # approach from below, then insert upward
    very_low_center_approach_pitch_deg: float = 0.0  # legacy; orientation follows the insertion vector
    low_center_forward_align_max_deg: float = 45.0    # align gripper local +Z with insertion direction
    low_center_yaw_align_enabled: bool = True         # face tool +Z horizontally toward the date
    low_center_yaw_align_max_deg: float = 35.0        # bounded world-Z correction
    dynamic_low_center_tool_roll: bool = False       # preserve current/aligned finger rotation
    low_center_tool_roll_default_deg: float = 0.0    # no added local +Z roll
    low_center_tool_roll_min_score: float = 0.60
    low_center_tool_roll_stable_frames: int = 4
    low_center_tool_roll_max_spread_deg: float = 8.0
    low_center_tool_roll_reset_jump_deg: float = 25.0


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




    # Lab vs outdoor: see _LAB_GRASP_OFFSETS / _OUTDOOR_GRASP_OFFSETS above.
    low_side_final_depth_offset: float = field(
        default_factory=lambda: _profile_offset("low_side_final_depth_offset"))
    low_side_final_y_offset: float = field(       # legacy alias for low_side_final_depth_offset
        default_factory=lambda: _profile_offset("low_side_final_depth_offset"))
    low_left_final_z_offset: float = field(       # m; left side-low gripper center offset
        default_factory=lambda: _profile_offset("low_left_final_z_offset"))
    low_right_final_z_offset: float = field(      # m; right side-low gripper center offset
        default_factory=lambda: _profile_offset("low_right_final_z_offset"))

    low_side_final_front_tilt_deg: float = 10.0 # max final +Z/front tilt toward fruit
    mid_center_approach_pitch_deg: float = 0.0 # local tool X pitch for MID/HIGH center; keep 0.0 to preserve approach→final orientation continuity

    # Scratch retune: stop 5mm shallower than the zero-depth baseline. Previous
    # tuned values were depth=-0.027m and Z=+0.007m for both centre classes.
    # Z remains zero and the physical TCP-to-closure correction is disabled.
    # Lab vs outdoor: see _LAB_GRASP_OFFSETS / _OUTDOOR_GRASP_OFFSETS above.
    low_center_final_depth_offset: float = field(
        default_factory=lambda: _profile_offset("low_center_final_depth_offset"))
    low_center_final_y_offset: float = field(   # legacy alias for low_center_final_depth_offset
        default_factory=lambda: _profile_offset("low_center_final_depth_offset"))
    low_center_final_z_offset: float = field(
        default_factory=lambda: _profile_offset("low_center_final_z_offset"))

    mid_center_final_depth_offset: float = field(
        default_factory=lambda: _profile_offset("mid_center_final_depth_offset"))
    mid_center_final_y_offset: float = field(   # legacy alias for mid_center_final_depth_offset
        default_factory=lambda: _profile_offset("mid_center_final_depth_offset"))
    mid_center_final_z_offset: float = field(
        default_factory=lambda: _profile_offset("mid_center_final_z_offset"))
    mid_center_slip_final_z_offset: float = field(  # m; slightly lower final target during slip retry
        default_factory=lambda: _profile_offset("mid_center_slip_final_z_offset"))

    # Tool-frame TCP-to-physical-closure correction. Lab vs outdoor: see
    # _LAB_GRASP_OFFSETS / _OUTDOOR_GRASP_OFFSETS above.
    closure_center_offset_tcp_m: List[float] = field(
        default_factory=lambda: _profile_offset("closure_center_offset_tcp_m"))
    envelop_closure_center_offset_tcp_m: List[float] = field(
        default_factory=lambda: _profile_offset("envelop_closure_center_offset_tcp_m"))
    # visual F1/F2/F3 -> hardware force-channel indices. Keep identity until
    # the controlled single-finger correspondence test establishes otherwise.
    grasp_visual_to_force_map: List[int] = field(
        default_factory=lambda: [0, 1, 2])
    phase4_safe_final_yaw_enabled: bool = False # keep finger roll fixed while testing approach yaw
    phase4_safe_final_yaw_max_deg: float = 35.0
    phase4_candidate_target_match_m: float = 0.05
    final_approach_yaw_enabled: bool = False     # deprecated unsafe FINAL-only experiment
    final_approach_yaw_max_deg: float = 30.0
    approach_date_axis_enabled: bool = False     # ellipse finger rotation disabled; logging remains available
    approach_date_axis_max_deg: float = 35.0
    approach_date_axis_min_confidence: float = 0.20
    approach_date_axis_stable_frames: int = 5
    approach_date_axis_max_spread_deg: float = 25.0
    approach_candidate_roll_enabled: bool = False # score/log only; keep calibrated finger orientation
    approach_candidate_roll_max_deg: float = 15.0
    approach_candidate_roll_target_match_m: float = 0.05
    approach_candidate_roll_min_score: float = 0.85
    approach_candidate_roll_min_finger_score: float = 0.75
    final_lock_measured_approach_orientation: bool = True # prevent wrist correction during straight insertion
    target_specific_approach_enabled: bool = False # superseded by explicit mid/high corridor candidates
    target_specific_approach_max_deg: float = 25.0
    mid_high_corridor_approach_enabled: bool = True
    mid_high_corridor_inner_angle_deg: float = 10.0
    mid_high_corridor_side_angle_deg: float = 20.0
    mid_high_corridor_wide_angle_deg: float = 30.0
    mid_high_corridor_outer_angle_deg: float = 45.0
    mid_high_direction_min_horizontal: float = 0.15
    corridor_date_axis_enabled: bool = True
    corridor_date_axis_min_confidence: float = 0.20
    # A fitted ellipse gives an UNDIRECTED major axis, so the mirrored LEFT/RIGHT
    # corridor can be equally valid -- that is what the mirror preflight exists to
    # resolve. But it used to run whenever the axis was merely valid, so a clearly
    # decided heading (e.g. match 0.997 vs mirror 0.814) still paid ~1s of mirror
    # IK per goal. Only treat polarity as ambiguous when the two direction_match
    # scores are within this margin of each other.
    corridor_polarity_margin: float = 0.10
    # Max single-joint change from the current state that a corridor's approach
    # IK may require. Above this the solution is a wrist/elbow flip, which would
    # swing the arm through the canopy -- rejected as APPROACH_BRANCH_SWITCH.
    # Previously only a getattr default in goals.py, so it could not be seen or
    # tuned. Value unchanged; raise it only with very good reason.
    corridor_preflight_approach_max_joint_delta_deg: float = 75.0
    # Diagnostic run (2026-09-09): emit the ranked corridor list AND promote the
    # per-corridor [CORRIDOR_PREFLIGHT] PASS/REJECT line (reason + branch_cost)
    # from debug to info. Set back to False once the corridor question is
    # settled -- it is several lines per goal.
    log_corridor_candidates: bool = True
    tool_axis_tip_aim_enabled: bool = True
    tool_axis_tip_max_age_s: float = 0.50
    # The detected tip is naturally displaced from the fruit centroid.  Keep
    # the proven 80 mm association gate; 15 mm rejected valid date tips and
    # forced a corridor-only grasp that contacted with one finger first.
    tool_axis_tip_max_goal_distance_m: float = 0.08
    # Corridor selection uses the vision date axis (/datefruit_direction), which
    # is published for whatever fruit vision currently rates best and is stored
    # without any link to a position. With several goals queued from one
    # detection pass that let goals 2..N inherit goal 1's axis. Require the axis
    # to be associated with the goal being approached -- same tip-to-goal test
    # the tool-axis aiming uses -- and fall back to the goal's own camera-ray
    # direction when it isn't. False restores the old unconditional behaviour.
    corridor_axis_require_goal_match: bool = True
    corridor_axis_max_age_s: float = 2.0
    # Matches subscribe_multi_goals' DISTINCT_DIST: detections >=5cm apart are
    # already treated as separate fruits, so an axis whose tip sits further than
    # this from the goal belongs to a different date. Tighter on purpose than
    # tool_axis_tip_max_goal_distance_m, which answers a different question
    # ("close enough to aim with", not "same fruit").
    corridor_axis_max_goal_distance_m: float = 0.05
    # Max age of a FROZEN_RETRY tip snapshot. Corridor preflight retries reuse
    # the tip captured on the first attempt so corridors are compared against a
    # fixed target -- but that snapshot had no expiry, so a tip from many
    # seconds and several retries ago still moved the grasp point. Observed:
    # "[CORRIDOR_AXIS] ... (tip 2.5s old); using goal-local camera-ray direction"
    # immediately followed by "[TOOL_AXIS_AIM] source=FROZEN_RETRY
    # goal_match=21.7mm" -- the same tip rejected as stale for corridor choice
    # and used anyway for aiming. Age is measured from the vision timestamp of
    # the frozen reading, so it counts both how old the reading was when frozen
    # and how long retrying has taken since.
    tool_axis_tip_frozen_max_age_s: float = 2.0
    tool_axis_tip_require_safe_candidate: bool = False
    tool_axis_tip_max_swing_deg: float = 45.0
    # Corridors to try before abandoning tip-aligned search and retrying from the
    # detected date centre. The preflight budget is 9, but at 2 the search gave
    # up after two rejections and fell through to a STRAIGHT approach 44deg off
    # the date axis, which closed on nothing. Raised to 5 so the ranked list is
    # actually worked down. Revert to 2 if the extra IK time is not paying for
    # itself -- each rejected corridor costs one preflight chain.
    tool_axis_tip_fallback_after_corridors: int = 5  # then retry from detected centre

#---------------------------------------------------------------------------------------------------------
    final_overshoot_threshold: float = 0.004    # m; correct only if TCP passes target by >4mm
    final_overshoot_max_backoff: float = 0.012  # m; max one-shot pullback before closing
    final_endpoint_tolerance: float = 0.004     # m; require FINAL TCP within 4mm before closing
    final_visual_verification_s: float = 0.4    # observation-only camera window at FINAL before close
    phase5_center_logging_enabled: bool = True  # estimate residual centering only; never command motion
    phase5_center_px_per_mm: float = 4.0        # provisional image scale; validate from controlled moves
    phase5_center_max_correction_mm: float = 3.0 # clamp each logged image-axis suggestion
    phase5_center_deadband_px: float = 4.0      # residual below this is reported as centered
    reverse_initial_wait: float = 0.08          # s; minimum wait after publishing partial reverse (capped at call site to 0.25*expected_duration)
    reverse_final_settle: float = 0.0           # direct handoff to pre-planned DROP-OFF
    # Reverse was slow for three stacked reasons: this 4x timestep, the velocity
    # cap below, and the decel tail. The timestep dominates -- 4.0 gave dt=40ms
    # vs HOME's 10ms. 2.0 (dt=20ms) halves the whole reverse AND halves the decel
    # tail's wall-clock time, without touching the tail's point count.
    reverse_dt_multiplier: float = 2.0          # dt = this * min_dt => 20ms
    # Target wall-clock duration for the partial reverse. The reverse replays the
    # stored approach path sample-for-sample, so without this its duration is a
    # side effect of how densely the approach was sampled (densifying the entry
    # leg to 148 states made the reverse LONGER despite dt being halved).
    # Resampling to this target keeps the path identical and only changes sample
    # spacing. 0 disables resampling and restores the old sample-for-sample replay.
    reverse_duration_s: float = 1.4
    # Halving dt doubles the central-difference velocities (measured 0.11-0.16
    # rad/s -> 0.22-0.32). At the old 0.22 the cap was 1.5*0.22 = 0.33 rad/s, so
    # they would land right on it and get clipped -- and clipping scales velocity
    # while positions stay put, making (pos, vel) inconsistent for the controller.
    # 0.35 (the code's own fallback default) restores headroom: cap = 0.525 rad/s.
    reverse_velocity_scale: float = 0.35        # dedicated joint-velocity cap for reverse only
    reverse_acceleration_scale: float = 0.20    # dedicated joint acceleration limit multiplier
    # Final path samples reshaped into a zero-slope ease-out (see ease_out_tail).
    # At the reverse dt of ~40ms, 20 points meant 0.8s of deliberate crawl -- about
    # the last third of the reverse's duration covering very little distance. 8
    # still decelerates smoothly into the forced zero-velocity endpoint (avoiding
    # the one-timestep hard brake this exists to prevent) in ~0.32s instead.
    # The stop itself only exists because reverse and DROP-OFF are published as
    # separate trajectories; merging them would remove the need for a tail at all.
    reverse_decel_tail_points: int = 8
    # FINAL runs at dt=max_dt (12.5Hz, see max_dt below) and hands off into
    # GRASP/REVERSE at a much higher rate. Ease FINAL's own tail the same way
    # REVERSE's tail already is, so it decelerates into that handoff instead of
    # hard-braking at the forced-zero-velocity endpoint. See plan_and_send().
    final_decel_tail_points: int = 8
    # ALIGNMENT is usually the first real motion after a HOME/return move (which
    # runs at min_dt=100Hz via a separate trajectory message, motions.py). Ease
    # ALIGNMENT's own head the mirror-image way FINAL's tail is eased, so the
    # first motion of a new cycle ramps up instead of jumping straight to speed.
    # See plan_and_send().
    alignment_ease_head_points: int = 8
    grasp_post_close_settle_s: float = 0.50      # visible hold after fingers finish closing
    grasp_pair_capture_timeout_s: float = 0.25   # bound each logging-only camera-frame wait
    skip_redundant_visual_grasp_after_temporal: bool = True  # reuse fresh after-reverse frame path
    post_reverse_verification_enabled: bool = False  # no camera/force pause; dispatch pre-planned dropoff immediately


    hold_check_settle_s: float = 0.03           # s; force settle before post-reverse hold samples
    hold_check_window_s: float = 0.20           # s; median force sample window after reverse
    hold_check_sample_dt: float = 0.04          # s; post-reverse force sample period
    hold_force_finger_threshold_n: float = 3.0  # above empty-close maximum (1.8N)
    hold_force_min_fingers: int = 2
    hold_force_strong_single_n: float = 8.0     # verified proper centre finger was ≥6N; outer ≥36N
    hold_force_sum_threshold_n: float = 12.0
    force_retry_after_reverse: bool = False     # enable only after ground-truthed success/miss calibration
    max_force_grasp_retries: int = 2
    dropoff_preplan_wait_s: float = 15.0        # wait for background plan before starting another planner


    pre_dropoff_z_offset: float = -0.1  # m above dropoff
    pre_dropoff_depth_offset: float = 0.25
    pre_dropoff_y_offset: float = 0.25  # legacy alias for pre_dropoff_depth_offset

    # === REACQUIRE ===
    reacquire_after_approach: bool = True   # re-detect fruit position after reaching approach standoff
    # 1.5s, not 1.0s. Reacquire was ~1.1s of a 22.3s harvest cycle, so latency
    # here is not the constraint -- and both field runs timed out at 0.9s with
    # the date already measured well inside tolerance.
    reacquire_timeout_s: float = 1.5
    require_reacquire_before_grasp: bool = False  # allow stable-seed fallback when close-view reacquisition times out
    reacquire_depth_settle_s: float = 0.10  # brief depth/RGB synchronization settle
    # One distinct published measurement is enough. Vision's own
    # MIN_FRAMES_TO_SHOW=2 already requires a fruit to appear in two consecutive
    # frames under the same ID before it is published at all, so demanding two
    # further distinct samples here double-counts the same confirmation. Field
    # runs consistently reached stable=1/2 and were rejected while the date sat
    # 4-8mm from the seed. Raise again once vision publishes stable tracks at a
    # steady rate (per-track association + filtering).
    reacquire_stable_frames: int = 1
    # Reacquire detection-quality gate. Until this existed, reacquire matched on
    # position proximity alone -- any detection inside the match radius was
    # accepted however bad it was, because the quality topics
    # (depth_diagnostics / fruit_score / bbox_norm) are published inside
    # _process_best_target, which the vision node SKIPS in reacquire mode.
    # Quality now arrives on /vision/all_fruit_quality, which is published in
    # reacquire mode too.
    #
    # Defaults are deliberately permissive: a rejected candidate costs a
    # reacquire (falling back to the seed, i.e. today's behaviour), so start by
    # only excluding clearly bad detections and tighten from field logs. Every
    # rejection is logged with the failing metric.
    reacquire_quality_gate: bool = True
    reacquire_min_vis_ratio: float = 0.25     # fraction of mask with valid depth
    reacquire_max_z_std_m: float = 0.025      # depth scatter across the fruit
    reacquire_min_confidence: float = 0.20    # YOLO confidence
    # 0.0 means the bbox touches the frame border, i.e. the date is cut off --
    # the close-range failure mode, and it also corrupts the bbox-derived radius
    # that feeds the surface->centre push.
    reacquire_min_edge_margin: float = 0.0
    slip_check_reacquire: bool = False      # query depth after grasp to detect fruit slip
    regrip_after_slip: bool = False         # attempt regrip correction when grip is weak after slip
    subscribe_goal_min_settle_s: float = 0.25  # ignore first depth samples after Subscribe
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
    concise_console_logs: bool = True  # operator view; warnings/errors and cycle results remain visible
    log_gripper_force_profile: bool = False  # enable only while collecting force calibration data

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
    grasp_mode: str = "AUTO"             # AUTO chooses NORMAL/ENVELOP per accepted goal
    auto_envelop_axis_from_vertical_deg: float = 45.0
    auto_envelop_min_axis_confidence: float = 0.30
    normal_depth_extra_m: float = 0.0
    normal_z_extra_m: float = 0.0
    envelop_depth_extra_m: float = 0.017 # place fruit 17mm deeper than NORMAL
    envelop_z_extra_m: float = 0.005
    adaptive_aperture_enabled: bool = False  # keep fully open during approach
    min_fingers_for_stop: int = 2
    stop_closing_on_force: bool = False  # always command the full closed posture for now
    closing_steps: int = 10
    step_delay_s: float = 0.05
    close_feedback_trim_enabled: bool = True
    close_feedback_trim_iterations: int = 4
    close_feedback_trim_tolerance_rad: float = 0.0175  # about 1 degree
    close_feedback_trim_max_correction_rad: float = 0.1745  # 10 degrees
    close_feedback_trim_settle_s: float = 0.25
    holding_force_enabled: bool = True
    # Squeeze applied to the three primary closing motors (M4/M7/M12) after the
    # position close finishes -- this is what actually holds the date, since the
    # close itself is position-controlled to the empty-closed posture and simply
    # stops where the fruit blocks it. At 50 (5 N) dates were closing to size and
    # then slipping. Raised to 70 (7 N); range is 0-200 (0-20 N), and too much
    # risks crushing the fruit, so step up rather than jumping to the ceiling.
    holding_force_command: int = 70  # DG-3F-M units: 0.1 N (70 = 7 N)
    holding_force_settle_s: float = 0.15
    holding_release_settle_s: float = 0.05
    opening_steps: int = 6
    opening_step_delay_s: float = 0.06
    open_settle_s: float = 0.10
    open_hold_repeats: int = 3
    open_hold_interval_s: float = 0.04
    # Non-tactile contact validation. Re-measured 2026-09-08 on the lab hand
    # against the corrected closed posture and the signed remaining_fraction
    # (empty close now reads ~0.00 instead of 0.15-0.43, so the position signal
    # finally means something). Directly measured, empty vs holding one date:
    #
    #             remaining_fraction        max current (mA)
    #   finger1    0.000 -> +0.155           96 -> 111
    #   finger2    0.000 -> -0.037          150 -> 150
    #   finger3    0.000 -> +0.206           41 ->  82
    #
    # POSITION is the usable signal and these are the midpoints of it. Finger 2
    # did not contact that date at all -- it closed PAST the empty posture, hence
    # negative -- so its bar stays high rather than being tuned to noise.
    #
    # CURRENT is deliberately permissive. Finger 1 moved only 15 mA on a 96 mA
    # idle (~15%, inside noise) and finger 2 moved 0 mA, so the old 220/120/220
    # thresholds were unreachable while finger 2's 120 sat BELOW its 150 mA
    # empty-hold and so read as gripping while holding nothing. Contact requires
    # current AND position, so these are set below the empty values to stop the
    # current term vetoing a real grasp; position carries the decision. Revisit
    # once the hold force change (5 N -> 7 N) is running, which should widen the
    # current separation and make a real current threshold worth setting.
    nontactile_contact_validation: bool = True
    nontactile_current_delta_a: List[float] = field(
        default_factory=lambda: [0.020, 0.020, 0.020])
    nontactile_remaining_fraction: List[float] = field(
        default_factory=lambda: [0.08, 0.50, 0.10])
    nontactile_min_contact_fingers: int = 2
    nontactile_feedback_max_age_s: float = 0.30

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
    semicircle_radius_m: float = 0.35
    semicircle_arc_deg: float = 180.0
    semicircle_use_base_angles: bool = True           # use fixed base-frame scan angles instead of centering arc on current TCP
    semicircle_start_deg: float = 0.0                 # front half in base frame; 0=+X, 90=+Y
    semicircle_end_deg: float = 180.0                 # opposite of the previous back-side sweep
    semicircle_z_offset_m: float = 0.0
    semicircle_face_target: bool = True               # yaw at every arc point so the LiDAR/tool axis faces the scan center
    semicircle_face_target_max_yaw_deg: float = 10.0  # small inward bias without forcing wrist/IK branch changes across the arc
    semicircle_fast_yaw_order: bool = True            # try the likely-safe yaw first by base angle; retain all yaw fallbacks
    semicircle_local_axis: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0])
    semicircle_adaptive_scan: bool = True             # allow skipped angles/variable radius instead of forcing a perfect arc
    semicircle_joint_bridge_step_deg: float = 10.0    # insert only as needed when a coarse arc step causes an IK-branch jump
    semicircle_radius_candidates_m: List[float] = field(
        default_factory=lambda: [0.22, 0.35])
    semicircle_min_points: int = 3
    semicircle_preflight_direction: str = "reverse"   # reverse is usually smoother for the base-frame front arc
    semicircle_preflight_fallback_opposite: bool = True
    semicircle_preview_cache_enabled: bool = True     # lidar_scan can reuse the most recent validated preview targets
    semicircle_preview_cache_max_age_s: float = 8.0
    semicircle_preview_cache_max_joint_delta_deg: float = 3.0
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
