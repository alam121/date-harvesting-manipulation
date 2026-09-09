"""Vision module configuration constants."""

import os
from pathlib import Path

from .calibration_profiles import (
    frame_from_profile,
    load_camera_profile,
    matrix_from_profile,
)

# Model paths — resolve from source tree (works from both src and install)
_SRC_MODELS = Path(__file__).resolve().parents[3] / "zed_date_detector" / "models"
if not _SRC_MODELS.exists():
    # Fallback: resolve from workspace src directory
    _SRC_MODELS = Path(os.getenv("COLCON_PREFIX_PATH", "")).parent / "src" / "zed_date_detector" / "models"
if not _SRC_MODELS.exists():
    _SRC_MODELS = Path("/home/datepalm2/manipulatorsdatepalm/ur_ws_new/src/zed_date_detector/models")
DEFAULT_MODELS_DIR = _SRC_MODELS
MODELS_DIR = os.getenv("UR10E_MODELS_DIR", str(DEFAULT_MODELS_DIR))


DEFAULT_WEIGHTS = os.path.join(
    MODELS_DIR, "yolo26_small_zedone4k_3classes_jetson.engine")

#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolo26_s_p2.engine")

# YOLO inference parameters
DEFAULT_CONF_THRES = 0.1          # confidence threshold (lower = more detections)
DEFAULT_IMG_SIZE = 1248            # fallback only; for .engine weights the real
                                   # size is read from the engine at load time
                                   # (see engine_imgsz in yolo_thread.py) and
                                   # this value is overridden to match.

# Scoring System Weights (tune these for your application)
SCORE_WEIGHTS = {
    "distance": 0.90,       # HIGHEST PRIORITY: closer to gripper is better
    "visibility": 0.15,     # higher vis_ratio is better
    "depth_quality": 0.05,  # lower z_std is better
    "confidence": 0.10,     # YOLO detection confidence
    "ellipse": 0.05,        # bonus for valid ellipse fit (orientation reliability)
    "center_bias": 0.05,    # prefer fruits near frame center (better depth data)
}

# Distance scoring parameters
DIST_MIN = 0.10  # best possible distance (m)
DIST_MAX = 0.60  # tighter range so small differences (e.g. 0.28 vs 0.31) matter more

# Depth quality parameters
Z_STD_IDEAL = 0.005   # ideal depth std (m)
Z_STD_WORST = 0.05    # worst acceptable depth std (m)

# Sticky bonus: how much to prefer the previous best fruit
STICKY_BONUS = 0.02   # small bonus — distance should override easily

# Hysteresis: only switch to new target if it beats current best by this margin
SWITCH_THRESHOLD = 0.02  # low threshold — switch quickly to closer fruit

# Best fruit tracking
BEST_REUSE_THRESH = 0.08  # 8 cm positional tolerance in base_link

# Collision avoidance parameters.
#
# NOTE: these are RADII, not diameters. The previous values (min 20mm radius =
# 40mm diameter) were larger than a real date, so estimate_fruit_radius() clamped
# every fruit to the floor and the estimate carried no information. They were
# also masking the hardcoded fx=700 in that function, which inflated every radius
# by fx_real/700 (~1.8x at QHDPLUS) -- the clamp hid the error.
# Set for the harvested cultivar: 20-25mm diameter, i.e. 10.0-12.5mm radius
# (confirmed 2026-09-09).
#
# MIN sits below the real range on purpose. A partially occluded date has a
# smaller visible bbox and so estimates small; clamping it up to 8mm under-states
# the radius, which under-states the surface->centre push below. That is the safe
# failure direction -- the goal stops short of the centre, still inside the
# fruit, rather than being driven out the back of it.
#
# MAX is the important guard. It is NOT "the largest date", it is the cap on how
# wrong a bad bounding box is allowed to make us. Two merged detections or a
# bbox that swallowed part of the bunch would otherwise estimate a huge radius,
# and FRUIT_SURFACE_TO_CENTER_FRACTION would push the goal that much past the
# surface. At 18mm the worst-case push is 12.6mm; at the old 60mm it was 42mm.
FRUIT_RADIUS_DEFAULT = 0.011  # default radius of a date fruit (22mm diameter)
FRUIT_RADIUS_MIN = 0.008      # floor: 16mm diameter (occluded/partial detections)
FRUIT_RADIUS_MAX = 0.018      # ceiling: 36mm diameter (caps bad-bbox over-push)

# Grasp goals must name the fruit CENTRE, but every depth backend measures the
# front surface facing the camera. For a sphere of radius r the median depth of
# the visible cap sits ~0.71*r in front of the centre, so the goal was
# consistently that far short. Push the measured point back along the camera ray
# by this fraction of the estimated radius. The trunk already gets the equivalent
# correction (see _publish_trunk_position). Set to 0.0 to disable.
FRUIT_SURFACE_TO_CENTER_FRACTION = 0.7
APPROACH_CHECK_DIST = 0.20    # how far back to check for collisions (20cm)
NUM_CANDIDATE_DIRS = 12       # number of directions to sample

# Fruit ID tracking (prevent re-targeting same fruit)
FRUIT_ID_TIMEOUT = 30.0       # seconds before forgetting a fruit ID
MAX_FRUIT_ATTEMPTS = 3        # max grasp attempts before blacklisting

# Temporal stabilization (reduce flickering)
MIN_FRAMES_TO_SHOW = 2        # Fruit must appear in N consecutive frames before showing
MAX_FRAMES_TO_KEEP = 2        # Keep fruit for N frames after last seen (persistence)

# Target lock state
TARGET_LOCK_RADIUS = 0.06     # 6cm - match locked target within this radius

# Rendering knobs to save CPU (publishing unaffected)
DRAW_ONLY_BEST = False
DRAW_TOP_N = 10
SHOW_REJECTED = True
SKIP_DRAW = False
SHOW_CLASSIFICATION_ZONES = False
SHOW_GAP_DEBUG = False
SHOW_FINGER_CONTACTS = True

# Selected-date three-finger contact visualisation.  These values affect only
# the predicted overlay/quality score; they do not alter gripper commands.
FINGERTIP_CONTACT_RADIUS_M = 0.006
FINGER_CONTACT_RADIAL_FRACTION = 0.68
FINGER_CONTACT_ROTATION_SAMPLES = 24
FINGER_CONTACT_MIN_SCORE = 0.60
CLASS_ZONE_MID_LEFT_THRESH = 0.20
CLASS_ZONE_MID_RIGHT_THRESH = 0.80


CLASS_ZONE_LOW_LEFT_THRESH = 0.32
CLASS_ZONE_LOW_RIGHT_THRESH = 0.68

# Z limit in base_link frame
Z_MAX = 1.50
Z_STD_IDEAL = 0.005   # ideal depth std (m)

# Camera frame name. The active calibration profile is selected by the launcher
# via UR10E_CAMERA_PROFILE / UR10E_CAMERA_MODE. Explicit frame env vars remain
# as emergency overrides for field debugging.
CAMERA_PROFILE = load_camera_profile()
CAM_FRAME = os.getenv(
    "UR10E_CAM_FRAME",
    frame_from_profile(CAMERA_PROFILE, "rgb_frame", "zed2_left_camera_frame"),
)
ZEDMINI_CAM_FRAME = os.getenv(
    "UR10E_ZEDMINI_CAM_FRAME",
    frame_from_profile(CAMERA_PROFILE, "depth_frame", "zed_mini_left_camera_frame"),
)

# Class filtering
# When enabled, only GOAL_CLASS_NAME is used as a grasp target.
# Other classes (e.g. trunk) are shown in visualization but never become goals.
# Set to False for single-class models (e.g. dates-only) where all detections are goals.
CLASS_FILTER_ENABLED = True
GOAL_CLASS_NAME = "date-fruits-77rw"  # class picked as grasp target (must match model.names)
VIZ_ONLY_CLASSES = ["trunk"]          # classes shown in visualization only

# Trunk depth correction along the configured forward/back axis.
# Positive value moves pole closer to robot. Tune per setup.
TRUNK_DEPTH_OFFSET = 0.0

# ── ZED X One Mono (SN57931814) — QHD+ / QHDPLUS with HDR ───────────────────
# QHDPLUS (3200x1800) is the maximum resolution that supports HDR on ZED X One.
# Intrinsics are read at runtime from zed.get_camera_information() — values below
# are kept as reference only (FHD1200 from SN57931814.conf).
ZEDXONE_IMAGE_TOPIC = "/zedxone/image_raw"
ZEDXONE_WIDTH  = 3200   # QHDPLUS
ZEDXONE_HEIGHT = 1800
ZEDXONE_FX = 738.615    # reference only — runtime values used instead
ZEDXONE_FY = 738.284
ZEDXONE_CX = 934.32
ZEDXONE_CY = 642.52
ZEDXONE_DIST = [-0.0137106, -0.0304117, 0.000282395, -0.000401063, 0.00817325]  # k1 k2 p1 p2 k3

# ── LiDAR integration ─────────────────────────────────────────────────────────
# When USE_LIDAR=True the ZED X One Mono provides images via ArgusBayerCapture
# (published by zedxone_ros node) and Livox Mid-70 provides all depth.
# Set to False for the stock ZED stereo depth pipeline.
USE_LIDAR = False
LIDAR_TOPIC = "/livox/lidar"
LIDAR_Z_MIN = 0.10    # minimum valid LiDAR depth (m)
LIDAR_Z_MAX = 5.0     # maximum valid LiDAR depth (m)
# Camera←LiDAR extrinsic (4×4, transforms points FROM LiDAR frame TO camera frame).
T_CAM_LIDAR = [
    [ 0.193945, -0.972218, -0.131065, -0.667081],
    [ 0.0420431, 0.141716, -0.989014, -0.305516],
    [ 0.980111,  0.186304,  0.0683602, -0.177982],
    [ 0.0,       0.0,       0.0,        1.0     ],
]

# ── Dual-camera: ZED One Mono (detection) + ZED X Mini (depth) ────────────────
# When --use_zed_mini is passed the ZED X One Mono supplies RGB to YOLO and the
# ZED X Mini is opened in-process (depth-only) to supply the point cloud.
# Both cameras are grabbed simultaneously; no LiDAR subscription is needed.
ZEDMINI_SERIAL = 0        # 0 = auto-detect (first available stereo ZED); set SN to pin
ZEDMINI_DEPTH_FPS = 15    # grab rate for the depth camera — must match ZED One (15 max at QHDPLUS)
ZEDMINI_RGBD_FPS = 30     # grab rate when ZED X Mini is used for both RGB + depth (mini-only mode; no ZED One pacing)
ZEDMINI_DEPTH_Z_MIN = 0.15   # minimum valid ZED Mini depth (m)
ZEDMINI_DEPTH_Z_MAX = 7.0    # reject distant background behind nearby fruit
# Fruit acceptance window. This used to sit at 0.20 -- 5cm above the sensor
# floor -- purely so the gripper's red fingertips could not be detected as ripe
# dates. The fingertips were recoloured on 2026-09-09, so that reason is gone and
# the window now matches the ZED X Mini's usable depth range exactly: anything
# the sensor can measure is eligible.
#
# This matters beyond a few extra centimetres. Hand-eye translation is ~16cm, so
# at the moment of grasp the camera sits roughly 20-30cm from the fruit -- right
# on the old boundary. At 0.20 the target was rejected exactly when the arm was
# closest to it, which is why the final approach was dead-reckoned from a frozen
# pose and why reacquire never had anything fresh to work with.
#
# Keep in step with ZEDMINI_DEPTH_Z_MIN: points below that are filtered out of
# the cloud upstream anyway, so a lower value here would have no effect.
FRUIT_CAMERA_Z_MIN = float(os.getenv("UR10E_FRUIT_CAMERA_Z_MIN", "0.15"))
FRUIT_CAMERA_Z_MAX = float(os.getenv("UR10E_FRUIT_CAMERA_Z_MAX", "0.70"))
if not 0.0 < FRUIT_CAMERA_Z_MIN < FRUIT_CAMERA_Z_MAX:
    raise ValueError(
        "Expected 0 < UR10E_FRUIT_CAMERA_Z_MIN < UR10E_FRUIT_CAMERA_Z_MAX"
    )
ZEDMINI_MAX_POINTS = 20000   # subsample dense depth cloud to this many points (match LiDAR density)
# Camera←ZedMini extrinsic (4×4, transforms points FROM ZED Mini left-cam frame
# TO ZED One Mono camera frame).  Fill in after extrinsic calibration.
_T_CAM_ZEDMINI_DEFAULT = [
    [  0.99997176,  -0.00567234,  -0.00492976,  -0.02432635],
    [  0.00516021,   0.99514099,  -0.09832485,   0.03795027],
    [  0.00546354,   0.09829664,   0.99514216,  -0.01018928],
    [  0.00000000,   0.00000000,   0.00000000,   1.00000000],
]  # extrinsic calibration 2026-04-26, 42 pairs, max residual 0.5mm
T_CAM_ZEDMINI = matrix_from_profile(
    CAMERA_PROFILE, "depth_to_rgb", _T_CAM_ZEDMINI_DEFAULT)
