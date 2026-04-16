"""Vision module configuration constants."""

import os
from pathlib import Path

# Model paths — resolve from source tree (works from both src and install)
_SRC_MODELS = Path(__file__).resolve().parents[3] / "zed_date_detector" / "models"
if not _SRC_MODELS.exists():
    # Fallback: resolve from workspace src directory
    _SRC_MODELS = Path(os.getenv("COLCON_PREFIX_PATH", "")).parent / "src" / "zed_date_detector" / "models"
if not _SRC_MODELS.exists():
    _SRC_MODELS = Path("/home/datepalm2/manipulatorsdatepalm/ur_ws_new/src/zed_date_detector/models")
DEFAULT_MODELS_DIR = _SRC_MODELS
MODELS_DIR = os.getenv("UR10E_MODELS_DIR", str(DEFAULT_MODELS_DIR))

#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolo_26_dates_trunk_bunch_seg.engine")
#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolo26_improved_exposure_data.engine")

DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolov26_small_zed_one4k.engine")

# YOLO inference parameters
DEFAULT_CONF_THRES = 0.1          # confidence threshold (lower = more detections)
DEFAULT_IMG_SIZE = 1504            # must match compiled TRT engine exactly

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
BEST_REUSE_THRESH = 0.05  # 5 cm positional tolerance in base_link

# Collision avoidance parameters
FRUIT_RADIUS_DEFAULT = 0.035  # default radius of a date fruit (3.5cm)
FRUIT_RADIUS_MIN = 0.020      # minimum fruit radius (2cm)
FRUIT_RADIUS_MAX = 0.060      # maximum fruit radius (6cm)
APPROACH_CHECK_DIST = 0.20    # how far back to check for collisions (20cm)
NUM_CANDIDATE_DIRS = 12       # number of directions to sample

# Fruit ID tracking (prevent re-targeting same fruit)
FRUIT_ID_TIMEOUT = 30.0       # seconds before forgetting a fruit ID
MAX_FRUIT_ATTEMPTS = 3        # max grasp attempts before blacklisting

# Temporal stabilization (reduce flickering)
MIN_FRAMES_TO_SHOW = 2        # Fruit must appear in N consecutive frames before showing
MAX_FRAMES_TO_KEEP = 1        # Keep fruit for N frames after last seen (persistence)

# Target lock state
TARGET_LOCK_RADIUS = 0.10     # 10cm - match locked target within this radius

# Rendering knobs to save CPU (publishing unaffected)
DRAW_ONLY_BEST = False
SHOW_REJECTED = True
SKIP_DRAW = False

# Z limit in base_link frame
Z_MAX = 1.50

# Camera frame name
CAM_FRAME = "zed2_left_camera_frame"

# Class filtering
# When enabled, only GOAL_CLASS_NAME is used as a grasp target.
# Other classes (e.g. trunk) are shown in visualization but never become goals.
# Set to False for single-class models (e.g. dates-only) where all detections are goals.
CLASS_FILTER_ENABLED = True
GOAL_CLASS_NAME = "date-fruits-77rw"  # class picked as grasp target (must match model.names)
VIZ_ONLY_CLASSES = ["trunk"]          # classes shown in visualization only

# Trunk depth correction: camera overestimates trunk distance (Y in base_link)
# Positive value moves pole closer to robot. Tune per setup.
TRUNK_Y_OFFSET = 0.0  # meters — added to detected Y (tune if camera depth is off)

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
ZEDMINI_DEPTH_FPS = 15    # grab rate for the depth camera
ZEDMINI_DEPTH_Z_MIN = 0.15   # minimum valid ZED Mini depth (m)
ZEDMINI_DEPTH_Z_MAX = 5.0    # maximum valid ZED Mini depth (m)
ZEDMINI_MAX_POINTS = 20000   # subsample dense depth cloud to this many points (match LiDAR density)
# Camera←ZedMini extrinsic (4×4, transforms points FROM ZED Mini left-cam frame
# TO ZED One Mono camera frame).  Fill in after extrinsic calibration.
T_CAM_ZEDMINI = [
    [  0.99994366,  -0.00709704,   0.00789350,  -0.02646186],
    [  0.00731502,   0.99958288,  -0.02793839,   0.03122307],
    [ -0.00769193,   0.02799455,   0.99957848,  -0.01490936],
    [  0.00000000,   0.00000000,   0.00000000,   1.00000000],
]
