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
#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "lab_dates_realbunch6.engine")
#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "weights_yolo26_v2_updated.engine")

#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolo_26_dates_trunk_seg.engine")

#DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolo_26_dates_trunk_seg_v2.engine")

DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "yolo_26_dates_trunk_bunch_seg.engine")

# YOLO inference parameters
DEFAULT_CONF_THRES = 0.2   # confidence threshold (lower = more detections)
DEFAULT_IMG_SIZE = 640      # inference size in pixels

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
SHOW_REJECTED = False
SKIP_DRAW = False

# Z limit in base_link frame
Z_MAX = 1.34

# Camera frame name
CAM_FRAME = "zed2_left_camera_frame"

# Class filtering
# When enabled, only GOAL_CLASS_NAME is used as a grasp target.
# Other classes (e.g. trunk) are shown in visualization but never become goals.
# Set to False for single-class models (e.g. dates-only) where all detections are goals.
CLASS_FILTER_ENABLED = True
GOAL_CLASS_NAME = "datefruit"  # class picked as grasp target (must match model.names)
VIZ_ONLY_CLASSES = ["trunk", "bunch"]  # classes shown in visualization only

# Trunk depth correction: camera overestimates trunk distance (Y in base_link)
# Positive value moves pole closer to robot. Tune per setup.
TRUNK_Y_OFFSET = 0.0  # meters — added to detected Y (tune if camera depth is off)
