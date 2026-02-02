"""Vision module configuration constants."""

import os

# Model paths
MODELS_DIR = "/home/datepalm2/manipulatorsdatepalm/ur_ws_new/src/zed_date_detector/models"
DEFAULT_WEIGHTS = os.path.join(MODELS_DIR, "lab_dates_realbunch6.engine")

# YOLO inference parameters
DEFAULT_CONF_THRES = 0.2   # confidence threshold (lower = more detections)
DEFAULT_IMG_SIZE = 640      # inference size in pixels

# Scoring System Weights (tune these for your application)
SCORE_WEIGHTS = {
    "distance": 0.90,       # HIGHEST PRIORITY: closer is always better
    "visibility": 0.05,     # higher vis_ratio is better
    "depth_quality": 0.02,  # lower z_std is better
    "confidence": 0.07,     # YOLO detection confidence
    "ellipse": 0.04,        # bonus for valid ellipse fit (orientation reliability)
    "center_bias": 0.02,    # prefer fruits near frame center (better depth data)
}

# Distance scoring parameters
DIST_MIN = 0.10  # best possible distance (m)
DIST_MAX = 1.50  # worst acceptable distance (m)

# Depth quality parameters
Z_STD_IDEAL = 0.005   # ideal depth std (m)
Z_STD_WORST = 0.05    # worst acceptable depth std (m)

# Sticky bonus: how much to prefer the previous best fruit
STICKY_BONUS = 0.15   # added to score if this was the previous best

# Hysteresis: only switch to new target if it beats current best by this margin
SWITCH_THRESHOLD = 0.08  # new best must be 8% better to trigger switch

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
