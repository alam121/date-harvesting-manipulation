"""Configuration for the grasp learning system (force-profile based)."""

import os

GRASP_LEARNING_ENABLED = True  # master switch to enable/disable grasp learning features

# Data storage
DATA_DIR = os.path.expanduser("~/grasp_learning_data")
GRASP_LOG_PATH = os.path.join(DATA_DIR, "grasp_log.csv")
MODEL_PATH = os.path.join(DATA_DIR, "grasp_model.pkl")

# Force profile detection
# "Early contact" = force delta > CONTACT_DELTA_THRESHOLD on any finger
# If this happens before step (total_steps - LATE_STEP_MARGIN), object is present.
CONTACT_DELTA_THRESHOLD = 0.5  # Newtons — delta above baseline to count as contact
LATE_STEP_MARGIN = 2  # last N steps are mechanical-stop territory (not real contact)

# Learning hyperparameters
MIN_SAMPLES_TO_LEARN = 3
LEARNING_RATE = 0.3  # EMA weight for contact step threshold
DEFAULT_CONTACT_STEP_THRESHOLD = 7  # steps 0-6 = early contact, 7+ = probably air
