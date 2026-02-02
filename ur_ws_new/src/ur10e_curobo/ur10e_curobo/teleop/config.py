"""Joystick teleop configuration."""

# =============================================================================
# LOGITECH EXTREME 3D PRO MAPPINGS (as reported by: jstest /dev/input/js0)
# =============================================================================
# Axes:
#   0: X        (stick left/right)
#   1: Y        (stick forward/back)
#   2: Rz       (twist / rudder)
#   3: Throttle (slider)
#   4: Hat0X
#   5: Hat0Y
#
# Buttons:
#   0: Trigger
#   1: ThumbBtn
#   2: ThumbBtn2
#   3: TopBtn
#   4: TopBtn2
#   5: PinkieBtn
#   6: BaseBtn
#   7: BaseBtn2
#   8: BaseBtn3
#   9: BaseBtn4
#   10: BaseBtn5
#   11: BaseBtn6
# =============================================================================

# =============================================================================
# AXIS MAPPINGS
# =============================================================================
AXIS_X = 0            # Left / Right
AXIS_Y = 1            # Forward / Back
AXIS_YAW = 2          # Twist rotation (Rz)
AXIS_Z = -1           # Use buttons for Z (safer than throttle)
AXIS_THROTTLE = 3     # Optional speed scaling input (absolute, non-centering)

# Axis inversion (set to -1.0 to invert direction)
INVERT_X = 1.0
INVERT_Y = -1.0       # usually needed: push forward gives negative on many sticks
INVERT_YAW = 1.0
INVERT_Z = 1.0

# =============================================================================
# BUTTON MAPPINGS
# =============================================================================
# Safety gate: hold to allow motion (recommended for real UR10e)
BUTTON_ENABLE = 0            # Trigger (deadman)

# Gripper commands (avoid conflict with deadman)
BUTTON_GRIPPER_OPEN = 1      # ThumbBtn
BUTTON_GRIPPER_CLOSE = 2     # ThumbBtn2 (or 5 for PinkieBtn if you prefer)

# Z motion (hold)
BUTTON_Z_UP = 3              # TopBtn
BUTTON_Z_DOWN = 4            # TopBtn2

# Speed scaling (use base buttons; top buttons already used)
BUTTON_SPEED_UP = 10         # BaseBtn5
BUTTON_SPEED_DOWN = 11       # BaseBtn6

# Utility / safety
BUTTON_HOME = 6              # BaseBtn
BUTTON_STOP = 7              # BaseBtn2 (emergency stop)

# =============================================================================
# FILTERING
# =============================================================================
DEADZONE = 0.25               # increased to handle joystick calibration offset and trigger crosstalk
SMOOTHING_ALPHA = 0.2         # 0..1 (lower = smoother, higher = more responsive)

# =============================================================================
# COMMAND RATE
# =============================================================================
UPDATE_RATE_HZ = 50

# =============================================================================
# VELOCITY SETTINGS (Twist convention: m/s and rad/s)
# =============================================================================
MAX_LINEAR_VELOCITY = 0.02    # m/s  (2 cm/s at full deflection)
MAX_ANGULAR_VELOCITY = 0.10   # rad/s (~5.7 deg/s at full twist)

MIN_VELOCITY_SCALE = 0.10
MAX_VELOCITY_SCALE = 1.00
VELOCITY_SCALE_STEP = 0.10

DEFAULT_VELOCITY_SCALE = 0.30 # start at 30% for safety

# =============================================================================
# OPTIONAL: THROTTLE-BASED SPEED CONTROL (needs code support in your node)
# =============================================================================
THROTTLE_AS_SPEED = False     # True -> map AXIS_THROTTLE to velocity_scale
