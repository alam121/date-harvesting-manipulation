#!/bin/bash
# Launch UR10e nodes in a single terminator window with panes
# Usage: launch_ur10e [main] [teleop] [vision] [gui]
# Layout for 5 panes:
#   +-------------+-------------+
#   | UR Bringup  |    Main     |
#   +------+------+------+------+
#   |Vision|Teleop| GUI  |      |
#   +------+------+------+------+

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$SCRIPT_DIR"
SOURCE_CMD="source $WS/install/setup.bash"

# Parse arguments
RUN_MAIN=false
RUN_TELEOP=false
RUN_VISION=false
RUN_GUI=false

RUN_CALIBRATE=false

for arg in "$@"; do
    case $arg in
        main)      RUN_MAIN=true ;;
        teleop)    RUN_TELEOP=true ;;
        vision)    RUN_VISION=true ;;
        gui)       RUN_GUI=true ;;
        calibrate) RUN_CALIBRATE=true ;;
        *)         echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if ! $RUN_MAIN && ! $RUN_TELEOP && ! $RUN_VISION && ! $RUN_GUI && ! $RUN_CALIBRATE; then
    echo "Usage: launch_ur10e [main] [teleop] [vision] [gui] [calibrate]"
    echo ""
    echo "Examples:"
    echo "  launch_ur10e main"
    echo "  launch_ur10e main teleop"
    echo "  launch_ur10e main vision gui"
    echo "  launch_ur10e calibrate          # grasp force calibration only"
    exit 1
fi

# Fixed order commands (order: ur_bringup, main, vision, teleop, gui)
CMD_UR="$SOURCE_CMD && ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:=false launch_rviz:=true"
CMD_MAIN="sleep 3 && $SOURCE_CMD && ros2 run ur10e_curobo main"
CMD_VISION="sleep 1 && $SOURCE_CMD && ros2 run ur10e_curobo vision"
CMD_TELEOP="sleep 6 && $SOURCE_CMD && ros2 run ur10e_curobo teleop"
CMD_GUI="sleep 8 && $SOURCE_CMD && ros2 run ur10e_curobo gui"
CMD_CALIBRATE="sleep 6 && $SOURCE_CMD && ros2 run ur10e_curobo calibrate"

# Count panes needed
NUM_PANES=1
$RUN_MAIN && ((NUM_PANES++))
$RUN_VISION && ((NUM_PANES++))
$RUN_TELEOP && ((NUM_PANES++))
$RUN_GUI && ((NUM_PANES++))
$RUN_CALIBRATE && ((NUM_PANES++))

echo "Launching $NUM_PANES panes..."

# Start terminator
terminator --maximise &
sleep 2

# Focus terminator
xdotool search --class "Terminator" windowactivate 2>/dev/null
sleep 0.5

# Helper function
type_cmd() {
    xdotool type --clearmodifiers --delay 5 "$1"
    sleep 0.1
    xdotool key --clearmodifiers Return
}

split_h() {
    xdotool key --clearmodifiers ctrl+shift+o
    sleep 0.4
}

split_v() {
    xdotool key --clearmodifiers ctrl+shift+e
    sleep 0.4
}

nav_up() {
    xdotool key --clearmodifiers alt+Up
    sleep 0.2
}

nav_down() {
    xdotool key --clearmodifiers alt+Down
    sleep 0.2
}

nav_left() {
    xdotool key --clearmodifiers alt+Left
    sleep 0.2
}

nav_right() {
    xdotool key --clearmodifiers alt+Right
    sleep 0.2
}

# Pane 1: UR Bringup (always)
type_cmd "$CMD_UR"

# Pane 2: Main (if requested)
if $RUN_MAIN; then
    split_v  # split right
    type_cmd "$CMD_MAIN"
fi

# Pane 3: Vision (if requested)
if $RUN_VISION; then
    nav_up 2>/dev/null || true
    nav_left 2>/dev/null || true
    split_h  # split down
    type_cmd "$CMD_VISION"
fi

# Pane 4: Teleop (if requested)
if $RUN_TELEOP; then
    split_v  # split right
    type_cmd "$CMD_TELEOP"
fi

# Pane 5: GUI (if requested)
if $RUN_GUI; then
    split_v  # split right
    type_cmd "$CMD_GUI"
fi

# Pane 6: Calibrate (if requested)
if $RUN_CALIBRATE; then
    split_v  # split right
    type_cmd "$CMD_CALIBRATE"
fi

echo "Done. Launched $NUM_PANES panes in single window."
