#!/bin/bash
# Launch UR10e nodes in tmux with panes
# Usage: launch_ur10e_tmux [main] [vision] [teleop] [gui]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$SCRIPT_DIR"
SOURCE_CMD="source $WS/install/setup.bash"
SESSION="ur10e"

# Parse arguments
RUN_MAIN=false
RUN_VISION=false
RUN_TELEOP=false
RUN_GUI=false

for arg in "$@"; do
    case $arg in
        main)   RUN_MAIN=true ;;
        vision) RUN_VISION=true ;;
        teleop) RUN_TELEOP=true ;;
        gui)    RUN_GUI=true ;;
        *)      echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if ! $RUN_MAIN && ! $RUN_VISION && ! $RUN_TELEOP && ! $RUN_GUI; then
    echo "Usage: launch_ur10e_tmux [main] [vision] [teleop] [gui]"
    echo ""
    echo "Examples:"
    echo "  launch_ur10e_tmux main"
    echo "  launch_ur10e_tmux main teleop"
    echo "  launch_ur10e_tmux main vision teleop gui"
    exit 1
fi

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null

# Create new session with UR bringup
tmux new-session -d -s $SESSION -n ur10e
tmux send-keys -t $SESSION "$SOURCE_CMD && ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:=false launch_rviz:=true" Enter

# Split and add panes
if $RUN_MAIN; then
    tmux split-window -h -t $SESSION
    tmux send-keys -t $SESSION "sleep 5 && $SOURCE_CMD && ros2 run ur10e_curobo main" Enter
fi

if $RUN_VISION; then
    tmux split-window -v -t $SESSION:0.0
    tmux send-keys -t $SESSION "sleep 7 && $SOURCE_CMD && ros2 run ur10e_curobo vision" Enter
fi

if $RUN_TELEOP; then
    tmux split-window -v -t $SESSION:0.1
    tmux send-keys -t $SESSION "sleep 6 && $SOURCE_CMD && ros2 run ur10e_curobo teleop" Enter
fi

if $RUN_GUI; then
    tmux split-window -h -t $SESSION
    tmux send-keys -t $SESSION "sleep 8 && $SOURCE_CMD && ros2 run ur10e_curobo gui" Enter
fi

# Balance panes
tmux select-layout -t $SESSION tiled

# Attach to session
tmux attach-session -t $SESSION
