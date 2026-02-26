
# ManipulatorsDatePalm

## Current Latest
- Branch: `dev_current`
- Vision: external `date_v1.9.py` node (publishes `/external_goal_pose`)
- Manipulator: `ros2 run ur10e_curobo main`

## Overview

ROS 2 Humble control stack for UR10e robot arm for autonomous date palm harvesting:

1. **cuRobo** — GPU-accelerated motion planning (collision-free trajectories)
2. **MoveIt** — trajectory management and execution
3. **Delto 3-Finger Gripper** — force-profile based grasp with learning
4. **ZED Camera + YOLO** — 3D detection and goal generation
5. **Voxel Obstacles** — depth-based collision avoidance using ZED point clouds
6. **Grasp Learning** — learns grasp success from force profile during closure

## Command Setup (One-Time Per Machine)

Use repo-managed commands from `<repo>/bin` so cloning on another machine keeps the same command names.

```bash
cd <repo_root>
./bin/install_shell.sh
source ~/.bashrc
```

This adds `<repo_root>/bin` to `PATH` and sets `MANIPULATOR_REPO`.
It also keeps repo commands ahead of older `~/bin` wrappers if both exist.

## Quick Start

### Launch everything (real robot)
```bash
launch_ur10e main vision teleop gui
```

### Launch with fake hardware (simulation)
```bash
launch_ur10e fake main
```

### Available launch nodes
| Node | Description |
|------|-------------|
| `main` | Main control pipeline (approach, grasp, dropoff) |
| `vision` | ZED/YOLO vision node |
| `teleop` | Joystick teleop control |
| `gui` | Desktop GUI control panel |
| `calibrate` | Standalone grasp force calibration tool |
| `hand_eye` | Hand-eye (camera-to-gripper) calibration |

### Hand-Eye Calibration
Calibrate the ZED camera-to-gripper transform using a chessboard:
```bash
launch_ur10e hand_eye teleop
```
1. Place a chessboard flat in the workspace
2. Move the robot to 15-20 different poses (vary rotation and translation)
3. Press `c` to capture at each pose, `q` when done
4. Result saved to `vision/hand_eye_calibration.yaml`

### Grasp Calibration
Train the force-profile grasp learner without running the full pipeline:
```bash
launch_ur10e calibrate teleop
```
Keys: `c`=close, `o`=open, `y`=success, `n`=fail, `s`=stats, `q`=quit

Data stored in `~/grasp_learning_data/` (CSV log + pickle model).

## RViz Keyboard Shortcuts
When the RViz window is focused:

| Key | Action |
|-----|--------|
| H | Home position |
| D | Dropoff position |
| E | Execute stored goals |
| S | Subscribe to vision goals |
| O | Open gripper |
| C | Close gripper |
| U | Update voxel obstacles |
| Y | Grasp feedback: success |
| N | Grasp feedback: fail |

## Harvesting Pipeline

```
HOME → APPROACH → REACQUIRE → FINAL GRASP → CLOSE → PARTIAL REVERSE → DROPOFF → HOME
```

1. **Approach** — cuRobo plans collision-free path to detected fruit
2. **Reacquire** — vision re-locks fruit position with stability check
3. **Final Grasp** — slow IK-based precision move to fruit
4. **Close & Evaluate** — gripper closes, force profile analyzed (early contact = grabbed)
5. **Partial Reverse** — pull back ~12cm to clear the date bunch
6. **Dropoff** — cuRobo plans collision-free path to dropoff, release

### Grasp Force Profile Detection
During the 10-step gripper closure, force deltas are recorded at each step:
- **Grabbed fruit**: forces rise at step 5-7 (fingers contact object mid-closure)
- **Closed on air**: forces only spike at step 9-10 (mechanical stop)

The grasp learner uses an EMA threshold on the first-contact step to predict success.

## Dependencies
1. Install `ros2 humble` via debian package
2. Install [cuRobo](https://curobo.org/get_started/5_docker_development.html#docker-dev) (native on Jetson)
3. Install ROS packages:
```bash
sudo apt install ros-humble-ur-msgs ros-humble-ur-client-library \
  ros-humble-moveit ros-humble-moveit-servo ros-humble-moveit-visual-tools \
  ros-humble-moveit-resources ros-humble-moveit-planners-ompl \
  ros-humble-moveit-ros-perception ros-humble-ros2-control \
  ros-humble-ros2-controllers ros-humble-controller-interface \
  ros-humble-controller-manager ros-humble-control-toolbox \
  ros-humble-realtime-tools ros-humble-rviz2 ros-humble-rviz-visual-tools \
  ros-humble-ackermann-msgs
pip install warp-lang==1.0.0
```

## Build
```bash
cd <repo_root>/ur_ws_new
colcon build --symlink-install --packages-select ur10e_curobo
source install/setup.bash
```

## FastDDS Buffer Configuration
The system uses a custom FastDDS config (`ur_ws_new/fastdds_config.xml`) to increase UDP buffer sizes and suppress startup warnings. This is set automatically by `launch_ur10e.py` via `FASTRTPS_DEFAULT_PROFILES_FILE`.

If you see `sequence size exceeds remaining buffer` warnings, increase the kernel UDP buffer limits:
```bash
sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608 net.core.rmem_default=8388608 net.core.wmem_default=8388608
```

To make it permanent:
```bash
echo -e "net.core.rmem_max=8388608\nnet.core.wmem_max=8388608\nnet.core.rmem_default=8388608\nnet.core.wmem_default=8388608" | sudo tee /etc/sysctl.d/10-fastdds.conf && sudo sysctl --system
```

## Setup UR10e Connection
```bash
sudo ip addr flush dev eth0
sudo ip addr add 192.168.1.101/24 dev eth0
sudo ip addr add 169.254.186.100/24 dev eth0
sudo ip link set eth0 up
```
Robot IP: `192.168.1.190`

## Run Individually

### UR10e Bringup
```bash
ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:=false launch_rviz:=true
```

### Delto Gripper
```bash
ros2 launch delto_3f_driver delto_3f_bringup.launch.py delto_ip:=169.254.186.72 delto_port:=502
```

### Main Node
```bash
ros2 run ur10e_curobo main
```

## Key Configuration

Speed and motion parameters in `ur10e_curobo/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `global_speed_multiplier` | 5.0 | Scales all motion speeds |
| `speed_approach` | 1.0 | Approach to fruit |
| `speed_final` | 0.2 | Slow precision grasp |
| `speed_predropoff` | 0.3 | Reverse trajectory speed |
| `speed_dropoff` | 2.0 | Move to dropoff |
| `speed_home` | 0.5 | Return to home |

Gripper parameters in `ur10e_curobo/config.py` (`Gripper` dataclass):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `closing_steps` | 10 | Gripper closure steps |
| `step_delay_s` | 0.05 | Delay between steps |
| `use_suction` | false | Enable suction mode |

## Project Structure

```
ur_ws_new/src/ur10e_curobo/ur10e_curobo/
  main.py                  # Entry point
  node.py                  # Main ROS2 node
  goals.py                 # Harvesting pipeline (approach/grasp/dropoff)
  motions.py               # Trajectory building and execution
  config.py                # All configuration parameters
  delto_gripper_controller.py  # Gripper control + force profile
  grasp_learner.py         # Force-profile grasp learning
  grasp_calibrate.py       # Standalone calibration tool
  grasp_learning_config.py # Grasp learning parameters
  grasp_outcome_classifier.py  # Template-based grasp classifier
  gripper.py               # Gripper initialization
  voxel_obstacle.py        # Depth-based collision avoidance
  utils.py                 # Helpers (TF, wait, IK)
  markers.py               # RViz visualization
  managers/
    config_manager.py      # Parameter management
    state_manager.py       # Robot state tracking
    motion_executor.py     # Trajectory execution + teleop
  vision/
    node.py                # Vision processing node
    hand_eye_calibration.py
  teleop/
    teleop_node.py         # Joystick teleop
```

## Headless Machine Setup
```bash
ssh -X username@remote_ip
sudo nano /etc/ssh/sshd_config
# Ensure: X11Forwarding yes, X11DisplayOffset 10, X11UseLocalhost yes
sudo systemctl restart ssh
sudo apt install xauth
```

## YOLOv8 Training
```bash
pip install ultralytics
yolo task=segment mode=train model=yolov8n-seg.pt data=data.yaml epochs=50 imgsz=720 batch=8
yolo export model=best.pt format=onnx opset=12 imgsz=640 dynamic=False
/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.trt --explicitBatch --fp16
```
