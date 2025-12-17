
# ManipulatorsDatePalm 🤖🌴

An automated date palm harvesting system using UR10e robot arm with GPU-accelerated motion planning and AI-based vision detection.

## Current Latest
- **Branch**: `dev2`
- **Repository**: Documents/manipulatorsdatepalm
- **Vision Code**: `date_v1.7.py`
- **Motion Planning**: `python3 -m ur10e_curobo.main`

## Overview

This system provides a complete ROS 2 Humble-based control stack for the UR10e robot arm integrated with:

1. **cuRobo** - GPU-accelerated motion planning and collision avoidance
2. **MoveIt!** - Trajectory management and motion execution
3. **Delto 3-Finger Suction Gripper** - Grasp control via Modbus TCP
4. **ZED 2i Camera** - 3D stereo vision and depth perception
5. **YOLOv8 Segmentation** - AI-based date detection and localization
6. **RViz** - Live visualization and debugging

## System Architecture

```
ZED Camera → YOLOv8 Detection → ROS Topics → cuRobo Planning → UR10e Execution
                                              ↓
                                     Delto Gripper Control
```

## Hardware Requirements

- **Robot**: Universal Robots UR10e
- **Camera**: ZED 2i Stereo Camera
- **Gripper**: Delto 3-Finger Suction Gripper
- **Computer**: NVIDIA Jetson (or similar with CUDA support)
- **OS**: Ubuntu 22.04 with ROS 2 Humble

## Prerequisites

### 1. ROS 2 Humble Installation
Install ROS 2 Humble via debian package:
```bash
# Follow official ROS 2 Humble installation guide
# https://docs.ros.org/en/humble/Installation.html
```

### 2. NVIDIA CUDA and PyTorch
For Jetson devices, install PyTorch with CUDA support:
```bash
export CUDA_VERSION=11.4
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# Verify installation
python3 -c "import torch; print('torch:', torch.__version__, 'CUDA Available:', torch.cuda.is_available())"
```

### 3. ROS 2 Dependencies
Install all required ROS 2 packages:
```bash
sudo apt install -y \
  ros-humble-ur-msgs \
  ros-humble-ur-client-library \
  ros-humble-moveit \
  ros-humble-moveit-servo \
  ros-humble-moveit-visual-tools \
  ros-humble-moveit-resources \
  ros-humble-moveit-planners-ompl \
  ros-humble-moveit-ros-perception \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-interface \
  ros-humble-controller-manager \
  ros-humble-control-toolbox \
  ros-humble-realtime-tools \
  ros-humble-rviz2 \
  ros-humble-rviz-visual-tools \
  ros-humble-ackermann-msgs
```

### 4. Python Dependencies
```bash
pip install warp-lang==1.0.0
pip install ultralytics  # YOLOv8
```

### 5. cuRobo Installation
Install [cuRobo](https://curobo.org/get_started/5_docker_development.html#docker-dev) in Docker for native usage on Jetson:
```bash
# Follow cuRobo installation guide for Docker setup
```

## Installation

### 1. Clone Repository
```bash
git clone https://gitlab.kaust.edu.sa/risc/manipulatorsdatepalm.git
cd manipulatorsdatepalm
```

### 2. Build Workspace
```bash
cd ur_ws_new
colcon build --symlink-install
source install/setup.bash
```

## Network Configuration

### Setup UR10e Network Connection
The UR10e robot requires two IP addresses on the same interface:
- `192.168.1.101/24` - Robot communication
- `169.254.186.100/24` - Gripper communication

**Note**: Replace `eno1` with your network interface (use `ip a` to find it, e.g., `eth0`, `enp0s31f6`)

```bash
# Check your network interface name
ip a

# Configure network (replace eno1 with your interface)
sudo ip addr flush dev eno1
sudo ip addr add 192.168.1.101/24 dev eno1
sudo ip addr add 169.254.186.100/24 dev eno1
sudo ip link set eno1 up
```

**Verify configuration**:
```bash
ip a show eno1
```

Expected output:
```
3: eno1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    inet 169.254.186.100/24 scope global eno1
    inet 192.168.1.101/24 scope global eno1
```

**Important**: If IPs are not showing, repeat the configuration steps. If you close the terminal, you will need to reconfigure the network.

## YOLOv8 Setup

### 1. Train YOLOv8 Segmentation Model
```bash
yolo task=segment mode=train model=yolov8n-seg.pt \
  data=/path/to/dateopalm_tree.v4i.yolov8-obb/data.yaml \
  epochs=50 imgsz=720 batch=8
```

### 2. Export Model to ONNX
```bash
yolo export model=best.pt format=onnx opset=12 imgsz=640 dynamic=False
```

### 3. Convert ONNX to TensorRT Engine (Optional)
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=best.onnx \
  --saveEngine=best.trt \
  --explicitBatch \
  --fp16
```

### Additional Resources
- [YOLOv8 Segmentation DeepStream](https://github.com/marcoslucianops/DeepStream-Yolo-Seg/tree/master)
- [ZED GStreamer](https://github.com/stereolabs/zed-gstreamer)
- [GStream Deep Reference](https://github.com/valdivj/gstream_Deep)

## Running the System

### Step-by-Step Demo Instructions

#### Step 0: Environment Setup
```bash
# Set ROS Domain ID
export ROS_DOMAIN_ID=5

# Source workspace (if something unexpected happens, source again)
cd ~/Documents/manipulatorsdatepalm/ur_ws_new
source install/setup.bash
```

#### Step 1: Launch UR10e Robot Control
Open **Terminal 1**:
```bash
ros2 launch ur_bringup ur_control.launch.py \
  ur_type:=ur10e \
  robot_ip:=192.168.1.190 \
  use_fake_hardware:=false \
  launch_rviz:=true
```

This will:
- Launch robot drivers
- Start RViz visualization
- Display robot arm and gripper

**Troubleshooting**: If something goes wrong, check network IPs and re-source the workspace.

#### Step 2: Launch Delto Gripper
Open **Terminal 2**:
```bash
ros2 launch delto_3f_driver delto_3f_bringup.launch.py \
  delto_ip:=169.254.186.72 \
  delto_port:=502
```

#### Step 3: Start Robot Program on UR Tablet
1. Power on the robot using the UR tablet
2. Start the robot and release the brake
3. Go to **Program** tab
4. Load the **URCA file** (external control program)
5. Press the **START** button

**Note**: If you see `Connection to reverse interface dropped`, restart the program on the tablet.

**Important**: Make sure the START button is pressed, otherwise you'll see:
```
[ERROR] [ur10e_curobo_moveit_node]: ❌ Cannot execute goals: robot program is OFF. Please turn it ON first.
```

#### Step 4: Start Motion Planning Node
Open **Terminal 3**:
```bash
cd ~/Documents/manipulatorsdatepalm/ur_ws_new/src
python3 -m ur10e_curobo.main
```

#### Step 5: Start Vision Detection Node
Open **Terminal 4**:
```bash
cd ~/Documents/manipulatorsdatepalm/ur_ws_new/src/zed_date_detector
python3 date_v1.7.py --weights models/lab_dates_realbunch.pt --conf_thres 0.5
```

This will:
- Open OpenCV window with detections
- Publish detected date positions to ROS topics
- Fill `/external_goal_pose` topic with goal poses

#### Step 6: Subscribe to Goals and Execute Motion
In the **motion planning terminal (Terminal 3)**, use keyboard controls:

**Keyboard Controls**:
- **`s`** - Subscribe to **one goal** from vision
- **`p`** - Subscribe to **multiple goals** for 10 seconds
- **`n`** - **Execute** motion to subscribed goals
- **`o`** - **Open** gripper
- **`c`** - **Close** gripper

**Workflow**:
1. Press `s` or `p` to collect goal poses
2. Verify in RViz that red dots (goals) are showing
3. Press `n` to execute the motion sequence
4. Use `o` and `c` to control the gripper during operation

## Manual Control

### Gripper Control via ROS Topics
```bash
# Write to gripper registers
ros2 topic pub /gripper/write_register std_msgs/msg/Int16MultiArray "{data: [68,30]}"

# Open gripper
ros2 topic pub /gripper/grasp std_msgs/Bool "{data: false}"

# Close gripper
ros2 topic pub /gripper/grasp std_msgs/Bool "{data: true}"
```

### Manual Goal Publishing
Publish custom goal poses directly:
```bash
# Example: Home position
ros2 topic pub /external_goal_pose geometry_msgs/Pose \
  "{position: {x: -0.036, y: 0.344, z: 0.527}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}"

# Example: Target position
ros2 topic pub /external_goal_pose geometry_msgs/Pose \
  "{position: {x: -0.033, y: 0.355, z: 0.300}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}"
```

## Testing with Fake Hardware

For testing without physical robot:

### Terminal 1: Launch with Fake Hardware
```bash
ros2 launch ur_bringup ur_control.launch.py \
  ur_type:=ur10e \
  robot_ip:=xxx.xxx.xxx.xxx \
  use_fake_hardware:=true \
  launch_rviz:=true
```

### Terminal 2: Launch MoveIt with Fake Hardware
```bash
ros2 launch ur_bringup ur_moveit.launch.py \
  ur_type:=ur10e \
  robot_ip:=xxx.xxx.xxx.xxx \
  use_fake_hardware:=true \
  launch_rviz:=true
```

## Headless Remote Operation

For running on a headless machine (e.g., remote Jetson):

### On Server (Headless Machine)
```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config
```

Add/uncomment these lines:
```
X11Forwarding yes
X11DisplayOffset 10
X11UseLocalhost yes
```

```bash
# Restart SSH service
sudo systemctl restart ssh

# Install xauth
sudo apt install xauth
```

### On Client (Your Computer)
```bash
# Connect with X11 forwarding
ssh -X username@remote_ip
```

## Running cuRobo in Docker

```bash
sudo docker run -it --rm --network host \
  --runtime=nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/curobo_ws:/root/curobo_ws \
  curobo_docker:aarch64
```

## Debugging Tools

### View TF Frames
```bash
ros2 run tf2_tools view_frames
```

### Check PyTorch Installation
```bash
python3 -c "import torch, torchvision; \
print('CUDA Available:', torch.cuda.is_available()); \
print('PyTorch Version:', torch.__version__); \
print('CUDA Version:', torch.version.cuda); \
print('torchvision Version:', torchvision.__version__); \
print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Check CUDA Version
```bash
nvcc --version
```

## Troubleshooting

### Network Issues
- If IPs don't show after configuration, repeat the network setup
- Network configuration is lost when terminal is closed - reconfigure if needed
- Use `ip a` to verify current network configuration

### Robot Connection Issues
- Ensure robot program is running on UR tablet
- Check that START button is pressed
- If "reverse interface dropped" error occurs, restart the program on tablet
- Verify IP addresses: Robot=`192.168.1.190`, Host=`192.168.1.101`

### Vision Issues
- Check camera connection and ZED SDK installation
- Verify model weights path in launch command
- Adjust `--conf_thres` if too many/few detections

### Motion Planning Issues
- Ensure cuRobo Docker container is running
- Check ROS_DOMAIN_ID matches across all terminals
- Verify goals are visible as red dots in RViz before executing

## Project Structure

```
manipulatorsdatepalm/
├── ur_ws_new/                    # ROS 2 workspace
│   └── src/
│       ├── ur10e_curobo/         # cuRobo motion planning
│       ├── zed_date_detector/    # Vision detection
│       ├── DELTO_ROS2/           # Gripper drivers
│       ├── ur_robot_driver/      # UR robot drivers
│       ├── ros2_controllers/     # ROS 2 controllers
│       └── ros2_control_demos/   # Demo examples
├── urdfs/                        # Robot URDF files
├── instructions                  # Setup instructions
├── commands                      # Command reference
├── gripper_test.py              # Gripper testing script
└── README.md                    # This file
```

## Key Files

- [date_v1.7.py](ur_ws_new/src/zed_date_detector/date_v1.7.py) - Latest vision detection code
- [ur10e_curobo/main.py](ur_ws_new/src/ur10e_curobo/ur10e_curobo/main.py) - Motion planning node
- [instructions](instructions) - Detailed setup notes
- [commands](commands) - Quick command reference

## Contributing

For questions or contributions, contact the RISC Lab team.

## License

Copyright KAUST RISC Lab

## References

- [Universal Robots ROS 2 Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)
- [cuRobo Documentation](https://curobo.org/)
- [MoveIt 2](https://moveit.picknik.ai/main/index.html)
- [YOLOv8 Ultralytics](https://docs.ultralytics.com/)
- [ZED SDK](https://www.stereolabs.com/developers)


