# UR10e cuRobo Control System

This README describes how to launch and operate the UR10e robot system with cuRobo motion planning, vision, teleop, and GUI.

For day-to-day operation from RViz, including reachability goals, LiDAR scanning,
camera recording, safe-zone behavior, and troubleshooting, read:

```text
OPERATOR_GUIDE.md
```

---

## Quick Start

```bash
# Open a dialog and choose robot/camera/depth options
launch_ur10e_gui

# Launch the normal outdoor/harvest stack
launch_harvest

# Same command using the explicit preset
launch_ur10e harvest

# Launch with fake/simulated hardware for testing
launch_ur10e fake harvest

# Launch all nodes
launch_ur10e harvest teleop gui
```

## Command Setup (One-Time Per Machine)

Use repo-managed commands from `<repo_root>/bin`:

```bash
cd <repo_root>
./bin/install_shell.sh
source ~/.bashrc
```

This keeps repo commands ahead of older `~/bin` wrappers if both exist.

---

## Prerequisites

### Build the Workspace

```bash
cd <repo_root>/ur_ws_new
colcon build --symlink-install
source install/setup.bash
```

Note: Use `--symlink-install` so Python changes take effect immediately without rebuilding.

---

## Launch Options

The `launch_ur10e` command opens a terminator window with split panes for each component.
Use `launch_ur10e_gui` when you want a dialog for real/fake robot, camera/depth,
and optional panes.

### Usage

```bash
launch_ur10e [fake] [harvest|field] [main] [vision] [zed_mini|lidar] [teleop] [gui]
```

### Options

| Option   | Description                                    |
|----------|------------------------------------------------|
| `harvest` | Preset for main + vision using ZED Mini depth |
| `field` | Alias for `harvest` |
| `fake`   | Use simulated hardware (no real robot needed)  |
| `main`   | Main control node with cuRobo motion planning  |
| `vision` | ZED camera + YOLO detection node               |
| `zed_mini` | Use ZED X Mini depth with ZED X One detection |
| `lidar` | Use Livox LiDAR depth with ZED X One detection |
| `teleop` | Joystick teleop control                        |
| `gui`    | Desktop GUI control panel                      |

### Examples

```bash
# Normal outdoor stack
launch_harvest

# Same thing, explicit preset
launch_ur10e harvest

# Outdoor stack + teleop + GUI
launch_ur10e harvest teleop gui

# Fake hardware for testing (no robot needed)
launch_ur10e fake harvest

# Real robot + main + vision with ZED stereo instead of ZED Mini
launch_ur10e main vision
```

### What Happens

1. **UR Bringup pane**: Runs network setup (`unet.sh`) and launches UR robot driver + RViz
2. **Main pane**: Launches cuRobo motion planning node
3. **Vision pane**: Launches ZED/YOLO vision detection
4. **Teleop pane**: Launches joystick teleop control
5. **GUI pane**: Launches desktop GUI

---

## Manual Launch (Alternative)

If you prefer running commands manually in separate terminals:

### 1. Configure Network Interface

```bash
# Run the network setup script (requires sudo)
<repo_root>/bin/unet.sh
```

Or manually:
```bash
sudo ip addr flush dev eth0
sudo ip addr add 192.168.1.101/24 dev eth0
sudo ip addr add 169.254.186.100/24 dev eth0
sudo ip link set eth0 up
```

### 2. Launch UR10e + RViz

```bash
source <repo_root>/ur_ws_new/install/setup.bash
ros2 launch ur_bringup ur_control.launch.py \
    ur_type:=ur10e \
    robot_ip:=192.168.1.190 \
    use_fake_hardware:=false \
    launch_rviz:=true
```

### 3. Launch Main Control Node

```bash
source <repo_root>/ur_ws_new/install/setup.bash
ros2 run ur10e_curobo main
```

### 4. Launch Vision Node

```bash
source <repo_root>/ur_ws_new/install/setup.bash
ros2 run ur10e_curobo vision
```

### 5. Launch Teleop Node

```bash
source <repo_root>/ur_ws_new/install/setup.bash
ros2 run ur10e_curobo teleop
```

### 6. Launch GUI Node

```bash
source <repo_root>/ur_ws_new/install/setup.bash
ros2 run ur10e_curobo gui
```

---

## UR Tablet Instructions

On the UR teach pendant:

1. Power on the robot
2. Release the brakes
3. Go to **Program**
4. Load the **ucra** file
5. Press **Start**

If you see `Connection to reverse interface dropped`, restart the program and press **Start** again.

---

## Keyboard Controls (Main Node)

| Key | Action                              |
|-----|-------------------------------------|
| `s` | Subscribe a single goal             |
| `p` | Subscribe multiple goals (10 sec)   |
| `n` | Execute planned motion              |
| `o` | Open gripper                        |
| `c` | Close gripper                       |

---

## Troubleshooting

### "Cannot execute goals: robot program is OFF"
Make sure the **Start** button on the UR tablet is pressed.

### Network issues
Re-run `unet.sh` or manually configure IPs. IPs reset when terminal closes.

### ROS nodes not found
Re-source the workspace: `source install/setup.bash`

---

## Static Terminator Layout

For a fixed 4-pane layout (ur_bringup + main + teleop + vision):

```bash
terminator -l ur10e
```
