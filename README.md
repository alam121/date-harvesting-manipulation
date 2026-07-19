# ManipulatorsDatePalm

Main repository for the UR10e date-palm harvesting system.

This repo contains the ROS 2 Humble workspace, UR10e/cuRobo control stack,
RViz operator panel, vision pipeline, Delto gripper integration, LiDAR scan
workflow, camera recording tools, and supporting robot-driver changes used for
date harvesting experiments.

## Start Here

For day-to-day robot operation, use the operator guide:

- [UR10e Operator Guide](ur_ws_new/src/ur10e_curobo/OPERATOR_GUIDE.md)

For package-level launch and configuration details:

- [UR10e cuRobo Package README](ur_ws_new/src/ur10e_curobo/readme.md)

For common install/runtime problems:

- [Troubleshooting Guide](troubleshoot.md)

## Documentation Map

Core system docs:

- [UR10e Operator Guide](ur_ws_new/src/ur10e_curobo/OPERATOR_GUIDE.md) - RViz workflow, reachability goals, safe zone, LiDAR scan, camera recording, troubleshooting.
- [UR10e cuRobo Package README](ur_ws_new/src/ur10e_curobo/readme.md) - launch commands, manual launch, UR pendant instructions, keyboard controls.
- [Codebase Architecture](ur_ws_new/src/ur10e_curobo/CODEBASE_ARCHITECTURE.md) - high-level package architecture.
- [Goals Architecture](ur_ws_new/src/ur10e_curobo/GOALS_ARCHITECTURE.md) - goal and harvesting pipeline organization.
- [Motion Control / Collision Avoidance](ur_ws_new/src/ur10e_curobo/C1_4_MOTION_CONTROL_COLLISION_AVOIDANCE_UPDATED.md) - planning and collision behavior.
- [Latency Analysis](ur_ws_new/src/ur10e_curobo/C1_5_LATENCY_ANALYSIS.md) - timing notes.
- [Harvesting Strategy](ur_ws_new/src/ur10e_curobo/E1_4_PROGRAMMED_HARVESTING_STRATEGY_UPDATED.md) - programmed harvesting approach.

Related package docs:

- [ZED Date Detector README](ur_ws_new/src/zed_date_detector/README.md)
- [UR Robot Driver README](ur_ws_new/src/ur_robot_driver/README.md)
- [UR Driver ROS Interface](ur_ws_new/src/ur_robot_driver/ur_robot_driver/doc/ROS_INTERFACE.md)
- [Delto ROS 2 README](ur_ws_new/src/DELTO_ROS2/README.md)
- [cuRobo README](curobo/README.md)

## Repository Layout

```text
.
├── bin/                         # Repo-managed launch/setup helper commands
├── curobo/                      # cuRobo source/vendor tree
├── ur_ws_new/                   # Main ROS 2 Humble workspace
│   ├── fastdds_config.xml       # FastDDS buffer profile
│   └── src/
│       ├── ur10e_curobo/        # Main UR10e/cuRobo harvesting package
│       ├── rviz_ur10e_panel/    # RViz operator panel plugin
│       ├── ur_robot_driver/     # UR driver source with local changes
│       ├── DELTO_ROS2/          # Delto gripper driver
│       └── zed_date_detector/   # ZED/date detector package
├── troubleshoot.md              # Runtime/build troubleshooting guide
└── README.md                    # This file
```

## Main Capabilities

- UR10e control through ROS 2 control and the UR robot driver.
- cuRobo GPU motion planning for joint-space and Cartesian motions.
- RViz operator panel with tabs for motion, goals, monitoring, camera, heat, and settings.
- Reachability cloud for clicking known-good goals from the current posture.
- Safe-zone walls and path validation for bounded in-zone work.
- LiDAR scan workflow with scan preview, preflight validation, and bag recording.
- Camera feed snapshot/video recording from `/vision/display`.
- ZED/YOLO vision pipeline for date/trunk perception.
- Delto 3-finger gripper control and force-profile grasp feedback.
- Stability/tracking recorder for joint, wrench, and trajectory comparison logs.

## Quick Start

One-time shell setup:

```bash
cd /home/datepalm2/manipulatorsdatepalm
./bin/install_shell.sh
source ~/.bashrc
```

Build:

```bash
cd /home/datepalm2/manipulatorsdatepalm/ur_ws_new
colcon build --symlink-install
source install/setup.bash
```

Open the launch dialog:

```bash
launch_ur10e_gui
```

Or launch the normal outdoor/harvest stack directly:

```bash
launch_harvest
```

This is the same as `launch_ur10e harvest`, which expands to `launch_ur10e main vision zed_mini`.

Launch fake hardware for testing:

```bash
launch_ur10e fake harvest
```

## Common Launch Options

```bash
launch_ur10e [fake] [harvest|field] [main] [vision] [zed_mini|lidar] [teleop] [gui]
```

Options:

| Option | Description |
| --- | --- |
| `launch_ur10e_gui` | Opens a dialog for real/fake robot, camera/depth, and optional panes. |
| `harvest` | Preset for `main vision zed_mini`. |
| `field` | Alias for `harvest`. |
| `fake` | Use fake/simulated hardware. |
| `main` | Start the main `ur10e_curobo` control node. |
| `vision` | Start the ZED/YOLO vision node. |
| `zed_mini` | Use ZED X Mini depth with ZED X One detection. |
| `lidar` | Use Livox LiDAR depth with ZED X One detection. |
| `teleop` | Start joystick teleop. |
| `gui` | Start the desktop GUI panel. |

For detailed launch and manual terminal commands, see:

- [UR10e cuRobo Package README](ur_ws_new/src/ur10e_curobo/readme.md)

## RViz Operator Workflow

The RViz panel is the primary operator surface.

Important tabs:

- Motion: HOME, dropoff, execute, gripper, plan confirmation.
- Goal: manual goals, goal capture, LiDAR scan, reachability/scan preview.
- Monitor: harvest results, stability recorder, joints, forces.
- Camera: save images and record videos from `/vision/display`.
- Heat: joint/tool temperature monitoring.
- Settings: velocity and process refresh controls.

Detailed operator instructions:

- [UR10e Operator Guide](ur_ws_new/src/ur10e_curobo/OPERATOR_GUIDE.md)

## Output Directories

Runtime logs and recordings are saved under the user home directory:

| Output | Directory |
| --- | --- |
| LiDAR scan bags | `~/lidar_scans` |
| Camera snapshots/videos | `~/camera_recordings` |
| Stability/tracking CSVs | `~/ur10e_stability` |
| Grasp learning data | `~/grasp_learning_data` |

## Hardware Notes

UR10e robot:

- Robot IP: `192.168.1.190`
- Control PC interface commonly uses `192.168.1.101/24`

Network setup:

```bash
sudo ip addr flush dev eth0
sudo ip addr add 192.168.1.101/24 dev eth0
sudo ip addr add 169.254.186.100/24 dev eth0
sudo ip link set eth0 up
```

UR pendant:

1. Power on the robot.
2. Release brakes.
3. Load the external control program.
4. Press Start.

If you see reverse-interface disconnects, restart the UR program and press
Start again.

## Build Notes

Build everything:

```bash
cd /home/datepalm2/manipulatorsdatepalm/ur_ws_new
colcon build --symlink-install
source install/setup.bash
```

Build only the RViz panel after C++ panel edits:

```bash
cd /home/datepalm2/manipulatorsdatepalm/ur_ws_new
colcon build --packages-select rviz_ur10e_panel
source install/setup.bash
```

Python-only edits in `ur10e_curobo` usually only require restarting the Python
node when using `--symlink-install`.

## Key Configuration

Primary config file:

```text
ur_ws_new/src/ur10e_curobo/ur10e_curobo/config.py
```

Important sections:

- `ROBOT_PROFILE`
- `ENVIRONMENT`
- stored HOME / side-home / dropoff joint presets
- `Planner`
- `LidarScan`
- safe-zone and reachability settings
- gripper settings

## Troubleshooting

Start with:

- [Troubleshooting Guide](troubleshoot.md)
- [UR10e Operator Guide - Troubleshooting](ur_ws_new/src/ur10e_curobo/OPERATOR_GUIDE.md#10-troubleshooting)

Quick checks:

```bash
ros2 node list
ros2 topic hz /vision/display
ros2 topic echo /joint_states --once
ros2 control list_controllers
```

Common issues:

- Robot program is not running on the UR pendant.
- Workspace was not sourced after rebuild.
- Safe-zone box rejects a path that leaves the work area.
- Goal is Cartesian-close but on a bad IK branch.
- Camera recording has no frames because `/vision/display` is not publishing.

## Development Notes

Before pushing code changes:

```bash
cd /home/datepalm2/manipulatorsdatepalm
python3 -m py_compile \
  ur_ws_new/src/ur10e_curobo/ur10e_curobo/node.py \
  ur_ws_new/src/ur10e_curobo/ur10e_curobo/lidar_scan.py \
  ur_ws_new/src/ur10e_curobo/ur10e_curobo/config.py
git diff --check
```

For RViz panel changes:

```bash
cd /home/datepalm2/manipulatorsdatepalm/ur_ws_new
colcon build --packages-select rviz_ur10e_panel
```

Do not commit generated `__pycache__` files.
