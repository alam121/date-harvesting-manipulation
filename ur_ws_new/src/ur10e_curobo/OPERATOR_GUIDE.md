# UR10e Date Harvesting Operator Guide

This guide is for running the UR10e harvesting stack from RViz and the ROS
nodes in this repository. It covers the normal operator workflow, RViz panel
tabs, reachability goals, LiDAR scanning, camera recording, and common failure
messages.

## 1. Start The System

From the repository root:

```bash
cd /home/datepalm2/manipulatorsdatepalm
source ur_ws_new/install/setup.bash
launch_ur10e main vision teleop gui
```

For fake hardware testing:

```bash
launch_ur10e fake main vision teleop gui
```

The usual real-robot panes are:

- UR driver / RViz
- main cuRobo control node
- vision node
- teleop node
- optional GUI

On the UR pendant, make sure the robot is powered, brakes are released, the
external control program is loaded, and the program is running.

## 2. RViz Panel Overview

Open the UR10e panel in RViz. The main status, robot state, active robot
profile, motion phase, reacquire status, and emergency stop are always visible.

Tabs:

- Motion: home, dropoff, execute, side-home, gripper, plan confirm/cancel.
- Goal: manual goals, goal capture, LiDAR scan, reachability/scan preview.
- Monitor: harvest result, robot stability recorder, joint state, forces.
- Camera: save images and record video from the camera feed.
- Heat: joint/tool temperature monitoring.
- Settings: velocity scale and process refresh buttons.

## 3. Motion Workflow

Basic sequence:

1. Press `Home`.
2. Confirm robot reaches a stable HOME posture.
3. Enable the safe zone if you are working inside a bounded workspace.
4. Add or click a reachable goal.
5. Press `Execute`.
6. Use `Dropoff` after a successful grasp.

Important behavior:

- HOME now requires the exact HOME joint branch. TCP-only HOME is disabled.
- Safe-zone walls are not cleared automatically.
- If a goal requires a very large IK jump, the node skips it instead of spending
  a long time trying unsafe plans.
- `Add Current Goal` saves the current joint posture, not only the end-effector
  pose.

## 4. Reachability Cloud

The reachability cloud shows which nearby TCP positions are currently easy or
hard to reach from the current robot posture.

Use it like this:

1. In RViz, enable `Reachability Cloud`.
2. Enable the `Publish Point` tool in RViz if it is not already in the toolbar.
3. Click a green point in the 3D view.
4. The node queues the nearest green reachability sample as a goal.
5. Press `Execute`.

Color meaning:

- Green: direct checked joint interpolation is expected to work.
- Yellow/red: reachable or sampled, but not preferred for direct execution.

Notes:

- Green points are not every physically possible point. They are sampled points
  that passed the current posture checks.
- Direction rays show the sampled TCP approach direction.
- Reachability-click goals execute slowly by default for safety.

## 5. Safe Zone

The safe zone is a box around the work area. When enabled, manual Cartesian
paths and scan paths are checked against the box.

Expected behavior:

- The Goal tab has Safe Zone buttons: `Enable`, `Disable`, `Snap TCP`, and
  `Snap Outdoor`.
- Use the RViz `Interact` tool to manipulate the safe-zone handles.
- Drag the yellow center handle to move the whole box.
- Drag the blue/orange corner handles to resize the box.
- Safe-zone walls remain enabled until you explicitly disable or clear them.
- HOME and recovery motions do not automatically clear safe-zone walls.
- If a path leaves the box, the node rejects it before execution.
- The default box extends below `base_link` so outdoor low fruit poses can be
  enclosed.

Useful right-click menu actions on either safe-zone corner handle:

- `Snap box around robot`: centers a compact box around the current TCP.
- `Snap outdoor/deep box`: builds a deeper outdoor box around the current TCP,
  including low poses below the RViz ground grid.
- `Enable safe zone`: applies the visible box as cuRobo keep-out walls.

Common message:

```text
rejected Cartesian path leaves safe-zone box
```

Meaning: the TCP path would leave the configured box. Move the target, adjust
the safe zone, or move the robot to a better starting posture.

## 6. LiDAR Scan

The `Lidar Scan` button is in the Goal tab.

The scan records:

- `/livox/lidar`
- `/livox/imu`

Output directory:

```bash
~/lidar_scans
```

Current scan behavior:

- The scan tries to build an arc-like path around the selected scan center.
- The path does not have to be a perfect semicircle.
- The scan prefers smooth, reachable viewpoints over exact geometric points.
- The tool orientation can stay close to the current orientation; it does not
  have to point exactly at the center.
- Preflight validates the path first.
- Execution reuses the exact preflight trajectory states so cuRobo does not
  replan into a different IK branch during the scan.

LiDAR scan preview:

- Enable `LiDAR Scan Preview` in the Goal tab.
- Preview markers show candidate scan points before pressing scan.
- If the preview is sparse, the robot may only have a few valid scan viewpoints
  from the current posture.

Important scan guards:

- Large joint branch jumps are rejected.
- Abrupt per-sample joint jumps are rejected.
- Recorded scan segments must stay close to the horizontal scan plane.
- The move to the first scan point happens before bag recording starts.

Useful messages:

```text
adaptive reverse sweep kept 6/7 angle slots
```

The scan found a valid sweep in reverse angle order.

```text
max joint change ... exceeds ...
```

The scan point would require a large joint branch jump and was rejected.

```text
TCP leaves horizontal plane
```

The recorded scan segment would climb or drop too much.

```text
no scan targets available
```

No safe scan path was found from the current posture. Move to a better posture
near the object and try again.

## 7. Camera Recording

The Camera tab records the `/vision/display` image stream.

Buttons:

- `Lab`: applies the lab exposure preset live. This keeps the current auto
  exposure / auto gain behavior.
- `Outdoor`: applies the outdoor exposure preset live. This uses manual low
  exposure and low gain to reduce overexposure in sunlight.
- `Auto Exposure`: when checked, the ZED controls exposure/gain automatically.
- `Exposure` and `Gain`: manual values used when `Auto Exposure` is unchecked.
- `Apply Exposure`: sends the current exposure controls to the live camera.
- `Save Image`: saves the latest frame as a PNG.
- `Start Video`: starts MP4 recording.
- `Stop Video`: stops MP4 recording.
- `Refresh Camera`: asks the vision process to refresh the camera.

Use `Lab` indoors. Use `Outdoor` before moving outside or whenever the image
looks washed out. The preset change is runtime-only; it does not require
restarting the camera.

Manual starting points:

- Bright direct sun: uncheck `Auto Exposure`, set `Exposure` to `5-8`, `Gain` to `0`.
- Outdoor shade: uncheck `Auto Exposure`, set `Exposure` to `10-15`, `Gain` to `0-10`.
- Lab/indoor: use `Lab` or leave `Auto Exposure` checked.

Output directory:

```bash
~/camera_recordings
```

File names:

```text
camera_YYYYMMDD_HHMMSS.png
camera_video_YYYYMMDD_HHMMSS.mp4
```

If `Save Image` warns that no frame has been received, confirm that the vision
node is running and publishing `/vision/display`.

Check camera feed:

```bash
ros2 topic hz /vision/display
```

## 8. Stability Recording

The Monitor tab includes `Record Stability / Tracking`.

It records:

- joint positions
- joint velocities
- tracking error
- TCP wrench
- commanded trajectory comparison

Output directory:

```bash
~/ur10e_stability
```

Press the button once to start recording. Press again to stop and calculate a
rating.

## 9. Configuration Files

Primary runtime configuration:

```text
ur_ws_new/src/ur10e_curobo/ur10e_curobo/config.py
```

Important sections:

- `ROBOT_PROFILE` and `ENVIRONMENT`
- HOME / side-home / dropoff joint presets
- `Planner`
- `LidarScan`
- safe-zone and reachability tuning

After Python-only changes, normally restart the Python node. If RViz panel C++
code changes, rebuild the RViz package:

```bash
cd /home/datepalm2/manipulatorsdatepalm/ur_ws_new
colcon build --packages-select rviz_ur10e_panel
source install/setup.bash
```

## 10. Troubleshooting

### Robot program is off

Start the external control program on the UR pendant.

### Plan failed for a nearby point

The end-effector position may be nearby, but the robot can still need a
different IK branch. Check the nearest IK delta in the logs.

### Goal skipped because IK delta is too high

Move to a better posture first, or click a green reachability point closer to
the current branch.

### HOME reaches TCP but not HOME joints

The TCP pose can be equivalent while the joints are on the wrong branch. The
system rejects TCP-only HOME and requires the exact HOME joint branch.

### LiDAR scan takes too long

The scan preflight is trying candidate viewpoints that are hard for the current
posture. Move closer to a good scan posture or reduce the requested scan span.

### Camera recording does not save

Check:

```bash
ros2 topic hz /vision/display
ls -lah ~/camera_recordings
```

### RViz panel did not update

Rebuild and re-source:

```bash
cd /home/datepalm2/manipulatorsdatepalm/ur_ws_new
colcon build --packages-select rviz_ur10e_panel
source install/setup.bash
```

Restart RViz after sourcing.
