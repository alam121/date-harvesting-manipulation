# UR10e cuRobo Control System

This branch uses **cuRobo v2**. See [installation, migration, and verification](CUROBO_V2.md)
before launching the motion node.

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
launch_ur10e [fake] [harvest|field] [main] [vision] [zed_mini|zedx_mini|lidar] [teleop] [gui]
```

### Options

| Option   | Description                                    |
|----------|------------------------------------------------|
| `harvest` | Preset for main + vision using ZED X Mini RGBD |
| `field` | Alias for `harvest` |
| `fake`   | Use simulated hardware (no real robot needed)  |
| `main`   | Main control node with cuRobo motion planning  |
| `vision` | ZED camera + YOLO detection node               |
| `zed_mini` | Use ZED X Mini depth with ZED X One detection |
| `zedx_mini` | Use ZED X Mini for both RGB detection and native stereo depth |
| `lidar` | Use Livox LiDAR depth with ZED X One detection |
| `teleop` | Joystick teleop control                        |
| `gui`    | Desktop GUI control panel                      |

### Camera Calibration Profiles

The launcher exports an explicit camera calibration profile for each camera mode.

| Camera mode | Launch option | Profile YAML |
|-------------|---------------|--------------|
| ZED X Mini RGBD | `zedx_mini` | `ur10e_curobo/vision/calibration_profiles/zedx_mini_rgbd.yaml` |
| ZED X One RGB + ZED X Mini depth | `zed_mini` | `ur10e_curobo/vision/calibration_profiles/zed_one_rgb_zedx_mini_depth.yaml` |

Use the same mode when calibrating:

```bash
launch_ur10e main hand_eye zedx_mini
launch_ur10e main hand_eye zed_mini
launch_ur10e extrinsic
```

The calibration tool saves the result to the selected profile and still offers
the existing URDF update prompt for the matching camera frame. The extrinsic
tool updates the dual-camera profile's `depth_to_rgb` matrix automatically.

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

## Record FINAL corrections in RViz

On the `demo-recorder` branch, build `ur10e_curobo` and `rviz_ur10e_panel`
with `colcon build --symlink-install --packages-select ur10e_curobo rviz_ur10e_panel`,
then restart the control node and RViz (`launch_harvest`).

1. Check **Demo: correct FINAL, then Close** in the RViz Motion tab.
2. Start a harvest attempt normally. Wait for the `[DEMO] holding at FINAL` log;
   the recorder first waits for a fresh crop and freezes it before you jog.
3. Jog the pendant to the correct grasp pose. Keep the gripper open.
4. Restore the robot's External Control program and press **Close** in RViz.
   During the hold, Close accepts the correction; the harvest sequence saves it,
   closes the gripper, and continues its normal reverse/drop-off/home workflow.
5. Inspect `~/harvest_logs/demos/<timestamp>/`: `crop_before.jpg` is the training
   input, `crop_after.jpg` is optional context, and `sample.json` contains the
   before/after poses, translation correction in base/tool metres, and jog path.

Demo mode waits indefinitely by default. Unchecking Demo during the hold discards
and stops that attempt (including automatic harvest). Stop, missing fresh imagery,
recording failure, or a configured positive `demo_capture_timeout_s` expiring also
ends the attempt without automatically closing. Outside a demo hold, Close keeps
its ordinary behavior. The checkbox reflects the running node and is off by default.

Validate pendant-to-ROS handoff and fruit visibility with supervised cycles before
collecting the dataset. The recorder requires a fresh crop but does not determine
whether the fruit is actually visible in it. It does not switch robot control modes.

## NuRec compressed stereo capture

Available in `--use_zedx_mini_only` mode, **off by default**. After rebuilding
`ur10e_curobo` and restarting the vision node, enable/disable it at runtime:

```bash
export ROS_DOMAIN_ID=6
ros2 topic pub --once /camera_command std_msgs/msg/String "data: 'nurec on'"
ros2 topic pub --once /camera_command std_msgs/msg/String "data: 'nurec off'"
```

Use the same ROS environment and Fast DDS profile as the harvest launcher.
For startup capture, append `--nurec_capture` to the vision command.
Optional arguments: `--nurec_width 1280 --nurec_hz 3 --nurec_jpeg_quality 80`.
These settings produce 1280×720 from the current 1920×1080 camera mode.

| Topic | Type |
|---|---|
| `/vision/nurec/left/image_rect/compressed` | `sensor_msgs/msg/CompressedImage` |
| `/vision/nurec/right/image_rect/compressed` | `sensor_msgs/msg/CompressedImage` |
| `/vision/nurec/left/camera_info` | `sensor_msgs/msg/CameraInfo` |
| `/vision/nurec/right/camera_info` | `sensor_msgs/msg/CameraInfo` |

Both rectified images come from the same SDK grab; all four messages share its
acquisition stamp. Intrinsics match the published image dimensions, distortion
is zero, and rectification rotation is identity. The right projection includes
`P[3] = -fx * baseline_m`; recover the baseline as `-P[3] / P[0]`.
The baseline comes from the SDK, using the camera's existing `UNIT.METER` setting
([SDK baseline units](https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CalibrationParameters.html),
[rectified calibration](https://www.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)).
The right image/info use the corresponding right camera frame name; this mode
provides stereo geometry through CameraInfo and does not add a right-camera TF.

Transport is reliable, volatile, keep-last 1. A separate JPEG/publishing worker
uses a one-pair queue, replacing stale queued pairs instead of blocking the SDK
capture thread. Disabled capture retrieves no additional images and publishes no
pairs. This is a live stream, not a lossless local recorder: synchronize by stamp
and discard incomplete pairs at the receiver. JPEG size and delivered frequency
depend on the scene, CPU load, and network; 1 MB/s is a target, not a guarantee.
Validate pair arrival and recovered scale on the remote recorder before a dataset run.

Existing `/vision/raw` is independently gated by `raw_stream on`; publisher
presence alone does not mean raw streaming has been enabled.

### Demo full-frame observations

Restart both vision and the motion node after this change; reload `view_robot.rviz`
to see **Demo Before (frozen)**. At FINAL, demo capture requires a fresh crop and
an unannotated full field-of-view image. `/vision/demo_full/compressed` publishes
JPEG at up to 3 Hz and 1280 pixels wide. `/demo/before_image` holds the exact
pre-jog full frame for RViz. Check that the target and fingertips are visible
before jogging; uncheck Demo to discard if they are obscured.

Samples retain `crop_before.jpg` and `crop_after.jpg`, and add required
`full_before.jpg` plus `full_after.jpg` when a frame received within one second
is available at confirmation. JSON records the before full-frame source and
receipt timestamps and dimensions. Crop and full-frame observations are fresh
but not guaranteed to be from the same camera exposure. The preview remains
frozen until the next capture; it is not a live jogging view. Existing demos
cannot be retroactively supplied with full images.

The frozen demo preview labels detected fruit IDs and suggests a target only
when its base-frame position is within 50 mm of the attempted goal and at least
20 mm closer than the next candidate. This is an unverified proximity match,
not persistent identity tracking; inspect it before accepting. Ambiguous or
missing matches are explicitly unconfirmed. `full_before_marked.jpg` preserves
the overlay; `full_before.jpg` stays clean. `before.target_label` records the
attempted goal, candidates, IDs and original/saved bounding boxes. Metadata is
matched to the full frame by the exact publication timestamp. These IDs should
not be treated as globally unique across runs. Restart vision and main to load
this change.

#### Operator target click

At the FINAL hold, click the intended fruit in the frozen image in the RViz
**Motion** panel (the separate Image display is not clickable). A green circle
shows the sent click. Wait for `[DEMO] Target click accepted`, jog, restore
External Control, then press Close. Close without a click is rejected. Clicking
again changes the selected pixel; a new capture resets selection and rejects
old-frame clicks. The marked image and JSON preserve the operator point even
when no detector box matches it. A fruit ID is assigned only for one matching
box; overlapping/no boxes leave the ID unknown, while retaining the clicked
point. Restart RViz and the motion node to load the click feature, and vision
if full-frame publishing is not yet running. This does not move the robot when
you click.

### Current demo flow: APPROACH observation (schema 2)

This supersedes the FINAL-hold instructions above. Enable **Demo: jog APPROACH
to grasp, then Close**. After the normal approach and target calculation, the
robot holds before automatic insertion. Click the intended date in the Motion
panel frozen image, jog to the actual grasp endpoint, restore External Control,
and press Close. No nominal FINAL movement is executed after the manual jog.
The recorded jog joint samples are appended to the withdrawal path.

Samples have `observation_stage: APPROACH` and `schema_version: 2`.
`desired_motion_base_m`/`desired_motion_tool_m` measure grasp minus APPROACH;
`residual_from_planned_final_base_m` measures grasp minus nominal FINAL.
The legacy correction fields now refer to the APPROACH displacement and are
explicitly tagged with `correction_reference`. Do not mix them with older
FINAL correction samples. The planned FINAL pose is retained under `before`.
These records still contain endpoint images and a joint/pose jog trace, not
synchronized video for ACT sequence training. Restart main and RViz to activate.

### Supervised level-insertion test

`planner.low_center_level_insertion_enabled` defaults to true. For LOW/VERY_LOW
center approaches only, FINAL TCP Z is set to the measured settled APPROACH TCP
Z after closure-center compensation. FINAL X/Y and orientation are preserved;
APPROACH geometry and MID/HIGH or side approaches are unchanged. Fallback
endpoints and in-flight corrections retain the same height. Existing IK,
collision and trajectory checks remain active. Watch `[LEVEL_INSERTION]` for
nominal versus commanded Z. This is not a fix for Y depth overshoot.

Runtime switch via `/ui_command` (`std_msgs/msg/String`):
`set_level_insertion true` or `set_level_insertion false`, effective when the
next FINAL target is calculated. Restart main to load the new implementation.
With Demo enabled, insertion is still manual. Disable Demo to test automatic
level insertion under supervision. Record five outcomes and remaining manual
X/Y corrections before assessing improvement. Demo JSON records the switch
and actual constrained height so these samples can be separated from older ones.

### Depth diagnostics for new demos

Restart vision and main to enable `/vision/depth_trace`. Each published goal
has a trace of the measured camera-space surface, the surface-to-center shift
(0.7 times estimated radius), camera-space center, cached camera-to-base transform
when available, unsmoothed base point, and smoothed published goal. Main matches
accepted/reacquired goals by XYZ within 1 micrometre in a bounded history; missing
matches remain null, never silently substituted with the latest frame. This is
coordinate matching, not a globally unique target identity.

Demo `before.depth_trace` freezes that provenance and records configured offsets,
tip-anchor versus goal-offset geometry, pre/post grasp-mode targets, closure
compensation, pre/post level FINAL TCP, and operator endpoint. Positive
`planned_excess_forward_m` means the nominal FINAL was farther into the fruit
than the demonstrated endpoint (+X or -Y according to robot configuration).
`[DEPTH_TRACE]` logs the Y stages and final excess insertion. Old demos cannot
recover missing camera-space measurements retroactively. These diagnostics
measure disagreement with an operator endpoint; they do not independently
prove sensor calibration error.

### Zero-correction ACT collection baseline (2026-10-04)

Lab and outdoor FINAL depth/Z offsets, slip-retry Z offset, Normal/Envelop
closure-center offsets, and mode-specific depth/Z extras default to zero.
RViz initialization/reset values and the Isaac launch override match.
Restart the motion node and RViz to load these defaults; existing runtime
overrides can still change them. Older demos retain their recorded offsets
and must not be interpreted as zero-correction demonstrations.

Camera calibration, surface-to-center target estimation, physical TCP geometry,
approach standoff, level insertion, and reactive grasp recovery remain active.
Collect in APPROACH demo mode so the operator demonstrates the final movement;
zero correction defaults do not mean ACT is trained or controlling the robot.

### Timestamped ACT pilot recordings

Each new demo includes `sequence/sequence.json` and full-frame JPEGs during the
APPROACH hold and pendant jog. Joint states retain source and receipt timestamps
and explicit joint order. Images use the vision capture timestamp, rather than
post-processing publication time. The publisher caps full frames at 10 Hz; actual
rate depends on perception throughput and the network.

Nearest joint states are matched by source time within 50 ms. Unmatched frames
remain recorded but are not assigned a fabricated state. The quality summary
reports matching coverage, frame rate, maximum gap and write errors. `timing_ok`
requires at least 10 frames, all matched, at least 5 Hz and no gap over 0.5 s.
This does not verify cross-machine clock alignment or target visibility;
`training_ready` remains false pending pilot review. Joint positions are measured
states, not pendant command targets. Future action labels require offline
trajectory conversion. The selected target is in sample.json's frozen-image
coordinates, not a tracked per-frame target mask. Closing occurs after this
sequence; this dataset initially trains positioning, not automatic closure.

Restart vision and the motion node before collecting one pilot. Restore any
operator-selected runtime offsets after restart and verify them before collecting.
Do not mix archived demos into the fresh dataset automatically.
