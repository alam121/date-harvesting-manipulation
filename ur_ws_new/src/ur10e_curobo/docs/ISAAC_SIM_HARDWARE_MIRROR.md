# Isaac Sim Hardware Mirror

This branch treats the hardware-ready date-palm stack as the source of truth.
Isaac simulation should mimic the UR10e/ZED/Delto hardware interfaces, not run a
separate toy controller.

## Goal

Run the same ROS 2 control stack used on hardware:

- `python3 -m ur10e_curobo.main`
- cuRobo planning and collision logic
- `/external_goal_pose` date targets
- `/vision/all_fruit_poses`, `/fruit_radius`, `/trunk_position`
- Delto gripper target/force topics
- `/joint_states`
- `/scaled_joint_trajectory_controller/joint_trajectory`
- `/io_and_status_controller/robot_program_running`
- RViz/UI/teleop topics

The simulation side should provide those interfaces from IsaacLab/Isaac Sim.

## Current Implementation

Added package module:

```text
ur10e_curobo/isaac_sim/bridge.py
```

Added executable:

```text
ros2 run ur10e_curobo isaac_mirror
```

Added launch file:

```text
ros2 launch ur10e_curobo isaac_sim_mirror.launch.py
```

The default launch uses a live Isaac Lab backend. ROS 2 runs under the system
Python 3.10 environment and Isaac Lab runs under its Python 3.11 conda
environment. Two localhost UDP ports connect them without mixing incompatible
Python extension modules:

```text
55000  ROS trajectory/gripper commands -> Isaac
55001  Isaac joints/forces/RGB-D perception -> ROS
```

The old lightweight kinematic backend remains available as
`backend:=kinematic` for ROS-only diagnostics.

## Hardware Contract Mirrored

### Consumed By The Mirror

The mirror subscribes to the same commands the real hardware receives:

```text
/scaled_joint_trajectory_controller/joint_trajectory  trajectory_msgs/JointTrajectory
/gripper/target_joint                                  std_msgs/Float32MultiArray
/io_and_status_controller/set_io                       ur_msgs/srv/SetIO
```

### Published By The Mirror

The mirror publishes the same signals the real hardware and vision stack provide:

```text
/joint_states                                          sensor_msgs/JointState
/io_and_status_controller/robot_program_running        std_msgs/Bool
/gripper/force                                         std_msgs/Float32MultiArray
/external_goal_pose                                    geometry_msgs/PoseStamped
/vision/all_fruit_poses                                std_msgs/Float32MultiArray
/fruit_radius                                          std_msgs/Float32
/trunk_position                                        geometry_msgs/PointStamped
/datefruit_direction                                   geometry_msgs/Vector3Stamped
/fruit_image_bbox_norm                                 std_msgs/Float32MultiArray
/vision/fruit_score                                    std_msgs/String
```

This means the real `UR10eCuroboMoveIt` node can run unchanged against simulated
hardware.

## Run

### Simulation-Only Mode

On the GPU workstation, start the ROS adapter first:

```bash
cd ~/manipulatorsdatepalm/ur_ws_new
export ROS_DOMAIN_ID=6
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER=10.76.154.75:11811
unset FASTRTPS_DEFAULT_PROFILES_FILE
colcon build --symlink-install
source install/setup.bash

ros2 launch ur10e_curobo isaac_sim_mirror.launch.py
```

In a second GPU terminal, start the actual PhysX scene:

```bash
cd ~/manipulatorsdatepalm
bin/launch_isaac_gpu
```

For a headless run:

```bash
bin/launch_isaac_gpu --headless --debug_bridge
```

The launch file applies these settings to its child processes. The existing
Fast DDS buffer profile is intentionally unset for this routed Wi-Fi/Ethernet
setup because it overrides discovery-server client configuration. The hardware
launcher continues using that profile on the Jetson.

By default this launch starts only the mirror on the GPU. The real control node
runs on the Jetson. For a single-machine development test, add
`start_control:=true`.

### Physical Robot Digital Twin

Use this mode when the physical UR10e is enabled and Isaac should follow its
measured joints. Start the GPU adapter with ROS joint-state authority and leave
the real UR/Delto/vision drivers authoritative:

```bash
cd ~/manipulatorsdatepalm/ur_ws_new
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ur10e_curobo isaac_sim_mirror.launch.py \
  joint_state_source:=ros publish_fake_vision:=false
```

Start `bin/launch_isaac_gpu` in the second GPU terminal as usual.

On the Jetson GUI, select **Physical robot + Isaac digital twin**. This uses the
normal real-hardware launcher, including UR bringup, cuRobo, vision, Terminator,
and the RViz control panel, while joining ROS domain 6 through the workstation
discovery server. Moving the physical robot then updates Isaac from measured
`/joint_states`.

The GPU mirror publishes measured Isaac feedback separately so it never
competes with the physical driver's `/joint_states`:

```text
/isaac/joint_states       sensor_msgs/JointState
/isaac/joint_sync_error   std_msgs/Float32MultiArray
```

The digital-twin Terminator layout also starts `isaac_joint_relay`. It samples
the local 500 Hz UR feedback and publishes only the newest complete arm state at
50 Hz on `/isaac/physical_joint_states` with best-effort, depth-1 QoS. This
prevents routed Wi-Fi/Ethernet DDS links from accumulating stale reliable joint
messages and delivering them in bursts.

In physical-twin mode, the GPU bridge forwards measured joint positions and
velocities to Isaac. Isaac writes that measured state directly to the simulated
arm and uses bounded velocity interpolation between samples. This keeps the
visual twin synchronized without making the simulated actuator controller
compete with the physical robot. Standalone simulation continues to use normal
physics position targets.

The sync array contains six signed joint errors followed by mean absolute error,
maximum absolute error, and physical-state age in seconds.

Record a passive 60-second validation session from a Jetson terminal:

```bash
cd ~/manipulatorsdatepalm/ur_ws_new
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur10e_curobo isaac_twin_recorder --duration 60
```

The recorder writes timestamped JSONL under `demos/hardware_twin/` and prints
mean/peak joint error plus mean state age when it exits. It only observes ROS
topics and does not send robot commands.

### Hardware Demonstration Recording

After twin synchronization is validated, record one complete harvesting attempt
per episode from a Jetson terminal:

```bash
cd ~/manipulatorsdatepalm/ur_ws_new
source /opt/ros/humble/setup.bash
source install/setup.bash

export ROS_DOMAIN_ID=6
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER=10.76.154.75:11811
unset FASTRTPS_DEFAULT_PROFILES_FILE

ros2 run ur10e_curobo hardware_demo_recorder \
  --notes "outdoor date grasp"
```

Start the recorder before commanding the approach. It records until `Ctrl+C`,
then prompts for `success`, `failure`, or `unknown`. The recorder is passive: it
subscribes to the hardware and Isaac topics but never sends motion or gripper
commands.

Each episode writes:

```text
demos/hardware_demos/demo_<timestamp>.jsonl
demos/hardware_demos/demo_<timestamp>.meta.json
demos/hardware_demos/demo_<timestamp>_images/*.jpg
```

The 20 Hz JSONL frames align physical and Isaac joints, interpolated arm command,
`base_link` to `tool0` pose, gripper state/target/force, structured date
perception, grasp events/results, camera frame reference, topic freshness, and
twin synchronization. The metadata sidecar contains the operator outcome,
notes, topic message counts, image count, and episode-level synchronization
summary. Disable JPEG recording with `--no-images`, or use a fixed duration such
as `--duration 45`.

### Guarded One-Shot Hardware Demo

Record and label one successful episode with the current recorder, then extract
the demonstrated target-relative closure and grasp settings:

```bash
DEMO=$(ls -t demos/hardware_demos/demo_*.jsonl | head -1)
ros2 run ur10e_curobo one_shot_grasp learn \
  --demo "$DEMO" \
  --out skills/one_shot/grasp_skill.json
```

With the physical robot stopped at `IDLE`, the robot program running, and the
same date visible, validate the learned skill without changing the controller:

```bash
ros2 run ur10e_curobo one_shot_grasp apply \
  --skill skills/one_shot/grasp_skill.json
```

The command defaults to dry-run. It rejects stale state, robot motion, a target
more than 30 mm from the demonstration, or any learned grasp-setting change over
5 mm. After a passing dry-run, explicitly apply the settings:

```bash
ros2 run ur10e_curobo one_shot_grasp apply \
  --skill skills/one_shot/grasp_skill.json \
  --execute
```

The applicator waits for the main controller to echo the exact settings before
reporting success. It never starts motion or publishes joint commands. Keep the
robot at low speed, then use the normal supervised GUI harvest command; cuRobo,
safe-zone, collision, grasp, lift, drop, and stop handling remain authoritative.
The settings remain active for later grasps until changed in the GUI or the main
node is restarted.

Do not run simulation-only and physical-twin Jetson launchers together. Stop old
launches first so only one cuRobo control node and one UR driver are active.

### Jetson GUI

After checking out and building the `isaac-sim` branch on the Jetson, run the
normal launcher:

```bash
launch_ur10e_gui
```

Select **Isaac Sim only (no physical robot)**, keep the real gripper profile selected,
and press **Start System**. This invokes `launch_isaac_jetson`, which starts only
the real cuRobo/control node plus `robot_state_publisher` for the URDF TF tree.
The control node uses its real ROS interfaces so trajectories and Delto commands
reach Isaac. It does not start the physical UR driver, Delto driver, ZED camera,
or local fake ros2_control hardware.

The equivalent manual command is:

```bash
launch_isaac_jetson \
  environment:=outdoor gripper_profile:=old gripper_enabled:=true
```

The Isaac scene uses the combined UR10e/Tesollo articulation, a hanging target
date and stem geometry, an RGB-D camera, and fingertip contact sensors. Trunk
geometry is temporarily omitted while validating hardware mirroring.
Its default target is:

```text
fruit_xyz = [0.62, -0.04, 0.82]
fruit_radius = 0.024
```

Use the ROS-only kinematic fallback for diagnostics:

```bash
ros2 launch ur10e_curobo isaac_sim_mirror.launch.py \
  backend:=kinematic fruit_x:=0.64 fruit_y:=-0.03 fruit_z:=0.84
```

Then use the normal operator flow:

- press `s` in the control node to subscribe a single `/external_goal_pose`
- press `n` to execute
- use RViz/GUI exactly like hardware

## Live Isaac Backend

`isaac/isaac_hardware_scene.py` now provides the physical backend:

- applies interpolated cuRobo trajectories to the UR10e articulation
- maps grouped Delto commands into the USD articulation joint order
- returns measured PhysX arm and gripper joint state
- returns three fingertip contact-force magnitudes
- detects the red target from rendered RGB-D and transforms it into `base_link`
- falls back to the simulator object pose only while the camera is warming up

## Why This Is Better Than The Previous Isaac Scripts

Previous IsaacLab scripts duplicated the robot intelligence:

- separate state machine
- separate target picking
- separate gripper timing
- separate imitation policy rollout
- separate success logic

That made sim success disconnected from hardware readiness.

The mirror approach keeps the real hardware logic alive:

- cuRobo remains the planner
- ZED/date detector topic contract remains the perception API
- Delto gripper target/force contract remains the grasp API
- RViz/operator workflow remains the same
- imitation/RL can train around the real interface instead of a toy controller

## Immediate Validation Checklist

1. Launch `isaac_sim_mirror.launch.py`.
2. Start `bin/launch_isaac_gpu` and confirm the bridge prints
   `Live PhysX backend ready`.
3. Confirm `/robot_program_running` is true and `/joint_states` is publishing:

   ```bash
   ros2 topic echo /joint_states --once
   ```

4. Confirm live vision target and source:

   ```bash
   ros2 topic echo /external_goal_pose --once
   ros2 topic echo /vision/all_fruit_poses --once
   ros2 topic echo /trunk_position --once
   ros2 topic echo /vision/fruit_score --once
   ```

5. Confirm the Jetson cuRobo control node receives goals.
6. Execute one goal and confirm Isaac follows the trajectory and publishes
   measured joint/contact state back to the Jetson.
