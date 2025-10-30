**UR10e CuRobo Integration**

**Node.py**

Purpose: UR10eCuroboMoveIt is the ROS 2 node that orchestrates the entire control and planning pipeline for the UR10e manipulator.

It:

- Initializes all ROS 2 publishers, subscribers, and TF listeners
- Loads configuration from config.py and environment variables
- Initializes cuRobo’s GPU-based motion planner (MotionGen)
- Handles real-time robot state, path visualization, and marker feedback
- Provides a keyboard interface for manual control
- Integrates gripper control, goal capture, marker tracking, and perception feedback

**Execution Flow Summary**

Startup
Initialize node → load configs → set up pubs/subs → warm up cuRobo

Idle Mode
Waits for joint states or keyboard input

**Goal Acquisition**

- From RViz (y)
- Manual input (m)
- Perception input (s)
- Goal Execution (n)


**motions.py**

- Provides the building blocks for robot movement:
- Plan in joint or Cartesian space
- Execute single poses or sequences
- Move to pre-defined positions (home, drop-off, etc.)
- Stop the robot safely
- Adjust wrist orientation or perform small retreat motions
- Wait until the robot reaches the target pose



**Command to Generate URDF**

Run the following command to convert the Xacro into a single flattened .urdf file:

```
ros2 run xacro xacro \
  ~/Documents/v3/manipulatorsdatepalm/ur_ws_new/src/ur_robot_driver/ur_description/urdf/ur.urdf.xacro \
  name:=ur10e \
  ur_type:=ur10e \
  prefix:="" \
  use_fake_hardware:=false \
  safety_limits:=true \
  safety_pos_margin:=0.15 \
  safety_k_position:=20 \
  joint_limits_parameters_file:=~/Documents/v3/manipulatorsdatepalm/ur_ws_new/src/ur_robot_driver/ur_description/config/ur10e/joint_limits.yaml \
  kinematics_parameters_file:=~/Documents/v3/manipulatorsdatepalm/ur_ws_new/src/ur_robot_driver/ur_description/config/ur10e/default_kinematics.yaml \
  physical_parameters_file:=~/Documents/v3/manipulatorsdatepalm/ur_ws_new/src/ur_robot_driver/ur_description/config/ur10e/physical_parameters.yaml \
  visual_parameters_file:=~/Documents/v3/manipulatorsdatepalm/ur_ws_new/src/ur_robot_driver/ur_description/config/ur10e/visual_parameters.yaml \
  > ~/ur10e_curobo.urdf
```
