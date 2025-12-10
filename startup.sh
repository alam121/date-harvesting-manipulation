#!/bin/bash

# Step 1: source and set IP
source install/setup.bash
./set_ip.sh

# Step 2: launch UR and Gripper
gnome-terminal -- bash -c "ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:=false launch_rviz:=true; exec bash"

gnome-terminal -- bash -c "ros2 launch delto_3f_driver delto_3f_bringup.launch.py delto_ip:=169.254.186.72 delto_port:=502; exec bash"

# Step 3: motion
gnome-terminal -- bash -c "python3 -m ur10e_curobo.main; exec bash"

# Step 4: vision
gnome-terminal -- bash -c "python3 date_v1.4.py --weights models/lab_dates_realbunch.pt --conf_thres 0.5; exec bash"

