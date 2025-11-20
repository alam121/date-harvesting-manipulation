# Demo Commands and System Launch Guide

This README describes the procedure to configure network settings, launch the UR10e robot and gripper, run the vision module, subscribe goals, and execute motions.

---

## 0. Repository Location

The main repository is located at: Humble_Refactor6.1_labdatesrealbunch 


## 1. Configure Network Interface

Run the following commands:
sudo ip addr flush dev eno1
sudo ip addr add 192.168.1.101/24 dev eno1
sudo ip addr add 169.254.186.100/24 dev eno1
sudo ip link set eno1 up


Expected output includes:

inet 169.254.186.100/24
inet 192.168.1.101/24


If these IPs do not appear, repeat the steps.

**Important:**  
Each time the terminal is closed, you must configure the IPs again.

---

## 2. Launch UR10e + Gripper + RViz
build: colcon build --cmake-args -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so (if colcon build dosn't work)
Run: ros2 launch ur_bringup ur_control.launch.py
ur_type:=ur10e
robot_ip:=192.168.1.190
use_fake_hardware:=false
launch_rviz:=true



This launches:
- UR10e ROS driver  
- Gripper interface  
- RViz with robot model  

If something fails, re-source the workspace and re-check IP configuration.

---

## 3. UR Tablet Instructions

On the UR tablet:

1. Power on the robot.
2. Release the brakes.
3. Go to **Program**.
4. Load the **ucra** file.
5. Press **Start**.

If you see: Connection to reverse interface dropped.


Restart the program on the tablet and press **Start** again.

---

## 4. Run Refactor Code (Control and Vision)

### 4.1 Control Node: 
python3 -m ur10e_curobo.main


### 4.2 Vision Node

Navigate to: /manipulatorsdatepalm/ur_ws_new/src/zed_date_detector


Run vision: python3 date_v1.2.py --weights models/lab_dates_realbunch.pt --conf_thres 0.5



An OpenCV window will open with detections, and external goals will be published to ROS 2 topics.

### Common Runtime Error

If you see: Cannot execute goals: robot program is OFF.



Make sure the **Start** button on the tablet is pressed.

---

## 5. Goal Subscription (Motion Window)

Press the following keys:

- `s` — subscribe a single goal  
- `p` — subscribe multiple goals for 10 seconds  

Check RViz to ensure:
- Red dot is visible  
- Goal markers are visible  

---

## 6. Execute Motion

Press: n


This executes the planned robot motion.

---

## 7. Gripper Manual Control

- `o` — open gripper  
- `c` — close gripper  

---









