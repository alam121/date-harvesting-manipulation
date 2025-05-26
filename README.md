# ManipulatorsDatePalm 

## Dependencies 
1. Install ```ros2 foxy``` via debian package
2. Install the following dependencies:
```
sudo apt install ros-foxy-ur-msgs
sudo apt install ros-foxy-ur-client-library
sudo apt install ros-foxy-moveit
sudo apt install ros-foxy-moveit-servo
sudo apt install ros-foxy-moveit-visual-tools
sudo apt install ros-foxy-moveit-resources
sudo apt install ros-foxy-moveit-planners-ompl
sudo apt install ros-foxy-moveit-ros-perception
sudo apt install ros-foxy-ros2-control
sudo apt install ros-foxy-ros2-controllers
sudo apt install ros-foxy-controller-interface
sudo apt install ros-foxy-controller-manager
sudo apt install ros-foxy-control-toolbox
sudo apt install ros-foxy-realtime-tools
sudo apt install ros-foxy-rviz2
sudo apt install ros-foxy-rviz-visual-tools
sudo apt install ros-foxy-ackermann-msgs
```
## Install workspace 

1. Clone the repository ```git clone https://gitlab.kaust.edu.sa/risc/manipulatorsdatepalm.git```
2. Move to ```cd manipulatordateplam/ur_ws_new/src``` and run ```colcon build --symlink-install```
