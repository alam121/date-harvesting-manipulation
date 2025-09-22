# ManipulatorsDatePalm 

## Dependencies 
1. Install ```ros2 foxy``` via debian package
2. Install [Curobo](https://curobo.org/get_started/5_docker_development.html#docker-dev) in docker for native usage on jetson
3. Install the following dependencies:
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
pip install warp-lang==1.0.0

```
## Install workspace 

1. Clone the repository ```git clone https://gitlab.kaust.edu.sa/risc/manipulatorsdatepalm.git```
2. Move to ```cd manipulatordateplam/ur_ws_new/src``` and run ```colcon build --symlink-install```

## Setup UR10 Connections 
```
sudo ip addr flush dev eth0
sudo ip addr add 192.168.1.101/24 dev eth0
sudo ip addr add 169.254.186.100/24 dev eth0
sudo ip link set eth0 up
ip a | grep eth0
```
Output should look like this:
```
eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    inet 169.254.186.100/24 scope global eth0
    inet 192.168.1.101/24 scope global eth0
```

## Installing Curobo
- Follow instructions at [CuRobo](https://curobo.org/get_started/1_install_instructions.html)
- May have to fix some of the dependencies version conflict with this ```python -m pip install -U scikit-build-core "setuptools>=71" "packaging>=24" "pybind11[global]" cmake ninja``` 


## YOLOv8 
1. NOTE: [Yolov8 Segmentation](https://github.com/marcoslucianops/DeepStream-Yolo-Seg/tree/master) (yet to try out)
2. Install YOLOv8 in vitual environment
```pip install ultralytics```
3. Download the labelled data from roboflow 
Train the nextwork
```yolo task=segment mode=train model=yolov8n-seg.pt data=/home/datepalm-ws/Documents/date_detection_maskrcnn/dateopalm_tree.v4i.yolov8-obb/data.yaml epochs=50 imgsz=720 batch=8```

4. ### Trained Model to ONNX 
```yolo export model=best.pt format=onnx opset=12 imgsz=640 dynamic=False```

5. ### Saving ONNX model to TRT Engine
```/usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.trt --explicitBatch --fp16```

6. Install [ZED-GSTREAMER](https://github.com/stereolabs/zed-gstreamer)
7. Consult [this](https://github.com/valdivj/gstream_Deep?tab=readme-ov-file)


## Run UR10 launch file
```ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:=false launch_rviz:=true```

## Run Delto gripper launch file
```ros2 launch delto_3f_driver delto_3f_bringup.launch.py delto_ip:=169.254.186.72 delto_port:=502 ```

## Setting Headless machine 
1. ```ssh -X username@remote_ip```
2. ```sudo nano /etc/ssh/sshd_config```
3. make sure following lines exit and are uncommented 
```X11Forwarding yes X11DisplayOffset 10 X11UseLocalhost yes```
4. ```sudo systemctl restart ssh```
5. ```sudo apt install xauth```

## RUNNING COLMAP on the NVIDIA Jetson AGX ORIN

1. Install dependencies:
sudo apt update
sudo apt install git cmake build-essential ninja-build libboost-all-dev libeigen3-dev /
libsuitesparse-dev qtbase5-dev libglew-dev libfreeimage-dev 
libgoogle-glog-dev libflags-dev libopencv-dev libatlas-base-dev 
libceres-dev libsqlite3-dev libcgal-dev libqt5opengl5-dev libmetis-dev


2. Download Colmap package from github and install: git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
ninja
sudo ninja


3. Check colmap
colmap -h
colmap gui
