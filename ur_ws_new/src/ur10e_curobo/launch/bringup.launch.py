# launch/bringup.launch.py
from launch import LaunchDescription
from launch.actions import TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node

def generate_launch_description():
    control_node = Node(
        package="ur10e_curobo",
        executable="main",              # your control entry point
        name="ur10e_control",
        output="screen",
        # env vars to keep memory low; change as you like
        env={
            "UR10E_DISABLE_PERCEPTION": "1" # make sure control doesn't spawn its own perception
        },
    )

    perception_node = Node(
        package="ur10e_curobo",
        executable="perception",        # your packaged perception entry point
        name="ur10e_perception",
        output="screen",
        env={
            "CUDA_VISIBLE_DEVICES": "",     # CPU for YOLO/ZED unless you have headroom
            "UR10E_YOLO_WEIGHTS": "yolov8n-seg.pt",
            "UR10E_YOLO_IMGSZ": "512",
            "UR10E_SHOW_VIEW": "0",         # set 1 if you want OpenCV windows
        },
    )

    # Delay perception by 5 seconds after launch starts
    delayed_perception = TimerAction(period=5.0, actions=[perception_node])

    return LaunchDescription([
        control_node,
        delayed_perception,
    ])

