"""Launch the real UR10e cuRobo stack against an Isaac/hardware mirror.

This does not launch the physical UR driver or physical ZED camera. Instead,
``isaac_mirror`` provides the ROS topics/services that the real hardware stack
expects. The control node therefore runs the same planner, goal logic, gripper
logic, UI, and RViz tooling used for the robot.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    fruit_x = DeclareLaunchArgument("fruit_x", default_value="0.62")
    fruit_y = DeclareLaunchArgument("fruit_y", default_value="-0.04")
    fruit_z = DeclareLaunchArgument("fruit_z", default_value="0.82")
    fruit_radius = DeclareLaunchArgument("fruit_radius", default_value="0.024")
    publish_fake_vision = DeclareLaunchArgument("publish_fake_vision", default_value="true")
    backend = DeclareLaunchArgument("backend", default_value="udp")
    joint_state_source = DeclareLaunchArgument(
        "joint_state_source", default_value="isaac"
    )
    debug_plan_preview = DeclareLaunchArgument("debug_plan_preview", default_value="false")
    start_control = DeclareLaunchArgument("start_control", default_value="false")
    discovery_server = DeclareLaunchArgument(
        "discovery_server", default_value="10.76.154.75:11811"
    )

    mirror = Node(
        package="ur10e_curobo",
        executable="isaac_mirror",
        name="isaac_hardware_mirror",
        output="screen",
        parameters=[{
            "fruit_x": ParameterValue(LaunchConfiguration("fruit_x"), value_type=float),
            "fruit_y": ParameterValue(LaunchConfiguration("fruit_y"), value_type=float),
            "fruit_z": ParameterValue(LaunchConfiguration("fruit_z"), value_type=float),
            "fruit_radius": ParameterValue(
                LaunchConfiguration("fruit_radius"), value_type=float
            ),
            "publish_fake_vision": ParameterValue(
                LaunchConfiguration("publish_fake_vision"), value_type=bool
            ),
            "backend": LaunchConfiguration("backend"),
            "joint_state_source": LaunchConfiguration("joint_state_source"),
        }],
    )

    control = Node(
        package="ur10e_curobo",
        executable="main",
        name="ur10e_control",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_control")),
        parameters=[{
            "planner.use_fake_hardware": True,
            "planner.debug_plan_preview": ParameterValue(
                LaunchConfiguration("debug_plan_preview"), value_type=bool
            ),
            "perception.enabled": False,
        }],
    )

    return LaunchDescription([
        discovery_server,
        SetEnvironmentVariable("ROS_DOMAIN_ID", "6"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable(
            "ROS_DISCOVERY_SERVER", LaunchConfiguration("discovery_server")
        ),
        UnsetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE"),
        fruit_x,
        fruit_y,
        fruit_z,
        fruit_radius,
        publish_fake_vision,
        backend,
        joint_state_source,
        debug_plan_preview,
        start_control,
        mirror,
        control,
    ])
