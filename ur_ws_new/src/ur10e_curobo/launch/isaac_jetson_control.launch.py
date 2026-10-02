"""Run the hardware control pipeline on Jetson against the remote Isaac mirror."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    fastdds_profile = DeclareLaunchArgument("fastdds_profile", default_value="")
    _profile_given = PythonExpression(["'", LaunchConfiguration("fastdds_profile"), "' != ''"])
    discovery_server = DeclareLaunchArgument(
        "discovery_server", default_value="10.76.154.75:11811"
    )
    environment = DeclareLaunchArgument("environment", default_value="outdoor")
    gripper_profile = DeclareLaunchArgument("gripper_profile", default_value="old")
    gripper_enabled = DeclareLaunchArgument("gripper_enabled", default_value="true")

    robot_description_content = Command([
        FindExecutable(name="xacro"),
        " ",
        PathJoinSubstitution([
            FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"
        ]),
        " name:=ur10e",
        " ur_type:=ur10e",
        " robot_profile:=old",
        " environment:=", LaunchConfiguration("environment"),
        " kinematics_params:=",
        PathJoinSubstitution([
            FindPackageShare("ur_description"),
            "config", "ur10e", "old_kinematics.yaml",
        ]),
    ])
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    control = Node(
        package="ur10e_curobo",
        executable="main",
        name="ur10e_control",
        output="screen",
        parameters=[{
            # Isaac supplies the real hardware ROS contract. Keeping this false
            # enables trajectory, Delto target/force, and UR program-state I/O.
            "planner.use_fake_hardware": False,
            "perception.enabled": False,
            # The lab profile's -0.015 m corrects the real ZED, which reads dates nearer than
            # they are at ~1.4 m (config.py _LAB_GRASP_OFFSETS). Isaac's wrist depth is exact,
            # so that correction left the closure in front of the date; +0.020 m (the outdoor
            # value) puts it at the date's centre.
            "planner.low_center_final_depth_offset": 0.020,
        }],
    )

    return LaunchDescription([
        fastdds_profile,
        discovery_server,
        environment,
        gripper_profile,
        gripper_enabled,
        SetEnvironmentVariable("ROS_DOMAIN_ID", "6"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "0"),
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        SetEnvironmentVariable(
            "ROS_DISCOVERY_SERVER", LaunchConfiguration("discovery_server")
        ),
        # A discovery-client profile that whitelists the route between the machines
        # (bin/fastdds_client_profile); without one, unset any buffer profile, which would
        # override the discovery-server client configuration.
        SetEnvironmentVariable(
            "FASTRTPS_DEFAULT_PROFILES_FILE", LaunchConfiguration("fastdds_profile"),
            condition=IfCondition(_profile_given)),
        UnsetEnvironmentVariable(
            "FASTRTPS_DEFAULT_PROFILES_FILE", condition=UnlessCondition(_profile_given)),
        SetEnvironmentVariable("UR10E_ROBOT_PROFILE", "old"),
        SetEnvironmentVariable("UR10E_ENVIRONMENT", LaunchConfiguration("environment")),
        # The wrist ZED X Mini as RGB-D, as launch_ur10e.py selects for the hardware: without these
        # the node fell back to the zed_one_rgb_zedx_mini_depth seed profile (zed2 frame) and the
        # refreshed lab hand-eye never loaded. The Isaac wrist camera uses the same extrinsic.
        SetEnvironmentVariable("UR10E_CAMERA_MODE", "zedx_mini"),
        SetEnvironmentVariable(
            "UR10E_CAMERA_PROFILE", [LaunchConfiguration("environment"), "_old_zedx_mini_rgbd"]),
        SetEnvironmentVariable("UR10E_CAM_FRAME", "zed_mini_left_camera_frame"),
        SetEnvironmentVariable("UR10E_ZEDMINI_CAM_FRAME", "zed_mini_left_camera_frame"),
        SetEnvironmentVariable(
            "UR10E_GRIPPER_PROFILE", LaunchConfiguration("gripper_profile")
        ),
        SetEnvironmentVariable(
            "UR10E_GRIPPER_ENABLED", LaunchConfiguration("gripper_enabled")
        ),
        robot_state_publisher,
        control,
    ])
