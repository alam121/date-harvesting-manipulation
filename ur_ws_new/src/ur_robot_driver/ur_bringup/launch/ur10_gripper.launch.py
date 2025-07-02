# Copyright (c) 2021 PickNik, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Denis Stogl

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = []

    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument("ur_type", description="Type/series of used UR robot.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_ip", description="IP address by which the robot can be reached.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("safety_limits", default_value="true", description="Enables safety limits if true.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("safety_pos_margin", default_value="0.15", description="Safety margin.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("safety_k_position", default_value="20", description="Safety k-position factor.")
    )

    # Delto gripper arguments
    declared_arguments.append(
        DeclareLaunchArgument("delto_ip", default_value="192.168.1.20", description="IP of Delto gripper.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("delto_port", default_value="502", description="Port of Delto gripper.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("p_gain", default_value="10.0", description="P-gain for Delto driver.")
    )
    declared_arguments.append(
        DeclareLaunchArgument("d_gain", default_value="1.0", description="D-gain for Delto driver.")
    )

    # General arguments
    declared_arguments.extend([
        DeclareLaunchArgument("runtime_config_package", default_value="ur_bringup", description='Config package.'),
        DeclareLaunchArgument("controllers_file", default_value="controllers.yaml", description="Controllers config file."),
        DeclareLaunchArgument("description_package", default_value="ur_description", description="Description package."),
        DeclareLaunchArgument("description_file", default_value="ur10_gripper.xacro", description="URDF/XACRO file."),
        DeclareLaunchArgument("prefix", default_value='""', description="Joint name prefix."),
        DeclareLaunchArgument("use_fake_hardware", default_value="false", description="Use fake hardware."),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false", description="Fake sensor cmds."),
        DeclareLaunchArgument("robot_controller", default_value="joint_trajectory_controller", description="Controller to start."),
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")
    ])

    # Launch configurations
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    delto_ip = LaunchConfiguration("delto_ip")
    delto_port = LaunchConfiguration("delto_port")
    p_gain = LaunchConfiguration("p_gain")
    d_gain = LaunchConfiguration("d_gain")
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    prefix = LaunchConfiguration("prefix")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    robot_controller = LaunchConfiguration("robot_controller")
    launch_rviz = LaunchConfiguration("launch_rviz")

    # Parameter file paths
    joint_limit_params = PathJoinSubstitution([
        FindPackageShare(description_package), "config", ur_type, "joint_limits.yaml"
    ])
    kinematics_params = PathJoinSubstitution([
        FindPackageShare(description_package), "config", ur_type, "default_kinematics.yaml"
    ])
    physical_params = PathJoinSubstitution([
        FindPackageShare(description_package), "config", ur_type, "physical_parameters.yaml"
    ])
    visual_params = PathJoinSubstitution([
        FindPackageShare(description_package), "config", ur_type, "visual_parameters.yaml"
    ])
    script_filename = PathJoinSubstitution([
        FindPackageShare("ur_robot_driver"), "resources", "ros_control.urscript"
    ])
    input_recipe_filename = PathJoinSubstitution([
        FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"
    ])
    output_recipe_filename = PathJoinSubstitution([
        FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"
    ])

    rviz_config_file = PathJoinSubstitution([
        FindPackageShare(description_package), "rviz", "view_robot.rviz"
    ])

    # Robot description including Delto gripper
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), ' ',
        PathJoinSubstitution([FindPackageShare(description_package), 'urdf', description_file]),
        ' robot_ip:=', robot_ip,
        ' joint_limit_params:=', joint_limit_params,
        ' kinematics_params:=', kinematics_params,
        ' physical_params:=', physical_params,
        ' visual_params:=', visual_params,
        ' safety_limits:=', safety_limits,
        ' safety_pos_margin:=', safety_pos_margin,
        ' safety_k_position:=', safety_k_position,
        ' prefix:=', prefix,
        ' use_fake_hardware:=', use_fake_hardware,
        ' fake_sensor_commands:=', fake_sensor_commands,
        ' script_filename:=', script_filename,
        ' input_recipe_filename:=', input_recipe_filename,
        ' output_recipe_filename:=', output_recipe_filename,
        ' delto_ip:=', delto_ip,
        ' delto_port:=', delto_port,
        ' p_gain:=', p_gain,
        ' d_gain:=', d_gain
    ])
    robot_description = {"robot_description": robot_description_content}

    # Controllers YAML
    robot_controllers = PathJoinSubstitution([
        FindPackageShare(runtime_config_package), "config", controllers_file
    ])

    # Nodes
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output={'stdout': 'screen', 'stderr': 'screen'}
    )

    dashboard_client_node = Node(
        package="ur_robot_driver",
        condition=UnlessCondition(use_fake_hardware),
        executable="dashboard_client",
        name="dashboard_client",
        output="screen",
        emulate_tty=True,
        parameters=[{"robot_ip": robot_ip}]
    )

    controller_stopper_node = Node(
        package="ur_robot_driver",
        executable="controller_stopper_node",
        name="controller_stopper",
        output="screen",
        emulate_tty=True,
        parameters=[{"consistent_controllers": [
            "io_and_status_controller",
            "force_torque_sensor_broadcaster",
            "joint_state_broadcaster",
            "speed_scaling_state_broadcaster"
        ]}]
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description]
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file]
    )

    # Spawners
    spawner_args = ['joint_state_broadcaster', '--controller-manager', '/controller_manager']
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner.py",
        arguments=spawner_args
    )

    io_and_status_controller_spawner = Node(
        package="controller_manager",
        executable="spawner.py",
        arguments=[
            'io_and_status_controller',
            '--controller-manager',
            '/controller_manager'
        ]
    )

    speed_scaling_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner.py",
        arguments=[
            'speed_scaling_state_broadcaster',
            '--controller-manager',
            '/controller_manager'
        ]
    )

    force_torque_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner.py",
        arguments=[
            'force_torque_sensor_broadcaster',
            '--controller-manager',
            '/controller_manager'
        ]
    )

    # UR10 arm controller spawner
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner.py",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"]
    )

    # Delto gripper controller spawner
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner.py",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"]
    )

    nodes_to_start = [
        control_node,
        dashboard_client_node,
        controller_stopper_node,
        robot_state_publisher_node,
        rviz_node,
        joint_state_broadcaster_spawner,
        io_and_status_controller_spawner,
        speed_scaling_state_broadcaster_spawner,
        force_torque_sensor_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner
    ]

    return LaunchDescription(declared_arguments + nodes_to_start)

