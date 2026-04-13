# launch/combined.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    main_arg = DeclareLaunchArgument(
        'main', default_value='true',
        description='Launch main control node'
    )
    teleop_arg = DeclareLaunchArgument(
        'teleop', default_value='false',
        description='Launch teleop node'
    )
    vision_arg = DeclareLaunchArgument(
        'vision', default_value='false',
        description='Launch vision node'
    )
    lidar_arg = DeclareLaunchArgument(
        'lidar', default_value='false',
        description='Use Livox LiDAR for depth instead of ZED stereo depth'
    )
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='192.168.1.190',
        description='Robot IP address'
    )
    use_fake_hardware_arg = DeclareLaunchArgument(
        'use_fake_hardware', default_value='false',
        description='Use fake hardware'
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='true',
        description='Launch RViz'
    )

    # UR Bringup
    ur_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ur_bringup'), '/launch/ur_control.launch.py'
        ]),
        launch_arguments={
            'ur_type': 'ur10e',
            'robot_ip': LaunchConfiguration('robot_ip'),
            'use_fake_hardware': LaunchConfiguration('use_fake_hardware'),
            'launch_rviz': LaunchConfiguration('launch_rviz'),
        }.items()
    )

    # Main control node
    main_node = Node(
        package='ur10e_curobo',
        executable='main',
        name='ur10e_control',
        output='screen',
        condition=IfCondition(LaunchConfiguration('main'))
    )

    # Teleop node
    teleop_node = Node(
        package='ur10e_curobo',
        executable='teleop',
        name='ur10e_teleop',
        output='screen',
        condition=IfCondition(LaunchConfiguration('teleop'))
    )

    # Vision node — pass --use_lidar flag when lidar:=true
    vision_args = PythonExpression([
        '"--use_lidar" if "', LaunchConfiguration('lidar'), '" == "true" else ""'
    ])
    vision_node = Node(
        package='ur10e_curobo',
        executable='vision',
        name='ur10e_vision',
        output='screen',
        arguments=[vision_args],
        condition=IfCondition(LaunchConfiguration('vision'))
    )

    # LiDAR driver node — only started when lidar:=true and vision:=true
    lidar_node = Node(
        package='livox_ros2_driver',
        executable='livox_ros2_driver_node',
        name='livox_lidar',
        output='screen',
        condition=IfCondition(PythonExpression([
            '"true" if "', LaunchConfiguration('lidar'), '" == "true" and "',
            LaunchConfiguration('vision'), '" == "true" else "false"'
        ]))
    )

    delayed_vision = TimerAction(period=3.0, actions=[vision_node])
    delayed_lidar = TimerAction(period=1.0, actions=[lidar_node])

    return LaunchDescription([
        # Arguments
        main_arg,
        teleop_arg,
        vision_arg,
        lidar_arg,
        robot_ip_arg,
        use_fake_hardware_arg,
        launch_rviz_arg,
        # Nodes
        ur_bringup,
        main_node,
        teleop_node,
        delayed_lidar,
        delayed_vision,
    ])
