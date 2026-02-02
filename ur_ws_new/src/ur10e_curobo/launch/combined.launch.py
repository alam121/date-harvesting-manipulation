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

    # Vision node (delayed to let other nodes initialize)
    vision_node = Node(
        package='ur10e_curobo',
        executable='vision',
        name='ur10e_vision',
        output='screen',
        condition=IfCondition(LaunchConfiguration('vision'))
    )
    delayed_vision = TimerAction(period=3.0, actions=[vision_node])

    return LaunchDescription([
        # Arguments
        main_arg,
        teleop_arg,
        vision_arg,
        robot_ip_arg,
        use_fake_hardware_arg,
        launch_rviz_arg,
        # Nodes
        ur_bringup,
        main_node,
        teleop_node,
        delayed_vision,
    ])
