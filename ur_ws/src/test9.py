import rclpy
from rclpy.node import Node
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState as ROSJointState
from visualization_msgs.msg import InteractiveMarkerFeedback
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from ros2_robotiqgripper.srv import RobotiqGripper 
from std_msgs.msg import Bool
import torch
import time
import sys
import termios
import tty
import select
import threading

class UR10eCuroboMoveIt(Node):
    def __init__(self):
        super().__init__("ur10e_curobo_moveit_node")

        # ROS2 publisher for MoveIt trajectory execution
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )

        # ROS2 subscriber to get the current joint state
        self.joint_state_sub = self.create_subscription(
            ROSJointState, "/joint_states", self.joint_state_callback, 10
        )
        
        self.planned_path_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )
        
        # New publisher for goal position markers
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)


        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)

        # Store the path points
        self.path_points = []

        # ROS2 subscriber for MoveIt interactive marker feedback
        self.marker_sub = self.create_subscription(
            InteractiveMarkerFeedback,
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback",
            self.marker_callback,
            10
        )

        # TF buffer and listener for end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.create_timer(0.1, self.track_robot_path)  # Updates path every 0.1 sec


        self.current_joint_positions = None
        self.latest_marker_pose = None  # Store latest marker pose
        self.goal_poses = []
        self.running = True  # Control flag for keyboard thread

        self.joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Load cuRobo motion planning config for UR10e
        world_config = {"cuboid": {"table": {"dims": [5.0, 5.0, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0]}}}

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            "ur10e.yml", world_config, interpolation_dt=0.005
        )

        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()

        self.get_logger().info("Waiting for joint states...")
        self.timer = self.create_timer(0.5, self.check_joint_states)
        self.keyboard_thread = threading.Thread(target=self.wait_for_key_press, daemon=True)
        self.keyboard_thread.start()

    def check_joint_states(self):
        """Check if joint states are received, then proceed with execution."""
        if self.current_joint_positions is not None:
            self.get_logger().info(f"Received initial joint positions: {self.current_joint_positions}")
            self.destroy_timer(self.timer)

    def joint_state_callback(self, msg):
        """Callback to update the latest joint positions and velocities."""
        joint_map = dict(zip(msg.name, msg.position))
        velocity_map = dict(zip(msg.name, msg.velocity))  # Extract velocity data

        self.current_joint_positions = [joint_map[joint] for joint in self.joint_order]
        self.current_joint_velocities = [velocity_map[joint] for joint in self.joint_order]  # Store velocities

        #self.get_logger().debug(f"Updated joint positions: {self.current_joint_positions}")
        #self.get_logger().debug(f"Updated joint velocities: {self.current_joint_velocities}")


    def marker_callback(self, msg):
        """Callback to store the latest marker pose from MoveIt."""
        self.latest_marker_pose = msg.pose
        #self.get_logger().info(f"Received marker feedback: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

    def get_key(self, timeout=0.1):
        """Reads a single key press with a timeout."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def wait_for_key_press(self):
        """Wait for user input: 'y' for interactive marker, 'm' for manual entry, 'n' to execute, 'q' to quit."""
        print("Press 'y' to store a marker position, 'm' to enter X, Y, Z manually.")
        print("Press 'n' to execute stored goals. Press 'q' to exit.")
        print("Do you want to (O)pen or (C)lose the gripper?")

        while self.running:
            key = self.get_key()

            if key == "y" and self.latest_marker_pose:
                # Store goal from interactive marker
                goal_position = [
                    self.latest_marker_pose.position.x,
                    self.latest_marker_pose.position.y,
                    self.latest_marker_pose.position.z,
                    self.latest_marker_pose.orientation.w,
                    self.latest_marker_pose.orientation.x,
                    self.latest_marker_pose.orientation.y,
                    self.latest_marker_pose.orientation.z,
                ]
                self.goal_poses.append(goal_position)
                print(f"Goal {len(self.goal_poses)} saved from interactive marker: {goal_position}")
                self.publish_goal_marker(goal_position[:3])

            elif key == "m":
                # Retrieve the current end-effector pose
                current_pose = self.get_end_effector_pose()
                
                if current_pose:
                    print(f"\nCurrent End-Effector Position: X={current_pose[0]:.3f}, "
                          f"Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
                else:
                    print("⚠️ Warning: Could not retrieve current end-effector pose!")

                # Ask for manual coordinates
                try:
                    x = float(input("Enter new X coordinate: "))
                    y = float(input("Enter new Y coordinate: "))
                    z = float(input("Enter new Z coordinate: "))

                    # Keep the same orientation as the current pose or default to identity quaternion
                    if current_pose:
                        goal_position = [x, y, z] + current_pose[3:]
                    else:
                        goal_position = [x, y, z, 1.0, 0.0, 0.0, 0.0]

                    self.goal_poses.append(goal_position)
                    print(f"Goal {len(self.goal_poses)} saved manually: {goal_position}")
                    self.publish_goal_marker(goal_position[:3])
                except ValueError:
                    print("⚠️ Invalid input! Please enter numerical values for X, Y, Z.")

            elif key == "n" and self.goal_poses:
                print("Executing motion...")
                self.plan_and_execute()
                self.get_logger().info("All goals executed. Waiting for gripper command...")

            elif key == "o":
                self.control_gripper("OPEN")

            elif key == "c":
                self.control_gripper("CLOSE")

            elif key == "q":
                print("Exiting...")
                self.running = False
                break



                
    def get_end_effector_pose(self):
        """Retrieve the end-effector pose using TF lookup."""
        self.get_logger().info("Retrieving end-effector pose...")

        source_frame, target_frame = "base_link", "tool0"

        try:
            transform = self.tf_buffer.lookup_transform(
                source_frame, target_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            position = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            orientation = [transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]
            self.get_logger().info(f"Current End-Effector Pose: {position}, {orientation}")
            return position + orientation
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get end-effector pose: {e}")
            return None

    def ask_goal_position(self, current_position):
        """Prompt user for new goal position."""
        print("\nCurrent End-Effector Position:", current_position[:3])
        x = float(input("Enter new X coordinate: "))
        y = float(input("Enter new Y coordinate: "))
        z = float(input("Enter new Z coordinate: "))

        goal_position = [x, y, z] + current_position[3:]  # Keep same orientation
        self.get_logger().info(f"New Goal Position: {goal_position[:3]}")
        return goal_position


    def plan_and_execute(self):
        """Execute motion plans sequentially while dynamically updating the path visualization and waiting for execution completion."""
        self.get_logger().info("Starting path planning for all goals sequentially")

        if self.current_joint_positions is None:
            self.get_logger().warn("Current joint state not received yet!")
            return

        if not self.goal_poses:
            self.get_logger().warn("No goal positions stored!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize start state from the current joint positions
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        while self.goal_poses and self.running:
            goal = self.goal_poses.pop(0)  # Get and remove the first goal
            self.get_logger().info(f"Processing goal: {goal}")

            goal_pose = Pose.from_list(goal)

            # Generate motion plan
            result = self.motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=5))

            if result.success:
                self.get_logger().info("Motion plan generated successfully! Executing...")

                trajectory_msg = JointTrajectory()
                trajectory_msg.joint_names = self.joint_order
                interpolated_plan = result.get_interpolated_plan()

                if isinstance(interpolated_plan, JointState):
                    interpolated_plan = interpolated_plan.position
                if not isinstance(interpolated_plan, torch.Tensor):
                    interpolated_plan = torch.tensor(interpolated_plan, dtype=torch.float32)
                interpolated_plan = interpolated_plan.to("cpu")

                if len(interpolated_plan) < 2:
                    self.get_logger().warn("Generated trajectory has too few points or no motion!")
                    continue

                time_from_start = 0.0
                for i, point in enumerate(interpolated_plan):
                    traj_point = JointTrajectoryPoint()
                    traj_point.positions = point.numpy().tolist()
                    traj_point.velocities = [0.1] * len(self.joint_order)
                    traj_point.time_from_start.sec = int(time_from_start)
                    traj_point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
                    time_from_start += 0.03
                    trajectory_msg.points.append(traj_point)

                    # Convert joint space to Cartesian space and update the path
                    cartesian_point = self.forward_kinematics(traj_point.positions)
                    if cartesian_point:
                        self.path_points.append(cartesian_point)
                        self.publish_path_marker()  # Update path dynamically

                # Publish trajectory for execution
                self.trajectory_pub.publish(trajectory_msg)

                # Wait for execution to complete before proceeding to the next goal
                self.wait_for_execution_completion(goal[:3])

                # Update start state to the last position of the executed trajectory
                start_state = JointState.from_position(
                    torch.tensor([trajectory_msg.points[-1].positions], dtype=torch.float32, device=device),
                    joint_names=self.joint_order,
                )
                self.get_logger().info(f"Updated start state for next goal: {start_state.position.tolist()}")

            else:
                self.get_logger().warn("Failed to generate a motion plan for goal!")

        self.get_logger().info("All goals executed. Waiting for new goals...")




    def publish_path_marker(self):
        """Publishes a Marker line strip to visualize the actual robot's movement path."""
        marker = Marker()
        marker.header.frame_id = "base_link"  # Adjust if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.01  # Line width

        # Set color (green)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Assign the stored path points to the marker
        marker.points = self.path_points

        # Publish the marker
        self.path_marker_pub.publish(marker)
        #self.get_logger().info("Published real-time path marker to RViz.")


        
    def forward_kinematics(self, joint_positions):
        """Compute the end-effector position in Cartesian space using forward kinematics."""
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return Point(
                x=transform.transform.translation.x,
                y=transform.transform.translation.y,
                z=transform.transform.translation.z,
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to compute forward kinematics: {e}")
            return None


    def track_robot_path(self):
        """Continuously track the robot's actual end-effector position and update the path in RViz."""
        try:
            # Get real-time end-effector position
            transform = self.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Convert to a ROS Point
            current_point = Point(
                x=transform.transform.translation.x,
                y=transform.transform.translation.y,
                z=transform.transform.translation.z,
            )

            # Only add if it's different from the last point (to avoid redundant updates)
            if not self.path_points or (self.path_points[-1].x != current_point.x or
                                        self.path_points[-1].y != current_point.y or
                                        self.path_points[-1].z != current_point.z):
                self.path_points.append(current_point)
                self.publish_path_marker()  # Update the visualization

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to track robot position: {e}")

            
            
            
    def wait_for_execution_completion(self, goal_position, tolerance=0.08):
        """Waits until the robot reaches the target position within a given tolerance."""
        self.get_logger().info(f"Waiting for robot to reach goal: {goal_position}")
        
        while self.running:
            current_position = self.get_end_effector_pose()
            
            if current_position is None:
                self.get_logger().warn("Failed to retrieve current position. Retrying...")
                time.sleep(0.1)
                continue

            # Compute Euclidean distance between current position and goal
            distance = ((current_position[0] - goal_position[0]) ** 2 +
                        (current_position[1] - goal_position[1]) ** 2 +
                        (current_position[2] - goal_position[2]) ** 2) ** 0.5

            if distance < tolerance:
                self.get_logger().info(f"Goal reached with distance {distance:.4f}. Proceeding to next goal.")
                break

            time.sleep(0.1)  # Check every 100ms
            
            
            
                
    def publish_path_marker(self):
        """Publishes a Marker line strip to visualize the actual robot's movement path."""
        marker = Marker()
        marker.header.frame_id = "base_link"  # Adjust if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # Keep a continuous path
        marker.action = Marker.ADD

        marker.scale.x = 0.01  # Line width

        # Set color (green for actual movement)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Assign the stored path points to the marker
        marker.points = self.path_points

        # Publish the marker
        self.path_marker_pub.publish(marker)
        #self.get_logger().info("Published real-time path marker to RViz.")
        
        
        
    def publish_goal_marker(self, position):
        """Publishes a marker in RViz for the goal position."""
        marker = Marker()
        marker.header.frame_id = "base_link"  # Adjust if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_positions"
        marker.id = len(self.goal_poses)  # Unique ID for each goal
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]

        # Marker size
        marker.scale.x = 0.05  # 5cm sphere
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # Set color (red for goals)
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Publish the marker
        self.goal_marker_pub.publish(marker)
        self.get_logger().info(f"Published goal marker at {position}.")

    def wait_until_motion_finishes(self):
        """Dynamically check if motion execution is complete before moving to the next goal."""
        self.get_logger().info("Monitoring execution for smooth transitions...")

        while self.running:
            if self.is_robot_moving():
                time.sleep(0.03)  # 🔹 Faster update for smooth blending
            else:
                self.get_logger().info("Motion execution finished. Moving to the next goal.")
                break



    def is_robot_moving(self, velocity_threshold=0.001):
        """Check if the robot is still moving by monitoring joint velocities."""
        
        print("Checking if the robot is moving...")

        if self.current_joint_positions is None:
            print("No joint state data available. Assuming robot is NOT moving.")
            return False  # No data → assume it's stopped

        try:
            # Retrieve joint velocities from the latest joint state message
            velocities = self.current_joint_velocities  # New variable to store velocities

            if velocities is None:
                print("No velocity data available. Assuming robot has stopped.")
                return False

            # Print velocity values for debugging
            print(f"Joint velocities: {velocities}")

            # If any velocity is above the threshold, the robot is still moving
            if any(abs(v) > velocity_threshold for v in velocities):
                print("Robot is still moving!")
                return True

        except Exception as e:
            print(f"Error checking robot movement: {e}")
            return False  # If we can't check → assume it's stopped

        print("Robot has stopped moving.")
        return False  # Default to False if nothing indicates movement



    def control_gripper(self, action):
        """Publish a command to open or close the Delto 3F gripper."""
        self.get_logger().info(f"Sending Delto 3F gripper command: {action}")

        self.gripper_pub = self.create_publisher(Bool, "/gripper/grasp", 10)
        msg = Bool()
        msg.data = True if action.upper() == "CLOSE" else False  # True = Close, False = Open
        self.gripper_pub.publish(msg)

        self.get_logger().info(f"Published gripper command '{msg.data}' to /gripper/grasp")




def main():
    print("Starting UR10e MoveIt Node...")
    rclpy.init()
    node = UR10eCuroboMoveIt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down node.")
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

