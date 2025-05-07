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
from geometry_msgs.msg import Point, Pose as ROSPose
from delto_gripper_controller import DeltoGripperController

import torch
import time
import sys
import termios
import tty
import select
import threading
import math

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
        
        self.wrist_publisher_ = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        self.planned_path_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )
        
        # New publisher for goal position markers
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)


        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)

        # Store the path points
        self.path_points = []


        self.stop_requested = False  # Flag to request stopping the current motion


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
            "wrist_3_joint"
        ]

        # Load cuRobo motion planning config for UR10e
        world_config = {"cuboid": {"table": {"dims": [5.0, 5.0, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0]}}}

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            "ur10e.yml", world_config, interpolation_dt=0.004) #,trajopt_dt=0.05, num_trajopt_seeds=5)
        
        

        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()


        self.get_logger().info("Waiting for joint states...")
        self.timer = self.create_timer(0.5, self.check_joint_states)
        #self.home_pose = [-0.0357, 0.3447, 0.5241, 0.0417, -0.7464, 0.6636, 0.0261]  # Cartesian home
        self.home_pose = [0.07565116137266159, -0.4599267244338989, 0.9337347745895386, 0.062273602932691574, -0.019624363631010056, -0.20094674825668335, 0.977423906326294] #Cartesian home        
        self.dropoff_pose = [-0.19983947277069092, -0.7162748575210571, 0.10007642209529877, 0.04266364127397537, -0.29519373178482056, -0.9115561842918396, 0.2830296754837036]

        self.gripper_controller = DeltoGripperController(self)

        self.move_to_home_position()
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

        self.get_logger().debug(f"Updated joint positions: {self.current_joint_positions}")
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
        print("Press 'y' to store a marker position, 'm' to enter X, Y, Z manually, press m to subsribe to a goal point")
        print("Press 'n' to execute stored goals. Press 'q' to exit.")
        print("Do you want to (O)pen or (C)lose the gripper?")

        while self.running:

            #self.get_end_effector_pose()
            #self.get_logger().info(f"joint positions: {self.current_joint_positions}")

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
                    
                    
            elif key == "s":
                    print("Subscribing to external goal pose topic...")
                    self.subscribe_to_goal_pose_topic()


            elif key == "n" and self.goal_poses:
                print("Executing motion...")
                
                if hasattr(self, 'goal_pose_sub'):
                    self.destroy_subscription(self.goal_pose_sub)
                    del self.goal_pose_sub
                    print("Unsubscribed from external goal poses.")
                    
                self.plan_and_execute()
                self.get_logger().info("All goals executed. Waiting for gripper command...")

            elif key == "o":
                self.control_gripper("OPEN")
                #self.move_backward(-0.05)
                #self.wait_until_motion_finishes()

            elif key == "c":
                self.control_gripper("CLOSE")
                self.move_backward(-0.01)
                self.rotate_wrist(90)
                self.rotate_wrist(-90)
                
                #self.wait_until_motion_finishes()
                
            elif key == "h":
                self.move_to_home_position()
                
            elif key == "d":
                print("Moving to drop-off zone...")
                self.move_to_dropoff_position()
                self.control_gripper("OPEN")

            elif key == "k":
                print("🛑 Stop requested! Interrupting current motion...")
                self.stop_requested = True


            elif key == "q":
                print("Exiting...")
                self.running = False
                break
              
    def get_end_effector_pose(self):
        """Compute current end-effector pose using CuRobo forward kinematics."""
        if self.current_joint_positions is None:
            self.get_logger().warn("Joint states not yet received.")
            return None

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            joint_state = JointState.from_position(
                torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order
            )

            ee_pose = self.motion_gen.rollout_fn.compute_kinematics(joint_state)

            position = ee_pose.ee_pos_seq[0].cpu().tolist()
            orientation = ee_pose.ee_quat_seq[0].cpu().tolist()

            self.get_logger().info(f"[FK] End-effector position: {position}, orientation: {orientation}")
            return position + orientation

        except Exception as e:
            self.get_logger().warn(f"FK computation failed: {e}")
            return None


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
            goal = self.goal_poses.pop(0)
            self.get_logger().info(f"Processing goal: {goal}")

            x, y, z = goal[:3]
            orientation = goal[3:]
            
            # 🔹 Step 1: Go to (x, y, z + offset) → approach pose
            approach_pose = [x, y, z - 0.10] + orientation  # You can adjust the Z offset

            # 🔹 Step 2: Then go to (x, y, z) → actual target
            ordered_goals = [approach_pose, goal]
            
            for sub_goal in ordered_goals:
                goal_pose = Pose.from_list(sub_goal)

                result = self.motion_gen.plan_single(
                    start_state,
                    goal_pose,
                    MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)
                )

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

                    time_from_start = 0.0
                    for i, point in enumerate(interpolated_plan):
                        if self.stop_requested:
                            self.get_logger().warn("🛑 Stop requested! Aborting current goal execution.")
                            self.stop_requested = False  # Reset for future use
                            return  # Exit execution early
                        traj_point = JointTrajectoryPoint()
                        traj_point.positions = point.numpy().tolist()
                        traj_point.velocities = [0.1] * len(self.joint_order)
                        traj_point.time_from_start.sec = int(time_from_start)
                        traj_point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
                        time_from_start += 0.03
                        trajectory_msg.points.append(traj_point)

                        cartesian_point = self.forward_kinematics(traj_point.positions)
                        if cartesian_point:
                            self.path_points.append(cartesian_point)
                            self.publish_path_marker()

                    self.trajectory_pub.publish(trajectory_msg)
                    self.wait_for_execution_completion(sub_goal[:3])

                    start_state = JointState.from_position(
                        torch.tensor([trajectory_msg.points[-1].positions], dtype=torch.float32, device=device),
                        joint_names=self.joint_order,
                    )
                else:
                    self.get_logger().warn("Failed to generate a motion plan for sub-goal!")
                    
                    
            
            
        self.get_logger().info("All goals executed. Waiting for new goals...")
        self.control_gripper("CLOSE")
        self.move_backward(-0.02)
        self.rotate_wrist(90)
        self.rotate_wrist(-90)
        self.move_to_dropoff_position()
        self.control_gripper("OPEN")
        self.move_to_home_position()





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
        """Use CuRobo's rollout_fn to compute end-effector pose."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            joint_tensor = torch.tensor([joint_positions], dtype=torch.float32, device=device)
            joint_state = JointState.from_position(joint_tensor, joint_names=self.joint_order)

            ee_pose = self.motion_gen.rollout_fn.compute_kinematics(joint_state)

            pos = ee_pose.ee_pos_seq.squeeze().tolist()
            # Optionally get orientation:
            # quat = ee_pose.ee_quat_seq.squeeze().tolist()
            return Point(x=pos[0], y=pos[1], z=pos[2])

        except Exception as e:
            self.get_logger().warn(f"CuRobo FK failed: {e}")
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

            
            
            
    def wait_for_execution_completion(self, goal_position, tolerance=0.005):
        self.get_logger().info(f"Waiting for robot to reach goal: {goal_position}")

        while self.running:
            # ———————————————— Check for interrupt ————————————————
            if self.stop_requested:
                self.get_logger().warn("🛑 Stop requested during execution. Sending stop command.")
                # Build a “stop” trajectory at the current joint positions
                stop_msg = JointTrajectory()
                stop_msg.joint_names = self.joint_order
                pt = JointTrajectoryPoint()
                pt.positions = list(self.current_joint_positions)
                pt.time_from_start.sec = 0
                pt.time_from_start.nanosec = 0
                stop_msg.points = [pt]
                self.trajectory_pub.publish(stop_msg)

                # Reset the flag and exit wait
                self.stop_requested = False
                break

            # ———————————————— Normal waiting ————————————————
            current = self.get_end_effector_pose()
            if current is None:
                time.sleep(0.1)
                continue

            dist = ((current[0] - goal_position[0])**2 +
                    (current[1] - goal_position[1])**2 +
                    (current[2] - goal_position[2])**2) ** 0.5
            if dist < tolerance:
                self.get_logger().info(f"Goal reached (dist={dist:.4f}).")
                break

            time.sleep(0.1)
            
            
            
                
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
        
        #print("Checking if the robot is moving...")

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
            #print(f"Joint velocities: {velocities}")

            # If any velocity is above the threshold, the robot is still moving
            if any(abs(v) > velocity_threshold for v in velocities):
                print("Robot is still moving!")
                return True

        except Exception as e:
            print(f"Error checking robot movement: {e}")
            return False  # If we can't check → assume it's stopped

        print("Robot has stopped moving.")
        return False  # Default to False if nothing indicates movement


    def goal_pose_callback(self, msg):
        """Callback to store the latest goal pose received from the topic."""
        goal_position = [msg.position.x, msg.position.y, msg.position.z, msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        self.goal_poses.append(goal_position)
        self.get_logger().info(f"Received goal pose from topic: {goal_position}")
        self.publish_goal_marker(goal_position[:3])
        
        
    def subscribe_to_goal_pose_topic_once_stationary(self):
        if not self.is_robot_moving():
            self.get_logger().info("🟢 Robot is now stationary. Subscribing to external goal pose.")
            self.subscribe_to_goal_pose_topic()

        
        
    def subscribe_to_goal_pose_topic(self):
        """Subscribe to external goal pose and move in pattern until a goal is received."""

        if self.is_robot_moving():
            self.get_logger().warn("⚠️ Robot is moving. Delaying subscription to external goal pose.")
            self.create_timer(2.0, self.subscribe_to_goal_pose_topic_once_stationary)
            return

        if hasattr(self, 'goal_pose_sub'):
            self.get_logger().info("Already subscribed. Unsubscribing first.")
            self.destroy_subscription(self.goal_pose_sub)
            del self.goal_pose_sub

        # Get current orientation
        current_pose = self.get_end_effector_pose()
        if current_pose:
            current_orientation = current_pose[3:]
        else:
            self.get_logger().warn("⚠️ Current pose not found. Using fallback orientation.")
            current_orientation = [0.9949, -0.0981, 0.0154, 0.0151]

        self.goal_received = False
        self.goal_poses = []

        def goal_callback(msg):
            # Stop idle motion if running
            if hasattr(self, 'idle_timer'):
                self.idle_timer.cancel()
                self.get_logger().info("🛑 Idle motion stopped.")

            if self.is_robot_moving():
                self.get_logger().warn("⚠️ Goal received while robot is moving. Ignoring.")
                return

            time.sleep(1)  # Let robot stabilize

            goal_position = [
                msg.position.x,
                msg.position.y,
                msg.position.z,
            ] + current_orientation

            # Check for duplicates
            duplicate = any(
                ((existing_goal[0] - goal_position[0]) ** 2 +
                 (existing_goal[1] - goal_position[1]) ** 2 +
                 (existing_goal[2] - goal_position[2]) ** 2) ** 0.5 < 0.01
                for existing_goal in self.goal_poses
            )

            if duplicate:
                self.get_logger().warn("⚠️ Duplicate goal detected. Skipping.")
            else:
                self.goal_poses.append(goal_position)
                self.publish_goal_marker(goal_position[:3])
                self.get_logger().info(f"✅ Goal received and saved: {goal_position}")
                self.goal_received = True

            # Unsubscribe
            if hasattr(self, 'goal_pose_sub'):
                self.destroy_subscription(self.goal_pose_sub)
                del self.goal_pose_sub
                self.get_logger().info("📴 Unsubscribed from /external_goal_pose.")

        # Subscribe to goal pose
        self.goal_pose_sub = self.create_subscription(ROSPose, '/external_goal_pose', goal_callback, 10)
        self.get_logger().info("🟢 Subscribed to /external_goal_pose")

        # Motion pattern: Up → Left → Right → Down → Left → Right
        self.idle_motion_sequence = [
            ('UP',    0.0,  0.0,  0.05),
            ('LEFT',  0.0,  0.05, 0.0),
            ('RIGHT', 0.0, -0.05, 0.0),
            ('DOWN',  0.0,  0.0, -0.05),
            ('LEFT',  0.0,  0.05, 0.0),
            ('RIGHT', 0.0, -0.05, 0.0),
        ]
        self.idle_motion_index = 0

        def idle_motion_callback():
            if not self.goal_received and self.idle_motion_index < len(self.idle_motion_sequence):
                current_pose = self.get_end_effector_pose()
                if current_pose:
                    direction, dx, dy, dz = self.idle_motion_sequence[self.idle_motion_index]
                    target_pose = [
                        current_pose[0] + dx,
                        current_pose[1] + dy,
                        current_pose[2] + dz,
                        *current_orientation
                    ]

                    if not self.is_robot_moving():
                        self.get_logger().info(
                            f"🔄 Idle motion {self.idle_motion_index + 1}/{len(self.idle_motion_sequence)}: moving {direction}"
                        )
                        self.execute_single_pose(target_pose)
                        self.idle_motion_index += 1

            elif self.idle_motion_index >= len(self.idle_motion_sequence):
                if hasattr(self, 'idle_timer'):
                    self.idle_timer.cancel()
                    self.get_logger().info("✅ Idle motion sequence complete. Timer stopped.")

        self.idle_timer = self.create_timer(5.0, idle_motion_callback)




    def execute_single_pose(self, pose):
        """Plan and execute a single target pose directly."""
        if self.current_joint_positions is None:
            self.get_logger().warn("Joint states not available. Cannot execute idle pose.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        goal_pose = Pose.from_list(pose)

        result = self.motion_gen.plan_single(start_state, goal_pose)
        if result.success:
            self.get_logger().info("✅ Idle motion plan successful. Executing...")

            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_order
            interpolated_plan = result.get_interpolated_plan()

            if isinstance(interpolated_plan, JointState):
                interpolated_plan = interpolated_plan.position
            if not isinstance(interpolated_plan, torch.Tensor):
                interpolated_plan = torch.tensor(interpolated_plan, dtype=torch.float32)

            interpolated_plan = interpolated_plan.to("cpu")
            time_from_start = 0.0

            for point in interpolated_plan:
            
                if self.stop_requested:
                   self.get_logger().warn("🛑 Stop requested! Aborting current trajectory.")
                   self.stop_requested = False
                   return
                traj_point = JointTrajectoryPoint()
                traj_point.positions = point.tolist()
                traj_point.velocities = [0.1] * len(self.joint_order)
                traj_point.time_from_start.sec = int(time_from_start)
                traj_point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
                time_from_start += 0.03
                trajectory_msg.points.append(traj_point)

            self.trajectory_pub.publish(trajectory_msg)
        else:
            self.get_logger().warn("❌ Idle motion planning failed.")

    def move_to_home_position(self, timeout=5.0):
        """Move the robot to a predefined Cartesian end-effector pose (home) after waiting for joint state."""
        self.get_logger().info("Waiting for joint state before moving to home pose...")

        # Wait until joint states are received or timeout
        start_time = time.time()
        while self.current_joint_positions is None and (time.time() - start_time < timeout):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.current_joint_positions is None:
            self.get_logger().warn("Joint state not received after waiting. Skipping home motion.")
            return

        self.get_logger().info("Joint state received. Moving to home pose...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        home_pose = Pose.from_list(self.home_pose)

        result = self.motion_gen.plan_single(start_state, home_pose)

        if result.success:
            self.get_logger().info("Successfully planned to home pose.")
            self.control_gripper("OPEN")

            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_order
            interpolated_plan = result.get_interpolated_plan()
            
            if isinstance(interpolated_plan, JointState):
                    interpolated_plan = interpolated_plan.position
            if not isinstance(interpolated_plan, torch.Tensor):
                    interpolated_plan = torch.tensor(interpolated_plan, dtype=torch.float32)
            interpolated_plan = interpolated_plan.to("cpu")

            time_from_start = 0.0
            for point in interpolated_plan:
                traj_point = JointTrajectoryPoint()
                traj_point.positions = point.tolist()
                traj_point.velocities = [0.1] * len(self.joint_order)
                traj_point.time_from_start.sec = int(time_from_start)
                traj_point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
                time_from_start += 0.03
                trajectory_msg.points.append(traj_point)

            self.trajectory_pub.publish(trajectory_msg)
            #self.wait_for_execution_completion(self.home_pose[:3])

        else:
            self.get_logger().warn("Failed to plan motion to home pose.")
            
        
    def rotate_wrist(self, degrees):
        """ Rotates the UR10's wrist (wrist_3_joint) by the given degrees. """
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state received yet. Cannot rotate wrist.")
            return

        radians = math.radians(degrees)  # Convert degrees to radians
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]  # Full joint list

        # Maintain current positions for other joints, modify only wrist_3_joint
        new_positions = self.current_joint_positions.copy()
        new_positions[5] += radians  # Modify wrist_3_joint (6th joint)

        point = JointTrajectoryPoint()
        point.positions = new_positions  # Set new positions
        point.time_from_start.sec = 2  # Execute within 2 seconds

        trajectory_msg.points.append(point)
        self.wrist_publisher_.publish(trajectory_msg)
        self.get_logger().info(f"UR10 wrist rotating by {degrees} degrees ({radians} radians)")
        time.sleep(2)
        
    def move_backward(self, backward_distance):
        """ Moves the UR10 slightly backward by adjusting shoulder_pan_joint. """
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state received yet. Cannot move.")
            return

        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                      "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]  # Full joint list

        new_positions = self.current_joint_positions.copy()
        new_positions[1] -= backward_distance  # Modify shoulder_pan_joint (1st joint)

        point = JointTrajectoryPoint()
        point.positions = new_positions  # Set new positions
        point.time_from_start.sec = 1  # Execute within 2 seconds

        trajectory_msg.points.append(point)
        self.wrist_publisher_.publish(trajectory_msg)
        self.get_logger().info(f"UR10 moved backward by {backward_distance} meters")
        time.sleep(2)
        
    def move_to_dropoff_position(self, timeout=5.0):
        """Move the robot to the predefined drop-off position."""
        if self.current_joint_positions is None:
            self.get_logger().warn("Joint state not available. Cannot move to drop-off.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        dropoff_pose = Pose.from_list(self.dropoff_pose)

        result = self.motion_gen.plan_single(start_state, dropoff_pose)

        if result.success:
            self.get_logger().info("Successfully planned to drop-off position.")
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_order
            interpolated_plan = result.get_interpolated_plan()

            if isinstance(interpolated_plan, JointState):
                interpolated_plan = interpolated_plan.position
            if not isinstance(interpolated_plan, torch.Tensor):
                interpolated_plan = torch.tensor(interpolated_plan, dtype=torch.float32)

            interpolated_plan = interpolated_plan.to("cpu")
            time_from_start = 0.0

            for point in interpolated_plan:
                traj_point = JointTrajectoryPoint()
                traj_point.positions = point.tolist()
                traj_point.velocities = [0.1] * len(self.joint_order)
                traj_point.time_from_start.sec = int(time_from_start)
                traj_point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
                time_from_start += 0.03
                trajectory_msg.points.append(traj_point)

            self.trajectory_pub.publish(trajectory_msg)
            self.wait_for_execution_completion(self.dropoff_pose[:3])

        else:
            self.get_logger().warn("Failed to plan to drop-off position.")
            
            
    def control_gripper(self, action):
        if action.upper() == "OPEN":
            self.gripper_controller.open_gripper()
        elif action.upper() == "CLOSE":
            self.gripper_controller.run_closure_loop()
            
            
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

