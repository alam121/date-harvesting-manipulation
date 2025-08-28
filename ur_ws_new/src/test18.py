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
from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Pose as ROSPose
from delto_gripper_controller import DeltoGripperController
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float32MultiArray
import numpy as np
from geometry_msgs.msg import Quaternion
from scipy.spatial.transform import Rotation as R


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
        

        self.force_sub = self.create_subscription(
            Float32MultiArray,
            '/gripper/force',
            self.force_callback,
            10
        )

                # --- Emergency stop topic ---
        self.create_subscription(
            Bool,
            "/emergency_stop",
            self._stop_callback,
            10
        )
        
        # New publisher for goal position markers
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)


        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)

        # Store the path points
        self.path_points = []
        self.last_force = None
        self.force_threshold = 0.04    # adjust to your slip‐sensitivity
        self.gripper_closed = False

        self.stop_requested = False  # Flag to request stopping the current motion
        self.slip_detection = False   # Flag to see slip
        self.abort_flag = False

        # ROS2 subscriber for MoveIt interactive marker feedback
        self.marker_sub = self.create_subscription(
            InteractiveMarkerFeedback,
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback",
            self.marker_callback,
            10
        )
        self.speed_scale = 1.8    
        self.alpha = 0.2                # smoothing factor (0 < α < 1)
        self.filtered_forces = None     # will become a list of length N_channels
        self.grap_miss = False 
        self.open_baseline = [-0.10, -0.09, -0.08]
        self.open_baseline = []
        self.goal_capture_active = False
        self.goal_capture_timer = None
        self.goal_capture_count = 0
        self.goal_sort_ref = None        # [x, y, z] reference for distance sorting
        self.goal_sort_ascending = True  # nearest-first; set False for farthest-first

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
        world_config = {
    "cuboid": {
        "table": {
            "dims": [5.0, 5.0, 0.2],          # 5×5×0.2 m table
            "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0]  # center at z=-0.1 so top is at z=0
        },
        "pole": {
            # full dims: 0.02 m thick in x, 0.02 m thick in y, 1.0 m tall in z
            "dims": [0.02, 0.02, 1.0],
            # center at x=0, y=0.65, z=0.5 (half of 1.0m)
            "pose": [0.0, -0.95, 0.5, 1, 0, 0, 0]
            }
         }
        }

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            "ur10e.yml", world_config, interpolation_dt=0.004) #,trajopt_dt=0.05, num_trajopt_seeds=5)
        
        

        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()


        self.get_logger().info("Waiting for joint states...")
        self.timer = self.create_timer(0.5, self.check_joint_states)
        #self.home_joints =  [-1.7429350058185022, -2.118960682545797, 2.262824058532715, -4.16720420518984, 4.74897575378418, 0.008626394905149937]

        #self.home_joints = [-1.730377499257223, -2.06564170518984, 2.350080728530884, -4.355106298123495, 4.711179733276367, 0.027907514944672585]
        #self.home_joints = [-1.7171486059771937, -2.121176067982809, 2.4437646865844727, -4.073089901600973, 4.712892055511475, 0.006838873028755188]
        self.home_joints =  [-1.5914891401873987, -1.7436569372760218, 2.196291923522949, -4.470575277005331, 4.604394912719727, -0.05696469942201787]
        self.dropoff_joints = [-2.16858417192568, -1.3347657362567347, 2.0885677337646484, -2.6394265333758753, 4.78283166885376, 0.013545919209718704]
        self.predropoff_joints = [-1.6832264105426233, -2.020153347645895, 2.238132953643799, -3.9681833426104944, 4.682962894439697, -0.010893646870748341]

        self.gripper_controller = DeltoGripperController(self)

        self.robot_running = False
        self.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            self._robot_running_callback,
            10
        )

        #self.move_to_home_position()
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


            elif key == "p":
                print("Capturing external goals for 30 seconds…")
                self.start_goal_capture(duration=10.0)

            elif key == "n" and self.goal_poses:
                # stop timed capture if running
                if self.goal_capture_active:
                    self.stop_goal_capture()
                if hasattr(self, 'goal_pose_sub'):
                    self.destroy_subscription(self.goal_pose_sub)
                    del self.goal_pose_sub
                    print("Unsubscribed from external goal poses.")
                print("Executing motion...")
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
                self._publish_stop_trajectory()


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
        """Execute goals one-by-one:
        approach -> target -> grip checks -> dropoff -> home, then next goal.
        """
        if not self.robot_running:
            self.get_logger().error("❌ Cannot execute goals: robot program is OFF. Please turn it ON first.")
            return

        if self.current_joint_positions is None:
            self.get_logger().warn("Current joint state not received yet!")
            return

        if not self.goal_poses:
            self.get_logger().warn("No goal positions stored!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # while there are goals, handle EACH goal end-to-end
        while self.goal_poses and self.running:
            # snapshot start state at the moment we begin this goal
            start_state = JointState.from_position(
                torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )

            goal = self.goal_poses.pop(0)
            self.get_logger().info(f"Processing goal: {goal}")
            x, y, z = goal[:3]
            # orientation depends on whether date is high or low
            if z > 1.15:
                # high → approach from SIDE, gripper faces inward
                orientation = goal[3:]#self.quaternion_from_approach(pitch_deg=-20.0)
                self.get_logger().info("Using SIDE orientation (high date).")
            else:
                # low → approach from ABOVE, gripper faces downward
                #orientation = self.quaternion_from_approach(pitch_deg=30.0)
                orientation = goal[3:]
                self.get_logger().info("Using TOP-DOWN orientation (low date).")


            # Decide approach offset based on height:
            # - If the point is high (z > 1.18), approach from the side (x - 0.10)
            # - Otherwise, approach from above (z - 0.10)
            if z > 1.15:
                approach = [x, y+0.10, z] + list(orientation)
                y = y-0.004
                z= z +0.02
                self.get_logger().info(f"High goal (z={z:.3f}) → side approach (x-0.10)")
            else:
                approach = [x, y +0.1, z - 0.10] + list(orientation)
                self.get_logger().info(f"Normal goal (z={z:.3f}) → top approach (z-0.10)")
                z= z + 0.02
                #y = y+0.8
                y = y-0.003

            sub_goals = [
                approach,                # approach first
                [x, y, z] + list(orientation)  # then the actual target
]


            for sub_goal in sub_goals:
                goal_pose = Pose.from_list(sub_goal)
                result = self.motion_gen.plan_single(
                    start_state,
                    goal_pose,
                    MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)
                )

                if not result.success:
                    self.get_logger().warn("Failed to generate a motion plan for sub-goal! Skipping this goal.")
                    # break out of sub-goals; continue with next saved goal
                    break

                # build & publish trajectory
                traj_msg = JointTrajectory()
                traj_msg.joint_names = self.joint_order

                interpolated = result.get_interpolated_plan()
                if isinstance(interpolated, JointState):
                    interpolated = interpolated.position
                if not isinstance(interpolated, torch.Tensor):
                    interpolated = torch.tensor(interpolated, dtype=torch.float32)
                interpolated = interpolated.to("cpu")

                tfs = 0.0
                for point in interpolated:
                    if self.stop_requested:
                        self.get_logger().warn("🛑 Stop requested! Aborting current goal execution.")
                        self.stop_requested = False
                        return  # hard exit from execution immediately

                    pt = JointTrajectoryPoint()
                    pt.positions = point.numpy().tolist()
                    pt.velocities = [0.1] * len(self.joint_order)
                    pt.time_from_start.sec = int(tfs)
                    pt.time_from_start.nanosec = int((tfs % 1) * 1e9)
                    tfs += 0.03
                    traj_msg.points.append(pt)

                    # live FK trail (optional)
                    cp = self.forward_kinematics(pt.positions)
                    if cp:
                        self.path_points.append(cp)
                        self.publish_path_marker()

                self.trajectory_pub.publish(traj_msg)
                self.wait_for_execution_completion(sub_goal[:3])

                # update start_state for next leg
                start_state = JointState.from_position(
                    torch.tensor([traj_msg.points[-1].positions], dtype=torch.float32, device=device),
                    joint_names=self.joint_order,
                )

            else:
                # Only runs if we didn't break: we reached the target successfully
                # 2) Grip or checks around the goal
                self.control_gripper("CLOSE")
                time.sleep(1)

                if self.abort_flag:
                    #self.get_logger().info("🚨 Abort triggered → opening gripper and returning home.")
                    self.control_gripper("OPEN")
                    self.move_to_home_position()
                    self.abort_flag = False
                    continue  # move on to the next saved goal

                if self.slip_detection or self.grap_miss:
                    self.get_logger().info("🛠 Slip/miss → nudge up, reopen/close to retry.")
                    self.control_gripper("OPEN")
                    cur = self.get_end_effector_pose()
                    if cur:
                        up_goal = [cur[0], cur[1], cur[2] + 0.015] + cur[3:]
                        self.execute_single_pose(up_goal)
                    self.slip_detection = False
                    self.grap_miss = False
                    self.control_gripper("CLOSE")

                # 3) Optional micro-motions
                self.move_backward(-0.02)
                self.rotate_wrist(90)
                self.rotate_wrist(-90)


                #retract
                self.move_to_predropoff_position()

                # 4) Drop-off and open gripper
                self.move_to_dropoff_position()
                self.control_gripper("OPEN")

                # 5) Return home before next goal
                self.move_to_home_position()

        self.get_logger().info("✅ Finished all goals.")



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

    def nudge_wrist2(self, delta_rad: float, duration: float = 1.0):
        """
        Move only the wrist_2_joint by delta_rad (radians) over `duration` seconds,
        then return to its original position.
        """
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state – cannot nudge wrist_2.")
            return

        # 1) Save original full joint state
        orig = list(self.current_joint_positions)

        # 2) Build target joint state, bump wrist_2 (index 4)
        target = orig.copy()
        target[4] += delta_rad  # wrist_2_joint is the 5th in joint_order

        # 3) Publish the 6-joint trajectory to move out
        traj_out = JointTrajectory()
        traj_out.joint_names = self.joint_order
        pt_out = JointTrajectoryPoint()
        pt_out.positions = target
        pt_out.time_from_start.sec     = int(duration)
        pt_out.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        traj_out.points = [pt_out]
        self.trajectory_pub.publish(traj_out)
        self.get_logger().info(f"Nudging wrist_2 by {math.degrees(delta_rad):.1f}°")

        # 4) Wait for motion to complete
        time.sleep(duration + 0.05)

        # 5) Publish the 6-joint trajectory to return to original
        traj_back = JointTrajectory()
        traj_back.joint_names = self.joint_order
        pt_back = JointTrajectoryPoint()
        pt_back.positions = orig
        pt_back.time_from_start.sec     = int(duration)
        pt_back.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        traj_back.points = [pt_back]
        self.trajectory_pub.publish(traj_back)
        self.get_logger().info("Returning wrist_2 to original orientation")

        traj2.points = [pt2]
        self.wrist_publisher_.publish(traj2)
        self.get_logger().info(f"wrist_2_joint → back to original over {duration:.1f}s")

        
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
            
               
        
    def publish_goal_marker(self, position, rank=None):
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

                # 🔹 Add text marker to show rank number
        if rank is not None:
            text_marker = Marker()
            text_marker.header.frame_id = "base_link"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "goal_labels"
            text_marker.id = 1000 + marker.id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = position[0]
            text_marker.pose.position.y = position[1]
            text_marker.pose.position.z = position[2] + 0.06  # a bit above sphere
            text_marker.scale.z = 0.05  # text size
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.text = str(rank)
            self.goal_marker_pub.publish(text_marker)
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

            time.sleep(0.5)  # Let robot stabilize

            goal_position = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
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
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/external_goal_pose', goal_callback, 10)
        self.get_logger().info("🟢 Subscribed to /external_goal_pose")

        # Motion pattern: Up → Left → Right → Down → Left → Right
        self.idle_motion_sequence = [
            ('UP',    0.0,  0.0,  0.1),
            ('LEFT',  0.0,  0.1, 0.0),
            ('RIGHT', 0.0, -0.1, 0.0),
            ('DOWN',  0.0,  0.0, -0.1),
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


    def move_to_home_position(self, cfg: MotionGenPlanConfig = None):
        """Plan & execute a joint-space path to self.home_joints, then wait until the end-effector arrives."""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state; skipping home move.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        goal_state = JointState.from_position(
            torch.tensor([self.home_joints], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        plan_cfg = cfg or MotionGenPlanConfig(
            max_attempts=20,
            enable_finetune_trajopt=True
        )
        result = self.motion_gen.plan_single_js(start_state, goal_state, plan_cfg)

        if not result.success:
            self.get_logger().warn("❌ Joint-space plan to HOME failed.")
            return

        # Build the ROS JointTrajectory exactly like execute_single_pose
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_order

        interpolated_plan = result.get_interpolated_plan()
        traj_tensor = (
            interpolated_plan.position
            if isinstance(interpolated_plan, JointState)
            else interpolated_plan
        ).to("cpu")

        time_from_start = 0.0
        for point in traj_tensor:
            if self.stop_requested:
                self.get_logger().warn("🛑 Stop requested! Aborting home motion.")
                self.stop_requested = False
                return

            traj_pt = JointTrajectoryPoint()
            traj_pt.positions = point.tolist()
            traj_pt.velocities = [0.1] * len(self.joint_order)
            traj_pt.time_from_start.sec     = int(time_from_start)
            traj_pt.time_from_start.nanosec = int((time_from_start % 1.0) * 1e9)
            trajectory_msg.points.append(traj_pt)

            # apply global speed scaling if you have one
            time_from_start += 0.03 / getattr(self, "speed_scale", 1.0)

        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info("🏠 Moving to HOME joints…")

        # ——— Wait until we really reach the home Cartesian pose ———
        final_joints = traj_tensor[-1].tolist()
        fk = self.forward_kinematics(final_joints)
        if fk:
            # wait_for_execution_completion expects [x, y, z]
            self.wait_for_execution_completion([fk.x, fk.y, fk.z])


    
    
        
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
        
    def move_to_dropoff_position(self, cfg: MotionGenPlanConfig = None):
        """Plan & execute a joint-space path to self.dropoff_joints."""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state; skipping drop-off move.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        goal_state = JointState.from_position(
            torch.tensor([self.dropoff_joints], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        plan_cfg = cfg or MotionGenPlanConfig(
            max_attempts=20,
            enable_finetune_trajopt=True
        )
        result = self.motion_gen.plan_single_js(start_state, goal_state, plan_cfg)

        if not result.success:
            self.get_logger().warn("❌ Joint-space plan to DROP-OFF failed.")
            return

        # build the ROS JointTrajectory exactly like execute_single_pose
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_order

        interpolated_plan = result.get_interpolated_plan()
        if isinstance(interpolated_plan, JointState):
            traj_tensor = interpolated_plan.position
        else:
            traj_tensor = interpolated_plan

        traj_tensor = traj_tensor.to("cpu")
        time_from_start = 0.0

        for point in traj_tensor:
            if self.stop_requested:
                self.get_logger().warn("🛑 Stop requested! Aborting drop-off motion.")
                self.stop_requested = False
                return

            pt = JointTrajectoryPoint()
            pt.positions = point.tolist()
            pt.velocities = [0.1] * len(self.joint_order)        # same 0.1 rad/s feed-forward
            pt.time_from_start.sec     = int(time_from_start)
            pt.time_from_start.nanosec = int((time_from_start % 1.0) * 1e9)

            trajectory_msg.points.append(pt)
            time_from_start += 0.03/self.speed_scale                                    # same 30 ms increment

        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info("📦 Moving to DROP-OFF joints…")
        
        final_joints = traj_tensor[-1].tolist()
        fk = self.forward_kinematics(final_joints)
        if fk:
            # wait_for_execution_completion expects a list [x, y, z]
            self.wait_for_execution_completion([fk.x, fk.y, fk.z])
            


    def move_to_predropoff_position(self, cfg: MotionGenPlanConfig = None):
        """Plan & execute a joint-space path to self.dropoff_joints."""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state; skipping drop-off move.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        goal_state = JointState.from_position(
            torch.tensor([self.predropoff_joints], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )

        plan_cfg = cfg or MotionGenPlanConfig(
            max_attempts=20,
            enable_finetune_trajopt=True
        )
        result = self.motion_gen.plan_single_js(start_state, goal_state, plan_cfg)

        if not result.success:
            self.get_logger().warn("❌ Joint-space plan to DROP-OFF failed.")
            return

        # build the ROS JointTrajectory exactly like execute_single_pose
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_order

        interpolated_plan = result.get_interpolated_plan()
        if isinstance(interpolated_plan, JointState):
            traj_tensor = interpolated_plan.position
        else:
            traj_tensor = interpolated_plan

        traj_tensor = traj_tensor.to("cpu")
        time_from_start = 0.0

        for point in traj_tensor:
            if self.stop_requested:
                self.get_logger().warn("🛑 Stop requested! Aborting drop-off motion.")
                self.stop_requested = False
                return

            pt = JointTrajectoryPoint()
            pt.positions = point.tolist()
            pt.velocities = [0.1] * len(self.joint_order)        # same 0.1 rad/s feed-forward
            pt.time_from_start.sec     = int(time_from_start)
            pt.time_from_start.nanosec = int((time_from_start % 1.0) * 1e9)

            trajectory_msg.points.append(pt)
            time_from_start += 0.03/self.speed_scale                                    # same 30 ms increment

        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info("📦 Moving to preDROP-OFF joints…")
        
        final_joints = traj_tensor[-1].tolist()
        fk = self.forward_kinematics(final_joints)
        if fk:
            # wait_for_execution_completion expects a list [x, y, z]
            self.wait_for_execution_completion([fk.x, fk.y, fk.z])


      
    def _publish_stop_trajectory(self):
        """Publish a zero-time trajectory at the current joint positions to halt immediately."""
        if self.current_joint_positions is None:
            return
        stop_msg = JointTrajectory()
        stop_msg.joint_names = self.joint_order
        pt = JointTrajectoryPoint()
        pt.positions = list(self.current_joint_positions)
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        stop_msg.points = [pt]
        self.trajectory_pub.publish(stop_msg)        
            
    def control_gripper(self, action):
        if action.upper() == "OPEN":
            self.gripper_controller.open_gripper()
            self.gripper_closed = False
            self.slip_detection = False
            self.grap_miss = False
            self.open_baseline = self.filtered_forces.copy()
            self.get_logger().info("Gripper opened → slip detection paused")
        elif action.upper() == "CLOSE":
            self.gripper_controller.run_closure_loop()           
            self.gripper_closed = True
            self.get_logger().info("Gripper closed → slip detection active")

    def force_callback(self, msg):
        raw = list(msg.data)

        # initialize filtered_forces on first pass
        if self.filtered_forces is None:
            self.filtered_forces = raw.copy()
        else:
            # EWMA: filtered = α·raw + (1–α)·prev_filtered
            self.filtered_forces = [
                self.alpha * r + (1 - self.alpha) * f
                for r, f in zip(raw, self.filtered_forces)
            ]

        # only check when gripper truly closed
        if not self.gripper_closed:
            return

        f = self.filtered_forces  # short name
        self.get_logger().info(f"Updated open Filtetrerd force: {self.filtered_forces}")
        self.get_logger().info(f"Updated open baseline: {self.open_baseline}")
        d = [fi - bi for fi, bi in zip(self.open_baseline, self.filtered_forces)]
        print("delta", d)

        # --- thresholds ---
        POS_TH = 0.02     # "pushing out" vs baseline
        NEG_TH = -0.08    # "squeezing in" vs baseline
        HARD_ABORT = -0.35

        # 1) Safety abort: very strong squeeze on any channel
        if any(di <= HARD_ABORT for di in d):
            if not self.abort_flag:
                self.get_logger().error(f"🛑 ABORT (hard): Δ={d}")
            self.abort_flag = False
            self.slip_detection = False
            self.grap_miss = False
            return

        # 2) Classify grip using deltas
        positives = sum(di > POS_TH for di in d)
        negatives = sum(di < NEG_TH for di in d)

        if negatives == len(d):
            # All channels squeezed inward: likely closed on air → MISS
            if not self.grap_miss:
                self.get_logger().warn(f"❗ Likely miss: Δ={d}")
            self.grap_miss = False
            self.slip_detection = False

        elif positives >= 1 and negatives >= 1:
            # Opposing forces: object pinched between fingers → GOOD
            if self.grap_miss:
                self.get_logger().info(f"✅ Good grip: Δ={d}")
            self.grap_miss = False

        else:
            # Not clearly miss nor strong opposing contact → BAD/weak
            if not self.grap_miss:
                self.get_logger().warn(f"⚠️ Weak/unstable grip: Δ={d}")
            self.grap_miss = False
            self.slip_detection = False

        # Optional: a friendly status when everything is calm & good
        if not self.grap_miss and not self.slip_detection and self.abort_flag is not False:
            self.get_logger().info("👍 Gripper stable.")

        # finally, stash raw if you want
        self.last_force = raw

    def start_goal_capture(self, duration: float = 30.0):
        """Subscribe to /external_goal_pose and collect unique goals for `duration` seconds, sorted by distance."""
        # restart if already active
        if self.goal_capture_active:
            self.stop_goal_capture()

        self.goal_poses.clear()
        self.goal_capture_count = 0

        # set distance reference at capture start
        ee = self.get_end_effector_pose()
        if ee:
            self.goal_sort_ref = ee[:3]
            self.get_logger().info(f"Sorting goals by distance to EE at capture start: {self.goal_sort_ref}")
        else:
            self.goal_sort_ref = [0.0, 0.0, 0.0]
            self.get_logger().warn("EE pose unavailable; sorting by distance to [0,0,0].")

        # subscribe
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/external_goal_pose', self._capture_goal_callback, 10
        )
        self.goal_capture_active = True
        self.goal_capture_count = 0
        self.get_logger().info(f"Started goal capture for {duration:.0f}s from /external_goal_pose")

        def _stop_once():
            if self.goal_capture_timer is not None:
                self.goal_capture_timer.cancel()
                self.goal_capture_timer = None
            self.stop_goal_capture()

        # one-shot timer (seconds)
        self.goal_capture_timer = self.create_timer(duration, _stop_once)

    def stop_goal_capture(self):
        """Stop timed goal capture if active."""
        if not self.goal_capture_active:
            return
        self.goal_capture_active = False

        if hasattr(self, 'goal_pose_sub'):
            self.destroy_subscription(self.goal_pose_sub)
            del self.goal_pose_sub

        if self.goal_capture_timer is not None:
            self.goal_capture_timer.cancel()
            self.goal_capture_timer = None

        # final sort to be sure
        self._sort_goals_by_distance()
        self.get_logger().info(f"Goal capture stopped. Collected {self.goal_capture_count} goals.")

    def _capture_goal_callback(self, msg: PoseStamped):
        """Append each unique incoming goal (1 cm de-dup) using the CURRENT EE orientation,
        then keep the list distance-sorted."""
        if self.is_robot_moving():
            return

        # incoming position
        gx, gy, gz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

        # grab current EE orientation now (fresh per goal)
        ee = self.get_end_effector_pose()
        if ee:
            # get_end_effector_pose() returns [x, y, z, qw, qx, qy, qz]
            qw, qx, qy, qz = ee[3], ee[4], ee[5], ee[6]
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            self.get_logger().warn("EE orientation unavailable; using default [1,0,0,0].")

        # 1 cm positional de-dup
        is_dup = any(
            ((g[0] - gx)**2 + (g[1] - gy)**2 + (g[2] - gz)**2) ** 0.5 < 0.01
            for g in (self.goal_poses or [])
        )
        if is_dup:
            return

        # save position + current EE orientation
        goal_position = [gx, gy, gz, qw, qx, qy, qz]
        self.goal_poses.append(goal_position)
        self._sort_goals_by_distance()  # keep list ordered after every insert

        self.goal_capture_count += 1
        self.publish_goal_marker(goal_position[:3])  # pass rank here if you want labels
        print(f"🎯 New goal #{self.goal_capture_count}: {goal_position}")
        self.get_logger().info(
            f"Captured goal #{self.goal_capture_count} (total={len(self.goal_poses)})"
        )

    def _goal_distance(self, goal_xyz, ref_xyz=None):
        """Euclidean distance from goal_xyz to ref_xyz (or self.goal_sort_ref)."""
        rx, ry, rz = (ref_xyz or self.goal_sort_ref or [0.0, 0.0, 0.0])
        gx, gy, gz = goal_xyz
        dx, dy, dz = gx - rx, gy - ry, gz - rz
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _sort_goals_by_distance(self, ref_xyz=None):
        """Sort self.goal_poses in-place by distance to ref_xyz (or self.goal_sort_ref)."""
        ref = ref_xyz or self.goal_sort_ref or [0.0, 0.0, 0.0]
        self.goal_poses.sort(
            key=lambda g: self._goal_distance(g[:3], ref),
            reverse=not self.goal_sort_ascending
        )

    def _robot_running_callback(self, msg: Bool):
        self.robot_running = msg.data
        if not self.robot_running:
            self.get_logger().warn(
                "⚠️ Robot program is NOT running! Please turn it on in the teach pendant."
            )
        else:
            self.get_logger().info("✅ Robot program is running.")


    def _stop_callback(self, msg: Bool):
        """Asynchronous stop: publish a 'hold' at current joints as soon as we receive True."""
        if not msg.data:
            return
        self.get_logger().warn("🛑 Emergency stop from /emergency_stop!")
        self.stop_requested = True
        # publish a hold point right now so the controller brakes immediately
        self._publish_stop_trajectory()


    def quaternion_from_approach(self, direction_xyz=None, pitch_deg=None, world=False):
        """
        If pitch_deg is provided: apply a pitch-only rotation (about Y axis)
        to the CURRENT end-effector orientation.

        - pitch_deg: degrees to tilt (+ forward / - backward)
        - world=False -> rotate about the TOOL's local Y (minimal arm motion)
        world=True  -> rotate about the WORLD Y

        Returns: [qw, qx, qy, qz]
        """
        import math
        import numpy as np

        # ---- Pitch-only path (preferred for minimal motion) ----
        if pitch_deg is not None:
            cur = self.get_end_effector_pose()
            if cur:
                qw, qx, qy, qz = cur[3], cur[4], cur[5], cur[6]
            else:
                # fallback to identity
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            half = math.radians(pitch_deg) / 2.0
            # delta quaternion for rotation about Y: (w, x, y, z)
            cw, cx, cy, cz = math.cos(half), math.sin(half),0.0, 0.0

            if world:
                # world-pitch: q_new = q_delta ⊗ q_cur
                nw = cw*qw - cx*qx - cy*qy - cz*qz
                nx = cw*qx + cx*qw + cy*qz - cz*qy
                ny = cw*qy - cx*qz + cy*qw + cz*qx
                nz = cw*qz + cx*qy - cy*qx + cz*qw
            else:
                # tool-pitch (local Y): q_new = q_cur ⊗ q_delta
                nw = qw*cw - qx*cx - qy*cy - qz*cz
                nx = qw*cx + qx*cw + qy*cz - qz*cy
                ny = qw*cy - qx*cz + qy*cw + qz*cx
                nz = qw*cz + qx*cy - qy*cx + qz*cw

            return [nw, nx, ny, nz]

        # ---- (Optional) fallback: align tool +Z to a direction vector ----
        if direction_xyz is None:
            direction_xyz = (0, 0, -1)

        d = np.array(direction_xyz, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            # invalid direction -> return identity
            return [1.0, 0.0, 0.0, 0.0]
        d /= n

        # Build orthonormal frame with z-axis = d and up = world Z
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        x_axis = np.cross(up, d)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(d, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = d

        R = np.array([[x_axis[0], y_axis[0], z_axis[0]],
                    [x_axis[1], y_axis[1], z_axis[1]],
                    [x_axis[2], y_axis[2], z_axis[2]]], dtype=float)

        # rotation matrix -> quaternion (qw,qx,qy,qz)
        t = R[0,0] + R[1,1] + R[2,2]
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2,1] - R[1,2]) / s
            qy = (R[0,2] - R[2,0]) / s
            qz = (R[1,0] - R[0,1]) / s
        else:
            # robust branches
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
                qw = (R[2,1] - R[1,2]) / s
                qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s
                qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
                qw = (R[0,2] - R[2,0]) / s
                qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s
                qz = (R[1,2] + R[2,1]) / s
            else:
                s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
                qw = (R[1,0] - R[0,1]) / s
                qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s
                qz = 0.25 * s

        return [qw, qx, qy, qz]




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

