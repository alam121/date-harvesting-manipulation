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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import json
from std_msgs.msg import String
from grasp_outcome_classifier import GraspOutcomeClassifier

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
        self.weak_grab = False
        # Store the path points
        self.path_points = []
        self.last_force = None
        self.force_threshold = 0.04    # adjust to your slip‐sensitivity
        self.gripper_closed = False

        self.stop_requested = False  # Flag to request stopping the current motion
        self.slip_detection = False   # Flag to see slip
        self.abort_flag = False

        self.grap_miss = False
        self.last_grasp_end_template = None   # e.g., "OPEN" | "CLOSED_NOTHING" | "WEAK" | "PROPER"


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
            "ur10e.yml", world_config, interpolation_dt=0.004,trajopt_dt=0.05, num_trajopt_seeds=5)
        
        

        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        self.qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    durability=DurabilityPolicy.VOLATILE,
                )

        self.get_logger().info("Waiting for joint states...")
        self.timer = self.create_timer(0.5, self.check_joint_states)
        #self.home_joints =  [-1.7429350058185022, -2.118960682545797, 2.262824058532715, -4.16720420518984, 4.74897575378418, 0.008626394905149937]

      
        #self.home_joints =  [-1.5916569868670862, -1.649402920399801, 2.113215446472168, -4.4819199482547205, 4.604945659637451, -0.05743295351137334]


        self.home_joints = [-1.5892723242389124, -1.696021858845846, 2.4897632598876953, -4.738286916409628, 4.602346420288086, -0.05749255815614873]


        self.dropoff_joints = [-2.16858417192568, -1.3347657362567347, 2.0885677337646484, -2.6394265333758753, 4.78283166885376, 0.013545919209718704]
        self.predropoff_joints = [-1.6832264105426233, -2.020153347645895, 2.238132953643799, -3.9681833426104944, 4.682962894439697, -0.010893646870748341]

        self.gripper_controller = DeltoGripperController(self)
        self.robot_running = False

        self.classifier = GraspOutcomeClassifier(
        on_outcome=self._on_grasp_outcome,    # callback below
        dead_time_thresh_s=1.40,
        hold_time_s=0.5
    )
    # call tick from a fast timer (20–50 Hz)
        self.create_timer(0.02, self.classifier.tick)  # 50 Hz
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

            elif key == "b":
                    print("going creazy")
                    self.descend_z_wrist_osc_simple(
                        dz=-0.06,       # go down 6 cm
                        duration=1.0,   # total time
                        yaw_amp_deg=40.0,
                        osc_hz=4.0,
                        max_points=180  # ~6 ms spacing over 1s
                    )

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
                self.rotate_wrist(90)
                time.sleep(0.7)
                self.descend_z_wrist_osc_simple(
                        dz=-0.13,       # go down 6 cm
                        duration=5.0,   # total time
                        yaw_amp_deg=90.0,
                        osc_hz=3.0,
                        max_points=180  # ~6 ms spacing over 1s
                    )
                
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
        
        #if self.gripper_closed:
            #self.control_gripper("OPEN")

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
            #return

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
            if z > 1.23:
                # high → approach from SIDE, gripper faces inward
                #orientation = goal[3:]
                orientation = self.quaternion_from_approach(pitch_deg=-35.0)
                self.get_logger().info("Using SIDE orientation (high date).")
            else:
                # low → approach from ABOVE, gripper faces downward
                #orientation = self.quaternion_from_approach(pitch_deg=30.0)
                orientation = goal[3:]
                self.get_logger().info("Using TOP-DOWN orientation (low date).")


            # Decide approach offset based on height:
            # - If the point is high (z > 1.18), approach from the side (x - 0.10)
            # - Otherwise, approach from above (z - 0.10)
            if z > 1.23:
                approach = [x, y+0.12, z] + list(orientation)
                y = y-0.004
                z= z-0.4
                self.get_logger().info(f"High goal (z={z:.3f}) → side approach (x-0.10)")
            else:
                approach = [x, y +0.1, z - 0.12] + list(orientation)
                self.get_logger().info(f"Normal goal (z={z:.3f}) → top approach (z-0.10)")
                z= z + 0.055
                #y = y+0.8
                y = y-0.003

            # ---------- 1) PLAN & EXECUTE APPROACH (pre-grasp) ----------
            goal_pose = Pose.from_list(approach)
            result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)
            )

            if not result.success:
                self.get_logger().warn("Failed to generate a motion plan for APPROACH. Skipping this goal.")
                continue

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
                    self.get_logger().warn("🛑 Stop requested during approach. Aborting.")
                    self.stop_requested = False
                    return
                pt = JointTrajectoryPoint()
                pt.positions = point.numpy().tolist()
                pt.velocities = [0.1] * len(self.joint_order)
                pt.time_from_start.sec = int(tfs)
                pt.time_from_start.nanosec = int((tfs % 1) * 1e9)
                #0.01
                
                tfs += 0.01
                traj_msg.points.append(pt)

            self.trajectory_pub.publish(traj_msg)
            self.wait_for_execution_completion(approach[:3])

            # Update start_state after approach
            start_state = JointState.from_position(
                torch.tensor([traj_msg.points[-1].positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )

            # ---------- 2) RE-DETECT / REACQUIRE AT PRE-GRASP ----------
            seed = [x, y, z]  # original target before refinement
            reacq = self.reacquire_goal_pose(seed_xyz=seed, timeout=3.5,
                                            stable_eps=0.004, stable_need=3)
            if reacq:
                x, y, z = reacq  # refine using the fresh detection
                self.publish_goal_marker([x, y, z])
            else:
                self.get_logger().warn("❌ No reacquire — skipping this goal, returning HOME, continuing to next.")
                continue
                
            # ---------- 3) PLAN & EXECUTE FINAL INSERT / GRASP ----------
            final_target = [x, y+0.01, z +0.040 ] + list(orientation)
            goal_pose = Pose.from_list(final_target)
            result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)
            )

            if not result.success:
                self.get_logger().warn("Failed to generate a motion plan for FINAL target. Skipping this goal.")
                continue

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
                    self.get_logger().warn("🛑 Stop requested during final move. Aborting.")
                    self.stop_requested = False
                    return
                pt = JointTrajectoryPoint()
                pt.positions = point.numpy().tolist()
                pt.velocities = [0.1] * len(self.joint_order)
                pt.time_from_start.sec = int(tfs)
                pt.time_from_start.nanosec = int((tfs % 1) * 1e9)
                tfs += 0.03
                traj_msg.points.append(pt)

            self.trajectory_pub.publish(traj_msg)
            self.wait_for_execution_completion(final_target[:3])

                # update start_state for next leg
            start_state = JointState.from_position(
                    torch.tensor([traj_msg.points[-1].positions], dtype=torch.float32, device=device),
                    joint_names=self.joint_order,
                )
            
            goal_quat = list(orientation)  # [qw, qx, qy, qz]
            self.converge_to_goal(
                goal_xyz=[x, y+0.01, z + 0.040],
                goal_quat=goal_quat,
                pos_tol=0.0015,         # ~1.5 mm
                step=0.003,             # 3 mm hops
                max_iters=25,
                max_secs=1.25,
                orient_quat=goal_quat,  # also converge orientation
                orient_tol_deg=2.0
            )


            # Only runs if we didn't break: we reached the target successfully
            # 2) Grip or checks around the goal
            self.control_gripper("CLOSE")
            time.sleep(0.7)
            self.descend_z_wrist_osc_simple(
                        dz=-0.13,       # go down 6 cm
                        duration=5.0,   # total time
                        yaw_amp_deg=90.0,
                        osc_hz=3.0,
                        max_points=180  # ~6 ms spacing over 1s
                    )
            # print(self.slip_detection ,self.grap_miss,self.weak_grab)

            # if self.slip_detection or self.grap_miss or self.weak_grab:
            #     self.get_logger().info("🛠 Slip/miss → nudge up, reopen/close to retry.")
            #     self.control_gripper("OPEN")
            #     cur = self.get_end_effector_pose()
            #     if cur:
            #         up_goal = [cur[0], cur[1], cur[2] + 0.015] + cur[3:]
            #         self.execute_single_pose(up_goal)
            #     self.slip_detection = False
            #     self.grap_miss = False
            #     self.control_gripper("CLOSE")

            # 3) Optional micro-motions
            #self.move_backward(-0.02)
            self.rotate_wrist(90)
            time.sleep(0.7)
            #self.rotate_wrist(-90)
            return


            #retract
            self.move_to_predropoff_position()
            #self.move_to_home_position()
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
            
               
        
    def publish_goal_marker(self, position, orientation=None, rank=None):
        """Publishes a marker in RViz for the goal position and orientation."""
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

        # --- Orientation marker (arrow) ---
        if orientation is not None:
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "base_link"
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = "goal_orientations"
            arrow_marker.id = 10000 + marker.id
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.pose.position.x = position[0]
            arrow_marker.pose.position.y = position[1]
            arrow_marker.pose.position.z = position[2]
            # Set orientation (expects [w, x, y, z] or [x, y, z, w])
            if len(orientation) == 4:
                # Assume [w, x, y, z] or [x, y, z, w] (try both)
                # Try [x, y, z, w] (ROS default)
                arrow_marker.pose.orientation.x = orientation[0]
                arrow_marker.pose.orientation.y = orientation[1]
                arrow_marker.pose.orientation.z = orientation[2]
                arrow_marker.pose.orientation.w = orientation[3]
            elif len(orientation) == 7:
                # If full pose, use last 4 as quaternion
                arrow_marker.pose.orientation.x = orientation[3]
                arrow_marker.pose.orientation.y = orientation[4]
                arrow_marker.pose.orientation.z = orientation[5]
                arrow_marker.pose.orientation.w = orientation[6]
            else:
                # Fallback: identity
                arrow_marker.pose.orientation.w = 1.0
            # Arrow size (shaft length, shaft diameter, head length, head diameter)
            arrow_marker.scale.x = 0.15  # shaft length
            arrow_marker.scale.y = 0.03  # shaft diameter
            arrow_marker.scale.z = 0.06  # head diameter
            # Color (blue for orientation)
            arrow_marker.color.a = 1.0
            arrow_marker.color.r = 0.0
            arrow_marker.color.g = 0.2
            arrow_marker.color.b = 1.0
            self.goal_marker_pub.publish(arrow_marker)

        # Publish orientation as an ARROW marker at the same position if orientation is available
        if hasattr(self, 'goal_poses') and self.goal_poses:
            # Try to get orientation from the latest goal (if present)
            latest_goal = self.goal_poses[-1]
            if len(latest_goal) >= 7:
                q = latest_goal[3:7]  # [w, x, y, z]
                arrow_marker = Marker()
                arrow_marker.header.frame_id = "base_link"
                arrow_marker.header.stamp = self.get_clock().now().to_msg()
                arrow_marker.ns = "goal_orientations"
                arrow_marker.id = 10000 + marker.id
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.pose.position.x = position[0]
                arrow_marker.pose.position.y = position[1]
                arrow_marker.pose.position.z = position[2]
                arrow_marker.pose.orientation.w = q[0]
                arrow_marker.pose.orientation.x = q[1]
                arrow_marker.pose.orientation.y = q[2]
                arrow_marker.pose.orientation.z = q[3]
                arrow_marker.scale.x = 0.15  # Arrow length
                arrow_marker.scale.y = 0.03  # Arrow shaft diameter
                arrow_marker.scale.z = 0.05  # Arrow head diameter
                arrow_marker.color.a = 1.0
                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 0.5
                arrow_marker.color.b = 1.0
                self.goal_marker_pub.publish(arrow_marker)
                self.get_logger().info(f"Published orientation arrow marker at {position}.")

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
            self.create_timer(0.5, self.subscribe_to_goal_pose_topic_once_stationary)
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
            self.goal_received = True            # prevents more patrol steps

            if hasattr(self, 'idle_timer'):
                self.idle_timer.cancel()
                self.get_logger().info("🛑 Idle motion stopped.")
                self._publish_stop_trajectory()

            if self.is_robot_moving():
                self.get_logger().warn("⚠️ Goal received while robot is moving. Ignoring.")
                return

            time.sleep(0.1)  # Let robot stabilize

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
                x, y, z = goal_position[:3]
                if z > 1.23:
                    self.get_logger().info(f"High goal (z={z:.3f}) → side approach (x-0.10)")
                else:
                    self.get_logger().info(f"Normal goal (z={z:.3f}) → top approach (z-0.10)")



            # Unsubscribe
            if hasattr(self, 'goal_pose_sub'):
                self.destroy_subscription(self.goal_pose_sub)
                del self.goal_pose_sub
                self.get_logger().info("📴 Unsubscribed from /external_goal_pose.")

        # Subscribe to goal pose
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/external_goal_pose', goal_callback, self.qos)
        self.get_logger().info("🟢 Subscribed to /external_goal_pose")

        # Motion pattern: Up → Left → Right → Down → Left → Right
        self.idle_motion_sequence = [
            ('UP',    0.0,  0.0,  0.2),
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


    def reacquire_goal_pose(self, seed_xyz, timeout=3.5, radius=0.08,
                            stable_eps=0.004, stable_need=2):
        """Try to reacquire near seed. On timeout, retreat a bit, then try once more."""
        import math, time
        latest = None
        stable = 0
        picked = None

        # ensure we're not moving
        start = time.time()
        while self.is_robot_moving() and (time.time() - start) < 0.25:
            time.sleep(0.01)

        def _try_once(seed, timeout_s, rad, need):
            nonlocal latest, stable, picked
            latest = None
            stable = 0
            picked = None
            last_inradius_ts = time.time()

            def _cb(msg):
                nonlocal latest, stable, picked, last_inradius_ts
                if self.is_robot_moving():
                    return
                x = msg.pose.position.x
                y = msg.pose.position.y
                z = msg.pose.position.z
                print(f"Reacquire got detection: {(x,y,z)}")
                dx, dy, dz = x - seed[0], y - seed[1], z - seed[2]
                d_xy = math.hypot(dx, dy)
                print(f"Reacquire checking radius: d_xy={d_xy:.4f} (<= {rad:.3f})")
                if d_xy > rad:
                    return
                last_inradius_ts = time.time()
                latest = (x, y, z)
                stable += 1
                print(f"Reacquire: in-radius hit #{stable} at {latest}")
                if stable >= max(1, need):
                    picked = latest

            sub = self.create_subscription(PoseStamped, '/external_goal_pose', _cb, self.qos)
            try:
                while picked is None:
                    if timeout_s is not None and (time.time() - last_inradius_ts) > timeout_s:
                        break
                    time.sleep(0.01)
            finally:
                self.destroy_subscription(sub)

            return picked

        # -------- first attempt --------
        first = _try_once(seed_xyz, timeout, radius, stable_need)
        if first is not None:
            self.get_logger().info(f"🔎 Reacquired near seed {seed_xyz} → {first}")
            return list(first)

        # -------- retreat then second attempt --------
        self.get_logger().warn("⚠️ Reacquire timed out; backing off, then retrying…")
        cur = self.get_end_effector_pose()
        if cur:
            # small lateral back-off; keep orientation
            retreat = [cur[0], cur[1] + 0.10, cur[2], *cur[3:]]
            self.execute_single_pose(retreat)
            # wait until the retreat finishes (prevents filtering while moving)
            self.wait_until_motion_finishes()

        # optional: expand search a bit & accept 1 stable hit
        retry_radius = max(radius, 0.15)
        retry_timeout = 2.0
        retry_need = max(1, stable_need - 1)

        second = _try_once(seed_xyz, retry_timeout, retry_radius, retry_need)

        if second is not None:
            self.get_logger().info(f"🔎 Reacquired after retreat → {second}")
            return list(second)

        self.get_logger().warn("❌ Reacquire failed after retreat.")
        return None



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
                #0.03
                time_from_start += 0.08
                trajectory_msg.points.append(traj_point)

            self.trajectory_pub.publish(trajectory_msg)
        else:
            self.get_logger().warn("❌ Idle motion planning failed.")


    def move_to_home_position(self, cfg: MotionGenPlanConfig = None):
        """Plan & execute a joint-space path to self.home_joints, then wait until the end-effector arrives."""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state; skipping home move.")
            return
        
        if not self.robot_running:
            self.get_logger().error("❌ Cannot execute goals: robot program is OFF. Please turn it ON first.")
            #return
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
            time_from_start += 0.08 / getattr(self, "speed_scale", 1.0)

        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info("🏠 Moving to HOME joints…")

        # ——— Wait until we really reach the home Cartesian pose ———
        final_joints = traj_tensor[-1].tolist()
        fk = self.forward_kinematics(final_joints)
        if fk:
            # wait_for_execution_completion expects [x, y, z]
            self.wait_for_execution_completion([fk.x, fk.y, fk.z])


    def rotate_wrist(self, degrees, duration=0.30):
        """Rotate wrist_3 by +degrees, then back to start, in one JointTrajectory."""
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state received yet. Cannot rotate wrist.")
            return

        # (optional but helpful) ensure we capture a stable start
        try:
            while self.is_robot_moving(velocity_threshold=0.01):
                time.sleep(0.01)
        except Exception:
            pass

        radians = math.radians(degrees)
        traj = JointTrajectory()
        traj.joint_names = self.joint_order

        start = self.current_joint_positions.copy()
        plus  = start.copy(); plus[5]  = start[5] + radians
        back  = start.copy()

        zeros = [0.0] * len(self.joint_order)

        # t = 0
        p0 = JointTrajectoryPoint()
        p0.positions = start
        p0.velocities = zeros
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0

        # t = duration  (go to +deg and STOP)
        p1 = JointTrajectoryPoint()
        p1.positions = plus
        p1.velocities = zeros
        p1.time_from_start.sec = int(duration)
        p1.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        # t = 2*duration (come back and STOP)
        p2 = JointTrajectoryPoint()
        p2.positions = back
        p2.velocities = zeros
        total = 2.0 * duration
        p2.time_from_start.sec = int(total)
        p2.time_from_start.nanosec = int((total - int(total)) * 1e9)

        # tiny final hold at back (2*duration + 0.05s)
        p3 = JointTrajectoryPoint()
        p3.positions = back
        p3.velocities = zeros
        hold_ns = p2.time_from_start.nanosec + 50_000_000
        p3.time_from_start.sec = p2.time_from_start.sec + (1 if hold_ns >= 1_000_000_000 else 0)
        p3.time_from_start.nanosec = hold_ns % 1_000_000_000

        traj.points = [p0, p1, p2, p3]
        self.trajectory_pub.publish(traj)
        self.get_logger().info(f"Wrist +{degrees}° then back in {2*duration:.2f}s (vel=0 at waypoints)")

        
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
            #0.008
            time_from_start += 0.08/self.speed_scale                                    # same 30 ms increment

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
        time_from_start = 0.08

        for point in traj_tensor:
            if self.stop_requested:
                self.get_logger().warn("🛑 Stop requested! Aborting drop-off motion.")
                self.stop_requested = False
                return

            pt = JointTrajectoryPoint()
            pt.positions = point.tolist()
            #pt.velocities = [0.1] * len(self.joint_order)        # same 0.1 rad/s feed-forward
            pt.time_from_start.sec     = int(time_from_start)
            pt.time_from_start.nanosec = int((time_from_start % 1.0) * 1e9)

            trajectory_msg.points.append(pt)
            #0.04
            time_from_start += 0.08/ getattr(self, "speed_scale", 1.0)                                 # same 30 ms increment

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
        pt.velocities = [0.0] * len(self.joint_order)     # <-- add
        pt.accelerations = [0.0] * len(self.joint_order)  # <-- add
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 1_000_000            # <-- tiny >0 duration helps preemption


        stop_msg.points = [pt]
        self.trajectory_pub.publish(stop_msg)        
            
    def control_gripper(self, action):
        if action.upper() == "OPEN":
            self.gripper_controller.open_gripper()
            self.gripper_closed = False
            self.slip_detection = False
            self.grap_miss = False
            self.classifier.start_opening()
            self.get_logger().info("Gripper opened → slip detection paused")

        elif action.upper() == "CLOSE":
            self.classifier.start_closing()
            self.gripper_controller.run_closure_loop()
            self.classifier.mark_close_done()            
            self.gripper_closed = True
            self.get_logger().info("Gripper closed → slip detection active")

    def force_callback(self, msg):
        raw = list(msg.data)
        self.classifier.on_force(raw[:3])

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
            PoseStamped, '/external_goal_pose', self._capture_goal_callback, self.qos
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


    def _on_grasp_outcome(self, outcome: str, end: str):
        # flags
        self.slip_detection = (outcome == "SLIPPED")
        self.grap_miss      = (outcome == "NO_GRAB")
        self.weak_grab      = (outcome == "GRABBED" and end == "WEAK")
        self.last_grasp_end_template = end

        self.get_logger().info(f"[grasp] outcome={outcome} end={end} "
                            f"-> slip={self.slip_detection} miss={self.grap_miss} weak={self.weak_grab}")



    def nudge_wrist1(self, delta_deg: float = -1.0, duration: float = 0.25):
        """
        Nudge wrist_1_joint (index 3) by delta_deg (degrees).
        Negative = “down” for most UR10e setups. Flip sign if direction is wrong.
        """
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state received yet. Cannot nudge wrist_1.")
            return

        # (optional) wait until robot is still
        try:
            while self.is_robot_moving(velocity_threshold=0.01):
                time.sleep(0.01)
        except Exception:
            pass

        delta_rad = math.radians(delta_deg)

        start = self.current_joint_positions.copy()
        target = start.copy()
        target[3] = start[3] + delta_rad   # wrist_1_joint is index 3

        traj = JointTrajectory()
        traj.joint_names = self.joint_order

        p0 = JointTrajectoryPoint()
        p0.positions = start
        p0.velocities = [0.0]*len(self.joint_order)
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0

        p1 = JointTrajectoryPoint()
        p1.positions = target
        p1.velocities = [0.0]*len(self.joint_order)
        p1.time_from_start.sec = int(duration)
        p1.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        # small hold at target
        p2 = JointTrajectoryPoint()
        p2.positions = target
        p2.velocities = [0.0]*len(self.joint_order)
        hold = duration + 0.05
        p2.time_from_start.sec = int(hold)
        p2.time_from_start.nanosec = int((hold - int(hold)) * 1e9)

        traj.points = [p0, p1, p2]
        self.trajectory_pub.publish(traj)
        self.get_logger().info(f"Wrist_1 nudged {delta_deg:+.2f}° in {duration:.2f}s")



    def converge_to_goal(self, goal_xyz, goal_quat, pos_tol=0.0015, step=0.003, max_iters=25,
                        max_secs=1.25, orient_quat=None, orient_tol_deg=2.0):
        """
        Micro-servo: iteratively 'nudge' toward goal using tiny plans.
        - pos_tol: meters (1.5 mm default)
        - step:    meters per hop (3 mm default)
        - orient_quat: optional [w,x,y,z] to enforce orientation simultaneously
        """
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state; skipping convergence.")
            return

        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prev_dist = None
        stagnation = 0
        orient_tol = math.radians(orient_tol_deg)

        for it in range(max_iters):
            if self.stop_requested or not self.running:
                break
            if (time.time() - t0) > max_secs:
                self.get_logger().info("Convergence time cap reached.")
                break

            cur = self.get_end_effector_pose()
            if not cur:
                break
            cx, cy, cz, cqw, cqx, cqy, cqz = cur
            ex, ey, ez = goal_xyz[0]-cx, goal_xyz[1]-cy, goal_xyz[2]-cz
            dist = math.sqrt(ex*ex + ey*ey + ez*ez)


            if dist <= pos_tol:
                self.get_logger().info(f"Converged: {dist*1000:.1f} mm")
                return

            # stagnation guard (<0.05 mm improvement over 3 iters)
            if prev_dist is not None and abs(prev_dist - dist) < 0.00005:
                stagnation += 1
            else:
                stagnation = 0
            prev_dist = dist
            if stagnation >= 3:
                self.get_logger().info("Convergence stagnating; stopping.")
                return

            # clamp step and build micro-target
            s = min(step, dist)
            nx, ny, nz = cx + (ex/dist)*s, cy + (ey/dist)*s, cz + (ez/dist)*s
            tiny_target = [nx, ny, nz, *goal_quat]

            # plan a tiny hop
            start_state = JointState.from_position(
                torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )
            goal_pose = Pose.from_list(tiny_target)
            res = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                MotionGenPlanConfig(max_attempts=6, enable_finetune_trajopt=True)
            )
            if not res.success:
                # reduce step and try next iteration
                step = max(0.0008, step * 0.6)
                continue

            # execute tiny hop with slow velocities
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_order
            interp = res.get_interpolated_plan()
            if isinstance(interp, JointState): interp = interp.position
            if not isinstance(interp, torch.Tensor):
                interp = torch.tensor(interp, dtype=torch.float32)
            interp = interp.to("cpu")

            tfs = 0.0
            for point in interp:
                if self.stop_requested:
                    break
                pt = JointTrajectoryPoint()
                pt.positions = point.numpy().tolist()
                pt.velocities = [0.03] * len(self.joint_order)  # slow for accuracy
                pt.time_from_start.sec = int(tfs)
                pt.time_from_start.nanosec = int((tfs % 1) * 1e9)
                tfs += 0.02
                traj_msg.points.append(pt)

            self.trajectory_pub.publish(traj_msg)
            # wait tightly on each micro-hop
            self.wait_for_execution_completion([nx, ny, nz])


    def descend_z_wrist_osc_simple(self,
                                dz=-0.06,          # meters (negative = down)
                                duration=1.0,      # seconds
                                yaw_amp_deg=40.0,  # ± amplitude around 0°
                                osc_hz=2.0,        # oscillations per second
                                max_points=200,    # cap points to keep it light
                                yaw_joint_index=5  # UR10e wrist_3
                                ):


        if self.current_joint_positions is None:
            self.get_logger().error("No joint state yet.")
            return

        # Current pose
        cur = self.get_end_effector_pose()
        if not cur:
            self.get_logger().warn("EE pose unavailable; aborting.")
            return
        x0, y0, z0 = cur[0], cur[1], cur[2]
        qw, qx, qy, qz = cur[3], cur[4], cur[5], cur[6]   # keep base orientation

        # Plan ONCE: start joints -> same (x,y) but z+dz, same orientation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_state = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        goal_pose = Pose.from_list([x0, y0, z0 + dz, qw, qx, qy, qz])
        res = self.motion_gen.plan_single(
            start_state, goal_pose,
            MotionGenPlanConfig(max_attempts=12, enable_finetune_trajopt=True)
        )
        if not res.success:
            self.get_logger().warn("Plan to final Z failed.")
            return

        interp = res.get_interpolated_plan()
        if isinstance(interp, JointState):
            interp = interp.position
        if not isinstance(interp, torch.Tensor):
            interp = torch.tensor(interp, dtype=torch.float32)
        pts = interp.to("cpu")
        N  = pts.shape[0]
        K  = min(max_points, N) if N > 1 else 1
        idxs = np.linspace(0, N - 1, num=K, dtype=int)

        # Build one JointTrajectory (seed t=0)
        traj = JointTrajectory()
        traj.joint_names = self.joint_order
        traj.header.stamp = self.get_clock().now().to_msg()

        p0 = JointTrajectoryPoint()
        p0.positions  = self.current_joint_positions.copy()
        p0.velocities = [0.0] * len(self.joint_order)
        p0.time_from_start.sec = 0
        p0.time_from_start.nanosec = 0
        traj.points.append(p0)

        amp = math.radians(yaw_amp_deg)
        for k, idx in enumerate(idxs):
            t = duration * (k + 1) / K
            q = pts[idx].numpy().tolist()

            # add sinusoid around 0° to wrist_3 (tool yaw)
            
            q[yaw_joint_index] += amp * math.sin(2.0 * math.pi * osc_hz * t)

            pt = JointTrajectoryPoint()
            pt.positions  = q
            pt.velocities = [0.0] * len(self.joint_order)  # let controller time-interpolate
            pt.time_from_start.sec     = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)

        # Small preempt to avoid any leftover motion, then publish once
        self._publish_stop_trajectory()
        time.sleep(0.02)
        self.trajectory_pub.publish(traj)

        self.get_logger().info(
            f"descend_z_wrist_osc_simple: dz={dz:.3f} m in {duration:.2f}s, "
            f"amp=±{yaw_amp_deg:.1f}°, f={osc_hz:.2f} Hz, points={len(traj.points)}"
        )




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

