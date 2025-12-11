# ruff: noqa
import threading
import rclpy
import os
import numpy as np

from rclpy.timer import Timer
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
from rclpy.node import Node
from sensor_msgs.msg import JointState as ROSJointState
from visualization_msgs.msg import InteractiveMarkerFeedback, Marker
from std_msgs.msg import Bool, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from .config import AppConfig, DEFAULT_QOS, WORLD_CONFIG, JOINT_ORDER

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.geom.types import Sphere, Cuboid
from .config import JOINT_ORDER, WORLD_CONFIG, DEFAULT_QOS
from .utils import read_key
from . import fk as fk_mod
from . import markers as markers_mod
from . import motions as motions_mod
from . import goals as goals_mod
from . import gripper as gripper_mod
from .motions import publish_stop_trajectory
from .dynamic_obstacle import DynamicObstacleManager
from .perception import ZedYoloPerception
from trajectory_msgs.msg import JointTrajectory
from ur_msgs.srv import SetIO
from .grasp_outcome_classifier import classify_triplet


# Callback	Trigger	Purpose

# _check_joint_states()	    timer	                  Confirms joint feedback received
# _joint_state_cb(msg)	    /joint_states	          Updates position & velocity arrays
# _marker_cb(msg)	        RViz interactive marker	  Stores last clicked pose
# _robot_running_cb(msg)	/robot_program_running	  Logs robot program state
# _stop_cb(msg)	            /emergency_stop	          Stops motion immediately
# _force_cb(msg)	        /gripper/force	          Sends force readings to classifier
# _classifier_tick()	    timer	                  Runs periodic classifier update


class UR10eCuroboMoveIt(Node):
    def __init__(self):
        super().__init__(
            "ur10e_curobo_moveit_node",
            automatically_declare_parameters_from_overrides=True
        )

        # ========= PARAMS: defaults → ROS params → env overrides =========
        self.cfg = AppConfig()  # 1) start with code defaults

        # 2) declare ROS parameters
        self.declare_parameter("planner.speed_scale", self.cfg.planner.speed_scale)
        self.declare_parameter("planner.urdf_config", self.cfg.planner.urdf_config)
        self.declare_parameter("planner.interpolation_dt", self.cfg.planner.interpolation_dt)

        self.declare_parameter("perception.enabled", self.cfg.perception.enabled)
        self.declare_parameter("perception.weights", self.cfg.perception.weights)
        self.declare_parameter("perception.img_size", self.cfg.perception.img_size)
        self.declare_parameter("perception.conf_thres", self.cfg.perception.conf_thres)
        self.declare_parameter("perception.cam_frame", self.cfg.perception.cam_frame)
        self.declare_parameter("perception.show_view", self.cfg.perception.show_view)
        self.declare_parameter("perception.use_gpu", self.cfg.perception.use_gpu)

        # gripper
        self.declare_parameter("gripper.force_threshold_left", self.cfg.gripper.force_threshold_left)
        self.declare_parameter("gripper.force_threshold_center", self.cfg.gripper.force_threshold_center)
        self.declare_parameter("gripper.force_threshold_right", self.cfg.gripper.force_threshold_right)
        self.declare_parameter("gripper.min_fingers_for_stop", self.cfg.gripper.min_fingers_for_stop)
        self.declare_parameter("gripper.closing_steps", self.cfg.gripper.closing_steps)
        self.declare_parameter("gripper.step_delay_s", self.cfg.gripper.step_delay_s)
        self.declare_parameter("gripper.use_suction", self.cfg.gripper.use_suction)

        #topics
        self.declare_parameter("topics.joint_traj", self.cfg.topics.traj_cmd)
        self.declare_parameter("topics.goal_marker", self.cfg.topics.goal_marker)
        self.declare_parameter("topics.path_marker", self.cfg.topics.path_marker)
        self.declare_parameter("topics.joint_states", self.cfg.topics.joint_states)
        #joints

        self.declare_parameter("joints.home", self.cfg.joints.home)
        self.declare_parameter("joints.dropoff", self.cfg.joints.dropoff)
        self.declare_parameter("joints.predropoff", self.cfg.joints.predropoff)
        
        self.declare_parameter("planner.pre_droffoff_z_offset", self.cfg.planner.pre_droffoff_z_offset)
        self.declare_parameter("planner.pre_droffoff_y_offset", self.cfg.planner.pre_droffoff_y_offset)

        # 3) read back ROS parameters
        self.cfg.planner.speed_scale = self.get_parameter("planner.speed_scale").value
        self.cfg.planner.urdf_config = self.get_parameter("planner.urdf_config").value
        self.cfg.planner.interpolation_dt = float(self.get_parameter("planner.interpolation_dt").value)

        self.cfg.perception.enabled    = bool(self.get_parameter("perception.enabled").value)
        self.cfg.perception.weights    = self.get_parameter("perception.weights").value
        self.cfg.perception.img_size   = int(self.get_parameter("perception.img_size").value)
        self.cfg.perception.conf_thres = float(self.get_parameter("perception.conf_thres").value)
        self.cfg.perception.cam_frame  = self.get_parameter("perception.cam_frame").value
        self.cfg.perception.show_view  = bool(self.get_parameter("perception.show_view").value)
        self.cfg.perception.use_gpu    = bool(self.get_parameter("perception.use_gpu").value)

        # gripper
        self.cfg.gripper.force_threshold_left   = float(self.get_parameter("gripper.force_threshold_left").value)
        self.cfg.gripper.force_threshold_center = float(self.get_parameter("gripper.force_threshold_center").value)
        self.cfg.gripper.force_threshold_right  = float(self.get_parameter("gripper.force_threshold_right").value)
        self.cfg.gripper.min_fingers_for_stop   = int(self.get_parameter("gripper.min_fingers_for_stop").value)
        self.cfg.gripper.closing_steps          = int(self.get_parameter("gripper.closing_steps").value)
        self.cfg.gripper.step_delay_s           = float(self.get_parameter("gripper.step_delay_s").value)
        self.cfg.gripper.use_suction            = bool(self.get_parameter("gripper.use_suction").value)

        #ros topics
        self.cfg.topics.traj_cmd      = self.get_parameter("topics.joint_traj").value
        self.cfg.topics.goal_marker    = self.get_parameter("topics.goal_marker").value
        self.cfg.topics.path_marker    = self.get_parameter("topics.path_marker").value
        self.cfg.topics.joint_states   = self.get_parameter("topics.joint_states").value


        # arrays come back as tuples in Foxy—cast to list
        self.cfg.joints.home       = list(self.get_parameter("joints.home").value)
        self.cfg.joints.dropoff    = list(self.get_parameter("joints.dropoff").value)
        self.cfg.joints.predropoff = list(self.get_parameter("joints.predropoff").value)
        
        self.cfg.planner.pre_droffoff_z_offset = float(self.get_parameter("planner.pre_droffoff_z_offset").value)
        self.cfg.planner.pre_droffoff_y_offset = float(self.get_parameter("planner.pre_droffoff_y_offset").value)

        # 4) env overrides (UR10E_*), e.g. UR10E_SHOW_VIEW=1
        self.cfg = AppConfig.from_env(self.cfg)

        # 5) expose to the rest of the class
        self.speed_scale       = self.cfg.planner.speed_scale
        self.home_joints       = self.cfg.joints.home
        self.dropoff_joints    = self.cfg.joints.dropoff
        self.predropoff_joints = self.cfg.joints.predropoff
        
        self.yoffset = self.cfg.planner.pre_droffoff_y_offset
        self.zoffset = self.cfg.planner.pre_droffoff_z_offset
        self.cam_frame = self.cfg.perception.cam_frame

        self.traj_cmd_topic    = self.cfg.topics.traj_cmd
        self.joint_states_topic = self.cfg.topics.joint_states
        self.goal_marker_topic  = self.cfg.topics.goal_marker
        self.path_marker_topic  = self.cfg.topics.path_marker
        # state: keep track of robot state, path history, and goals in memory.
        self.qos = DEFAULT_QOS
        self.goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        # ======== pubs/subs after config so QoS/params exist ========

        self.trajectory_pub = self.create_publisher(JointTrajectory, self.traj_cmd_topic, 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)
        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)

        self.create_subscription(ROSJointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(
            InteractiveMarkerFeedback,
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback",
            self._marker_cb, 10
        )
        self.create_subscription(Float32MultiArray, "/gripper/force", self._force_cb, 10)
        self.create_subscription(Bool, "/emergency_stop", self._stop_cb, 10)
        self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self._robot_running_cb, 10)
        self.create_subscription(String, "ui_command", self._ui_command_cb, 10)
        
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        #while not self.io_client.wait_for_service(timeout_sec=1.0):
            #self.get_logger().info("Waiting for /set_io service...")
        
        self.goal_tracker_sub = self.create_subscription(
            PoseStamped,
            '/external_goal_pose',        # always listen to new poses
            self._continuous_goal_tracker,  # callback function below
            self.goal_qos
        )
        
        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # timers
        self.create_timer(0.1, lambda: markers_mod.track_robot_path(self)) #Track & update RViz path markers
        self.create_timer(0.02, self._classifier_tick) #Tick classifier loop (gripper ML logic)
        self.timer_wait_js = self.create_timer(0.5, self._check_joint_states) #Check if joint states received

       # 1. Load cuRobo config
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.cfg.planner.urdf_config, WORLD_CONFIG,
            interpolation_dt=self.cfg.planner.interpolation_dt
        )

        # 2. Create MotionGen
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        self.get_logger().info("cuRobo warmup done")


        self.obstacles = DynamicObstacleManager(
            node=self,
            motion_gen=self.motion_gen,
            world_model=self.motion_gen.world_model
        )

        # 4. Add a dynamic sphere
        self.obstacles.add_sphere("dyn_sphere", radius=0.1)
        self.obstacles.add_sphere("fruit_obstacle", radius=0.06)
        

        self.joint_order = JOINT_ORDER
        
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self._js_missing_warned = False
        self.latest_marker_pose = None
        
        self.goal_poses = []
        self.path_points = []
        
        
        self.running = True
        self.stop_requested = False
        self.robot_running = False
        
        # --- capture state ---
        self.goal_capture_active = False
        self.goal_capture_timer = None
        self.goal_pose_sub = None
        self.goal_capture_count = 0
        
        self.goal_sort_ref = None
        self.goal_sort_ascending = True
        
        self.tf_warning_printed = False
        self.tf_printed = False

        self.latest_goal_pose = None        # [x, y, z, qw, qx, qy, qz]
        self.latest_goal_time = 0.0         # timestamp of last valid pos


        self.best_goal_xyz = None
        self.best_goal_quat = None  # Track best orientation quaternion [w, x, y, z]
        self.best_goal_score = float("inf")
        self.goal_seed_xy = None

        self.last_frames = [] # last few frames for stability checking




        # gripper/classifier (uses config values)
        gripper_mod.init_gripper(self)

        # keyboard
        self.keyboard_thread = threading.Thread(target=self._wait_for_key_press, daemon=True)
        self.keyboard_thread.start()
        self.get_logger().info("UR10e cuRobo node initialized. Waiting for joint states…")
        
        # # Perception: ZED + YOLO
        #self._maybe_start_perception()

    # callbacks
    def _check_joint_states(self):  #Confirms joint feedback received
        if self.current_joint_positions is not None:
            self.get_logger().info("Initial joints received."); self.destroy_timer(self.timer_wait_js)

    def _joint_state_cb(self, msg):  #Updates position & velocity arrays
        jm = dict(zip(msg.name, msg.position))
        vm = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}

        missing = [j for j in self.joint_order if j not in jm]
        if missing:
            if not self._js_missing_warned:
                self.get_logger().warn(f"JointState missing joints {missing}; waiting for full state.")
                self._js_missing_warned = True
            return

        self._js_missing_warned = False
        
        self.current_joint_positions = [jm[j] for j in self.joint_order]
        self.current_joint_velocities = [vm.get(j, 0.0) for j in self.joint_order]

    def _marker_cb(self, msg): ##Stores last clicked pose
        self.latest_marker_pose = msg.pose

    def _robot_running_cb(self, msg): ##Logs robot program state
        self.robot_running = msg.data
        self.get_logger().info("✅ Robot program is running." if msg.data else "⚠️ Robot program is NOT running.")

    def _stop_cb(self, msg): ##Stops motion immediately
        if not msg.data: return
        self.get_logger().warn("🛑 Emergency stop!"); self.stop_requested = True
        publish_stop_trajectory(self)

    def _force_cb(self, msg): ##Sends force readings to classifier
        forces = list(msg.data)[:3]
        self.classifier.on_force(forces)
        if hasattr(self, "visualizer"):
            self.visualizer.update_forces(forces)
            tpl = classify_triplet(forces)
            self.visualizer.update_classifier(self.classifier.phase, tpl)

    def _classifier_tick(self):   ##Runs periodic classifier update
        self.classifier.tick()

    def _ui_command_cb(self, msg: String):
        cmd = (msg.data or "").strip()
        if not cmd:
            return

        parts = cmd.lower().split()
        action = parts[0]
        args = parts[1:]

        try:
            if action == "home":
                motions_mod.move_to_home_position(self)
            elif action == "dropoff":
                motions_mod.move_to_dropoff_position(self)
                # Wait for robot to stabilize before opening gripper
                import time
                time.sleep(0.5)
                gripper_mod.control_gripper(self, "OPEN")
            elif action == "execute":
                self._prep_and_execute()
            elif action == "stop":
                self.stop_requested = True
                publish_stop_trajectory(self)
                self.goal_poses.clear()
            elif action == "open":
                gripper_mod.control_gripper(self, "OPEN")
            elif action == "close":
                gripper_mod.control_gripper(self, "CLOSE")
            elif action == "capture":
                duration = float(args[0]) if args else 10.0
                self.start_goal_capture(duration)
            elif action == "capture_stop":
                self.stop_goal_capture()
            elif action == "clear":
                self.goal_poses.clear()
                self.get_logger().info("Cleared stored goals.")
            elif action == "debug_world":
                self.debug_print_world()
            else:
                self.get_logger().warn(f"UI command '{cmd}' not recognized.")
        except Exception as e:
            self.get_logger().error(f"UI command '{cmd}' failed: {e}")

    # methods used by helpers (so helpers can call like node.get_end_effector_pose())
    def get_end_effector_pose(self):
        return fk_mod.get_end_effector_pose(self)
    
    

    # keyboard UI
    # keyboard UI (verbose)
    def _wait_for_key_press(self):
        def ts():
            # short timestamp for prints
            import time
            return time.strftime("%H:%M:%S")

        self.get_logger().info("Keys: y=save marker, m=manual, s=subscribe, p=capture, n=execute, o=open, c=close, h=home, d=dropoff, k=stop, q=quit")
        last_key = None
        while self.running:
            key = read_key()
            if key is None:
                continue

            # echo keypress
            self.get_logger().debug(f"key='{key}'")

            if key == 'y':
                if self.latest_marker_pose:

                    from .goals import pose_to_vec7
                    g = pose_to_vec7(self.latest_marker_pose)
                    self.goal_poses.append(g)
                    self.get_logger().info(f"Saved goal #{len(self.goal_poses)} from marker: {g}")

                    markers_mod.publish_goal_marker(self, g[:3])
                else:
                    self.get_logger().warn("No latest_marker_pose yet; press 'y' again after moving the interactive marker in RViz.")

            elif key == 'm':
                self.get_logger().info("Manual goal entry requested")
                curr_pose = self.get_end_effector_pose()
                self.get_logger().info(f"Current pose: {curr_pose}")
                self._manual_goal()
                self.get_logger().info(f"Manual goal entry done. Total goals={len(self.goal_poses)}")

            elif key == 's':
                self.get_logger().info("Subscribing to /external_goal_pose")
                goals_mod.subscribe_to_goal_pose(self)
                self.goal_capture_active = False
                self.get_logger().info("Subscribe called. Waiting for external goal")

            elif key == 'p':
                self.get_logger().info("Starting timed goal capture (10s)")
                self.start_goal_capture(10.0)
                self.get_logger().info(f"Capture armed. Current collected={getattr(self, 'goal_capture_count', 0)}")

            elif key == 'n':
                if self.goal_poses:
                    self.get_logger().info(f"Executing {len(self.goal_poses)} stored goal(s)")
                    self._prep_and_execute()
                    self.get_logger().info(f"Execute finished. Remaining goals={len(self.goal_poses)}")
                else:
                    self.get_logger().info("No goals to execute. Add with 'y', 'm', or 's'.")

            elif key == 'o':
                self.get_logger().info("Gripper → OPEN")
                gripper_mod.control_gripper(self, 'OPEN')

            elif key == 'c':
                self.get_logger().info("Gripper → CLOSE; nudge back & rotate wrist")
                gripper_mod.control_gripper(self, 'CLOSE')
                motions_mod.rotate_wrist(self, 65, duration_s=0.38)
                self.get_logger().info("Post-close micro-motions done")

            elif key == 'h':
                self.get_logger().info("Going HOME")
                motions_mod.move_to_home_position(self)
                self.get_logger().info("Reached HOME (or attempted)")

            elif key == 'd':
                self.get_logger().info("Going to DROPOFF")
                motions_mod.move_to_dropoff_position(self)
                gripper_mod.control_gripper(self, 'OPEN')
                self.get_logger().info("At drop-off; gripper opened")

            elif key == 'k':
                self.get_logger().info("STOP requested → publishing hold trajectory")
                self.stop_requested = True
                motions_mod.publish_stop_trajectory(self)
                # Clear any queued goals so execution loop can exit quickly
                self.goal_poses.clear()

            elif key == 'q':
                self.get_logger().info("Quitting")
                self.running = False
                self.stop_requested = True
                rclpy.shutdown()
                break
            elif key == "f":
                self.get_logger().info("Finding current joint positions")
                self.get_logger().info(str(self.current_joint_positions))
                self.get_logger().info("Finding current end-effector pose")
                self.get_logger().info(str(self.get_end_effector_pose()))

            elif key == "t":
                self.get_logger().info("Update dynamic obstacle position:")

                pos = self.obstacles.ask_user_position("dyn_sphere")
                if pos is not None:
                    self.obstacles.update_pose("dyn_sphere", pos)
                    self.obstacles.print_world()
            else:
                # unknown key helper
                if key != last_key:  # avoid spamming if someone holds a key
                    self.get_logger().info(f"Key '{key}' has no action. Valid keys: y m s p n o c h d k q")
            last_key = key


    def _manual_goal(self):
        cur = self.get_end_effector_pose()
        try:
            x = float(input("Enter X: "))
            y = float(input("Enter Y: "))
            z = float(input("Enter Z: "))
        except ValueError:
            self.get_logger().warn("Invalid input.")
            return
        goal = [x, y, z] + (cur[3:] if cur else [1.0,0.0,0.0,0.0])
        self.goal_poses.append(goal)
        markers_mod.publish_goal_marker(self, goal[:3])
        self.get_logger().info("Manual goal saved.")

    def _continuous_goal_tracker(self, msg: PoseStamped):

        if self.goal_seed_xy is None:
            return

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        seed_x, seed_y = self.goal_seed_xy

        # ------------------------------------------------------
        # 1. Strict fruit-lock (XY constraint)
        # Allow small drift (max 5–6 cm is reasonable)
        # ------------------------------------------------------
        if math.hypot(x - seed_x, y - seed_y) > 0.01:
            return

        # ------------------------------------------------------
        # 2. Depth check with softer limit
        # Allow 3.5–4 cm variation
        # ------------------------------------------------------
        if self.best_goal_xyz and z > self.best_goal_xyz[2] + 0.01:
            return

        # ------------------------------------------------------
        # 3. Distance to EE (we want CLOSER = BETTER)
        # ------------------------------------------------------
        ee = self.get_end_effector_pose()
        if ee:
            dx = x - ee[0]
            dy = y - ee[1]
            dz = z - ee[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            dist = z

        # ------------------------------------------------------
        # 4. Variance over last few frames (stability)
        # ------------------------------------------------------
        self.last_frames.append([x, y, z])
        if len(self.last_frames) > 8:
            self.last_frames.pop(0)

        variance = np.var(self.last_frames, axis=0).sum()

        # ------------------------------------------------------
        # 5. NEW Scoring: smaller distance & smaller noise is better
        # ------------------------------------------------------
        score = dist + variance * 3.0   # POSITIVE penalty

        # Lower score = better (changed sign logic!)
        if score < self.best_goal_score - 0.002:
            self.best_goal_score = score
            self.best_goal_xyz = [x, y, z]

            # Extract and apply relative rotation (same as in goals.py)
            rotation_quat_w = msg.pose.orientation.w
            rotation_quat_z = msg.pose.orientation.z

            # Get current EE orientation
            current_ee_pose = self.get_end_effector_pose()
            if current_ee_pose:
                current_quat = current_ee_pose[3:]  # [w, x, y, z]
            else:
                current_quat = [1.0, 0.0, 0.0, 0.0]

            # Apply relative rotation
            from scipy.spatial.transform import Rotation as R
            current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            relative_rot = R.from_quat([0, 0, rotation_quat_z, rotation_quat_w])
            final_rot = current_rot * relative_rot
            final_quat_scipy = final_rot.as_quat()
            self.best_goal_quat = [final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]]

            self.get_logger().debug(f"Updated BEST (continuous) = {self.best_goal_xyz}, score={score}")


    # goal capture (timed)
    def start_goal_capture(self, duration: float = 30.0):
        if self.goal_capture_active: 
            self.stop_goal_capture()
            
        self.goal_poses.clear()
        self.goal_capture_count = 0
        ee = self.get_end_effector_pose()
        self.goal_sort_ref = ee[:3] if ee else [0.0,0.0,0.0]
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            '/external_goal_pose',
            self._capture_goal_cb,
            getattr(self, "goal_qos", self.qos),
        )
        self.goal_capture_active = True
        self.get_logger().info(f"Started goal capture for {duration:.0f}s")
        
        def _stop_once():
            if self.goal_capture_timer: self.goal_capture_timer.cancel()
            self.stop_goal_capture()
        self.goal_capture_timer = self.create_timer(duration, _stop_once)

    def stop_goal_capture(self):
        if not self.goal_capture_active: 
            return
        self.goal_capture_active = False
    
        if hasattr(self, 'goal_pose_sub'):
            self.destroy_subscription(self.goal_pose_sub); del self.goal_pose_sub
        self.goal_poses.sort(key=lambda g: __import__('math').dist(g[:3], self.goal_sort_ref or [0,0,0]))
        self.get_logger().info(f"Goal capture stopped. Collected {self.goal_capture_count} goals.")

    def _capture_goal_cb(self, msg: PoseStamped):
        if goals_mod.is_robot_moving(self): 
            return
        gx,gy,gz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        cur = self.get_end_effector_pose()
        qw,qx,qy,qz = (cur[3:] if cur else [1.0,0.0,0.0,0.0])
        
        if any(math.dist([gx,gy,gz], g[:3]) < 0.01 for g in (self.goal_poses or [])): 
            return
        goal = [gx,gy,gz,qw,qx,qy,qz]
        self.goal_poses.append(goal)
        self.goal_poses.sort(key=lambda g: math.dist(g[:3], self.goal_sort_ref or [0,0,0]))
        self.goal_capture_count += 1
        markers_mod.publish_goal_marker(self, goal[:3])

    def _prep_and_execute(self):
        
        if self.goal_capture_active: 
            self.stop_goal_capture()
        if hasattr(self, 'goal_pose_sub'): 
            self.destroy_subscription(self.goal_pose_sub)
            del self.goal_pose_sub
            
        self.get_logger().info("Executing stored goals…")
        goals_mod.plan_and_execute(self)
        self.get_logger().info("Done.")

    # expose some helpers for external callers
    def control_gripper(self, action: str):
        gripper_mod.control_gripper(self, action)

    def debug_print_world(self):
        wm = self.motion_gen.world_model

        self.get_logger().info("\n========== CURRENT CUROBO WORLD ==========")

        # ----- SPHERES -----
        self.get_logger().info("SPHERES:")
        if wm.sphere:
            for s in wm.sphere:
                self.get_logger().info(f"  - name={s.name}, pose={s.pose}, radius={s.radius}")
        else:
            self.get_logger().info("  (none)")

        # ----- CUBOIDS -----
        self.get_logger().info("\nCUBOIDS:")
        if wm.cuboid:
            for c in wm.cuboid:
                self.get_logger().info(f"  - name={c.name}, dims={c.dims}, pose={c.pose}")
        else:
            self.get_logger().info("  (none)")

        # ----- CAPSULES -----
        self.get_logger().info("\nCAPSULES:")
        if wm.capsule:
            for cap in wm.capsule:
                self.get_logger().info(f"  - name={cap.name}, dims={cap.dims}, pose={cap.pose}")
        else:
            self.get_logger().info("  (none)")

        # ----- CYLINDERS -----
        self.get_logger().info("\nCYLINDERS:")
        if wm.cylinder:
            for cyl in wm.cylinder:
                self.get_logger().info(f"  - name={cyl.name}, dims={cyl.dims}, pose={cyl.pose}")
        else:
            self.get_logger().info("  (none)")

        # ----- MESH -----
        self.get_logger().info("\nMESHES:")
        if wm.mesh:
            for m in wm.mesh:
                self.get_logger().info(f"  - name={m.name}, pose={m.pose}")
        else:
            self.get_logger().info("  (none)")

        # ----- VOXEL -----
        self.get_logger().info("\nVOXEL GRIDS:")
        if wm.voxel:
            for v in wm.voxel:
                self.get_logger().info(f"  - name={v.name}, pose={v.pose}")
        else:
            self.get_logger().info("  (none)")

        self.get_logger().info("============================================\n")

    def _start_perception_once(self):
        # run exactly once
        self.perception_timer.cancel()
        try:
            from .perception import ZedYoloPerception

            # resolve defaults safely (tiny model; no window)
            weights = os.getenv("UR10E_YOLO_WEIGHTS", "exp_aug.pt")
            imgsz   = int(os.getenv("UR10E_YOLO_IMGSZ", "640"))
            conf    = float(os.getenv("UR10E_YOLO_CONF", "0.45"))
            cam_fr  = os.getenv("UR10E_CAM_FRAME", "zed2_left_camera_frame")
            show    = bool(int(os.getenv("UR10E_SHOW_VIEW", "1")))
            cpu_only = os.getenv("CUDA_VISIBLE_DEVICES", "") == ""

            self.get_logger().info(
                f"Starting perception (weights={weights}, imgsz={imgsz}, conf={conf}, "
                f"cam={cam_fr}, show={int(show)}, cpu_only={cpu_only})"
            )

            self.perception = ZedYoloPerception(
                self,
                weights=weights,
                img_size=imgsz,
                conf_thres=conf,
                cam_frame=cam_fr,
                show_view=show,
            )
            self.perception.start()
            self.get_logger().info("Perception started.")
        except Exception as e:
            self.get_logger().error(f"Perception failed to start: [{type(e).__name__}] {e}")
            self.perception = None  # don’t crash the whole node

    def _maybe_start_perception(self):
        if os.getenv("UR10E_DISABLE_PERCEPTION", "0") == "1":
            self.get_logger().info("Perception disabled by UR10E_DISABLE_PERCEPTION=1")
            return
        # Delay startup to avoid RAM spikes colliding with cuRobo init
        self.perception_timer: Timer = self.create_timer(5.0, self._start_perception_once)
