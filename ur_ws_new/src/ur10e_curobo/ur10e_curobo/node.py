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
from std_msgs.msg import Bool, Float32MultiArray
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
from .dynamic_obstacle import DynamicObstacleManager
from .perception import ZedYoloPerception
from trajectory_msgs.msg import JointTrajectory
from ur_msgs.srv import SetIO


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
        
        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')
        #while not self.io_client.wait_for_service(timeout_sec=1.0):
            #self.get_logger().info("Waiting for /set_io service...")
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
        
        # state: keep track of robot state, path history, and goals in memory.
        self.qos = DEFAULT_QOS
        self.goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.joint_order = JOINT_ORDER
        
        self.current_joint_positions = None
        self.current_joint_velocities = None
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
        self.best_goal_score = float("inf")
        self.goal_seed_xy = None

        self.last_frames = [] # last few frames for stability checking

        self.goal_tracker_sub = self.create_subscription(
            PoseStamped,
            '/external_goal_pose',        # always listen to new poses
            self._continuous_goal_tracker,  # callback function below
            self.goal_qos
        )


        # gripper/classifier
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
        
        self.current_joint_positions = [jm[j] for j in self.joint_order if j in jm]
        self.current_joint_velocities = [vm.get(j, 0.0) for j in self.joint_order]

    def _marker_cb(self, msg): ##Stores last clicked pose
        self.latest_marker_pose = msg.pose

    def _robot_running_cb(self, msg): ##Logs robot program state
        self.robot_running = msg.data
        self.get_logger().info("✅ Robot program is running." if msg.data else "⚠️ Robot program is NOT running.")

    def _stop_cb(self, msg): ##Stops motion immediately
        if not msg.data: return
        self.get_logger().warn("🛑 Emergency stop!"); self.stop_requested = True; from .motions import publish_stop_trajectory; publish_stop_trajectory(self)

    def _force_cb(self, msg): ##Sends force readings to classifier
        self.classifier.on_force(list(msg.data)[:3])

    def _classifier_tick(self):   ##Runs periodic classifier update
        self.classifier.tick()

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

        print("[{}] Keys: y=save marker, m=manual, s=subscribe, p=capture, n=execute, o=open, c=close, h=home, d=dropoff, k=stop, q=quit".format(ts()))
        last_key = None
        while self.running:
            key = read_key()
            if key is None:
                continue

            # echo keypress
            print(f"[{ts()}] key='{key}'")

            if key == 'y':
                if self.latest_marker_pose:
                    
                    from .goals import pose_to_vec7
                    g = pose_to_vec7(self.latest_marker_pose)
                    self.goal_poses.append(g)
                    print(f"[{ts()}] saved goal #{len(self.goal_poses)} from marker: {g}")
                    
                    markers_mod.publish_goal_marker(self, g[:3])
                else:
                    print(f"[{ts()}] WARN: no latest_marker_pose yet; press 'y' again after moving the interactive marker in RViz.")

            elif key == 'm':
                print(f"[{ts()}] manual goal entry requested…")
                curr_pose = self.get_end_effector_pose()
                print(f"  current pose: {curr_pose}")
                self._manual_goal()
                print(f"[{ts()}] manual goal entry done. total goals={len(self.goal_poses)}")

            elif key == 's':
                print(f"[{ts()}] subscribing to /external_goal_pose…")
                goals_mod.subscribe_to_goal_pose(self)
                self.goal_capture_active = False 
                print(f"[{ts()}] subscribe called. waiting for external goal…")

            elif key == 'p':
                print(f"[{ts()}] starting timed goal capture (10s)…")
                self.start_goal_capture(10.0)
                print(f"[{ts()}] capture armed. current collected={getattr(self, 'goal_capture_count', 0)}")

            elif key == 'n':
                if self.goal_poses:
                    print(f"[{ts()}] executing {len(self.goal_poses)} stored goal(s)…")
                    self._prep_and_execute()
                    print(f"[{ts()}] execute finished. remaining goals={len(self.goal_poses)}")
                else:
                    print(f"[{ts()}] INFO: no goals to execute. add with 'y', 'm', or 's'.")

            elif key == 'o':
                print(f"[{ts()}] gripper → OPEN")
                gripper_mod.control_gripper(self, 'OPEN')

            elif key == 'c':
                print(f"[{ts()}] gripper → CLOSE; nudge back & rotate wrist")
                gripper_mod.control_gripper(self, 'CLOSE')
                #motions_mod.move_backward(self, -0.01)
                #motions_mod.rotate_wrist(self, 120)
                print(f"[{ts()}] post-close micro-motions done")

            elif key == 'h':
                print(f"[{ts()}] going HOME…")
                motions_mod.move_to_home_position(self)
                print(f"[{ts()}] reached HOME (or attempted)")

            elif key == 'd':
                print(f"[{ts()}] going to DROPOFF…")
                motions_mod.move_to_dropoff_position(self)
                gripper_mod.control_gripper(self, 'OPEN')
                print(f"[{ts()}] at drop-off; gripper opened")

            elif key == 'k':
                print(f"[{ts()}] STOP requested → publishing hold trajectory")
                self.stop_requested = True
                motions_mod.publish_stop_trajectory(self)

            elif key == 'q':
                print(f"[{ts()}] quitting…")
                self.running = False
                self.stop_requested = True
                rclpy.shutdown()
                break
            elif key == "f":
                print("finding current joint positions…")
                print(self.current_joint_positions)
                print("finding current end-effector pose…")
                print(self.get_end_effector_pose()) 
                
            elif key == "t":
                print("Update dynamic obstacle position:")

                pos = self.obstacles.ask_user_position("dyn_sphere")
                if pos is not None:
                    self.obstacles.update_pose("dyn_sphere", pos)
                    self.obstacles.print_world()
            else:
                # unknown key helper
                if key != last_key:  # avoid spamming if someone holds a key
                    print(f"[{ts()}] NOTE: key '{key}' has no action. valid keys: y m s p n o c h d k q")
            last_key = key


    def _manual_goal(self):
        cur = self.get_end_effector_pose()
        try:
            x = float(input("Enter X: "))
            y = float(input("Enter Y: "))
            z = float(input("Enter Z: "))
        except ValueError:
            print("Invalid input."); return
        goal = [x, y, z] + (cur[3:] if cur else [1.0,0.0,0.0,0.0])
        self.goal_poses.append(goal)
        markers_mod.publish_goal_marker(self, goal[:3]) 
        print("Manual goal saved.")

    def _continuous_goal_tracker(self, msg: PoseStamped):

        # 0. seed check
        if self.goal_seed_xy is None:
            return

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        seed_x, seed_y = self.goal_seed_xy

        # 1. Strict XY lock (prevent switching fruits)
        if math.hypot(x - seed_x, y - seed_y) > 0.045:
            return  

        # 2. Depth sanity check
        if self.best_goal_xyz and z > self.best_goal_xyz[2] + 0.025:
            return

        # 3. Distance to EE
        ee = self.get_end_effector_pose()
        if ee:
            dx = x - ee[0]
            dy = y - ee[1]
            dz = z - ee[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            dist = z

        # 4. Stability buffer (last 8 frames)
        self.last_frames.append([x, y, z])
        if len(self.last_frames) > 8:
            self.last_frames.pop(0)

        variance = np.var(self.last_frames, axis=0).sum()

        # 5. Scoring
        score = dist - variance * 2.5


        if score > self.best_goal_score + 0.002:
            self.best_goal_score = score
            self.best_goal_xyz = [x, y, z]
            print(f"Updated BEST (far) goal = {self.best_goal_xyz}, score={score}")



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

        print("\n========== CURRENT CUROBO WORLD ==========")

        # ----- SPHERES -----
        print("SPHERES:")
        if wm.sphere:
            for s in wm.sphere:
                print(f"  - name={s.name}, pose={s.pose}, radius={s.radius}")
        else:
            print("  (none)")

        # ----- CUBOIDS -----
        print("\nCUBOIDS:")
        if wm.cuboid:
            for c in wm.cuboid:
                print(f"  - name={c.name}, dims={c.dims}, pose={c.pose}")
        else:
            print("  (none)")

        # ----- CAPSULES -----
        print("\nCAPSULES:")
        if wm.capsule:
            for cap in wm.capsule:
                print(f"  - name={cap.name}, dims={cap.dims}, pose={cap.pose}")
        else:
            print("  (none)")

        # ----- CYLINDERS -----
        print("\nCYLINDERS:")
        if wm.cylinder:
            for cyl in wm.cylinder:
                print(f"  - name={cyl.name}, dims={cyl.dims}, pose={cyl.pose}")
        else:
            print("  (none)")

        # ----- MESH -----
        print("\nMESHES:")
        if wm.mesh:
            for m in wm.mesh:
                print(f"  - name={m.name}, pose={m.pose}")
        else:
            print("  (none)")

        # ----- VOXEL -----
        print("\nVOXEL GRIDS:")
        if wm.voxel:
            for v in wm.voxel:
                print(f"  - name={v.name}, pose={v.pose}")
        else:
            print("  (none)")

        print("============================================\n")

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
