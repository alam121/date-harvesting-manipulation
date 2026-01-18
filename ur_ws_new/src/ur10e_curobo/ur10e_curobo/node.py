# ruff: noqa
import json
import threading
import rclpy
import os
import numpy as np
import math

from rclpy.timer import Timer
from rclpy.qos import QoSProfile
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray, String, Float32
from geometry_msgs.msg import PoseStamped
from .config import AppConfig

from .utils import read_key
from . import fk as fk_mod
from . import markers as markers_mod
from . import motions as motions_mod
from . import goals as goals_mod
from .goals import ThreadSafeGoalList
from . import gripper as gripper_mod
from ur_msgs.srv import SetIO
from .grasp_outcome_classifier import classify_triplet

# Manager imports
from .managers import ConfigManager, StateManager, MotionExecutor


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

        # ========= PHASE 1: ConfigManager =========
        self._config_mgr = ConfigManager(self)
        self._config_mgr.initialize()

        # ========= PHASE 2: StateManager =========
        self._state_mgr = StateManager(self, self._config_mgr)
        self._state_mgr.initialize()

        # ========= PHASE 3: MotionExecutor =========
        self._motion_mgr = MotionExecutor(self, self._config_mgr, self._state_mgr)
        self._motion_mgr.initialize()

        # ======== Remaining pubs/subs (not handled by managers yet) ========

        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)
        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)

        self.create_subscription(Float32MultiArray, "/gripper/force", self._force_cb, 10)

        # GUI integration: command subscriber and info publishers
        self.create_subscription(String, "/ui_command", self._ui_command_cb, 10)
        self.velocity_scale_pub = self.create_publisher(Float32, "/velocity_scale", 10)
        self.goal_info_pub = self.create_publisher(String, "/goal_info", 10)

        # Timer to publish goal info periodically
        self.create_timer(0.2, self._publish_goal_info)  # 5Hz

        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')

        self.goal_tracker_sub = self.create_subscription(
            PoseStamped,
            '/external_goal_pose',        # always listen to new poses
            self._continuous_goal_tracker,  # callback function below
            self.goal_qos
        )

        # timers
        self.create_timer(0.1, lambda: markers_mod.track_robot_path(self)) #Track & update RViz path markers
        self.create_timer(0.02, self._classifier_tick) #Tick classifier loop (gripper ML logic)

        # Note: cuRobo, obstacles, trajectory_pub, teleop, static obstacles moved to MotionExecutor

        self.goal_poses = ThreadSafeGoalList()  # Thread-safe for ROS callback + main thread access
        # Note: _motion_lock moved to MotionExecutor

        # --- capture state ---
        self.goal_capture_active = False
        self.goal_capture_timer = None
        self.goal_pose_sub = None
        self.goal_capture_count = 0

        self.goal_sort_ref = None
        self.goal_sort_ascending = True

        self.latest_goal_pose = None        # [x, y, z, qw, qx, qy, qz]
        self.latest_goal_time = 0.0         # timestamp of last valid pos

        self.best_goal_xyz = None
        self.best_goal_score = float("inf")
        self.goal_seed_xy = None

        self.last_frames = [] # last few frames for stability checking

        # gripper/classifier
        gripper_mod.init_gripper(self, suction=False)

        # keyboard
        self.keyboard_thread = threading.Thread(target=self._wait_for_key_press, daemon=True)
        self.keyboard_thread.start()
        self.get_logger().info("UR10e cuRobo node initialized. Waiting for joint states…")

        # Note: Teleop state, subscription, and timer moved to MotionExecutor

    def reset_goal_tracking(self):
        """Reset all goal tracking state for a fresh cycle."""
        self.goal_seed_xy = None
        self.best_goal_xyz = None
        self.best_goal_score = float("inf")
        self.last_frames.clear()
        self.latest_goal_pose = None
        self.latest_goal_time = 0.0
        self.goal_received = False

    # callbacks
    # Note: _check_joint_states, _joint_state_cb, _marker_cb, _direction_cb,
    # _robot_running_cb, _stop_cb moved to StateManager

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
        """Handle commands from the GUI."""
        import json
        import threading
        cmd = msg.data.strip()
        self.get_logger().info(f"UI command received: {cmd}")

        # Run blocking motion commands in separate thread to avoid blocking ROS callbacks
        # Use mutex to prevent concurrent execution of motion commands
        def run_home():
            if not self._motion_lock.acquire(blocking=False):
                self.get_logger().warn("Motion already in progress, ignoring HOME command")
                return
            try:
                motions_mod.move_to_home_position(self)
            finally:
                self._motion_lock.release()

        def run_dropoff():
            if not self._motion_lock.acquire(blocking=False):
                self.get_logger().warn("Motion already in progress, ignoring DROPOFF command")
                return
            try:
                motions_mod.move_to_dropoff_position(self)
                gripper_mod.control_gripper(self, 'OPEN')
            finally:
                self._motion_lock.release()

        def run_execute():
            if not self._motion_lock.acquire(blocking=False):
                self.get_logger().warn("Motion already in progress, ignoring EXECUTE command")
                return
            try:
                if self.goal_poses:
                    self._prep_and_execute()
            finally:
                self._motion_lock.release()

        if cmd == "home":
            threading.Thread(target=run_home, daemon=True).start()
        elif cmd == "dropoff":
            threading.Thread(target=run_dropoff, daemon=True).start()
        elif cmd == "execute":
            threading.Thread(target=run_execute, daemon=True).start()
        elif cmd == "clear":
            self.goal_poses.clear()
            self.get_logger().info("Goals cleared")
        elif cmd == "open":
            gripper_mod.control_gripper(self, 'OPEN')
        elif cmd == "close":
            gripper_mod.control_gripper(self, 'CLOSE')
        elif cmd == "stop":
            self.stop_requested = True
            motions_mod.publish_stop_trajectory(self)
            self.goal_poses.clear()
        elif cmd.startswith("capture "):
            try:
                duration = float(cmd.split()[1])
                self.start_goal_capture(duration)
            except (ValueError, IndexError):
                self.start_goal_capture(10.0)
        elif cmd == "capture_stop":
            self.stop_goal_capture()
        elif cmd.startswith("set_speeds "):
            # Format: "set_speeds home dropoff approach predropoff"
            try:
                parts = cmd.split()
                self.cfg.planner.speed_home = float(parts[1])
                self.cfg.planner.speed_dropoff = float(parts[2])
                self.cfg.planner.speed_approach = float(parts[3])
                if len(parts) > 4:
                    self.cfg.planner.speed_predropoff = float(parts[4])
                self.get_logger().info(f"Speeds updated: home={parts[1]}, dropoff={parts[2]}, approach={parts[3]}, predropoff={self.cfg.planner.speed_predropoff}")
            except (ValueError, IndexError) as e:
                self.get_logger().warn(f"Invalid set_speeds format: {e}")
        elif cmd.startswith("set_velocity_scale "):
            # Format: "set_velocity_scale 1.5"
            try:
                scale = float(cmd.split()[1])
                scale = max(0.1, min(scale, 10.0))  # Clamp between 0.1 and 10.0
                self.cfg.planner.global_speed_multiplier = scale
                self.speed_scale = scale
                self.get_logger().info(f"Velocity scale set to {scale}")
                # Publish updated scale
                scale_msg = Float32()
                scale_msg.data = scale
                self.velocity_scale_pub.publish(scale_msg)
            except (ValueError, IndexError) as e:
                self.get_logger().warn(f"Invalid set_velocity_scale format: {e}")
        elif cmd == "debug_world":
            self.debug_print_world()
        else:
            self.get_logger().warn(f"Unknown UI command: {cmd}")

    def _publish_goal_info(self):
        """Publish goal information for GUI consumption."""
        # Convert ThreadSafeGoalList to a list safely
        if hasattr(self.goal_poses, "data"):  # check if 'data' attribute exists
            safe_goals = list(self.goal_poses.data)
        elif hasattr(self.goal_poses, "_list"):  # check for '_list'
            safe_goals = list(self.goal_poses._list)
        elif hasattr(self.goal_poses, "get_list"):  # check for 'get_list' method
            safe_goals = list(self.goal_poses.get_list())
        else:
            # Fallback for unknown implementation
            try:
                safe_goals = [g for g in self.goal_poses]  # if iteration works
            except Exception:
                safe_goals = []

        msg_data = {
            "goal_count": len(safe_goals),
            "goals": [[round(v, 4) for v in g[:3]] for g in safe_goals[:5]],  # First 5 goals, XYZ only
            "latest_goal": (
                [round(v, 4) for v in self.latest_goal_pose[:3]]
                if self.latest_goal_pose else None
            ),
            "best_goal_xyz": (
                [round(v, 4) for v in self.best_goal_xyz]
                if self.best_goal_xyz else None
            ),
            "capture_active": self.goal_capture_active,
            "capture_count": getattr(self, "goal_capture_count", 0),
            "velocity_scale": self.cfg.planner.global_speed_multiplier,
            "speed_home": self.cfg.planner.speed_home,
            "speed_dropoff": self.cfg.planner.speed_dropoff,
            "speed_approach": self.cfg.planner.speed_approach,
            "speed_predropoff": self.cfg.planner.speed_predropoff,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.goal_info_pub.publish(msg)

    # Note: _publish_static_obstacles moved to MotionExecutor

    # methods used by helpers (so helpers can call like node.get_end_effector_pose())
    def get_end_effector_pose(self):
        return self._motion_mgr.get_end_effector_pose()
    
    

    # keyboard UI
    # keyboard UI (verbose)
    def _wait_for_key_press(self):
        def ts():
            # short timestamp for prints
            import time
            return time.strftime("%H:%M:%S")

        print("[{}] Keys: y=save marker, m=manual, s=subscribe, p=capture, n=execute, o=open, c=close, h=home, d=dropoff, k=stop, z=teleop-toggle, q=quit".format(ts()))
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
                #motions_mod.rotate_wrist(self, 70, rotate_time=0.6, hold_time=0.05, return_time=0.6)
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
                # Clear any queued goals so execution loop can exit quickly
                self.goal_poses.clear()

            elif key == 'z':
                # Toggle teleop enable
                self.teleop_enabled = not getattr(self, 'teleop_enabled', False)
                if self.teleop_enabled:
                    # clear any cached twist so we don't immediately servo stale commands
                    self.latest_teleop_twist = None
                    self.last_teleop_time = 0.0
                    print(f"[{ts()}] TELEOP enabled. Waiting for incoming teleop deltas...")
                else:
                    print(f"[{ts()}] TELEOP disabled.")

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
        # Always store latest goal pose unconditionally for immediate access
        import time as _time
        self.latest_goal_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        self.latest_goal_time = _time.time()

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
        # Reject if Z deviates too much from best (both higher AND lower)
        # ------------------------------------------------------
        if self.best_goal_xyz:
            z_diff = abs(z - self.best_goal_xyz[2])
            if z_diff > 0.05:  # 5cm tolerance
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
        self._motion_mgr.debug_print_world()

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

    # Note: _teleop_cb and _teleop_servo_tick moved to MotionExecutor

    # ============ BACKWARD COMPATIBILITY PROPERTIES (Phase 1: ConfigManager) ============

    @property
    def cfg(self) -> AppConfig:
        return self._config_mgr.cfg

    @property
    def qos(self) -> QoSProfile:
        return self._config_mgr.qos

    @property
    def goal_qos(self) -> QoSProfile:
        return self._config_mgr.goal_qos

    @property
    def joint_order(self) -> list:
        return self._config_mgr.joint_order

    @property
    def home_joints(self) -> list:
        return self._config_mgr.home_joints

    @property
    def dropoff_joints(self) -> list:
        return self._config_mgr.dropoff_joints

    @property
    def predropoff_joints(self) -> list:
        return self._config_mgr.predropoff_joints

    @property
    def speed_scale(self) -> float:
        return self._config_mgr.speed_scale

    @speed_scale.setter
    def speed_scale(self, value: float):
        self._config_mgr.speed_scale = value

    @property
    def yoffset(self) -> float:
        return self._config_mgr.yoffset

    @property
    def zoffset(self) -> float:
        return self._config_mgr.zoffset

    @property
    def cam_frame(self) -> str:
        return self._config_mgr.cam_frame

    @property
    def traj_cmd_topic(self) -> str:
        return self._config_mgr.traj_cmd_topic

    @property
    def joint_states_topic(self) -> str:
        return self._config_mgr.joint_states_topic

    @property
    def goal_marker_topic(self) -> str:
        return self._config_mgr.goal_marker_topic

    @property
    def path_marker_topic(self) -> str:
        return self._config_mgr.path_marker_topic

    # ============ BACKWARD COMPATIBILITY PROPERTIES (Phase 2: StateManager) ============

    @property
    def current_joint_positions(self):
        return self._state_mgr.current_joint_positions

    @current_joint_positions.setter
    def current_joint_positions(self, value):
        self._state_mgr.current_joint_positions = value

    @property
    def current_joint_velocities(self):
        return self._state_mgr.current_joint_velocities

    @current_joint_velocities.setter
    def current_joint_velocities(self, value):
        self._state_mgr.current_joint_velocities = value

    @property
    def tf_buffer(self):
        return self._state_mgr.tf_buffer

    @property
    def running(self) -> bool:
        return self._state_mgr.running

    @running.setter
    def running(self, value: bool):
        self._state_mgr.running = value

    @property
    def stop_requested(self) -> bool:
        return self._state_mgr.stop_requested

    @stop_requested.setter
    def stop_requested(self, value: bool):
        self._state_mgr.stop_requested = value

    @property
    def robot_running(self) -> bool:
        return self._state_mgr.robot_running

    @robot_running.setter
    def robot_running(self, value: bool):
        self._state_mgr.robot_running = value

    @property
    def path_points(self) -> list:
        return self._state_mgr.path_points

    @property
    def latest_marker_pose(self):
        return self._state_mgr.latest_marker_pose

    @latest_marker_pose.setter
    def latest_marker_pose(self, value):
        self._state_mgr.latest_marker_pose = value

    @property
    def tf_warning_printed(self) -> bool:
        return self._state_mgr.tf_warning_printed

    @tf_warning_printed.setter
    def tf_warning_printed(self, value: bool):
        self._state_mgr.tf_warning_printed = value

    @property
    def tf_printed(self) -> bool:
        return self._state_mgr.tf_printed

    @tf_printed.setter
    def tf_printed(self, value: bool):
        self._state_mgr.tf_printed = value

    @property
    def fruit_direction(self):
        return self._state_mgr.fruit_direction

    @fruit_direction.setter
    def fruit_direction(self, value):
        self._state_mgr.fruit_direction = value

    # ============ BACKWARD COMPATIBILITY PROPERTIES (Phase 3: MotionExecutor) ============

    @property
    def motion_gen(self):
        return self._motion_mgr.motion_gen

    @property
    def motion_gen_config(self):
        return self._motion_mgr.motion_gen_config

    @property
    def trajectory_pub(self):
        return self._motion_mgr.trajectory_pub

    @property
    def env_marker_pub(self):
        return self._motion_mgr.env_marker_pub

    @property
    def obstacles(self):
        return self._motion_mgr.obstacles

    @property
    def static_obstacles(self):
        return self._motion_mgr.static_obstacles

    @property
    def _motion_lock(self):
        return self._motion_mgr._motion_lock

    @property
    def teleop_enabled(self) -> bool:
        return self._motion_mgr.teleop_enabled

    @teleop_enabled.setter
    def teleop_enabled(self, value: bool):
        self._motion_mgr.teleop_enabled = value

    @property
    def latest_teleop_twist(self):
        return self._motion_mgr.latest_teleop_twist

    @latest_teleop_twist.setter
    def latest_teleop_twist(self, value):
        self._motion_mgr.latest_teleop_twist = value

    @property
    def last_teleop_time(self) -> float:
        return self._motion_mgr.last_teleop_time

    @last_teleop_time.setter
    def last_teleop_time(self, value: float):
        self._motion_mgr.last_teleop_time = value
