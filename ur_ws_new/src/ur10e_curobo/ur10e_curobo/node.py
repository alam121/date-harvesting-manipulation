# ruff: noqa
import json
import copy
import threading
import rclpy
import os
import time
import numpy as np
import math
from collections import deque

from rclpy.timer import Timer
from rclpy.qos import QoSProfile
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray, String, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from .config import AppConfig, ROBOT_PROFILE, ENVIRONMENT

from .utils import read_key
from . import fk as fk_mod
from . import markers as markers_mod
from . import motions as motions_mod
from . import goals as goals_mod
from . import goal_marker as goal_marker_mod
from . import safe_zone as safe_zone_mod
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
    # Speed multiplier for "Execute Queue" goal moves, relative to the active
    # velocity scale. <1.0 makes these waypoint moves slower than normal motions.
    GOAL_MOVE_SPEED_FACTOR = 0.5

    def __init__(self):
        super().__init__(
            "ur10e_curobo_moveit_node",
            automatically_declare_parameters_from_overrides=False
        )

        # Thread-safety: lock held during plan_single / plan_single_js
        # FK checks this (non-blocking) to skip CUDA ops during graph capture
        self._planning_lock = threading.Lock()
        self._reachability_worker_lock = threading.Lock()
        self._reachability_worker_active = False
        self._reachability_clear_pending = True
        self._lidar_preview_clear_pending = True

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
        self.reachability_marker_pub = self.create_publisher(Marker, "/reachability_cloud", 10)

        self.create_subscription(Float32MultiArray, "/gripper/force", self._force_cb, 10)

        # Cache latest heatmap 3D data from vision (for goal marker rendering)
        self._latest_heatmap_data = None
        self.create_subscription(
            Float32MultiArray, "/vision/heatmap_3d_data",
            self._heatmap_data_cb, 10
        )

        # GUI integration: command subscriber and info publishers
        self.create_subscription(String, "/ui_command", self._ui_command_cb, 10)
        self.create_subscription(Image, "/vision/raw", self._camera_raw_image_cb, 10)
        self.create_subscription(Image, "/vision/display", self._camera_display_image_cb, 10)
        self.velocity_scale_pub = self.create_publisher(Float32, "/velocity_scale", 10)
        self.goal_info_pub = self.create_publisher(String, "/goal_info", 10)
        self.exclude_pub = self.create_publisher(Float32MultiArray, "/exclude_fruit_positions", 10)
        self._vision_mode_pub = self.create_publisher(String, "/vision/mode", 10)
        # GPU handoff handshake: the vision node confirms it has paused and drained
        # any in-flight YOLO inference by publishing "paused" on /vision/mode_state.
        self._vision_mode_state = None
        self._vision_paused_event = threading.Event()
        self.create_subscription(String, "/vision/mode_state", self._vision_mode_state_cb, 10)
        self.calib_check_pub = self.create_publisher(String, "/calib_check_result", 10)
        # Active robot type + environment for the RViz panel. Republished periodically so
        # the panel shows it regardless of who started first.
        self.robot_config_pub = self.create_publisher(String, "/robot_config_info", 10)
        self.create_timer(2.0, lambda: self.robot_config_pub.publish(
            String(data=f"Robot: {ROBOT_PROFILE}  |  Env: {ENVIRONMENT}")))

        # Timer to publish goal info periodically
        self.create_timer(0.2, self._publish_goal_info)  # 5Hz

        self.io_client = self.create_client(SetIO, '/io_and_status_controller/set_io')

        self.goal_tracker_sub = self.create_subscription(
            PoseStamped,
            '/external_goal_pose',        # always listen to new poses
            self._continuous_goal_tracker,  # callback function below
            self.goal_qos
        )
        self.create_subscription(
            PoseStamped,
            '/manual_goal_pose',
            self._manual_goal_pose_cb,
            self.goal_qos,
        )
        self.create_subscription(
            PointStamped,
            '/clicked_point',
            self._clicked_reachability_point_cb,
            10,
        )

        # Subscribe to fruit radius from vision for adaptive gripper
        self.create_subscription(
            Float32, '/fruit_radius',
            lambda msg: setattr(self, 'latest_fruit_radius', msg.data),
            10
        )

        # Subscribe to all visible fruit positions, score-sorted by vision.
        # Reacquire searches all entries; goal multi queues the top entries.
        # Format: flat [x0,y0,z0, x1,y1,z1, ...] in base_link frame.
        def _all_fruits_cb(msg):
            data = msg.data
            self.all_fruit_poses = [
                [data[i], data[i+1], data[i+2]]
                for i in range(0, len(data) - 2, 3)
            ]
        self.create_subscription(Float32MultiArray, '/vision/all_fruit_poses', _all_fruits_cb, 10)

        # timers
        self.create_timer(0.1, lambda: markers_mod.track_robot_path(self)) #Track & update RViz path markers
        self.create_timer(0.02, self._classifier_tick) #Tick classifier loop (gripper ML logic)

        # Note: cuRobo, obstacles, trajectory_pub, teleop, static obstacles moved to MotionExecutor

        self.goal_poses = ThreadSafeGoalList()  # Thread-safe for ROS callback + main thread access
        # Note: _motion_lock moved to MotionExecutor

        self.goal_received = False

        # --- capture state ---
        self.goal_capture_active = False
        self.goal_capture_timer = None
        self.goal_pose_sub = None
        self.goal_capture_count = 0

        self.goal_sort_ref = None
        self.goal_sort_ascending = True

        self.latest_goal_pose = None        # [x, y, z, qw, qx, qy, qz]
        self.latest_goal_time = 0.0         # timestamp of last valid pos
        self.latest_fruit_radius = None     # estimated fruit radius from vision (meters)
        self.all_fruit_poses = []           # list of [x,y,z] for ALL visible fruits (not just best)

        self.motion_phase: str = "IDLE"                   # current robot phase for GUI
        self.grasp_history: deque = deque(maxlen=15)      # last 15 grasp outcomes for GUI
        self.reacquire_result: str = ""                   # "OK" | "NUDGE" | "FAIL" | ""

        self.best_goal_xyz = None
        self.best_goal_score = float("inf")
        self.goal_seed_xy = None
        self._last_goal_queue_items = []
        self._last_goal_queue_metadata = {}
        self._home_joints_display = ""

        self.last_frames = [] # last few frames for stability checking

        # gripper/classifier
        gripper_mod.init_gripper(self, suction=None)  # uses config value

        # grasp learning
        from .grasp_learner import GraspLearner
        self.grasp_learner = GraspLearner(grasp_cfg=self.cfg.grasp)
        self.pending_grasp_record = None
        self._grasp_feedback = None  # set by GUI: True=success, False=fail, None=pending

        # System control publishers
        self._refresh_camera_pub = self.create_publisher(String, "/camera_command", 10)
        self._camera_lock = threading.Lock()
        self._camera_latest_raw_msg = None
        self._camera_latest_display_msg = None
        self._camera_bridge = None
        self._camera_video_writer = None
        self._camera_video_path = ""
        self._camera_video_recording = False
        self._camera_video_frames = 0
        self._camera_video_fps = 15.0

        # perception disabled - using external date_v1.9.py instead
        self.perception = None

        # Draggable RViz goal marker (3D-viewport counterpart to the panel buttons)
        goal_marker_mod.setup_goal_marker(self)
        safe_zone_mod.setup_safe_zone(self)
        self._reachability_sample_pattern = None
        self._latest_reachability_samples = []
        self._reachability_goal_metadata = {}
        self.create_timer(
            max(0.25, float(getattr(
                self.cfg.planner, "reachability_cloud_period_s", 1.0))),
            self._schedule_reachability_cloud_publish,
        )

        # keyboard
        self.keyboard_thread = threading.Thread(target=self._wait_for_key_press, daemon=True)
        self.keyboard_thread.start()
        self.get_logger().info("UR10e cuRobo node initialized. Waiting for joint states…")

        # Note: Perception is handled by external date_v1.9.py node
        # Voxel obstacles subscribe to /zed_depth_pointcloud from that node

        # Note: Teleop state, subscription, and timer moved to MotionExecutor

    def set_vision_mode(self, mode: str):
        """Switch the vision node between 'full' (approach) and 'reacquire' modes."""
        if mode == "paused":
            # Arm the handshake before commanding pause so we only accept a fresh
            # "paused" ack that arrives after this request.
            self._vision_paused_event.clear()
        msg = String()
        msg.data = mode
        self._vision_mode_pub.publish(msg)

    def _vision_mode_state_cb(self, msg: String):
        """Handle the vision node's effective-state report (GPU handoff ack)."""
        self._vision_mode_state = msg.data
        if msg.data == "paused":
            self._vision_paused_event.set()

    def wait_for_vision_paused(self, timeout: float = 0.30) -> bool:
        """Block until the vision node confirms it has paused and drained in-flight
        YOLO inference, freeing the GPU for cuRobo. Returns True on confirmation,
        False on timeout (caller should then fall back to a fixed delay)."""
        ev = getattr(self, "_vision_paused_event", None)
        if ev is None:
            return False
        return ev.wait(timeout=timeout)

    def _reachability_offsets(self, samples: int, cap_rad: float):
        """Deterministic joint-offset samples around zero, scaled by max joint delta."""
        cached = getattr(self, "_reachability_sample_pattern", None)
        if cached is not None and cached[0] == samples:
            pattern = cached[1]
        else:
            rng = np.random.default_rng(7)
            raw = rng.normal(size=(max(samples - 13, 0), len(self.joint_order)))
            norm = np.max(np.abs(raw), axis=1, keepdims=True)
            norm[norm < 1e-6] = 1.0
            radii = rng.uniform(0.15, 1.0, size=(raw.shape[0], 1))
            pattern = (raw / norm) * radii
            axes = [np.zeros(len(self.joint_order))]
            for j in range(len(self.joint_order)):
                for sign in (-1.0, 1.0):
                    v = np.zeros(len(self.joint_order))
                    v[j] = sign
                    axes.append(v)
            pattern = np.vstack([np.array(axes), pattern])[:samples]
            self._reachability_sample_pattern = (samples, pattern)
        return pattern * cap_rad

    @staticmethod
    def _quat_rotate_vec(q, v):
        qw, qx, qy, qz = [float(x) for x in q]
        vx, vy, vz = [float(x) for x in v]
        # q * [0, v] * q^-1, expanded to avoid an extra dependency.
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return [
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        ]

    def _schedule_reachability_cloud_publish(self):
        if self._reachability_worker_active:
            return
        self._reachability_worker_active = True

        def _run():
            try:
                self._publish_reachability_cloud()
            finally:
                self._reachability_worker_active = False

        threading.Thread(target=_run, daemon=True).start()

    def _publish_reachability_cloud(self):
        planner = self.cfg.planner
        if not bool(getattr(planner, "reachability_cloud_enabled", True)):
            if getattr(self, "_reachability_clear_pending", True):
                markers_mod.publish_reachability_cloud(self, [], [])
                self._reachability_clear_pending = False
            if getattr(self, "_lidar_preview_clear_pending", True):
                try:
                    markers_mod.publish_lidar_scan_preview(self, [], valid=False)
                except Exception as e:
                    self.get_logger().debug(f"Lidar scan preview clear skipped: {e}")
                self._lidar_preview_clear_pending = False
            return
        if not getattr(self, "joint_order", None):
            return
        if self.current_joint_positions is None:
            return
        if getattr(self, "motion_phase", "IDLE") != "IDLE":
            return
        lock = getattr(self, "_planning_lock", None)
        if lock is not None and lock.locked():
            return

        current = np.array(self.current_joint_positions, dtype=float)
        max_delta_deg = float(getattr(
            planner,
            "reachability_cloud_max_delta_deg",
            getattr(planner, "goal_reachability_skip_delta_deg", 100.0)))
        max_delta_rad = math.radians(max_delta_deg)
        samples = max(16, int(getattr(planner, "reachability_cloud_samples", 320)))

        offsets = self._reachability_offsets(samples, max_delta_rad)
        joint_samples = (current.reshape(1, -1) + offsets).tolist()
        poses = fk_mod.forward_kinematics_pose_batch(self, joint_samples)
        if not poses:
            return
        points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            for p in poses
        ]

        deltas = np.max(np.abs(offsets), axis=1)
        keep_points = []
        keep_deltas = []
        keep_samples = []
        keep_directions = []
        in_zone = getattr(self, "in_safe_zone", lambda _xyz: True)
        green_cap = float(getattr(
            planner, "safe_zone_verified_interp_max_delta_deg", 80.0))
        yellow_cap = float(getattr(
            planner, "shortest_ik_plan_max_delta_deg", 80.0))
        valid_direct = [False for _ in joint_samples]
        green_indices = [
            i for i, d in enumerate(deltas)
            if math.degrees(float(d)) <= green_cap
        ]
        if green_indices and bool(getattr(
                planner, "reachability_cloud_validate_green", True)):
            green_targets = [joint_samples[i] for i in green_indices]
            green_valid = motions_mod.validate_joint_interpolations_batch(
                self,
                current.tolist(),
                green_targets,
                max_delta_deg=green_cap,
                step_deg=float(getattr(
                    planner, "safe_zone_verified_interp_step_deg", 1.0)),
            )
            for i, ok in zip(green_indices, green_valid):
                valid_direct[i] = bool(ok)
        else:
            for i in green_indices:
                valid_direct[i] = True

        direction_axis = list(getattr(
            planner, "reachability_direction_axis", [0.0, 0.0, 1.0]))
        direction_stride = max(1, int(getattr(
            planner, "reachability_direction_stride", 4)))
        show_directions = bool(getattr(
            planner, "reachability_direction_enabled", True))
        green_seen = 0
        for i, (pose, p, d, joints) in enumerate(zip(poses, points, deltas, joint_samples)):
            xyz = [p.x, p.y, p.z]
            if not in_zone(xyz):
                continue
            keep_points.append(p)
            delta_deg = math.degrees(float(d))
            display_delta_deg = delta_deg
            if delta_deg <= green_cap and not valid_direct[i]:
                display_delta_deg = max(yellow_cap + 1.0, delta_deg)
            keep_deltas.append(delta_deg)
            keep_samples.append({
                "xyz": xyz,
                "pose": list(pose[:7]),
                "delta_deg": delta_deg,
                "display_delta_deg": display_delta_deg,
                "valid_direct": bool(valid_direct[i]),
                "joints": list(joints),
            })
            if display_delta_deg <= green_cap and valid_direct[i]:
                green_seen += 1
                if show_directions and (green_seen - 1) % direction_stride == 0:
                    keep_directions.append(
                        self._quat_rotate_vec(pose[3:7], direction_axis))
                else:
                    keep_directions.append(None)
            else:
                keep_directions.append(None)
        self._latest_reachability_samples = keep_samples
        self._reachability_clear_pending = True
        self._lidar_preview_clear_pending = True
        markers_mod.publish_reachability_cloud(
            self,
            keep_points,
            [s["display_delta_deg"] for s in keep_samples],
            keep_directions)
        try:
            from . import lidar_scan as lidar_scan_mod
            lidar_scan_mod.publish_lidar_scan_preview(self)
        except Exception as e:
            self.get_logger().debug(f"Lidar scan preview update skipped: {e}")

    def _clicked_reachability_point_cb(self, msg: PointStamped):
        """Queue the nearest green reachability-cloud sample from an RViz click."""
        if msg.header.frame_id and msg.header.frame_id != "base_link":
            self.get_logger().warn(
                f"Reachability click must be in base_link, got {msg.header.frame_id!r}.")
            return
        samples = list(getattr(self, "_latest_reachability_samples", []) or [])
        if not samples:
            self.get_logger().warn(
                "Reachability click ignored: no reachability cloud samples yet.")
            return

        clicked = [float(msg.point.x), float(msg.point.y), float(msg.point.z)]
        best = min(
            samples,
            key=lambda s: math.dist(clicked, s["xyz"]),
        )
        dist = math.dist(clicked, best["xyz"])
        planner = self.cfg.planner
        max_dist = float(getattr(planner, "reachability_click_max_distance_m", 0.06))
        if dist > max_dist:
            self.get_logger().warn(
                f"Reachability click ignored: nearest sample is {dist*100:.1f}cm away "
                f"(limit {max_dist*100:.1f}cm).")
            return

        green_cap = float(getattr(
            planner, "safe_zone_verified_interp_max_delta_deg", 80.0))
        if (
            bool(getattr(planner, "reachability_click_green_only", True))
            and (
                best.get("display_delta_deg", best["delta_deg"]) > green_cap
                or not bool(best.get("valid_direct", True))
            )
        ):
            reason = (
                f"delta {best['delta_deg']:.1f}deg exceeds green cap {green_cap:.0f}deg"
                if best.get("display_delta_deg", best["delta_deg"]) > green_cap
                else "cuRobo direct-path validation failed"
            )
            self.get_logger().warn(
                f"Reachability click ignored: nearest sample is not green-valid "
                f"({reason}).")
            return
        if self.current_joint_positions is None:
            self.get_logger().warn("Reachability click ignored: joint state unavailable.")
            return
        valid_now = motions_mod.validate_joint_interpolations_batch(
            self,
            list(self.current_joint_positions),
            [best["joints"]],
            max_delta_deg=green_cap,
            step_deg=float(getattr(
                planner, "safe_zone_verified_interp_step_deg", 1.0)),
        )
        if not valid_now or not valid_now[0]:
            self.get_logger().warn(
                "Reachability click ignored: selected sample no longer has a "
                "cuRobo-valid direct path from the current posture.")
            return

        pose = fk_mod.pose_from_joints(self, best["joints"])
        if not pose or len(pose) < 7:
            self.get_logger().warn("Reachability click ignored: FK pose unavailable.")
            return
        goal = [float(v) for v in pose[:7]]
        if self.goal_poses.any_within_distance(goal[:3], 0.01):
            self.get_logger().info(
                "Reachability click already queued (within 1cm) — skipping")
            return
        self.latest_goal_pose = list(goal)
        self.goal_poses.append(goal)
        self._reachability_goal_metadata[self._goal_meta_key(goal)] = {
            "joints": list(best["joints"]),
            "delta_deg": float(best["delta_deg"]),
        }
        markers_mod.publish_goal_marker(self, goal[:3])
        self.get_logger().info(
            f"Queued goal #{len(self.goal_poses)} from reachability click: "
            f"[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}] "
            f"(delta={best['delta_deg']:.1f}deg, snap={dist*100:.1f}cm)")

    @staticmethod
    def _goal_meta_key(goal):
        return tuple(round(float(v), 4) for v in goal[:7])

    @staticmethod
    def _is_gripper_queue_item(item):
        return isinstance(item, dict) and item.get("type") == "gripper"

    @staticmethod
    def _is_pose_queue_item(item):
        return isinstance(item, (list, tuple)) and len(item) >= 7

    def _snapshot_goal_queue(self, reason: str = "manual"):
        items = self.goal_poses.snapshot()
        if not items:
            return False
        self._last_goal_queue_items = copy.deepcopy(items)
        metadata = {}
        for item in items:
            if not self._is_pose_queue_item(item):
                continue
            key = self._goal_meta_key(item)
            meta = getattr(self, "_reachability_goal_metadata", {}).get(key)
            if meta is not None:
                metadata[key] = copy.deepcopy(meta)
        self._last_goal_queue_metadata = metadata
        self.get_logger().info(
            f"Saved last goal queue ({len(items)} item(s), reason={reason}).")
        return True

    def _restore_last_goal_queue(self):
        items = copy.deepcopy(getattr(self, "_last_goal_queue_items", []) or [])
        if not items:
            self.get_logger().warn("No previous goal queue to reuse.")
            return
        self.goal_poses.clear()
        getattr(self, "_reachability_goal_metadata", {}).clear()
        markers_mod.clear_goal_markers(self)
        for item in items:
            self.goal_poses.append(item)
            if self._is_pose_queue_item(item):
                key = self._goal_meta_key(item)
                meta = getattr(self, "_last_goal_queue_metadata", {}).get(key)
                if meta is not None:
                    self._reachability_goal_metadata[key] = copy.deepcopy(meta)
                markers_mod.publish_goal_marker(self, item[:3])
        self.get_logger().info(
            f"Restored previous goal queue ({len(items)} item(s)).")

    def _queue_gripper_action(self, action: str, position: str):
        action = action.upper()
        if action not in ("OPEN", "CLOSE"):
            self.get_logger().warn(f"Invalid queued gripper action: {action}")
            return
        item = {"type": "gripper", "action": action}
        if position == "front":
            items = self.goal_poses.snapshot()
            self.goal_poses.clear()
            self.goal_poses.append(item)
            for existing in items:
                self.goal_poses.append(existing)
        else:
            self.goal_poses.append(item)
        where = "start" if position == "front" else "end"
        self.get_logger().info(
            f"Queued gripper {action.lower()} at {where} of current goal queue.")
        self._snapshot_goal_queue(f"queue_gripper_{action.lower()}")

    def _show_home_joints(self):
        joints = [float(v) for v in self.home_joints]
        deg = [math.degrees(v) for v in joints]
        rad_str = ", ".join(f"{v:.4f}" for v in joints)
        deg_str = ", ".join(f"{v:.1f}" for v in deg)
        self._home_joints_display = f"rad=[{rad_str}] deg=[{deg_str}]"
        self.get_logger().info(f"Current HOME joints: {self._home_joints_display}")

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

    def _heatmap_data_cb(self, msg):
        """Cache latest heatmap 3D points from vision node."""
        self._latest_heatmap_data = list(msg.data)

    def _force_cb(self, msg): ##Sends force readings to classifier
        if not hasattr(self, 'classifier'):
            return
        forces = list(msg.data)[:3]
        self.classifier.on_force(forces)
        if hasattr(self, "visualizer"):
            self.visualizer.update_forces(forces)
            tpl = classify_triplet(forces)
            self.visualizer.update_classifier(self.classifier.phase, tpl)

    def _classifier_tick(self):   ##Runs periodic classifier update
        if hasattr(self, 'classifier'):
            self.classifier.tick()

    def _camera_output_dir(self):
        path = os.path.expanduser("~/camera_recordings")
        os.makedirs(path, exist_ok=True)
        return path

    def _get_camera_bridge(self):
        if self._camera_bridge is None:
            from cv_bridge import CvBridge
            self._camera_bridge = CvBridge()
        return self._camera_bridge

    def _camera_msg_to_bgr(self, msg):
        bridge = self._get_camera_bridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _record_camera_frame_locked(self, msg):
        try:
            import cv2
            frame = self._camera_msg_to_bgr(msg)
            if self._camera_video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._camera_video_writer = cv2.VideoWriter(
                    self._camera_video_path,
                    fourcc,
                    self._camera_video_fps,
                    (w, h),
                )
                if not self._camera_video_writer.isOpened():
                    self.get_logger().error(
                        f"Camera video: failed to open {self._camera_video_path}")
                    self._camera_video_recording = False
                    self._camera_video_writer = None
                    return
            self._camera_video_writer.write(frame)
            self._camera_video_frames += 1
        except Exception as e:
            self.get_logger().error(f"Camera video recording failed: {e}")
            if self._camera_video_writer is not None:
                self._camera_video_writer.release()
            self._camera_video_writer = None
            self._camera_video_recording = False

    def _camera_raw_image_cb(self, msg):
        with self._camera_lock:
            self._camera_latest_raw_msg = msg
            if self._camera_video_recording:
                self._record_camera_frame_locked(msg)

    def _camera_display_image_cb(self, msg):
        with self._camera_lock:
            self._camera_latest_display_msg = msg

    def _save_camera_snapshot(self):
        with self._camera_lock:
            msg = self._camera_latest_raw_msg
            source = "raw"
            if msg is None:
                msg = self._camera_latest_display_msg
                source = "display"
        if msg is None:
            self.get_logger().warn(
                "Camera snapshot: no /vision/raw or /vision/display frame received yet")
            return
        try:
            import cv2
            frame = self._camera_msg_to_bgr(msg)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._camera_output_dir(), f"camera_{source}_{stamp}.png")
            if not cv2.imwrite(path, frame):
                self.get_logger().error(f"Camera snapshot: failed to save {path}")
                return
            if source != "raw":
                self.get_logger().warn(
                    "Camera snapshot used /vision/display fallback; restart the vision "
                    "node to enable raw /vision/raw snapshots.")
            self.get_logger().info(f"Camera {source} snapshot saved: {path}")
        except Exception as e:
            self.get_logger().error(f"Camera snapshot failed: {e}")

    def _start_camera_video_recording(self):
        with self._camera_lock:
            if self._camera_video_recording:
                self.get_logger().warn(
                    f"Camera video already recording: {self._camera_video_path}")
                return
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._camera_video_path = os.path.join(
                self._camera_output_dir(), f"camera_raw_video_{stamp}.mp4")
            self._camera_video_writer = None
            self._camera_video_frames = 0
            self._camera_video_recording = True
        self.get_logger().info(
            f"Raw camera video recording started: {self._camera_video_path}")

    def _stop_camera_video_recording(self):
        with self._camera_lock:
            if not self._camera_video_recording and self._camera_video_writer is None:
                self.get_logger().warn("Camera video: not recording")
                return
            path = self._camera_video_path
            frames = self._camera_video_frames
            writer = self._camera_video_writer
            self._camera_video_recording = False
            self._camera_video_writer = None
            self._camera_video_path = ""
            self._camera_video_frames = 0
        try:
            if writer is not None:
                writer.release()
        finally:
            self.get_logger().info(
                f"Camera video recording stopped: {path} ({frames} frames)")

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
                if getattr(self, "motion_phase", "") == "LIDAR_SCAN_HOME":
                    self.get_logger().warn(
                        "Already returning HOME after LiDAR scan; ignoring duplicate HOME.")
                    return
                self.stop_requested = True
                self.goal_poses.clear()
                getattr(self, "_reachability_goal_metadata", {}).clear()
                motions_mod.publish_stop_trajectory(self)
                self.get_logger().warn(
                    "Motion already in progress; requested stop for HOME. "
                    "Press HOME again after the active planner returns.")
                return
            try:
                self.stop_requested = False  # clear any prior stop before homing
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
                    self._snapshot_goal_queue("execute")
                    if any(
                        self._is_gripper_queue_item(item)
                        for item in self.goal_poses.snapshot()
                    ):
                        self.get_logger().info(
                            "Queue contains gripper actions — using ordered queue executor.")
                        self._execute_goals_no_grasp()
                    else:
                        self._prep_and_execute()
            finally:
                self._motion_lock.release()

        def run_execute_moves():
            if not self._motion_lock.acquire(blocking=False):
                self.get_logger().warn(
                    "Motion already in progress, ignoring EXECUTE MOVES command")
                return
            try:
                self._snapshot_goal_queue("execute_moves")
                self._execute_goals_no_grasp()
            finally:
                self._motion_lock.release()

        def run_side_home(which):
            # Manually move to a stored side-home joint config (for testing/calibrating
            # home_left/home_right per robot profile). Same path as the auto side approach.
            label = which.upper()
            attr = {
                "home_left": "home_left_joints",
                "home_right": "home_right_joints",
                "home_left_low": "home_left_low_joints",
                "home_right_low": "home_right_low_joints",
            }[which]
            if not self._motion_lock.acquire(blocking=False):
                self.get_logger().warn(f"Motion already in progress, ignoring {label} command")
                return
            try:
                self.stop_requested = False
                target = list(getattr(self, attr))
                cur = self.current_joint_positions
                if cur is not None:
                    target = motions_mod.nearest_joint_config(list(cur), target)
                ok = motions_mod.plan_execute_js(
                    self, target, label=label, motion_type="home", speed_factor=0.5)
                if not ok:
                    self.get_logger().warn(
                        f"{label}: move failed — cuRobo could not plan to this config "
                        f"(likely in collision or unreachable for the active robot profile). "
                        f"target joints={[round(v, 4) for v in target]}")
            finally:
                self._motion_lock.release()

        if cmd == "home":
            threading.Thread(target=run_home, daemon=True).start()
        elif cmd == "dropoff":
            threading.Thread(target=run_dropoff, daemon=True).start()
        elif cmd == "execute":
            threading.Thread(target=run_execute, daemon=True).start()
        elif cmd == "execute_moves":
            threading.Thread(target=run_execute_moves, daemon=True).start()
        elif cmd in ("home_left", "home_right", "home_left_low", "home_right_low"):
            threading.Thread(target=lambda c=cmd: run_side_home(c), daemon=True).start()
        elif cmd == "add_current_goal":
            self._add_current_as_goal()
        elif cmd == "reuse_last_goal_queue":
            self._restore_last_goal_queue()
        elif cmd in ("queue_gripper_open", "queue_gripper_open_start"):
            self._queue_gripper_action("OPEN", "back")
        elif cmd in ("queue_gripper_close", "queue_gripper_close_end"):
            self._queue_gripper_action("CLOSE", "back")
        elif cmd == "show_home_joints":
            self._show_home_joints()
        elif cmd == "clear":
            self._snapshot_goal_queue("clear")
            self.goal_poses.clear()
            getattr(self, "_reachability_goal_metadata", {}).clear()
            markers_mod.clear_goal_markers(self)
            markers_mod.clear_path_markers(self)
            markers_mod.clear_plan_preview(self)
            self.get_logger().info("Goals cleared")
        elif cmd == "open":
            gripper_mod.control_gripper(self, 'OPEN')
        elif cmd == "close":
            gripper_mod.control_gripper(self, 'CLOSE')
        elif cmd == "stop":
            self.stop_requested = True
            motions_mod.publish_stop_trajectory(self)
            self.goal_poses.clear()
            getattr(self, "_reachability_goal_metadata", {}).clear()
            markers_mod.clear_goal_markers(self)
            markers_mod.clear_path_markers(self)
            markers_mod.clear_plan_preview(self)
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
        elif cmd == "subscribe":
            goals_mod.subscribe_to_goal_pose(self)
            self.goal_capture_active = False
            self.get_logger().info("Subscribed to /external_goal_pose (via GUI)")
        elif cmd.startswith("subscribe_multi"):
            try:
                parts = cmd.split()
                max_goals = int(parts[1]) if len(parts) > 1 else 3
                max_goals = max(1, min(max_goals, 10))
            except (ValueError, IndexError):
                max_goals = 3
            goals_mod.subscribe_multi_goals(self, max_goals=max_goals)
            self.goal_capture_active = False
            self.get_logger().info(f"Subscribe multi: collecting up to {max_goals} goals in 10s")
        elif cmd == "update_voxel":
            self._update_voxel_snapshot()
        elif cmd == "set_home_current":
            self._set_current_as_home()
        elif cmd == "set_dropoff_current":
            self._set_current_as_dropoff()
        elif cmd == "safe_zone_enable":
            getattr(self, "enable_safe_zone", lambda: None)()
        elif cmd == "safe_zone_disable":
            getattr(self, "disable_safe_zone", lambda: None)()
        elif cmd == "safe_zone_snap":
            getattr(self, "snap_safe_zone", lambda: None)()
        elif cmd == "safe_zone_snap_deep":
            getattr(self, "snap_deep_safe_zone", lambda: None)()
        elif cmd == "grasp_success":
            self._grasp_feedback = True
            self.get_logger().info("Grasp feedback: SUCCESS")
        elif cmd == "grasp_fail":
            self._grasp_feedback = False
            self.get_logger().info("Grasp feedback: FAIL")
        elif cmd == "exit":
            self.get_logger().info("Exit requested from GUI — killing all nodes")
            import os, signal, subprocess, time as _time
            # Kill the entire launch process group (all terminator panes)
            subprocess.Popen(["pkill", "-f", "ros2"])
            subprocess.Popen(["pkill", "-f", "rviz2"])
            _time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGKILL)
        elif cmd == "refresh_main":
            self.get_logger().info("Refresh requested — restarting main node")
            import os, sys
            os.execvp(sys.executable, [sys.executable] + sys.argv)
        elif cmd == "refresh_camera":
            self._refresh_camera_pub.publish(String(data="refresh"))
            self.get_logger().info("Camera refresh requested")
        elif cmd == "camera_preset_lab":
            self._refresh_camera_pub.publish(String(data="preset lab"))
            self.get_logger().info("Camera exposure preset requested: lab")
        elif cmd == "camera_preset_outdoor":
            self._refresh_camera_pub.publish(String(data="preset outdoor"))
            self.get_logger().info("Camera exposure preset requested: outdoor")
        elif cmd.startswith("camera_settings "):
            self._refresh_camera_pub.publish(String(data=cmd.replace("camera_", "", 1)))
            self.get_logger().info(f"Camera settings requested: {cmd}")
        elif cmd.startswith("camera_model "):
            self._refresh_camera_pub.publish(String(data=cmd.replace("camera_", "", 1)))
            self.get_logger().info(f"Camera model requested: {cmd}")
        elif cmd == "camera_snapshot":
            self._save_camera_snapshot()
        elif cmd == "camera_video_start":
            self._start_camera_video_recording()
        elif cmd == "camera_video_stop":
            self._stop_camera_video_recording()
        elif cmd == "plan_confirm":
            self._state_mgr.plan_confirmed = True
            self._state_mgr.plan_confirm_event.set()
            self.get_logger().info("Plan confirmed (GUI)")
        elif cmd == "plan_cancel":
            self._state_mgr.plan_confirmed = False
            self._state_mgr.plan_confirm_event.set()
            self.get_logger().info("Plan cancelled (GUI)")
        elif cmd.startswith("set_debug_preview "):
            val = cmd.split()[1].lower()
            self.cfg.planner.debug_plan_preview = val in ("true", "1", "yes")
            self.get_logger().info(f"Debug plan preview set to {self.cfg.planner.debug_plan_preview}")
        elif cmd.startswith("set_reachability_cloud "):
            val = cmd.split()[1].lower()
            enabled = val in ("true", "1", "yes", "on")
            self.cfg.planner.reachability_cloud_enabled = enabled
            if not enabled:
                self._latest_reachability_samples = []
                self._reachability_clear_pending = True
                self._lidar_preview_clear_pending = True
                markers_mod.publish_reachability_cloud(self, [], [])
                markers_mod.publish_lidar_scan_preview(self, [], valid=False)
                self._reachability_clear_pending = False
                self._lidar_preview_clear_pending = False
            else:
                self._reachability_clear_pending = True
                self._schedule_reachability_cloud_publish()
            self.get_logger().info(
                f"Reachability cloud {'enabled' if enabled else 'disabled'}")
        elif cmd.startswith("set_lidar_scan_preview "):
            val = cmd.split()[1].lower()
            enabled = val in ("true", "1", "yes", "on")
            self.cfg.lidar_scan.semicircle_preview_enabled = enabled
            if not enabled:
                self._lidar_preview_clear_pending = True
                markers_mod.publish_lidar_scan_preview(self, [], valid=False)
                self._lidar_preview_clear_pending = False
            self.get_logger().info(
                f"LiDAR scan preview {'enabled' if enabled else 'disabled'}")
        elif cmd == "debug_world":
            self.debug_print_world()
        elif cmd == "check_calibration":
            threading.Thread(target=self._run_calib_check, daemon=True).start()
        elif cmd == "lidar_scan":
            def run_lidar_scan_cmd():
                if not self._motion_lock.acquire(blocking=False):
                    self.get_logger().warn("Motion already in progress, ignoring LIDAR_SCAN command")
                    return
                try:
                    from . import lidar_scan as lidar_scan_mod
                    lidar_scan_mod.run_lidar_scan(self)
                finally:
                    self._motion_lock.release()
            threading.Thread(target=run_lidar_scan_cmd, daemon=True).start()
        else:
            self.get_logger().warn(f"Unknown UI command: {cmd}")

    def _run_calib_check(self):
        """
        Check calibration accuracy using the trunk as a static reference.

        The trunk position in base_link (from vision) should match the trunk
        position computed by transforming the raw camera-frame point through
        the current T_gripper2cam calibration manually.

        We compare:
          A) node.trunk_xyz  — already transformed by the live TF chain
          B) A 10-sample average of /trunk_position over ~2 seconds

        Then we report the stability (std) as a proxy for calibration quality:
        a well-calibrated system gives a consistent trunk position regardless
        of arm motion. We also report the XYZ value so the user can compare
        against a known physical measurement.
        """
        import math, time
        import numpy as np
        from std_msgs.msg import String as StdString

        def pub(text):
            self.calib_check_pub.publish(StdString(data=text))
            self.get_logger().info(f"[CALIB CHECK] {text}")

        pub("RUNNING — collecting trunk samples for 5s, keep arm still...")

        from geometry_msgs.msg import PointStamped as _PointStamped
        samples_base = []   # trunk in base_link (full chain: ZED Mini + T_CAM_ZEDMINI + hand-eye)
        samples_cam  = []   # trunk in ZED One cam frame (ZED Mini + T_CAM_ZEDMINI only)

        _cam_sub_data = []
        def _cam_cb(msg):
            _cam_sub_data.append((msg.point.x, msg.point.y, msg.point.z))

        _cam_sub = self.create_subscription(
            _PointStamped, "/trunk_position_cam", _cam_cb, 10
        )

        t_end = time.time() + 5.0
        while time.time() < t_end:
            xyz = self.trunk_xyz
            if xyz is not None:
                samples_base.append(xyz)
            if _cam_sub_data:
                samples_cam.append(_cam_sub_data[-1])
            time.sleep(0.1)

        self.destroy_subscription(_cam_sub)

        def _report(samples, label):
            if len(samples) < 5:
                return None, f"{label}: FAIL — not enough samples ({len(samples)})"
            arr = np.array(samples)
            mean_xyz = arr.mean(axis=0)
            dists = np.linalg.norm(arr - mean_xyz, axis=1) * 1000
            rms_err = float(np.sqrt(np.mean(dists**2)))
            max_err = float(dists.max())
            if rms_err < 5.0:
                quality = "GOOD"
            elif rms_err < 15.0:
                quality = "ACCEPTABLE"
            else:
                quality = "POOR"
            return rms_err, (
                f"{label}: {quality} | "
                f"({mean_xyz[0]:.3f}, {mean_xyz[1]:.3f}, {mean_xyz[2]:.3f}) m | "
                f"RMS={rms_err:.1f}mm  max={max_err:.1f}mm  n={len(samples)}"
            )

        rms_cam,  msg_cam  = _report(samples_cam,  "[ZED Mini+T_CAM_ZEDMINI]")
        rms_base, msg_base = _report(samples_base, "[Full chain/hand-eye]   ")

        pub(msg_cam  or "[ZED Mini+T_CAM_ZEDMINI]: no /trunk_position_cam — vision node updated?")
        pub(msg_base or "[Full chain/hand-eye]: no /trunk_position — is vision node running?")

        # Diagnosis
        if rms_cam is not None and rms_base is not None:
            if rms_cam < 5.0 and rms_base < 5.0:
                pub("DIAGNOSIS: Both cameras and hand-eye calibration are good.")
            elif rms_cam < 5.0 and rms_base >= 5.0:
                pub("DIAGNOSIS: T_CAM_ZEDMINI OK — hand-eye calibration has drifted. Redo checkerboard calibration.")
            elif rms_cam >= 5.0 and rms_base < 5.0:
                pub("DIAGNOSIS: Hand-eye OK — T_CAM_ZEDMINI or ZED Mini depth is unstable.")
            else:
                pub("DIAGNOSIS: POOR on both — check trunk visibility, ZED Mini depth, and both calibrations.")

    def _update_voxel_snapshot(self):
        """Update voxel obstacles from latest depth data (triggered by 'u' key).
        VoxelObstacleManager subscribes to /zed_depth_pointcloud and caches points
        in _latest_points. snapshot() reads from that cache."""
        if not hasattr(self, 'voxel_obstacles') or self.voxel_obstacles is None:
            self.get_logger().warn("No voxel obstacle manager available.")
            return
        vo = self.voxel_obstacles
        if vo._latest_points is None:
            try:
                sub_count = vo._depth_sub.get_publisher_count()
            except Exception:
                sub_count = -1
            self.get_logger().warn(
                "No depth data available yet for voxel update. "
                f"/zed_depth_pointcloud publishers={sub_count}. "
                "Start/restart the vision node and wait for the initial depth-cloud burst."
            )
            return
        try:
            if vo.snapshot():
                self.get_logger().info("Voxel obstacles updated from latest depth.")
            else:
                self.get_logger().warn("Voxel snapshot returned False.")
        except Exception as e:
            self.get_logger().warn(f"Voxel update failed: {e}")

    def _set_current_as_home(self):
        """Set the active HOME joint preset to the latest measured joint state."""
        joints = self.current_joint_positions
        if joints is None or len(joints) != len(self.joint_order):
            self.get_logger().warn(
                "Cannot set HOME: current joint state is incomplete or unavailable.")
            return
        self._config_mgr.set_home_joints(joints)
        joints_str = ", ".join(f"{v:.6f}" for v in joints)
        self.get_logger().info(
            f"HOME updated from current joints: [{joints_str}] "
            "(runtime/ROS parameter only; edit config.py to make it permanent)")

    def _set_current_as_dropoff(self):
        """Set the active DROPOFF joint preset to the latest measured joint state."""
        joints = self.current_joint_positions
        if joints is None or len(joints) != len(self.joint_order):
            self.get_logger().warn(
                "Cannot set DROPOFF: current joint state is incomplete or unavailable.")
            return
        self._config_mgr.set_dropoff_joints(joints)
        joints_str = ", ".join(f"{v:.6f}" for v in joints)
        self.get_logger().info(
            f"DROPOFF updated from current joints: [{joints_str}] "
            "(runtime/ROS parameter only; edit config.py to make it permanent)")

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

        goal_summary = []
        goal_display_lines = []
        for i, item in enumerate(safe_goals[:8], 1):
            if self._is_gripper_queue_item(item):
                action = str(item.get("action", "GRIPPER")).upper()
                goal_summary.append(action)
                goal_display_lines.append(f"{i}: GRIPPER {action}")
            elif self._is_pose_queue_item(item):
                xyz = [round(float(v), 4) for v in item[:3]]
                goal_summary.append(xyz)
                goal_display_lines.append(
                    f"G{i}: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
            else:
                goal_summary.append("UNKNOWN")
                goal_display_lines.append(f"{i}: UNKNOWN")

        home_joints = [float(v) for v in self.home_joints]
        home_rad_str = ", ".join(f"{v:.4f}" for v in home_joints)
        home_deg_str = ", ".join(f"{math.degrees(v):.1f}" for v in home_joints)
        self._home_joints_display = f"rad=[{home_rad_str}] deg=[{home_deg_str}]"

        msg_data = {
            "goal_count": len(safe_goals),
            "goals": goal_summary,
            "goal_display": "\n".join(goal_display_lines),
            "latest_goal": (
                [round(v, 4) for v in self.latest_goal_pose[:3]]
                if self.latest_goal_pose else None
            ),
            "best_goal_xyz": (
                [round(v, 4) for v in self.best_goal_xyz]
                if self.best_goal_xyz else None
            ),
            "goal_classification": getattr(self, "latest_goal_classification", None),
            "capture_active": self.goal_capture_active,
            "capture_count": getattr(self, "goal_capture_count", 0),
            "velocity_scale": self.cfg.planner.global_speed_multiplier,
            "speed_home": self.cfg.planner.speed_home,
            "speed_dropoff": self.cfg.planner.speed_dropoff,
            "speed_approach": self.cfg.planner.speed_approach,
            "speed_predropoff": self.cfg.planner.speed_predropoff,
            "home_joints": [round(v, 4) for v in home_joints],
            "home_joints_display": self._home_joints_display,
            "debug_plan_preview": self.cfg.planner.debug_plan_preview,
            "reachability_cloud_enabled": self.cfg.planner.reachability_cloud_enabled,
            "lidar_scan_preview_enabled": self.cfg.lidar_scan.semicircle_preview_enabled,
            "plan_waiting_confirm": self._state_mgr.plan_waiting,
            "motion_phase": self.motion_phase,
            "grasp_history": list(self.grasp_history),
            "reacquire_result": self.reacquire_result,
            "gripper_stopped_early": getattr(getattr(self, 'gripper_controller', None), 'closure_stopped_early', False),
            "gripper_first_contact": getattr(getattr(self, 'gripper_controller', None), 'closure_first_contact_step', -1),
            "gripper_steps": getattr(getattr(self, 'gripper_controller', None), 'steps', 10),
            "gripper_closure_step": getattr(getattr(self, 'gripper_controller', None), 'closure_step_stopped', -1),
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

            # Plan preview confirmation intercept
            if self._state_mgr.plan_waiting:
                if key in ('\r', '\n', ''):
                    print(f"[{ts()}] Plan CONFIRMED (ENTER)")
                    self._state_mgr.plan_confirmed = True
                    self._state_mgr.plan_confirm_event.set()
                    continue
                elif key == 'k':
                    print(f"[{ts()}] Plan CANCELLED (k)")
                    self._state_mgr.plan_confirmed = False
                    self._state_mgr.plan_confirm_event.set()
                    continue

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
                print(f"[{ts()}] manual goal command dispatched.")

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
                getattr(self, "_reachability_goal_metadata", {}).clear()

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
        self._start_manual_goal(goal)
        print("Manual goal sent directly.")

    def _manual_goal_pose_cb(self, msg: PoseStamped):
        """Execute an RViz manual pose directly, without the harvest sequence."""
        if msg.header.frame_id and msg.header.frame_id != "base_link":
            self.get_logger().warn(
                f"Manual goal frame must be base_link, got {msg.header.frame_id!r}.")
            return
        goal = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        self._start_manual_goal(goal)

    def _start_manual_goal(self, goal):
        """Plan once from the current state directly to a manual Cartesian goal."""
        markers_mod.publish_goal_marker(self, goal[:3])

        def run_manual_goal():
            if not self._motion_lock.acquire(blocking=False):
                self.get_logger().warn(
                    "Motion already in progress, ignoring manual goal.")
                return
            try:
                self.stop_requested = False
                self.motion_phase = "MANUAL"
                self.get_logger().info(
                    "Executing direct manual goal: "
                    f"[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
                if motions_mod.execute_single_pose(self, goal, motion_type="manual"):
                    self.get_logger().info("Direct manual goal reached.")
                else:
                    self.get_logger().warn("Direct manual goal failed or was stopped.")
            finally:
                self.motion_phase = "IDLE"
                self._motion_lock.release()

        threading.Thread(target=run_manual_goal, daemon=True).start()

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
        # 5. Scoring: smaller distance & smaller noise is better.
        # Require at least 3 frames before committing — with fewer
        # frames variance=0 is meaningless (single-sample artifact).
        # Also gate on variance threshold: sum of per-axis variances
        # < 0.0003 ≈ 10mm std per axis with the dual-camera setup.
        # This makes commitment quick when stable, patient when noisy.
        # ------------------------------------------------------
        if len(self.last_frames) < 3:
            return
        if variance > 0.0003:
            return

        score = dist + variance * 3.0

        if score < self.best_goal_score - 0.002:
            self.best_goal_score = score
            self.best_goal_xyz = [x, y, z]


    # goal capture (timed)
    def start_goal_capture(self, duration: float = 30.0):
        if self.goal_capture_active: 
            self.stop_goal_capture()
            
        self.goal_poses.clear()
        getattr(self, "_reachability_goal_metadata", {}).clear()
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
        # Keep insertion order — the queue executes in the order goals were captured.
        self.get_logger().info(f"Goal capture stopped. Collected {self.goal_capture_count} goals.")

    def _capture_goal_cb(self, msg: PoseStamped):
        if goals_mod.is_robot_moving(self): 
            return
        gx,gy,gz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        cur = self.get_end_effector_pose()
        qw,qx,qy,qz = (cur[3:] if cur else [1.0,0.0,0.0,0.0])
        
        if self.goal_poses.any_within_distance([gx,gy,gz], 0.01):
            return
        goal = [gx,gy,gz,qw,qx,qy,qz]
        self.goal_poses.append(goal)
        # Keep insertion order — no distance sort, so the queue runs in capture order.
        self.goal_capture_count += 1
        markers_mod.publish_goal_marker(self, goal[:3])

    def _add_current_as_goal(self):
        """Append the robot's current joint posture to the goal queue.

        Lets the operator jog/move the arm to a posture, snapshot it as a goal,
        repeat to build a multi-goal sequence, then run them all with Execute.
        Insertion order is preserved (no distance sort) so the queue runs in
        the order postures were added. A TCP pose is still stored for RViz
        markers, but execution uses the exact captured joint configuration.
        """
        if self.current_joint_positions is None:
            self.get_logger().warn(
                "Cannot add current joint posture as goal: joint state unavailable")
            return
        cur = self.get_end_effector_pose()
        if not cur or len(cur) < 7:
            self.get_logger().warn(
                "Cannot add current joint posture as goal: end-effector pose unavailable")
            return
        joints = [float(v) for v in self.current_joint_positions]
        for meta in getattr(self, "_reachability_goal_metadata", {}).values():
            if meta.get("source") != "current_joint":
                continue
            saved = list(meta.get("joints", []))
            if len(saved) == len(joints):
                max_err = max(abs(a - b) for a, b in zip(saved, joints))
                if max_err < math.radians(1.0):
                    self.get_logger().info(
                        "Current joint posture already in goal queue "
                        "(within 1deg) — skipping")
                    return
        goal = [float(v) for v in cur[:7]]
        self.goal_poses.append(goal)
        self._reachability_goal_metadata[self._goal_meta_key(goal)] = {
            "joints": joints,
            "delta_deg": 0.0,
            "source": "current_joint",
        }
        markers_mod.publish_goal_marker(self, goal[:3])
        self.get_logger().info(
            f"Added current joint posture as goal #{len(self.goal_poses)}: "
            f"tcp=[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")

    def _execute_goals_no_grasp(self):
        """Run every queued goal as a plain Cartesian move — no grasp behavior.

        Runs queued poses and optional gripper action items. If the queue contains
        gripper actions, insertion order is strict so OPEN/CLOSE brackets remain
        where the operator placed them.
        Honors stop_requested and aborts the rest if a move fails.
        """
        if self.goal_capture_active:
            self.stop_goal_capture()
        if not self.goal_poses:
            self.get_logger().info("No queued goals to execute.")
            return
        self.stop_requested = False
        total = len(self.goal_poses)
        strict_order = any(
            self._is_gripper_queue_item(item)
            for item in self.goal_poses.snapshot()
        )
        idx = 0
        while self.goal_poses and getattr(self, 'running', True):
            if getattr(self, 'stop_requested', False):
                self.get_logger().warn("Stop requested — aborting goal moves.")
                break
            if strict_order:
                goal = self.goal_poses.pop(0)
                reach_delta = None
            else:
                goal, reach_delta = self._pop_next_reachable_goal()
            if goal is None:
                break
            idx += 1
            if self._is_gripper_queue_item(goal):
                action = str(goal.get("action", "")).upper()
                if action in ("OPEN", "CLOSE"):
                    self.motion_phase = "GRIPPER"
                    self.get_logger().info(
                        f"Queue item {idx}/{total}: gripper {action.lower()}")
                    if action == "OPEN":
                        pause_s = 1.5
                        self.get_logger().info(
                            f"Queued gripper open: pausing {pause_s:.1f}s before opening")
                        time.sleep(pause_s)
                        if getattr(self, 'stop_requested', False):
                            self.get_logger().warn(
                                "Stop requested during pre-open pause — aborting goal moves.")
                            break
                    gripper_mod.control_gripper(self, action)
                    continue
                self.get_logger().warn(
                    f"Queue item {idx}/{total}: invalid gripper action {action!r}; skipping.")
                continue
            if not self._is_pose_queue_item(goal):
                self.get_logger().warn(
                    f"Queue item {idx}/{total}: unsupported item; skipping.")
                continue
            goal_key = self._goal_meta_key(goal)
            self.motion_phase = "MOVING"
            markers_mod.publish_goal_marker(self, goal[:3])
            self.get_logger().info(
                f"Move {idx}/{total} → "
                f"[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
            if not getattr(self, "in_safe_zone", lambda _x: True)(goal[:3]):
                self.get_logger().warn(
                    f"GOAL{idx}: target [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}] is "
                    f"OUTSIDE the safe zone — skipping.")
                getattr(self, "_reachability_goal_metadata", {}).pop(goal_key, None)
                continue
            skip_delta = float(getattr(
                self.cfg.planner, "goal_reachability_skip_delta_deg", 100.0))
            if reach_delta is not None and reach_delta > skip_delta:
                self.get_logger().warn(
                    f"GOAL{idx}: skipped because nearest IK delta "
                    f"{reach_delta:.1f}deg exceeds {skip_delta:.0f}deg.")
                getattr(self, "_reachability_goal_metadata", {}).pop(goal_key, None)
                continue
            meta = getattr(self, "_reachability_goal_metadata", {}).get(goal_key)
            cur_pose = self.get_end_effector_pose()
            if meta is not None and meta.get("source") == "current_joint":
                cur_joints = self.current_joint_positions
                saved_joints = list(meta.get("joints", []))
                if cur_joints is not None and len(saved_joints) == len(cur_joints):
                    target = motions_mod.nearest_joint_config(
                        list(cur_joints), saved_joints)
                    joint_err = max(abs(t - c) for c, t in zip(target, cur_joints))
                    if joint_err < math.radians(0.5):
                        self.get_logger().info(
                            f"GOAL{idx}: already at saved joint posture "
                            f"(max_err={math.degrees(joint_err):.1f}deg); skipping motion.")
                        getattr(self, "_reachability_goal_metadata", {}).pop(goal_key, None)
                        continue
            elif cur_pose is not None and len(cur_pose) >= 7:
                pos_err = math.dist(cur_pose[:3], goal[:3])
                quat_dot = abs(sum(a * b for a, b in zip(cur_pose[3:7], goal[3:7])))
                if pos_err < 0.01 and quat_dot > 0.999:
                    self.get_logger().info(
                        f"GOAL{idx}: already at target "
                        f"(pos_err={pos_err*100:.1f}cm); skipping motion.")
                    getattr(self, "_reachability_goal_metadata", {}).pop(goal_key, None)
                    continue
            move_speed_factor = self.GOAL_MOVE_SPEED_FACTOR
            if meta is not None and meta.get("source") != "current_joint":
                move_speed_factor = float(getattr(
                    self.cfg.planner, "reachability_goal_speed_factor", 0.20))
            if meta is not None:
                source = str(meta.get("source", "reachability"))
                if source == "current_joint":
                    self.get_logger().info(
                        f"GOAL{idx}: using saved current joint posture "
                        f"(cached delta={float(meta.get('delta_deg', 0.0)):.1f}deg, "
                        f"speed={move_speed_factor:.2f}x).")
                    ok = motions_mod.plan_execute_js(
                        self,
                        list(meta["joints"]),
                        label=f"GOAL{idx}_JOINTS",
                        motion_type="manual",
                        speed_factor=move_speed_factor)
                else:
                    self.get_logger().info(
                        f"GOAL{idx}: using reachability-click joint sample "
                        f"(cached delta={float(meta.get('delta_deg', 0.0)):.1f}deg, "
                        f"speed={move_speed_factor:.2f}x).")
                    ok = motions_mod.execute_known_joint_goal(
                        self,
                        list(meta["joints"]),
                        label=f"GOAL{idx}_REACHABLE",
                        motion_type="manual",
                        speed_factor=move_speed_factor)
            else:
                ok = motions_mod.execute_pose_shortest(
                    self, list(goal), label=f"GOAL{idx}", motion_type="manual",
                    speed_factor=move_speed_factor)
            current_joint_failed = (
                not ok and meta is not None and meta.get("source") == "current_joint")
            if current_joint_failed:
                self.get_logger().warn(
                    f"GOAL{idx}: saved joint-posture move failed; not falling back "
                    "to TCP-pose planning.")
            if not ok and not current_joint_failed and not getattr(self, 'stop_requested', False):
                # The marker's orientation can force a far IK flip even when the position
                # is close. Retry keeping the marker POSITION but using the arm's CURRENT
                # tool orientation, so IK stays near current and the move is short.
                cur = self.get_end_effector_pose()
                if cur is not None and len(cur) >= 7:
                    quat_dot = abs(sum(
                        a * b for a, b in zip(goal[3:7], cur[3:7])))
                    if quat_dot < 0.999:
                        retry_pose = list(goal[:3]) + list(cur[3:7])
                        self.get_logger().warn(
                            f"GOAL{idx}: failed with marker orientation — retrying with "
                            f"current tool orientation (position only).")
                        ok = motions_mod.execute_pose_shortest(
                            self, retry_pose, label=f"GOAL{idx}_FREEORI",
                            motion_type="manual",
                            speed_factor=move_speed_factor)
                    else:
                        self.get_logger().warn(
                            f"GOAL{idx}: marker orientation already matches current "
                            f"tool orientation; skipping duplicate FREEORI retry.")
            recovery_cap = float(getattr(
                self.cfg.planner, "goal_recovery_max_ik_delta_deg", 100.0))
            recovery_allowed = (
                not current_joint_failed
                and (
                    reach_delta is None
                    or math.isfinite(reach_delta) and reach_delta <= recovery_cap
                )
            )
            if (
                not ok
                and not current_joint_failed
                and not getattr(self, 'stop_requested', False)
                and not recovery_allowed
            ):
                self.get_logger().warn(
                    f"GOAL{idx}: nearest IK delta {reach_delta:.1f}deg exceeds "
                    f"recovery cap {recovery_cap:.0f}deg; skipping staging/posture "
                    "recovery instead of grinding planner retries.")
            if (
                not ok
                and not getattr(self, 'stop_requested', False)
                and recovery_allowed
                and bool(getattr(
                    self.cfg.planner, "goal_recovery_local_staging", True))
            ):
                self.get_logger().warn(
                    f"GOAL{idx}: trying target-local staging before posture recovery.")
                ok = motions_mod.execute_goal_via_local_staging(
                    self,
                    list(goal),
                    label=f"GOAL{idx}_LOCAL_STAGE",
                    motion_type="manual",
                    speed_factor=move_speed_factor)
            if (
                not ok
                and not getattr(self, 'stop_requested', False)
                and recovery_allowed
                and bool(getattr(
                    self.cfg.planner, "goal_failure_recover_to_posture", True))
            ):
                self.get_logger().warn(
                    f"GOAL{idx}: failed from current posture — moving to nearest "
                    f"good posture, then retrying once.")
                if motions_mod.move_to_nearest_good_posture(
                        self, label=f"GOAL{idx}_RECOVERY",
                        goal_pose=list(goal)):
                    ok = motions_mod.execute_pose_shortest(
                        self, list(goal), label=f"GOAL{idx}_RETRY",
                        motion_type="manual",
                        speed_factor=move_speed_factor)
                    if not ok and not getattr(self, 'stop_requested', False):
                        cur = self.get_end_effector_pose()
                        if cur is not None and len(cur) >= 7:
                            quat_dot = abs(sum(
                                a * b for a, b in zip(goal[3:7], cur[3:7])))
                            if quat_dot < 0.999:
                                retry_pose = list(goal[:3]) + list(cur[3:7])
                                self.get_logger().warn(
                                    f"GOAL{idx}_RETRY: failed with marker orientation — "
                                    f"retrying with current tool orientation.")
                                ok = motions_mod.execute_pose_shortest(
                                    self, retry_pose, label=f"GOAL{idx}_RETRY_FREEORI",
                                    motion_type="manual",
                                    speed_factor=move_speed_factor)
                            else:
                                self.get_logger().warn(
                                    f"GOAL{idx}_RETRY: marker orientation already "
                                    f"matches current tool orientation; skipping "
                                    f"duplicate FREEORI retry.")
                    if (
                        not ok
                        and not getattr(self, 'stop_requested', False)
                        and bool(getattr(
                            self.cfg.planner,
                            "goal_recovery_cartesian_fallback",
                            True))
                    ):
                        self.get_logger().warn(
                            f"GOAL{idx}: posture retry still needs a large IK jump — "
                            f"trying Cartesian recovery plan.")
                        if bool(getattr(
                                self.cfg.planner,
                                "goal_recovery_local_staging",
                                True)) and bool(getattr(
                                    self.cfg.planner,
                                    "goal_recovery_repeat_staging_after_posture",
                                    False)):
                            ok = motions_mod.execute_goal_via_local_staging(
                                self,
                                list(goal),
                                label=f"GOAL{idx}_RECOVERED_STAGE",
                                motion_type="manual",
                                speed_factor=move_speed_factor)
                        if not ok:
                            ok = motions_mod.execute_single_pose(
                                self,
                                list(goal),
                                motion_type="manual",
                                speed_factor=move_speed_factor)
                        if not ok and not getattr(self, 'stop_requested', False):
                            cur = self.get_end_effector_pose()
                            if cur is not None and len(cur) >= 7:
                                quat_dot = abs(sum(
                                    a * b for a, b in zip(goal[3:7], cur[3:7])))
                                if quat_dot < 0.999:
                                    cart_pose = list(goal[:3]) + list(cur[3:7])
                                    self.get_logger().warn(
                                        f"GOAL{idx}: Cartesian recovery failed with "
                                        f"marker orientation — retrying position-only.")
                                    ok = motions_mod.execute_single_pose(
                                        self,
                                        cart_pose,
                                        motion_type="manual",
                                        speed_factor=move_speed_factor)
                else:
                    self.get_logger().warn(
                        f"GOAL{idx}: recovery posture move failed; not retrying.")
            if not ok:
                self.get_logger().warn(
                    f"Move {idx}/{total} failed or stopped; "
                    f"aborting remaining goals.")
                getattr(self, "_reachability_goal_metadata", {}).pop(goal_key, None)
                break
            getattr(self, "_reachability_goal_metadata", {}).pop(goal_key, None)
        self.motion_phase = "IDLE"
        self.get_logger().info("Goal moves finished.")

    def _pop_next_reachable_goal(self):
        """Pick the queued goal that is easiest from the current posture."""
        if not bool(getattr(self.cfg.planner, "dynamic_goal_ordering", True)):
            return self.goal_poses.pop(0), None

        goals = (
            self.goal_poses.snapshot()
            if hasattr(self.goal_poses, "snapshot")
            else [g for g in self.goal_poses]
        )
        if not goals:
            return None, None

        scored = []
        for i, goal in enumerate(goals):
            if not getattr(self, "in_safe_zone", lambda _x: True)(goal[:3]):
                scored.append((float("inf"), i, goal, "outside"))
                continue
            meta = getattr(self, "_reachability_goal_metadata", {}).get(
                self._goal_meta_key(goal))
            if meta is not None and self.current_joint_positions is not None:
                target = motions_mod.nearest_joint_config(
                    self.current_joint_positions, list(meta["joints"]))
                delta = math.degrees(max(
                    abs(t - c) for c, t in zip(target, self.current_joint_positions)))
            else:
                delta = motions_mod.estimate_nearest_ik_delta_deg(self, list(goal))
            scored.append((delta, i, goal, "ok"))

        scored.sort(key=lambda item: (item[0], item[1]))
        best_delta, best_idx, best_goal, status = scored[0]
        skip_delta = float(getattr(
            self.cfg.planner, "goal_reachability_skip_delta_deg", 100.0))
        if status == "outside":
            self.get_logger().warn("All queued goals are outside the safe zone.")
        elif best_delta > skip_delta:
            self.get_logger().warn(
                f"Best queued goal still needs {best_delta:.1f}deg IK delta "
                f"(cap {skip_delta:.0f}deg); skipping instead of grinding recovery.")
        else:
            self.get_logger().info(
                f"Selected queued goal #{best_idx + 1}/{len(goals)} "
                f"(nearest IK delta={best_delta:.1f}deg)")
        return self.goal_poses.pop(best_idx), best_delta

    def _prep_and_execute(self):

        if self.goal_capture_active:
            self.stop_goal_capture()
        if hasattr(self, 'goal_pose_sub'):
            try:
                self.destroy_subscription(self.goal_pose_sub)
            except Exception:
                pass
            try:
                del self.goal_pose_sub
            except Exception:
                pass
        # Cancel multi-subscribe timeout timer if active
        if hasattr(self, '_multi_timeout_timer'):
            try:
                self._multi_timeout_timer.cancel()
                del self._multi_timeout_timer
            except Exception:
                pass
            
        self.get_logger().info("Executing stored goals…")
        goals_mod.plan_and_execute(self)

    # expose some helpers for external callers
    def control_gripper(self, action: str, fruit_radius: float = None):
        gripper_mod.control_gripper(self, action, fruit_radius=fruit_radius)

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
    def home_left_joints(self) -> list:
        return self._config_mgr.home_left_joints

    @property
    def home_left_low_joints(self) -> list:
        return self._config_mgr.home_left_low_joints

    @property
    def home_right_low_joints(self) -> list:
        return self._config_mgr.home_right_low_joints

    @property
    def home_right_joints(self) -> list:
        return self._config_mgr.home_right_joints

    @property
    def trunk_x(self):
        return self._state_mgr.trunk_x

    @property
    def trunk_xyz(self):
        return self._state_mgr.trunk_xyz

    @property
    def plan_confirm_event(self):
        return self._state_mgr.plan_confirm_event

    @property
    def plan_confirmed(self):
        return self._state_mgr.plan_confirmed

    @plan_confirmed.setter
    def plan_confirmed(self, val):
        self._state_mgr.plan_confirmed = val

    @property
    def plan_waiting(self):
        return self._state_mgr.plan_waiting

    @plan_waiting.setter
    def plan_waiting(self, val):
        self._state_mgr.plan_waiting = val

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

    @property
    def fruit_image_norm(self):
        return self._state_mgr.fruit_image_norm

    @property
    def fruit_bunch_rel_x(self):
        return self._state_mgr.fruit_bunch_rel_x

    @property
    def fruit_bunch_rel_y(self):
        return self._state_mgr.fruit_bunch_rel_y

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
    def voxel_obstacles(self):
        return self._motion_mgr.voxel_obstacles

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
