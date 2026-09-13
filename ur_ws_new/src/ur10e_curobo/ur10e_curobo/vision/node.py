"""Main VisionNode class for date fruit detection."""

import math
import os
import queue
import sys
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep, time
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from std_msgs.msg import Float32, Float32MultiArray, String as StdString
from sensor_msgs.msg import PointCloud2, Image as ROSImage
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 - Required for transform registration

from .config import (
    CAMERA_PROFILE, CAM_FRAME, ZEDMINI_CAM_FRAME, Z_MAX,
    BEST_REUSE_THRESH, SWITCH_THRESHOLD, TARGET_LOCK_RADIUS,
    TRUNK_DEPTH_OFFSET,
    LIDAR_TOPIC, LIDAR_Z_MIN, LIDAR_Z_MAX, T_CAM_LIDAR,
    ZEDMINI_SERIAL, ZEDMINI_DEPTH_FPS, ZEDMINI_RGBD_FPS,
    ZEDMINI_DEPTH_Z_MIN, ZEDMINI_DEPTH_Z_MAX, ZEDMINI_MAX_POINTS, T_CAM_ZEDMINI,
    FRUIT_CAMERA_Z_MIN, FRUIT_CAMERA_Z_MAX,
    FRUIT_SURFACE_TO_CENTER_FRACTION,
    SHOW_CLASSIFICATION_ZONES, SHOW_GAP_DEBUG,
    SHOW_FINGER_CONTACTS, FINGERTIP_CONTACT_RADIUS_M,
    FINGER_CONTACT_RADIAL_FRACTION, FINGER_CONTACT_ROTATION_SAMPLES,
    FINGER_CONTACT_MIN_SCORE,
)
from ..perception_lidar import (
    parse_pointcloud2, project_lidar_to_image,
)
from ..config import X_FORWARD_Y_LATERAL
from .math_utils import quat_align_x_to_axis
from .ros_utils import wait_for_transform, create_pointcloud2_msg
from .zed_utils import (
    apply_zed_one_manual_exposure,
    apply_zed_one_exposure_preset,
    apply_zed_one_hdr,
    apply_zed_one_settings,
    apply_zed_mini_settings,
    apply_zed_stereo_settings,
)
apply_zed_camera_settings = apply_zed_one_settings  # used by older call sites below
from .tracking import FruitTracker

# Per-date depth diagnostics, throttled to ~2 Hz while resolving range-dependent
# foreground/background selection. Disable after field validation.
DEBUG_DEPTH_SAMPLING = True
# Periodic loop/render timing is useful for profiling but too noisy for normal
# field operation. Enable temporarily when benchmarking perception performance.
DEBUG_PERFORMANCE = False
from .scoring import (
    compute_fruit_score, compute_collision_free_direction,
    estimate_fruit_radius,
)
from .contact_points import estimate_three_finger_contacts
from .yolo_thread import YoloThread
from .visualization import VisionVisualizer


def _tf_stamped_to_Rt(T):
    """Convert TransformStamped → (R: 3×3, t: 3,) float64 numpy arrays."""
    tr = T.transform.translation
    q  = T.transform.rotation
    w, x, y, z = q.w, q.x, q.y, q.z
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
    return R, t


class VisionNode:
    """ROS2 node for date fruit detection using ZED camera and YOLO."""

    def __init__(self, args):
        self.args = args
        self.exit_signal = False

        # State
        self.best_target_prev: Optional[Dict[str, Any]] = None
        self.prev_heat_point: Optional[np.ndarray] = None
        self.prev_direction_base: Optional[np.ndarray] = None
        self.direction_history = deque(maxlen=5)
        self.best_history = deque(maxlen=5)

        # Heatmap throttling — only recompute every N frames
        self._heatmap_frame_count = 0
        self._heatmap_interval = 10  # recompute every 10th frame
        self._cached_heatmaps = {}  # key: target index → (heatmap, best_point, best_dir2d, best_point_3d)
        # Selected-date contact geometry changes only when the camera/target
        # moves. Cache it so the visual overlay adds no steady-state loop cost.
        self._contact_cache: Dict[tuple, Dict[str, Any]] = {}
        self._grasp_candidate_history = deque(maxlen=8)
        self._latest_grasp_candidate_record: Optional[Dict[str, Any]] = None

        # Per-frame TF cache — refreshed once at top of each loop iteration.
        # Avoids 8+ expensive tf_buffer.transform calls per frame (each ~20ms on Jetson).
        self._cached_tf_base: Optional[tuple] = None  # (R: 3×3, t: 3,) cam → base_link
        self._cached_tf_grip: Optional[tuple] = None  # (R: 3×3, t: 3,) cam → gripper_tip
        # Camera position at the moment the last YOLO result was accepted.
        # Used to detect cumulative drift (slow moves that never cross the per-frame threshold).
        self._yolo_accepted_cam_t: Optional[np.ndarray] = None

        # Detection mode: "full" (approach) or "reacquire" (lightweight — fruit position only).
        # Set by the main node via /vision/mode topic.
        self.detection_mode: str = "full"
        self._reacquire_frame_skip: int = 0  # frame counter for YOLO throttle in reacquire mode
        # True once we have confirmed the paused state (inference drained) to the
        # motion node. Reset when leaving paused so the next pause re-acks.
        self._vision_paused_acked: bool = False


        # Target lock
        self.target_lock_position: Optional[List[float]] = None
        self.target_lock_active = False

        # Exclusion zones (for multi-subscribe: skip already-accepted fruits)
        self.excluded_positions: List[List[float]] = []
        self._prev_excluded_count = 0  # track changes to clear hysteresis

        # Publishing state
        self.latest_goal_msg: Optional[PoseStamped] = None
        self.latest_dir_msg: Optional[Vector3Stamped] = None
        # Wall time when perception last produced this goal.  The publishing
        # timer must never make an old measurement look fresh merely by
        # replacing its ROS timestamp.
        self._latest_goal_update_time = 0.0
        self.pub_lock = Lock()

        # Sparse point cloud (used when --use_lidar)
        self._latest_cloud: Optional[np.ndarray] = None
        self._cloud_lock = Lock()

        # Dense depth map (H×W×3 XYZ) at display resolution (used when --use_zed_mini)
        self._latest_depth_map: Optional[np.ndarray] = None
        # Boolean mask of originally projected pixels (True = real data, False = EDT fill)
        self._latest_depth_map_orig: Optional[np.ndarray] = None
        self._depth_map_lock = Lock()

        # Active camera frame for TF and published camera-frame points.
        self.cam_frame = (
            ZEDMINI_CAM_FRAME
            if getattr(self.args, "use_zedx_mini_only", False)
            else CAM_FRAME
        )

        # Raw ZED Mini points + UV in ZED One image (no EDT fill).
        # pts_one: ZED One (CAM_FRAME) coordinates — used for centroid so TF to base_link is correct.
        # pts_mini: ZED Mini frame — kept for depth_map visualisation only.
        # Used by _extract_target_3d for direct point queries — avoids scatter/fill artefacts.
        self._mini_pts_buffer: list = []           # ring buffer: [(pts_one, uv_one, ts), ...]
        self._mini_pts_lock = Lock()
        self._mini_pts_buffer_size: int = 7       # keep last 7 Mini depth frames
        self._latest_zed_one_ts: float = 0.0      # wall-clock time of last ZED One grab
        # ROS time of the same grab. TF is looked up at THIS stamp rather
        # than at 'latest', because detections are consumed one or more
        # frames after the image they came from (YOLO runs in its own
        # thread and results are deliberately deferred a frame when the
        # GPU is busy). Transforming an old image with a current TF turns
        # pipeline latency into position error whenever the arm moves.
        self._latest_image_stamp = None           # rclpy Time of last grab

        # ZED X One image from ROS topic (used when --use_lidar)
        self._latest_ros_image: Optional[np.ndarray] = None
        self._ros_image_lock = Lock()

        # Open ZED cameras. _zed is the live image camera; _zed_mini is depth-only.
        self._zed = None
        self._zed_mini = None
        self._camera_type = "not open"
        self._camera_resolution = "-"
        self._camera_mode = "-"
        self._camera_fps = "-"
        self._camera_hdr_enabled = bool(int(getattr(self.args, "hdr", 1)))
        self._depth_camera_text = "-"
        self._camera_status_pub = None
        self._refresh_argv = None
        self._refresh_reason = "camera refresh"
        self._raw_stream_enabled = bool(int(os.getenv("UR10E_RAW_IMAGE_STREAM", "0")))
        self._raw_stream_until = 0.0

        # Components
        self.tracker = FruitTracker()
        self.yolo_thread: Optional[YoloThread] = None
        self.visualizer: Optional[VisionVisualizer] = None
        self._show_classification_zones = SHOW_CLASSIFICATION_ZONES
        self._show_gap_debug = SHOW_GAP_DEBUG

        # ROS2
        self.node = None
        self.executor = None
        self.tf_buffer = None

    @staticmethod
    def _argv_with_option(argv, option, value):
        updated = list(argv)
        if option in updated:
            idx = updated.index(option)
            if idx + 1 < len(updated):
                updated[idx + 1] = value
            else:
                updated.append(value)
            return updated
        insert_at = updated.index("--ros-args") if "--ros-args" in updated else len(updated)
        updated[insert_at:insert_at] = [option, value]
        return updated

    def _request_vision_restart(self, reason: str, argv=None):
        if argv is None:
            argv = self._argv_with_option(sys.argv, "--weights", str(getattr(self.args, "weights", "")))
            argv = self._argv_with_option(argv, "--hdr", str(int(self._camera_hdr_enabled)))
        self._refresh_requested = True
        self._refresh_reason = reason
        self._refresh_argv = list(argv)
        self.exit_signal = True
        _exec = self.executor
        Thread(target=_exec.shutdown, daemon=True).start()

    def _request_exit(self):
        """Stop both perception workers and the blocking ROS executor."""
        if self.exit_signal:
            return
        self.exit_signal = True
        executor = self.executor
        if executor is not None:
            Thread(target=executor.shutdown, daemon=True).start()

    def run(self):
        """Main entry point."""
        import torch
        with torch.no_grad():
            self._run_main()

    def _run_main(self):
        """Main execution loop. ROS2 is initialized once; ZED can be refreshed
        inside the perception thread without restarting the executor."""
        self.exit_signal = False
        self._refresh_requested = False

        # ROS2 setup — init once for the lifetime of the process
        rclpy.init()
        self.node = rclpy.create_node("zed_date_detector_ros")
        self.node.declare_parameter("vision.log_grasp_candidate_details", False)
        self.executor = MultiThreadedExecutor(num_threads=4)

        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        goal_pub = self.node.create_publisher(PoseStamped, "/external_goal_pose", fast_qos)
        dir_pub = self.node.create_publisher(Vector3Stamped, "/datefruit_direction", 10)
        self.date_tip_pub = self.node.create_publisher(
            PointStamped, "/datefruit_tip_point", 10)
        depth_pub = self.node.create_publisher(PointCloud2, "/zed_depth_pointcloud", 10)
        trunk_pub     = self.node.create_publisher(PointStamped, "/trunk_position",     10)
        trunk_cam_pub = self.node.create_publisher(PointStamped, "/trunk_position_cam", 10)
        self.radius_pub = self.node.create_publisher(Float32, "/fruit_radius", 10)
        self.gap_info_pub = self.node.create_publisher(Float32MultiArray, "/datefruit_gap_info", 10)
        self.depth_diag_pub = self.node.create_publisher(
            Float32MultiArray, "/vision/depth_diagnostics", 10)
        self.bbox_norm_pub = self.node.create_publisher(Float32MultiArray, "/fruit_image_bbox_norm", 10)
        self.all_fruits_pub = self.node.create_publisher(Float32MultiArray, "/vision/all_fruit_poses", 10)
        # Per-fruit detection quality, in the SAME score-sorted order as
        # /vision/all_fruit_poses, 5 floats per fruit. Published in reacquire
        # modes too, unlike the quality topics inside _process_best_target
        # (depth_diagnostics / fruit_score / bbox_norm), which are skipped there.
        # Without this, reacquire matched on position proximity alone: any
        # detection landing within the match radius was accepted regardless of
        # how bad it was.
        self.all_fruit_quality_pub = self.node.create_publisher(
            Float32MultiArray, "/vision/all_fruit_quality", 10)
        # Crop around the selected date, wide enough to include the fingertips.
        # Published in EVERY mode, unlike /fruit_image_bbox_norm which lives in
        # _process_best_target and so is silent exactly when the arm is closing
        # in. The motion node caches the latest and the episode recorder saves
        # it, so training data shows the date RELATIVE TO THE GRIPPER rather
        # than as an absolute position -- which is the point: it sidesteps the
        # hand-eye chain instead of inheriting its error.
        self.grasp_crop_pub = self.node.create_publisher(
            ROSImage, "/vision/grasp_crop", 10)
        self.grasp_crop_meta_pub = self.node.create_publisher(
            Float32MultiArray, "/vision/grasp_crop_meta", 10)
        self._last_grasp_crop_t = 0.0
        # Large camera frames must never back-pressure perception. A depth-1,
        # best-effort stream lets RViz consume the newest image and drops stale
        # frames when rendering cannot keep up.
        self.image_pub = self.node.create_publisher(
            ROSImage, "/vision/display", fast_qos)
        self.raw_image_pub = self.node.create_publisher(
            ROSImage, "/vision/raw", fast_qos)
        self.fingertip_verification_pub = self.node.create_publisher(
            StdString, "/vision/fingertip_verification", 10)
        self.grasp_candidates_pub = self.node.create_publisher(
            StdString, "/vision/grasp_candidates", 10)
        self.grasp_candidate_selection_pub = self.node.create_publisher(
            Float32MultiArray, "/vision/grasp_candidate_selection", 10)
        self._camera_status_pub = self.node.create_publisher(StdString, "/camera_status", 10)
        self.node.create_timer(2.0, self._publish_camera_status)
        self.heatmap_data_pub = self.node.create_publisher(Float32MultiArray, "/vision/heatmap_3d_data", 10)
        from std_msgs.msg import String as _Str
        self.score_pub = self.node.create_publisher(_Str, "/vision/fruit_score", 10)
        self.cv_bridge = CvBridge()

        # Timer-based publishing callback (50Hz)
        def publish_timer_cb():
            with self.pub_lock:
                goal_age = time() - self._latest_goal_update_time
                if self.latest_goal_msg is not None and goal_age <= 0.25:
                    goal_pub.publish(self.latest_goal_msg)
                elif self.latest_goal_msg is not None:
                    # Perception has not refreshed this target for several
                    # camera frames.  Remove it instead of replaying it.
                    self.latest_goal_msg = None
                    self.latest_dir_msg = None
                if self.latest_dir_msg is not None and goal_age <= 0.25:
                    dir_pub.publish(self.latest_dir_msg)

        self.node.create_timer(0.02, publish_timer_cb)

        # Subscribers
        def grasp_attempt_cb(msg):
            self.tracker.increment_fruit_attempt([msg.point.x, msg.point.y, msg.point.z])

        self.node.create_subscription(PointStamped, "/fruit_grasp_attempt", grasp_attempt_cb, 10)

        def target_lock_cb(msg):
            if msg.point.x == 0.0 and msg.point.y == 0.0 and msg.point.z == 0.0:
                self.target_lock_active = False
                self.target_lock_position = None
                print("Target lock released")
            else:
                self.target_lock_position = [msg.point.x, msg.point.y, msg.point.z]
                self.target_lock_active = True
                self._publish_locked_grasp_candidates(self.target_lock_position)
                print(f"Target locked at [{msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}]")

        self.node.create_subscription(PointStamped, "/target_lock", target_lock_cb, 10)

        # Exclusion zones for multi-subscribe
        def exclude_cb(msg):
            # Float32MultiArray: data = [x1,y1,z1, x2,y2,z2, ...] or empty to clear
            data = list(msg.data)
            positions = []
            for i in range(0, len(data) - 2, 3):
                positions.append([data[i], data[i+1], data[i+2]])
            self.excluded_positions = positions
            if positions:
                print(f"Exclusion zones set: {len(positions)} position(s)")
            else:
                print("Exclusion zones cleared")

        self.node.create_subscription(Float32MultiArray, "/exclude_fruit_positions", exclude_cb, 10)

        # Detection mode — "full" (approach) or "reacquire" (lightweight).
        # Main node publishes to /vision/mode to switch modes on the fly.
        def _mode_cb(msg):
            mode = msg.data.strip()
            if mode in ("full", "reacquire", "reacquire_fast", "paused"):
                self.detection_mode = mode
        self.node.create_subscription(StdString, "/vision/mode", _mode_cb, 10)
        # Report the effective detection state back to the motion node. Used for a
        # deterministic GPU handoff: we publish "paused" only after in-flight YOLO
        # inference has drained, so cuRobo can plan without a fixed guess delay.
        self._mode_state_pub = self.node.create_publisher(StdString, "/vision/mode_state", 10)

        # Runtime visualization overlays. Kept separate from /vision/mode so
        # debugging UI does not change detection behavior.
        def _overlay_cb(msg):
            cmd = msg.data.strip().lower()
            if cmd in ("classification_zones true", "zones true", "on", "true"):
                self._show_classification_zones = True
                if self.visualizer is not None:
                    self.visualizer.show_classification_zones = True
                self.node.get_logger().info("[vision] classification zone overlay ON")
            elif cmd in ("classification_zones false", "zones false", "off", "false"):
                self._show_classification_zones = False
                if self.visualizer is not None:
                    self.visualizer.show_classification_zones = False
                self.node.get_logger().info("[vision] classification zone overlay OFF")
            elif cmd in ("gap_debug true", "gaps true"):
                self._show_gap_debug = True
                if self.visualizer is not None:
                    self.visualizer.show_gap_debug = True
                self.node.get_logger().info("[vision] branch-gap debug overlay ON")
            elif cmd in ("gap_debug false", "gaps false"):
                self._show_gap_debug = False
                if self.visualizer is not None:
                    self.visualizer.show_gap_debug = False
                self.node.get_logger().info("[vision] branch-gap debug overlay OFF")
        self.node.create_subscription(StdString, "/vision/overlay_command", _overlay_cb, 10)

        self.node.create_timer(5.0, self.tracker.cleanup_old_fruit_ids)

        # Camera refresh command — restart the entire process.
        # ZED X Mini cannot reopen in-process (driver doesn't release cleanly);
        # the only reliable refresh is a full process restart via exec().
        def camera_cmd_cb(msg):
            cmd = msg.data.strip()
            if cmd == "refresh":
                self.node.get_logger().info("Camera refresh requested — restarting process...")
                self._request_vision_restart("camera refresh")
            elif cmd in ("preset lab", "preset outdoor"):
                preset = cmd.split()[1]
                if self._zed is None:
                    self.node.get_logger().warn(
                        f"Camera preset {preset}: ZED camera is not open yet")
                    return
                applied = apply_zed_one_exposure_preset(self._zed, preset)
                self.node.get_logger().info(
                    f"Camera exposure preset applied: {applied}")
            elif cmd.startswith("settings "):
                if self._zed is None:
                    self.node.get_logger().warn(
                        "Camera settings: ZED camera is not open yet")
                    return
                values = {}
                for token in cmd.split()[1:]:
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    values[key.strip().lower()] = value.strip()
                try:
                    auto = values.get("auto", "0").lower() in ("1", "true", "yes", "on")
                    hdr = values.get("hdr", str(int(self._camera_hdr_enabled))).lower() in (
                        "1", "true", "yes", "on")
                    exposure = int(values.get("exposure", "8"))
                    gain = int(values.get("gain", "0"))
                except ValueError:
                    self.node.get_logger().warn(f"Camera settings ignored: bad command '{cmd}'")
                    return
                self._camera_hdr_enabled = bool(hdr)
                self.args.hdr = int(hdr)
                apply_zed_one_manual_exposure(self._zed, auto, exposure, gain)
                hdr_ok = apply_zed_one_hdr(self._zed, hdr)
                if not hdr_ok:
                    self.node.get_logger().warn(
                        "Camera HDR runtime toggle may require Refresh Camera to take effect.")
                self.node.get_logger().info(
                    f"Camera settings applied: auto={int(auto)} hdr={int(hdr)} "
                    f"exposure={exposure} gain={gain}")
                self._publish_camera_status()
            elif cmd.startswith("model "):
                values = {}
                for token in cmd.split()[1:]:
                    if "=" not in token:
                        continue
                    key, value = token.split("=", 1)
                    values[key.strip().lower()] = value.strip()
                model_path = values.get("path", "")
                if not model_path:
                    self.node.get_logger().warn(f"Camera model ignored: bad command '{cmd}'")
                    return
                path = Path(model_path).expanduser()
                allowed = {".engine", ".pt", ".onnx"}
                if not path.exists() or path.suffix.lower() not in allowed:
                    self.node.get_logger().warn(
                        f"Camera model ignored: not a valid model file: {path}")
                    return
                self.args.weights = str(path)
                argv = self._argv_with_option(sys.argv, "--weights", str(path))
                argv = self._argv_with_option(argv, "--hdr", str(int(self._camera_hdr_enabled)))
                self.node.get_logger().info(
                    f"Camera model change requested: {path.name}; restarting vision.")
                self._request_vision_restart(f"model change to {path.name}", argv=argv)
            elif cmd.startswith("inference "):
                values = {}
                for token in cmd.split()[1:]:
                    if "=" in token:
                        key, value = token.split("=", 1)
                        values[key.strip().lower()] = value.strip()
                try:
                    confidence = float(values.get("conf", self.args.conf_thres))
                    max_detections = int(values.get("max_det", self.args.max_det))
                except ValueError:
                    self.node.get_logger().warn(
                        f"Inference settings ignored: bad command '{cmd}'")
                    return
                if not 0.0 < confidence <= 1.0 or not 1 <= max_detections <= 100:
                    self.node.get_logger().warn(
                        "Inference settings ignored: confidence must be in (0,1] "
                        "and max_det in [1,100]")
                    return
                self.args.conf_thres = confidence
                self.args.max_det = max_detections
                worker = self.yolo_thread
                if worker is not None:
                    worker.conf_thres = confidence
                    worker.max_det = max_detections
                    direct = getattr(worker, "_direct_model", None)
                    if direct is not None:
                        direct.conf = confidence
                        direct.max_det = max_detections
                self.node.get_logger().info(
                    f"YOLO settings applied live: confidence={confidence:.2f}, "
                    f"max_detections={max_detections}")
                self._publish_camera_status()
            elif cmd.startswith("raw_stream "):
                parts = cmd.split()
                mode = parts[1].lower() if len(parts) > 1 else ""
                if mode in ("on", "true", "1", "start"):
                    self._raw_stream_enabled = True
                    self.node.get_logger().info("Raw camera stream ON")
                elif mode in ("off", "false", "0", "stop"):
                    self._raw_stream_enabled = False
                    self._raw_stream_until = 0.0
                    self.node.get_logger().info("Raw camera stream OFF")
                elif mode in ("snapshot", "burst"):
                    duration = 1.0
                    for token in parts[2:]:
                        if token.startswith("duration="):
                            try:
                                duration = max(0.2, min(5.0, float(token.split("=", 1)[1])))
                            except ValueError:
                                pass
                    self._raw_stream_until = max(self._raw_stream_until, time() + duration)
                    self.node.get_logger().info(
                        f"Raw camera stream burst requested ({duration:.1f}s)")

        self.node.create_subscription(StdString, "/camera_command", camera_cmd_cb, 10)

        raw_yolo_view = bool(getattr(self.args, "raw_yolo_view", False))
        # Raw inspection is camera-local and must neither require nor wait for
        # a robot TF tree. Normal harvesting retains the existing TF setup.
        self.tf_buffer = None
        if not raw_yolo_view:
            self.tf_buffer = Buffer()
            TransformListener(self.tf_buffer, self.node)
            required_tfs = [("base_link", self.cam_frame), ("gripper_tip", self.cam_frame)]
            for target, source in required_tfs:
                wait_for_transform(self.tf_buffer, target, source, self.node, timeout=5.0)

        use_lidar   = getattr(self.args, "use_lidar",    False)
        use_zed_mini = getattr(self.args, "use_zed_mini", False)
        use_zedx_mini_only = getattr(self.args, "use_zedx_mini_only", False)
        use_mono_depth = use_lidar or use_zed_mini  # ZED One Mono + external depth source
        _pending_depth_thread = None  # set to a callable if ZED Mini warp thread is needed
        zed = self._init_zed_and_yolo()

        if use_mono_depth:
            # ── ZED X One Mono path (LiDAR or ZED Mini depth) ───────────────
            # Read intrinsics directly from the opened camera (resolution-independent)
            if zed is None:
                rclpy.shutdown()
                return
            camera_infos = zed.get_camera_information()
            cam_res = camera_infos.camera_configuration.resolution
            cam_w, cam_h = cam_res.width, cam_res.height
            left_cam = camera_infos.camera_configuration.calibration_parameters
            fx, fy = float(left_cam.fx), float(left_cam.fy)
            cx, cy = float(left_cam.cx), float(left_cam.cy)
            disto  = left_cam.disto
            print(f"[ZedXOne] {cam_w}x{cam_h}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

            if use_lidar and not raw_yolo_view:
                def _cloud_cb(msg):
                    pts = parse_pointcloud2(msg)
                    with self._cloud_lock:
                        self._latest_cloud = pts
                self.node.create_subscription(PointCloud2, LIDAR_TOPIC, _cloud_cb, fast_qos)
                self.node.get_logger().info(f"[LiDAR] subscribed to {LIDAR_TOPIC}")
            elif not raw_yolo_view:
                # ZED Mini depth — warp into ZED One's frame and image space.
                # ZED Mini XYZ is in ZED Mini's frame; we must transform every point
                # into ZED One's frame (T_CAM_ZEDMINI) and project to ZED One's image.
                # The result is a dense depth map where bboxes from ZED One YOLO align
                # directly and all XYZ values are in the ZED One camera frame.
                # Thread will be started after image_scale / _K_full / disp_w / disp_h
                # are defined below — store the function now, start it later.
                try:
                    from scipy.ndimage import distance_transform_edt as _edt
                    _have_edt = True
                except ImportError:
                    _have_edt = False

                def _zed_mini_depth_thread():
                    # All closure variables (image_scale, _K_full, _dist_coeffs,
                    # _T_cam_lidar, disp_w, disp_h) are defined before the thread
                    # is actually started, so reading them here is safe.
                    sx, sy = image_scale
                    _K_disp_one = _K_full.copy()
                    _K_disp_one[0, 0] *= sx;  _K_disp_one[0, 2] *= sx
                    _K_disp_one[1, 1] *= sy;  _K_disp_one[1, 2] *= sy
                    pc_mat = sl.Mat()
                    while not self.exit_signal:
                        if self._zed_mini is None:
                            sleep(0.05)
                            continue
                        if self._zed_mini.grab() != sl.ERROR_CODE.SUCCESS:
                            sleep(0.01)
                            continue
                        # Retrieve at native HD1080 — matches ZED One resolution
                        self._zed_mini.retrieve_measure(
                            pc_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
                        raw = pc_mat.get_data()[:, :, :3]   # (H_mini, W_mini, 3) in ZED Mini frame

                        # Flatten to sparse valid points
                        pts = raw.reshape(-1, 3).astype(np.float32)
                        valid_mask_flat = (np.isfinite(pts).all(axis=1) &
                                 (pts[:, 2] > ZEDMINI_DEPTH_Z_MIN) &
                                 (pts[:, 2] < ZEDMINI_DEPTH_Z_MAX))
                        pts_valid = pts[valid_mask_flat]

                        # Transform ZED Mini → ZED One frame and project to ZED One image
                        depth_map = np.full((disp_h, disp_w, 3), np.nan, dtype=np.float32)
                        _raw_mini_pts = np.empty((0, 3), np.float32)
                        _raw_mini_uv  = np.empty((0, 2), np.float32)
                        if pts_valid.shape[0] > 0:
                            pts_one, uv_one = project_lidar_to_image(
                                pts_valid, _K_disp_one, _dist_coeffs, _T_cam_lidar,
                                disp_w, disp_h,
                                z_min=ZEDMINI_DEPTH_Z_MIN, z_max=ZEDMINI_DEPTH_Z_MAX,
                            )
                            if pts_one.shape[0] > 0:
                                xs = uv_one[:, 0].astype(np.int32)
                                ys = uv_one[:, 1].astype(np.int32)
                                # Where multiple points hit the same pixel keep the closest
                                order = np.argsort(pts_one[:, 2])[::-1]  # far→close
                                # Transform back to ZED Mini frame — pixel alignment stays
                                # in ZED One image space but 3D coords are in ZED Mini frame
                                # so the well-calibrated ZED Mini hand-eye TF applies directly.
                                T_mini_one = np.linalg.inv(_T_cam_lidar)
                                ones_col = np.ones((pts_one.shape[0], 1), dtype=np.float32)
                                pts_one_h = np.hstack([pts_one, ones_col])
                                pts_mini = (T_mini_one @ pts_one_h.T).T[:, :3].astype(np.float32)
                                depth_map[ys[order], xs[order]] = pts_mini[order]
                                # ── Raw points for direct query (no EDT fill) ─────
                                # pts_one: ZED One (CAM_FRAME) 3D coords — correct frame for TF to base_link.
                                # uv_one:  corresponding pixel positions in ZED One image.
                                # Subsample to ZEDMINI_MAX_POINTS — bbox filtering in the
                                # main loop iterates all points per target, so keeping
                                # hundreds of thousands of points makes it very slow.
                                if pts_one.shape[0] > ZEDMINI_MAX_POINTS:
                                    _sub_idx = np.random.choice(pts_one.shape[0], ZEDMINI_MAX_POINTS, replace=False)
                                    _raw_mini_pts = pts_one[_sub_idx]
                                    _raw_mini_uv  = uv_one[_sub_idx]
                                else:
                                    _raw_mini_pts = pts_one
                                    _raw_mini_uv  = uv_one

                        # ── Hole filling (for heatmap visualisation only) ─────
                        # Both cameras now at HD1080 with similar FOVs, so
                        # scatter coverage is high (~95%+). Fill remaining gaps
                        # (occlusion boundaries, textureless regions) up to 50px.
                        # Save the pre-fill mask so RViz only shows real points.
                        _valid_z = np.isfinite(depth_map[:, :, 2])
                        orig_valid = _valid_z.copy()
                        n_valid = int(_valid_z.sum())
                        if _have_edt and n_valid > 0 and not _valid_z.all():
                            _dist, _idx = _edt(
                                ~_valid_z,
                                return_distances=True,
                                return_indices=True,
                            )
                            fill = (_dist > 0) & (_dist <= 50.0)
                            r_src = _idx[0][fill]
                            c_src = _idx[1][fill]
                            depth_map[fill, 0] = depth_map[r_src, c_src, 0]
                            depth_map[fill, 1] = depth_map[r_src, c_src, 1]
                            depth_map[fill, 2] = depth_map[r_src, c_src, 2]

                        with self._depth_map_lock:
                            self._latest_depth_map = depth_map
                            self._latest_depth_map_orig = orig_valid
                        with self._mini_pts_lock:
                            self._mini_pts_buffer.append((_raw_mini_pts, _raw_mini_uv, time()))
                            if len(self._mini_pts_buffer) > self._mini_pts_buffer_size:
                                self._mini_pts_buffer.pop(0)
                _pending_depth_thread = _zed_mini_depth_thread
        else:
            # ── ZED stereo path, including ZED X Mini-only RGBD mode ────────
            if zed is None:
                rclpy.shutdown()
                return
            camera_infos = zed.get_camera_information()
            cam_res = camera_infos.camera_configuration.resolution
            cam_w, cam_h = cam_res.width, cam_res.height
            left_cam = camera_infos.camera_configuration.calibration_parameters.left_cam
            fx, fy = float(left_cam.fx), float(left_cam.fy)
            cx, cy = float(left_cam.cx), float(left_cam.cy)
            disto  = left_cam.disto

        intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        # Intrinsics matrix and depth-sensor extrinsic — used for point-cloud projection
        _K_full = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        _dist_coeffs = np.array(disto[:5], dtype=np.float64)
        if use_zed_mini:
            _T_cam_lidar = np.array(T_CAM_ZEDMINI, dtype=np.float64)
        else:
            _T_cam_lidar = np.array(T_CAM_LIDAR, dtype=np.float64)

        disp_w = cam_w
        disp_h = cam_h
        self._disp_w = disp_w
        self._disp_h = disp_h
        display_resolution = sl.Resolution(disp_w, disp_h)
        image_left_ocv = np.full((disp_h, disp_w, 4), [245, 239, 239, 255], np.uint8)
        image_scale = [disp_w / cam_w, disp_h / cam_h]
        display_scale = 0.5
        image_left = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters() if not use_mono_depth else None
        objects = sl.Objects() if not use_mono_depth else None
        point_cloud = sl.Mat() if not use_mono_depth else None

        self.visualizer = VisionVisualizer(
            intrinsics,
            image_scale,
            display_scale,
            clean_harvest_overlay=True,
        )
        self.visualizer.show_classification_zones = self._show_classification_zones
        self.visualizer.show_gap_debug = self._show_gap_debug

        # Start ZED Mini depth warp thread now that all closure variables are defined
        if _pending_depth_thread is not None:
            self.node.get_logger().info("[ZedMini] depth warp thread starting")
            Thread(target=_pending_depth_thread, daemon=True).start()

        loop_fps = 0.0
        # Report camera acquisition throughput over a short rolling window.
        # A single frame that consumes a fresh detection can legitimately take
        # longer, but that must not make the UI report that one interval as the
        # throughput of the entire acquisition loop.
        _grab_times = deque(maxlen=30)

        # Viz queue: main loop drops frames here; background thread renders + publishes.
        # maxsize=1 means the main loop never blocks — old frames are dropped automatically.
        _viz_queue: queue.Queue = queue.Queue(maxsize=1)

        def _viz_worker():
            """Background thread: render and publish vision images without blocking main loop."""
            while not self.exit_signal:
                try:
                    frame_data = _viz_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                (img_ocv, tgts, rej_tgts, b_idx, net_fps, l_fps,
                 viz_only, uv_lidar, pts_lidar, raw_timing) = frame_data
                try:
                    _vt0 = time()
                    if bool(getattr(self.args, "raw_yolo_view", False)):
                        # Deliberately isolated inspection view: no VisionVisualizer,
                        # ROS image conversion/publication, fingertip verification,
                        # target tracking, depth, or TF work.
                        display_image = cv2.cvtColor(img_ocv, cv2.COLOR_BGRA2BGR)
                        for item in viz_only:
                            polygon = item.get("polygon")
                            if polygon is not None and len(polygon) > 2:
                                cv2.polylines(
                                    display_image,
                                    [np.asarray(polygon, dtype=np.int32)],
                                    True, (0, 220, 255), 2)
                            x1, y1, x2, y2 = item["bb"]
                            cv2.rectangle(display_image, (x1, y1), (x2, y2),
                                          (0, 255, 0), 2)
                            label = f"{item['class']} {item['conf']:.2f}"
                            cv2.putText(display_image, label, (x1, max(18, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                        (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(
                            display_image, f"YOLO {net_fps:.1f} FPS",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 0), 2, cv2.LINE_AA)
                        if raw_timing:
                            age_ms = max(
                                0.0, (time() - raw_timing["capture_time"]) * 1000.0)
                            timing_text = (
                                f"frame {raw_timing['frame_id']} age {age_ms:.0f}ms | "
                                f"pre {raw_timing['preprocess_ms']:.1f} "
                                f"infer {raw_timing['inference_ms']:.1f} "
                                f"post {raw_timing['postprocess_ms']:.1f} "
                                f"maskCPU {raw_timing['decode_mask_ms']:.1f}ms | "
                                f"dropped/30 {raw_timing['dropped_interval']}")
                            cv2.putText(
                                display_image, timing_text, (12, 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (255, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("Raw YOLO Model View", display_image)
                        key = cv2.waitKey(1) & 0xFF
                        try:
                            window_closed = cv2.getWindowProperty(
                                "Raw YOLO Model View", cv2.WND_PROP_VISIBLE) < 1
                        except cv2.error:
                            window_closed = True
                        if key in (ord("q"), 27) or window_closed:
                            self._request_exit()
                        continue
                    stamp = self.node.get_clock().now().to_msg()
                    raw_stream_active = (
                        self._raw_stream_enabled or time() < self._raw_stream_until)
                    if raw_stream_active:
                        if len(img_ocv.shape) == 3 and img_ocv.shape[2] == 4:
                            raw_pub_image = cv2.cvtColor(img_ocv, cv2.COLOR_BGRA2BGR)
                        else:
                            raw_pub_image = img_ocv
                        raw_msg = self.cv_bridge.cv2_to_imgmsg(raw_pub_image, encoding="bgr8")
                        raw_msg.header.stamp = stamp
                        self.raw_image_pub.publish(raw_msg)
                    if not getattr(self, "_printed_bottom_pixel_pre_render", False):
                        _bottom = img_ocv[-12:, :, :3]
                        print(
                            "bottom_pre_render mean",
                            np.round(_bottom.mean(axis=(0, 1)), 1).tolist(),
                            "min",
                            _bottom.min(axis=(0, 1)).tolist(),
                            "max",
                            _bottom.max(axis=(0, 1)).tolist(),
                        )
                        self._printed_bottom_pixel_pre_render = True
                    display_image = self.visualizer.render_frame(
                        img_ocv, tgts, rej_tgts,
                        b_idx, net_fps, l_fps,
                        viz_only=viz_only,
                        lidar_uv=uv_lidar,
                        lidar_pts_cam=pts_lidar,
                    )
                    verification = self.visualizer.last_fingertip_verification
                    self.fingertip_verification_pub.publish(
                        StdString(data=verification))
                    _vt1 = time()
                    if not getattr(self, "_printed_bottom_pixel_post_render", False):
                        _bottom = display_image[-12:, :, :3]
                        print(
                            "bottom_post_render mean",
                            np.round(_bottom.mean(axis=(0, 1)), 1).tolist(),
                            "min",
                            _bottom.min(axis=(0, 1)).tolist(),
                            "max",
                            _bottom.max(axis=(0, 1)).tolist(),
                        )
                        self._printed_bottom_pixel_post_render = True
                    if len(display_image.shape) == 3 and display_image.shape[2] == 4:
                        pub_image = cv2.cvtColor(display_image, cv2.COLOR_BGRA2BGR)
                    else:
                        pub_image = display_image
                    _vt2 = time()
                    if not getattr(self, "_printed_bottom_pixel_pub_image", False):
                        _bottom = pub_image[-12:, :, :3]
                        print(
                            "bottom_pub_image mean",
                            np.round(_bottom.mean(axis=(0, 1)), 1).tolist(),
                            "min",
                            _bottom.min(axis=(0, 1)).tolist(),
                            "max",
                            _bottom.max(axis=(0, 1)).tolist(),
                        )
                        self._printed_bottom_pixel_pub_image = True
                    img_msg = self.cv_bridge.cv2_to_imgmsg(pub_image, encoding="bgr8")
                    _vt3 = time()
                    img_msg.header.stamp = stamp
                    self.image_pub.publish(img_msg)
                    _vt4 = time()
                    _viz_count = getattr(self, '_viz_perf_count', 0) + 1
                    self._viz_perf_count = _viz_count
                    if DEBUG_PERFORMANCE and _viz_count % 20 == 0:
                        print(f"[VIZ_PERF] render={(_vt1-_vt0)*1000:.0f}ms  "
                              f"cvt={(_vt2-_vt1)*1000:.0f}ms  "
                              f"encode={(_vt3-_vt2)*1000:.0f}ms  "
                              f"publish={(_vt4-_vt3)*1000:.0f}ms  "
                              f"total={(_vt4-_vt0)*1000:.0f}ms  "
                              f"img={pub_image.shape[1]}x{pub_image.shape[0]}")
                except Exception as e:
                    if not getattr(self, '_img_pub_err_logged', False):
                        print(f"[WARN] Failed to publish vision image: {e}")
                        self._img_pub_err_logged = True

        Thread(target=_viz_worker, daemon=True).start()

        def perception_loop():
            nonlocal loop_fps, zed
            t_prev = time()
            current_dets = None
            current_raw_viz = []
            cached_processed_targets = None
            trunk_boxes  = []
            bunch_boxes  = []
            _YOLO_STALE_DRIFT = 0.008  # 8 mm — discard cached dets if camera drifted this far
            _last_viz_t = 0.0          # wall time of last visualization enqueue
            # Keep RViz/control local on Jetson, but cap only the expensive
            # annotated camera encoding/publication to 5 FPS. Detection, robot
            # state, commands and motion feedback retain their normal rates.
            _viz_fps = max(
                0.0, float(os.getenv("UR10E_VISION_DISPLAY_FPS", "5.0")))
            _VIZ_MIN_INTERVAL = 1.0 / _viz_fps if _viz_fps > 0.0 else float("inf")
            self.node.get_logger().info(
                f"Annotated vision display capped at {_viz_fps:.1f} FPS "
                "(set UR10E_VISION_DISPLAY_FPS to override).")
            _initial_voxel_cloud_sent = False
            _initial_voxel_cloud_burst_remaining = 3
            _depth_capture_pending = False
            _full_profile_count = 0

            def _enqueue_viz_frame(targets, rejected_targets, best_idx, viz_only,
                                   uv_lidar=None, pts_lidar=None,
                                   source_image=None, raw_timing=None):
                """Queue one visualization frame without blocking the perception loop."""
                nonlocal _last_viz_t
                _now = time()
                if _now - _last_viz_t < _VIZ_MIN_INTERVAL or _viz_queue.full():
                    return
                try:
                    _viz_queue.put_nowait((
                        (source_image.copy() if source_image is not None
                         else image_left_ocv.copy()),
                        targets, rejected_targets,
                        best_idx,
                        self.yolo_thread.net_fps,
                        loop_fps,
                        viz_only,
                        uv_lidar,
                        pts_lidar,
                        raw_timing,
                    ))
                    _last_viz_t = _now
                except queue.Full:
                    pass

            def _enqueue_raw_viz_frame():
                _enqueue_viz_frame([], [], None, [], None, None)

            # ZED X Mini capture must not be paced by per-date geometry.  Keep
            # all SDK grab/retrieve calls on one dedicated thread (the ZED SDK
            # objects are not shared across threads) and expose a depth-1 latest
            # packet queue to perception.  If target processing is slower than
            # the camera, stale RGB-D packets are replaced instead of delaying
            # the next camera grab.
            _capture_queue = queue.Queue(maxsize=1)
            _capture_depth_request = Event()
            _capture_thread = None
            if use_zedx_mini_only:
                def _capture_zedx_mini():
                    _cap_image = sl.Mat()
                    _cap_cloud = sl.Mat()
                    _cap_times = deque(maxlen=30)
                    _cap_runtime = sl.RuntimeParameters()
                    while not self.exit_signal:
                        _want_depth = (
                            not bool(getattr(self.args, "raw_yolo_view", False))
                            and _capture_depth_request.is_set())
                        _cap_runtime.enable_depth = _want_depth
                        if zed.grab(_cap_runtime) != sl.ERROR_CODE.SUCCESS:
                            self.exit_signal = True
                            break
                        _cap_t = time()
                        _cap_times.append(_cap_t)
                        _cap_fps = 0.0
                        if len(_cap_times) >= 2:
                            _span = _cap_times[-1] - _cap_times[0]
                            if _span > 0.0:
                                _cap_fps = (len(_cap_times) - 1) / _span
                        zed.retrieve_image(
                            _cap_image, sl.VIEW.LEFT, sl.MEM.CPU,
                            sl.Resolution(disp_w, disp_h))
                        if _want_depth:
                            zed.retrieve_measure(
                                _cap_cloud, sl.MEASURE.XYZ, sl.MEM.CPU,
                                sl.Resolution(disp_w, disp_h))
                            _cap_pc = _cap_cloud.get_data()[:, :, :3].copy()
                        else:
                            _cap_pc = None
                        _packet = (
                            _cap_t,
                            _cap_fps,
                            _cap_image.get_data().copy(),
                            _cap_pc,
                        )
                        try:
                            _capture_queue.put_nowait(_packet)
                        except queue.Full:
                            try:
                                _capture_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                _capture_queue.put_nowait(_packet)
                            except queue.Full:
                                pass
                        if _want_depth:
                            _capture_depth_request.clear()
                            # Do not let the following RGB-only packet replace
                            # this requested depth packet before perception has
                            # consumed it.
                            _wait_deadline = time() + 0.2
                            while (not self.exit_signal and
                                   not _capture_queue.empty() and
                                   time() < _wait_deadline):
                                sleep(0.001)

                _capture_thread = Thread(
                    target=_capture_zedx_mini,
                    name="zedx_mini_capture",
                    daemon=True,
                )
                _capture_thread.start()

            while not self.exit_signal:
                fresh_detections = False
                # ZED X Mini neural depth is the other major GPU workload. In
                # full harvesting mode, compute it only for a frame that will
                # consume a newly completed YOLO result. RGB grabbing remains
                # at camera rate. If inference completes during a no-depth grab,
                # defer that result for one iteration and capture depth next.
                _depth_enabled_this_grab = True
                _captured_pc_np = None
                if use_zedx_mini_only:
                    try:
                        t_now, loop_fps, image_left_ocv, _captured_pc_np = (
                            _capture_queue.get(timeout=1.0))
                    except queue.Empty:
                        continue
                else:
                    grab_status = zed.grab() if use_mono_depth else zed.grab(runtime_params)
                    if grab_status != sl.ERROR_CODE.SUCCESS:
                        self.exit_signal = True
                        break
                    t_now = time()
                self._latest_zed_one_ts = t_now
                # Stamp this frame with when it was CAPTURED, not when the loop
                # got round to it. _lookup_tf_at_image() exists precisely so a
                # detection is transformed with the arm pose from the moment of
                # the photo; handing it "now" fed it the very staleness it was
                # written to remove.
                #
                # The gap is not hypothetical in mini-only mode: t_now is
                # recorded in the capture thread right after grab() returns, and
                # the packet then crosses a depth-1 queue whose depth-request
                # handshake can hold it for up to 200ms (see _wait_deadline
                # above). At 0.3 m/s tool speed every 10ms of stamp error is
                # ~3mm of position error.
                #
                # Residual: the sensor->grab-return latency inside the SDK is
                # still counted as zero. zed.get_timestamp(TIME_REFERENCE.IMAGE)
                # would remove that too, but it has to be read on the capture
                # thread and carried in the packet; t_now removes the large,
                # variable term, which is the one that matters.
                self._latest_image_stamp = self._ros_time_from_wall(t_now)
                if not use_zedx_mini_only:
                    _grab_times.append(t_now)
                    if len(_grab_times) >= 2:
                        _grab_span = _grab_times[-1] - _grab_times[0]
                        loop_fps = ((len(_grab_times) - 1) / _grab_span
                                    if _grab_span > 0.0 else 0.0)
                t_prev = t_now

                _t0 = time()
                # Retrieve once at display resolution — YOLO resizes internally so
                # full QHDPLUS is wasted bandwidth. One retrieve serves both YOLO and display.
                if use_zedx_mini_only:
                    pass  # RGB and aligned depth were copied by the capture owner.
                elif use_mono_depth:
                    zed.retrieve_image(image_left,
                                       resolution=sl.Resolution(disp_w, disp_h))
                else:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU,
                                       sl.Resolution(disp_w, disp_h))
                # This SDK buffer stays valid until the next grab. Synchronous
                # depth/target processing finishes before then; background YOLO
                # and visualization make owned copies when queued. Avoid an
                # extra full-HD BGRA copy on every camera frame.
                if not use_zedx_mini_only:
                    image_left_ocv = image_left.get_data()
                if not getattr(self, "_printed_zed_buffer_shape", False):
                    print("buffer", image_left_ocv.shape, "zed", image_left_ocv.shape)
                    self._printed_zed_buffer_shape = True
                # "paused" mode: no inference at all (arm is moving, GPU needed for cuRobo).
                # "reacquire" mode: every 3rd frame (planner busy).
                # "reacquire_fast": every frame (arm stopped, GPU free).
                # "full" mode: every frame.
                _mode = self.detection_mode
                _paused = _mode == "paused"
                _reacquire = _mode in ("reacquire", "reacquire_fast")
                # YOLO submission throttle while reacquiring. The every-3rd-frame
                # version exists to leave GPU headroom for cuRobo, and at the
                # slip-check site that is real: _bg_preplan_dropoff is still
                # planning while that reacquire runs. At the approach sites it
                # protects nothing -- the arm is stopped and no preplan thread
                # exists yet -- and it starves reacquire of exactly the samples
                # it needs, which is why "[REACQ] no new perception sample" fires.
                # The caller picks, via reacquire_goal_pose(planner_busy=...).
                _reacq_skip_n = 1 if _mode == "reacquire_fast" else 3
                if _paused:
                    # Keep the display live during robot motion, but skip stale
                    # detections/depth/heatmap work so cuRobo keeps the GPU.
                    # On entering paused, wait for any in-flight inference to drain,
                    # then confirm to the motion node that the GPU is free.
                    if not self._vision_paused_acked:
                        self.yolo_thread.paused = True
                        self.yolo_thread.wait_until_idle(timeout=0.5)
                        self._publish_mode_state("paused")
                        self._vision_paused_acked = True
                    current_dets = None
                    _enqueue_raw_viz_frame()
                    continue
                else:
                    if self._vision_paused_acked:
                        self.yolo_thread.paused = False
                        self._vision_paused_acked = False
                    self._reacquire_frame_skip = (
                        (self._reacquire_frame_skip + 1) % _reacq_skip_n)
                    _waiting_for_depth = (
                        use_zedx_mini_only and
                        self.yolo_thread.dets_ready.is_set())
                    if ((not _reacquire or self._reacquire_frame_skip == 0) and
                            not _waiting_for_depth):
                        # The YOLO worker owns a newest-frame queue, so this is
                        # non-blocking and replaces stale pending RGB frames.
                        # Submit before depth/geometry work to overlap inference
                        # with CPU target extraction.
                        self.yolo_thread.set_image(image_left_ocv, capture_time=t_now)
                _t1 = time()

                # Raw model inspection intentionally stops here. It uses only
                # RGB acquisition, TensorRT inference and the small CV2 overlay.
                # In particular, do not touch TF, depth, tracking or goals.
                if bool(getattr(self.args, "raw_yolo_view", False)):
                    if self.yolo_thread.dets_ready.is_set():
                        self.yolo_thread.dets_ready.clear()
                        raw_result = self.yolo_thread.get_raw_result()
                        if raw_result is not None:
                            current_raw_viz = raw_result["detections"]
                            _enqueue_viz_frame(
                                [], [], None, current_raw_viz,
                                source_image=raw_result["image"],
                                raw_timing=raw_result["timing"])
                    continue

                # Refresh per-frame TF cache — one lookup per frame instead of
                # one per detection (was 8+ tf_buffer.transform calls at ~20ms each).
                try:
                    self._cached_tf_base = self._lookup_tf_at_image("base_link")
                except Exception:
                    pass  # keep previous cached value
                try:
                    self._cached_tf_grip = self._lookup_tf_at_image("gripper_tip")
                except Exception:
                    pass

                # Motion detection: when the camera has moved >5mm since the last
                # frame, clear all temporal state so detections update immediately.
                # This prevents stale bboxes, old heatmaps, and smoothing history
                # from lagging behind after a robot repositioning move.
                if self._cached_tf_base is not None:
                    _cur_t = self._cached_tf_base[1]
                    _prev_t = getattr(self, '_prev_cam_t', None)
                    if _prev_t is not None and float(np.linalg.norm(_cur_t - _prev_t)) > 0.005:
                        self.tracker.detection_history.clear()
                        self.direction_history.clear()
                        self.best_history.clear()
                        self._cached_heatmaps.clear()
                        self._contact_cache.clear()
                        self.prev_heat_point = None
                        self.prev_direction_base = None
                        current_dets = None  # discard stale bboxes; wait for fresh YOLO
                    self._prev_cam_t = _cur_t.copy()

                # Use fresh YOLO dets when ready, otherwise reuse cached dets so
                # the loop runs at camera fps rather than YOLO inference fps.
                _cur_t_now = self._cached_tf_base[1] if self._cached_tf_base is not None else None
                if self.yolo_thread.dets_ready.is_set():
                    if use_zedx_mini_only and _captured_pc_np is None:
                        # Hold this completed detection and ask the capture owner
                        # for exactly one aligned depth packet. Do not start a
                        # newer inference until the held result is consumed.
                        _capture_depth_request.set()
                        continue
                    self.yolo_thread.dets_ready.clear()
                    _depth_capture_pending = False
                    current_dets = self.yolo_thread.get_detections()
                    current_raw_viz = self.yolo_thread.get_raw_viz()
                    _latest_yolo_timing = self.yolo_thread.get_latest_timing()
                    fresh_detections = True
                    trunk_boxes  = self.yolo_thread.get_trunk_boxes()
                    # Record where the camera was when this result was accepted
                    self._yolo_accepted_cam_t = _cur_t_now.copy() if _cur_t_now is not None else None
                elif current_dets is None:
                    # No YOLO result yet — publish raw frame so display stays live.
                    _enqueue_raw_viz_frame()
                    continue
                elif use_zedx_mini_only:
                    # Geometry and goals update only for a fresh detection. RGB
                    # acquisition continues independently at camera rate.
                    continue
                else:
                    # Cumulative-drift check: discard cached dets if camera has drifted
                    # since the last accepted YOLO result (handles slow/incremental moves
                    # that never cross the per-frame 5 mm threshold).
                    if (_cur_t_now is not None and self._yolo_accepted_cam_t is not None and
                            float(np.linalg.norm(_cur_t_now - self._yolo_accepted_cam_t)) > _YOLO_STALE_DRIFT):
                        current_dets = None
                        # Camera has moved — publish raw frame so display stays live during motion.
                        _enqueue_raw_viz_frame()
                        continue
                    if cached_processed_targets is not None:
                        # No new YOLO observation: the target/depth result is
                        # already cached. Previously we still retrieved and
                        # traversed the full HD1080 XYZ map at camera FPS, then
                        # discarded the duplicate processing result. Wait for
                        # the next inference result instead. This preserves all
                        # new-detection updates while freeing CPU/memory bandwidth.
                        continue
                bunch_boxes = self.yolo_thread.get_bunch_boxes()

                pc_np_orig = None
                _depth_t0 = time()
                if use_zedx_mini_only:
                    # Already aligned to this packet by the capture owner.
                    pts_cam_l = np.empty((0, 3), np.float32)
                    uv_l = np.empty((0, 2), np.float32)
                    pc_np = _captured_pc_np
                elif use_zed_mini:
                    # ── ZED Mini dense depth path ─────────────────────────
                    # Dense HxW depth map at ZED One display resolution —
                    # same pixel grid as the image, so bboxes align directly.
                    # Uses the full heatmap pipeline (surface normal, approach
                    # direction, clearance) identical to the ZED stereo path.
                    pts_cam_l = np.empty((0, 3), np.float32)
                    uv_l = np.empty((0, 2), np.float32)
                    with self._depth_map_lock:
                        pc_np = self._latest_depth_map
                        pc_np_orig = self._latest_depth_map_orig
                elif use_lidar:
                    # ── LiDAR sparse depth path ───────────────────────────
                    pc_np = None
                    sx, sy = image_scale
                    K_disp = _K_full.copy()
                    K_disp[0, 0] *= sx; K_disp[0, 2] *= sx
                    K_disp[1, 1] *= sy; K_disp[1, 2] *= sy
                    with self._cloud_lock:
                        pts_lidar = self._latest_cloud
                    if pts_lidar is not None and pts_lidar.shape[0] > 0:
                        pts_cam_l, uv_l = project_lidar_to_image(
                            pts_lidar, K_disp, _dist_coeffs, _T_cam_lidar,
                            disp_w, disp_h,
                            z_min=LIDAR_Z_MIN, z_max=LIDAR_Z_MAX,
                        )
                    else:
                        pts_cam_l = np.empty((0, 3), np.float32)
                        uv_l = np.empty((0, 2), np.float32)
                else:
                    # ── ZED stereo depth path ─────────────────────────────
                    pts_cam_l = np.empty((0, 3), np.float32)
                    uv_l = np.empty((0, 2), np.float32)
                    # Mini-only RGBD is already pixel-aligned and YOLO already
                    # supplies the fruit boxes and masks.  Sending those masks
                    # through ZED custom-object ingestion/tracking adds another
                    # per-object pass and makes loop time grow with date count.
                    # Keep that SDK path only for the legacy stereo mode.
                    if not use_zedx_mini_only:
                        zed.ingest_custom_mask_objects(current_dets)
                        zed.retrieve_custom_objects(objects, obj_runtime_param)
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU,
                                        sl.Resolution(disp_w, disp_h))
                    pc_np = point_cloud.get_data()[:, :, :3]
                _depth_ms = (time() - _depth_t0) * 1000.0

                # Process detected objects
                self._heatmap_frame_count += 1
                # Heatmap is computed once per fruit (see target_key logic below);
                # periodic clear removed — camera-movement clear (above) is the
                # correct invalidation trigger when the robot repositions.
                _t2 = time()
                targets, rejected_targets, viz_only = self._process_objects(
                    current_dets if (use_mono_depth or use_zedx_mini_only) else objects,
                    pc_np, image_left_ocv, image_scale, display_resolution, intrinsics,
                    pts_cam=pts_cam_l, uv=uv_l,
                    use_lidar=use_lidar,       # True only for actual LiDAR
                    use_zed_mini=use_zed_mini, # raw dets + dense depth heatmap
                    use_raw_detections=use_zedx_mini_only,

                )
                _t3 = time()

                if fresh_detections:
                    # Update temporal state once per new inference result, not
                    # once per reused camera-loop detection.
                    targets = self.tracker.stabilize_detections(targets)

                if fresh_detections:
                    _full_profile_count += 1
                    if _full_profile_count % 30 == 0:
                        _yt = _latest_yolo_timing or {}
                        _target_ms = (_t3 - _t2) * 1000.0
                        _total_ms = (
                            float(_yt.get("predict_wall_ms", 0.0)) +
                            _depth_ms + _target_ms)
                        print(
                            "[VISION_PIPELINE_TIMING] "
                            f"id={_yt.get('frame_id', -1)} "
                            f"yolo={float(_yt.get('predict_wall_ms', 0.0)):.1f}ms "
                            f"depth={_depth_ms:.1f}ms "
                            f"targets={_target_ms:.1f}ms "
                            f"serial_sum={_total_ms:.1f}ms "
                            f"yolo_fps={self.yolo_thread.net_fps:.1f} "
                            f"loop_fps={loop_fps:.1f} "
                            f"detections={len(current_dets) if current_dets is not None else 0} "
                            f"extracted={getattr(self, '_last_extracted_target_count', 0)} "
                            f"tracked={len(targets)} rejected={len(rejected_targets)} "
                            f"objects_ms=[{','.join(f'{v:.1f}' for v in getattr(self, '_last_object_processing_ms', []))}]"
                        )
                _tp1 = time()

                # --- mode-gated operations -------------------------------------------
                # "full" mode  : all operations (approach, scoring, heatmap, viz)
                # "reacquire"  : fruit positions only — skip trunk, heatmap direction, viz

                # Publish trunk position every 5 frames — skip in reacquire mode
                trunk_published = False
                if not _reacquire and self._heatmap_frame_count % 5 == 0:
                    self._publish_trunk_position(
                        trunk_boxes, pc_np, image_scale, image_left_ocv, trunk_pub,
                        pts_cam=pts_cam_l, uv=uv_l, use_lidar=use_lidar,
                        trunk_cam_pub=trunk_cam_pub,
                    )
                if (not _reacquire and not _initial_voxel_cloud_sent and
                        _initial_voxel_cloud_burst_remaining > 0 and pc_np is not None):
                    # Do not spend the one-shot burst before the motion node has
                    # subscribed; otherwise manual voxel updates have no cached depth.
                    _has_depth_sub = depth_pub.get_subscription_count() > 0
                    _published_depth = False
                    if _has_depth_sub:
                        _published_depth = self._publish_depth_cloud(
                            pc_np, depth_pub,
                            orig_mask=pc_np_orig if use_zed_mini else None,
                        )
                    if _published_depth:
                        _initial_voxel_cloud_burst_remaining -= 1
                    if _initial_voxel_cloud_burst_remaining <= 0:
                        _initial_voxel_cloud_sent = True
                        self.node.get_logger().info(
                            "Initial voxel depth-cloud burst complete; disabling continuous depth cloud publish."
                        )
                _tp2 = time()

                # Select best fruit
                best_idx = self._select_best_fruit(targets)
                _tp3 = time()

                # Compute approach direction and publish — skip in reacquire mode
                if best_idx is not None and not _reacquire:
                    self._process_best_target(
                        targets, best_idx, intrinsics, bunch_boxes, pc_np=pc_np)
                _tp4 = time()

                # Publish visible fruit positions sorted by score, highest first.
                # Reacquire can still search all visible fruits, and goal multi can
                # queue top-scoring dates deterministically.
                # Format: [x0,y0,z0, x1,y1,z1, ...] in base_link frame.
                _all_flat = []
                _targets_by_score = sorted(
                    targets,
                    key=lambda _t: float(_t.get("score", 0.0)),
                    reverse=True,
                )
                # Quality per fruit, same order, 5 floats each:
                #   vis_ratio, z_std, confidence, score, edge_margin_norm
                # edge_margin_norm is how far the bbox sits from the nearest
                # image border, as a fraction of the half-min image dimension.
                # 0.0 means the box is clipped by the frame -- the direct signal
                # for "the date is not fully visible", which is the failure mode
                # at close range as the off-axis camera converges on the fruit.
                _qual_flat = []
                for _t in _targets_by_score:
                    _pb = _t.get("pt_base")
                    if _pb is None:
                        continue
                    _all_flat.extend([_pb.point.x, _pb.point.y, _pb.point.z])
                    _bb = _t.get("bb")
                    _iw = float(_t.get("img_width", 0) or 0)
                    _ih = float(_t.get("img_height", 0) or 0)
                    if _bb is not None and _iw > 0 and _ih > 0:
                        _bx1, _by1, _bx2, _by2 = (float(v) for v in _bb)
                        _edge_px = min(_bx1, _by1, _iw - _bx2, _ih - _by2)
                        _half = max(1.0, min(_iw, _ih) / 2.0)
                        _edge_norm = max(0.0, min(1.0, _edge_px / _half))
                    else:
                        _edge_norm = 1.0   # unknown -> do not penalise
                    _qual_flat.extend([
                        float(_t.get("vis_ratio", 0.0)),
                        float(_t.get("z_std", 0.0)),
                        float(_t.get("confidence", 0.0)),
                        float(_t.get("score", 0.0)),
                        float(_edge_norm),
                    ])
                # Publish EVERY processed frame, including when nothing was
                # detected. Suppressing the message on "no detections" left
                # every consumer holding the last non-empty list indefinitely,
                # with no way to distinguish "nothing is visible" from "vision
                # has not looked yet".
                #
                # That is what made reacquire report drift=(0,0,0)mm: the frozen
                # list still contained the very detection that produced the
                # seed, so reacquire matched the seed against itself and
                # "succeeded" having learned nothing. An empty publish makes it
                # time out honestly and fall back to the seed instead, which is
                # the correct outcome when there is nothing to see.
                #
                # goal-multi guards with `if visible:` and reacquire iterates,
                # so an empty list is safe for both consumers.
                _af_msg = Float32MultiArray()
                _af_msg.data = _all_flat
                self.all_fruits_pub.publish(_af_msg)
                _aq_msg = Float32MultiArray()
                _aq_msg.data = _qual_flat
                self.all_fruit_quality_pub.publish(_aq_msg)

                # Crop around the selected date for episode recording. The
                # margin is generous on purpose: the fingertips must be in frame
                # for the crop to say anything about gripper-relative geometry.
                # Throttled -- the recorder only needs the one at grasp time.
                _now_crop = time()
                if (best_idx is not None and 0 <= best_idx < len(targets)
                        and _now_crop - self._last_grasp_crop_t > 0.2):
                    try:
                        _bx1, _by1, _bx2, _by2 = (
                            int(v) for v in targets[best_idx]["bb"])
                        _ih, _iw = image_left_ocv.shape[:2]
                        _mw = max(8, _bx2 - _bx1)
                        _mh = max(8, _by2 - _by1)
                        # Slide the window instead of shrinking it when it
                        # hits a border. A VERY_LOW date sits near the bottom of
                        # the frame, so the naive clamp returned 213 of the 318
                        # requested rows and cut the lower gripper off. Sliding
                        # keeps the crop a consistent size for training and
                        # recovers the context on the opposite side.
                        _cw = min(_iw, (_bx2 - _bx1) + 2 * _mw)
                        _ch = min(_ih, (_by2 - _by1) + 2 * _mh)
                        _cx1 = int(min(max(0, _bx1 - _mw), _iw - _cw))
                        _cy1 = int(min(max(0, _by1 - _mh), _ih - _ch))
                        _cx2 = _cx1 + _cw; _cy2 = _cy1 + _ch
                        if _cx2 > _cx1 + 8 and _cy2 > _cy1 + 8:
                            _crop = image_left_ocv[_cy1:_cy2, _cx1:_cx2]
                            if _crop.ndim == 3 and _crop.shape[2] == 4:
                                _crop = cv2.cvtColor(_crop, cv2.COLOR_BGRA2BGR)
                            _cm = self.cv_bridge.cv2_to_imgmsg(
                                np.ascontiguousarray(_crop), encoding="bgr8")
                            _cm.header.stamp = self._image_stamp_msg()
                            _cm.header.frame_id = self.cam_frame
                            self.grasp_crop_pub.publish(_cm)
                            # date bbox INSIDE the crop, then the crop origin and
                            # full image size, so a label can be reconstructed
                            # either way round.
                            _mm = Float32MultiArray()
                            _mm.data = [
                                float(_bx1 - _cx1), float(_by1 - _cy1),
                                float(_bx2 - _cx1), float(_by2 - _cy1),
                                float(_cx1), float(_cy1),
                                float(_iw), float(_ih),
                            ]
                            self.grasp_crop_meta_pub.publish(_mm)
                            self._last_grasp_crop_t = _now_crop
                    except Exception as _ce:
                        _now_w = time()
                        if _now_w - getattr(self, "_last_crop_warn_t", 0.0) > 5.0:
                            self._last_crop_warn_t = _now_w
                            self.node.get_logger().warn(
                                f"[GRASP_CROP] not published: {_ce}")

                if best_idx is None:
                    # No valid target exists in this frame.  Stop publishing the
                    # previous target regardless of whether exclusions are active.
                    with self.pub_lock:
                        self.latest_goal_msg = None
                        self.latest_dir_msg = None
                        self._latest_goal_update_time = 0.0

                # Build viz entries — skip heavy overlays in reacquire mode but still
                # publish the live frame so the display doesn't freeze.
                if _reacquire:
                    _t4 = time()
                    self._perf_count = getattr(self, '_perf_count', 0) + 1
                    # Strip cached heatmap data so the viz thread doesn't draw stale overlays
                    _HEATMAP_KEYS = ("heatmap", "best_point2d", "best_point2d_smooth",
                                     "approach_dir", "approach_dir_cam", "approach_axis")
                    _tgts_lite = [
                        {k: v for k, v in t.items() if k not in _HEATMAP_KEYS}
                        for t in targets
                    ]
                    _enqueue_viz_frame(
                        _tgts_lite, rejected_targets,
                        best_idx,
                        [],   # no trunk/bunch polygon overlays in reacquire mode
                        uv_l if use_lidar else None,
                        pts_cam_l if use_lidar else None,
                    )
                    continue

                trunk_viz = []
                for bx1, by1, bx2, by2 in trunk_boxes:
                    tx1 = max(0, min(int(bx1 * image_scale[0]), image_left_ocv.shape[1] - 1))
                    ty1 = max(0, min(int(by1 * image_scale[1]), image_left_ocv.shape[0] - 1))
                    tx2 = max(0, min(int(bx2 * image_scale[0]), image_left_ocv.shape[1]))
                    ty2 = max(0, min(int(by2 * image_scale[1]), image_left_ocv.shape[0]))
                    trunk_viz.append({"bb": (tx1, ty1, tx2, ty2), "class": "trunk", "conf": 1.0})
                for b in bunch_boxes:
                    bx1, by1, bx2, by2 = b["bb"]
                    tx1 = max(0, min(int(bx1 * image_scale[0]), image_left_ocv.shape[1] - 1))
                    ty1 = max(0, min(int(by1 * image_scale[1]), image_left_ocv.shape[0] - 1))
                    tx2 = max(0, min(int(bx2 * image_scale[0]), image_left_ocv.shape[1]))
                    ty2 = max(0, min(int(by2 * image_scale[1]), image_left_ocv.shape[0]))
                    polygon = b.get("polygon")
                    polygon_scaled = None
                    if polygon is not None:
                        polygon_scaled = (polygon * np.array([[image_scale[0], image_scale[1]]])).astype(np.int32)
                    trunk_viz.append({"bb": (tx1, ty1, tx2, ty2), "class": "bunch", "conf": b["conf"], "polygon": polygon_scaled})

                _t4 = time()
                if (DEBUG_PERFORMANCE and
                        getattr(self, '_perf_count', 0) % 30 == 0):
                    print(f"[PERF] retrieve={(_t1-_t0)*1000:.0f}ms  "
                          f"process={(_t3-_t2)*1000:.0f}ms  "
                          f"post={(_t4-_t3)*1000:.0f}ms"
                          f"(stab={(_tp1-_t3)*1000:.0f} trunk={(_tp2-_tp1)*1000:.0f}"
                          f" sel={(_tp3-_tp2)*1000:.0f} best={(_tp4-_tp3)*1000:.0f})  "
                          f"total={(_t4-_t0)*1000:.0f}ms")
                self._perf_count = getattr(self, '_perf_count', 0) + 1

                # Enqueue a frame for the background viz thread.
                # Non-blocking: if the worker is busy, skip before copying the image.
                _enqueue_viz_frame(
                    targets, rejected_targets,
                    best_idx,
                    trunk_viz,
                    uv_l if use_lidar else None,
                    pts_cam_l if use_lidar else None,
                )

            if _capture_thread is not None:
                _capture_thread.join(timeout=2.0)

        perception_thread = Thread(target=perception_loop, daemon=True)
        perception_thread.start()

        try:
            self.executor.add_node(self.node)
            self.executor.spin()
        finally:
            self.exit_signal = True
            if bool(getattr(self.args, "raw_yolo_view", False)):
                cv2.destroyAllWindows()
            perception_thread.join(timeout=3.0)
            if self.yolo_thread is not None:
                self.yolo_thread.stop()
                self.yolo_thread.stopped.wait(timeout=5.0)
                self.yolo_thread = None
            if self._zed_mini is not None:
                self._zed_mini.close()
                self._zed_mini = None
            if zed is not None:
                zed.close()
            # Safe for CV-window exit, Ctrl-C and executor-triggered shutdown.
            # More than one path may have already shut the shared context down.
            self.node.context.try_shutdown()
            if getattr(self, '_refresh_requested', False):
                restart_argv = getattr(self, "_refresh_argv", None) or sys.argv
                reason = getattr(self, "_refresh_reason", "camera refresh")
                print(f"[Vision] Restarting process for {reason}...")
                sleep(4.0)  # let ZED/Argus driver release fully before exec
                os.execv(sys.executable, [sys.executable] + restart_argv)

    def _init_zed_and_yolo(self):
        """Initialize ZED camera then start YOLO thread. Returns zed or None on failure.
        ZED must be fully open before YOLO TRT loads — they share the GPU and
        simultaneous TRT initialization causes a segfault."""
        use_lidar    = getattr(self.args, "use_lidar",    False)
        use_zed_mini = getattr(self.args, "use_zed_mini", False)
        use_zedx_mini_only = getattr(self.args, "use_zedx_mini_only", False)
        use_mono_depth = use_lidar or use_zed_mini

        if use_mono_depth:
            # ZED X One Mono — CameraOne API (pyzed.sl)
            print("Initializing ZED X One Mono (detection camera)...")
            zed = sl.CameraOne()
            init_params = sl.InitParametersOne()
            init_params.camera_resolution = sl.RESOLUTION.QHDPLUS
            init_params.camera_fps = 15
            init_params.coordinate_units = sl.UNIT.METER
            init_params.sdk_verbose = 1
            init_params.enable_hdr = bool(self._camera_hdr_enabled)

            # Retry loop — daemon may need time to settle after restart
            for attempt in range(1, 11):
                status = zed.open(init_params)
                if status == sl.ERROR_CODE.SUCCESS:
                    break
                print(f"[ZedXOne] Open attempt {attempt}/10 failed: {repr(status)}, retrying in 3s...")
                sleep(3)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[ZedXOne] Failed to open after 10 attempts: {repr(status)}")
                return None
            print("ZED X One Mono initialized")
            self._zed = zed
            self._camera_type = "ZED X One Mono"
            self._camera_resolution = "QHDPLUS"
            self._camera_fps = "15 requested"
            if bool(getattr(self.args, "raw_yolo_view", False)):
                self._camera_mode = "raw RGB + TensorRT only (depth/TF/ROS image off)"
            elif use_lidar:
                self._camera_mode = "mono RGB + Livox depth"
            elif use_zed_mini:
                self._camera_mode = "mono RGB + ZED X Mini depth"
            else:
                self._camera_mode = "mono RGB"
            apply_zed_camera_settings(zed)

            if use_zed_mini and not bool(getattr(self.args, "raw_yolo_view", False)):
                # ZED X Mini — stereo Camera opened for depth only
                print("Initializing ZED X Mini (depth camera)...")
                zed_mini = sl.Camera()
                init_mini = sl.InitParameters()
                if ZEDMINI_SERIAL > 0:
                    init_mini.input.set_from_serial_number(ZEDMINI_SERIAL)
                # HD720 is NOT supported on ZED X Mini — use SVGA (fast, sufficient for depth)
                init_mini.camera_resolution = sl.RESOLUTION.HD1080
                init_mini.camera_fps = ZEDMINI_DEPTH_FPS
                init_mini.coordinate_units = sl.UNIT.METER
                init_mini.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # higher accuracy than NEURAL_LIGHT
                init_mini.depth_minimum_distance = ZEDMINI_DEPTH_Z_MIN
                init_mini.depth_maximum_distance = ZEDMINI_DEPTH_Z_MAX
                init_mini.sdk_verbose = 1

                for attempt in range(1, 6):
                    status_mini = zed_mini.open(init_mini)
                    if status_mini == sl.ERROR_CODE.SUCCESS:
                        break
                    print(f"[ZedMini] Open attempt {attempt}/5 failed: {repr(status_mini)}, retrying in 3s...")
                    sleep(3)
                if status_mini != sl.ERROR_CODE.SUCCESS:
                    print(f"[ZedMini] Failed to open: {repr(status_mini)}")
                    zed.close()
                    return None
                apply_zed_mini_settings(zed_mini)
                print("ZED X Mini initialized")
                self._zed_mini = zed_mini
                self._depth_camera_text = f"ZED X Mini HD1080 @ {ZEDMINI_DEPTH_FPS}fps"
            else:
                self._depth_camera_text = (
                    "disabled (raw YOLO mode)"
                    if bool(getattr(self.args, "raw_yolo_view", False))
                    else ("Livox" if use_lidar else "-")
                )

        else:
            # ZED stereo — standard Camera API. In Mini-only mode this opens the
            # ZED X Mini as the image and depth camera, so RGB/depth are native
            # to the same sensor and no ZED X One warp is used.
            input_type = sl.InputType()
            if self.args.svo:
                input_type.set_from_svo_file(self.args.svo)
            zed = sl.Camera()
            init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
            if use_zedx_mini_only and ZEDMINI_SERIAL > 0 and not self.args.svo:
                init_params.input.set_from_serial_number(ZEDMINI_SERIAL)
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            if use_zedx_mini_only:
                init_params.camera_fps = ZEDMINI_RGBD_FPS
            init_params.coordinate_units = sl.UNIT.METER
            # Mini-only mode previously achieved reliable front-view date depth
            # with the full NEURAL model. NEURAL_LIGHT was introduced with the
            # dual-camera performance work and produced multi-layer depth jumps
            # on overlapping front views. Keep this change isolated to native
            # ZED X Mini RGBD; ZED One + external-depth modes are unaffected.
            init_params.depth_mode = (
                sl.DEPTH_MODE.NONE
                if bool(getattr(self.args, "raw_yolo_view", False))
                else (sl.DEPTH_MODE.NEURAL
                      if use_zedx_mini_only
                      else sl.DEPTH_MODE.NEURAL_LIGHT)
            )
            init_params.depth_minimum_distance = (
                ZEDMINI_DEPTH_Z_MIN if use_zedx_mini_only else 0.15
            )
            init_params.depth_maximum_distance = (
                ZEDMINI_DEPTH_Z_MAX if use_zedx_mini_only else 50.0
            )
            init_params.sdk_verbose = 1

            print(
                "Initializing ZED X Mini RGBD camera..."
                if use_zedx_mini_only else
                "Initializing Camera..."
            )
            status = zed.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                return None
            print("ZED X Mini RGBD initialized" if use_zedx_mini_only else "Camera Initialized")
            self._zed = zed
            self._camera_type = "ZED X Mini" if use_zedx_mini_only else "ZED stereo"
            self._camera_resolution = "HD1080"
            self._camera_fps = (
                f"{ZEDMINI_RGBD_FPS} requested"
                if use_zedx_mini_only else
                "camera default"
            )
            self._camera_mode = (
                "raw RGB + TensorRT only (depth/TF/ROS image off)"
                if bool(getattr(self.args, "raw_yolo_view", False)) else
                ("ZED X Mini stereo RGB + ZED X Mini depth"
                 if use_zedx_mini_only else "stereo RGB + ZED depth")
            )
            self._camera_hdr_enabled = False
            self._depth_camera_text = (
                "disabled (raw YOLO mode)"
                if bool(getattr(self.args, "raw_yolo_view", False)) else
                ("ZED X Mini native stereo depth"
                 if use_zedx_mini_only else "ZED stereo depth")
            )
            apply_zed_stereo_settings(zed)
            # Positional tracking is the SDK's visual odometry: once enabled it
            # runs on every grab. Nothing here consumes it -- there is no
            # get_position()/sl.Pose call anywhere in the package, and the arm
            # pose comes from TF, which is exact. It was previously enabled in
            # BOTH modes, so mini-only paid for a camera trajectory it computed
            # and threw away every frame, on an arm-mounted camera watching a
            # wind-blown palm (the input VO is worst at).
            #
            # It is NOT dead in the stereo path: enable_object_detection() below
            # sets enable_tracking=True, and the SDK requires positional tracking
            # for that. So gate it on exactly the object-detection condition
            # rather than removing it -- keep these two ifs in step.
            _use_object_detection = (
                not use_zedx_mini_only
                and not bool(getattr(self.args, "raw_yolo_view", False)))
            if _use_object_detection:
                zed.enable_positional_tracking(sl.PositionalTrackingParameters())
            if _use_object_detection:
                obj_param = sl.ObjectDetectionParameters()
                obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
                obj_param.enable_tracking = True
                obj_param.enable_segmentation = False
                zed.enable_object_detection(obj_param)

        # ZED fully initialized — now safe to load YOLO TRT engine
        self.yolo_thread = YoloThread(
            weights=self.args.weights,
            img_size=self.args.img_size,
            conf_thres=self.args.conf_thres,
            max_det=self.args.max_det,
            raw_view=bool(getattr(self.args, "raw_yolo_view", False)),
            use_numpy_masks=use_zedx_mini_only,
        )
        Thread(target=self.yolo_thread.run, daemon=True).start()
        self._publish_camera_status()
        return zed

    def _ros_time_from_wall(self, t_wall: float):
        """rclpy Time for a time.time() reading, on the node clock's time base.

        time.time() and the ROS system clock are both Unix epoch seconds, so
        this is a straight conversion. Split whole seconds from the fraction
        before scaling: t_wall is ~1.8e9, and float64 cannot carry nanosecond
        resolution at that magnitude, while the sub-second remainder is well
        under 1.0 and converts with room to spare.

        clock_type is taken from the node clock so the returned Time can be
        compared against it inside tf2 (mixing ROS_TIME and SYSTEM_TIME raises).
        Assumes use_sim_time is false, which it must be -- this node opens
        physical ZED cameras.
        """
        try:
            _sec = int(t_wall)
            _nsec = int(round((t_wall - _sec) * 1e9))
            if _nsec >= 1_000_000_000:      # rounding can carry into the second
                _sec += 1
                _nsec -= 1_000_000_000
            return rclpyTime(seconds=_sec, nanoseconds=_nsec,
                             clock_type=self.node.get_clock().clock_type)
        except Exception:
            return self.node.get_clock().now()

    def _image_stamp_msg(self):
        """Stamp of the image currently being processed, for TF and headers.

        Falls back to zero time ("latest available") before the first grab.
        """
        stamp = self._latest_image_stamp
        return stamp.to_msg() if stamp is not None else rclpyTime().to_msg()

    def _lookup_tf_at_image(self, target_frame: str):
        """cam_frame -> target_frame at the current image's acquisition time.

        Falls back to the latest available transform when the buffer cannot
        resolve the image stamp (startup, a TF gap, or a stamp already aged out
        of the cache). The fallback is the old behaviour, so a miss is never
        worse than before -- it is just no longer the default.
        """
        stamp = self._latest_image_stamp
        if stamp is not None:
            try:
                return _tf_stamped_to_Rt(self.tf_buffer.lookup_transform(
                    target_frame, self.cam_frame, stamp,
                    timeout=rclpyDuration(seconds=0.01)))
            except Exception:
                _now = time()
                if _now - getattr(self, "_last_tf_stamp_warn_t", 0.0) > 5.0:
                    self._last_tf_stamp_warn_t = _now
                    self.node.get_logger().warn(
                        f"[TF] no {target_frame}<-{self.cam_frame} at image stamp; "
                        "using latest (adds pipeline latency to position error)")
        return _tf_stamped_to_Rt(self.tf_buffer.lookup_transform(
            target_frame, self.cam_frame, rclpyTime()))

    def _publish_camera_status(self) -> None:
        pub = getattr(self, "_camera_status_pub", None)
        if pub is None:
            return
        model_path = str(getattr(self.args, "weights", ""))
        model_name = Path(model_path).name if model_path else "-"
        status = (
            f"Camera: {self._camera_type}\n"
            f"Resolution: {self._camera_resolution}  FPS: {self._camera_fps}\n"
            f"Mode: {self._camera_mode}\n"
            f"Calibration: {CAMERA_PROFILE.get('camera_profile', '-')}\n"
            f"RGB frame: {CAM_FRAME}  Depth frame: {ZEDMINI_CAM_FRAME}\n"
            f"HDR: {'ON' if self._camera_hdr_enabled else 'OFF'}\n"
            f"Depth: {self._depth_camera_text}\n"
            f"Model: {model_name}\n"
            f"YOLO confidence: {float(self.args.conf_thres):.2f}\n"
            f"Maximum detections: {int(self.args.max_det)}\n"
            f"Path: {model_path}"
        )
        msg = StdString()
        msg.data = status
        pub.publish(msg)

    def _publish_mode_state(self, state: str) -> None:
        """Report the effective detection state to the motion node (GPU handoff ack)."""
        pub = getattr(self, "_mode_state_pub", None)
        if pub is None:
            return
        msg = StdString()
        msg.data = state
        pub.publish(msg)

    def _publish_depth_cloud(self, pc_np: np.ndarray, depth_pub,
                             orig_mask: Optional[np.ndarray] = None) -> bool:
        """Publish depth point cloud for voxel obstacle avoidance.
        orig_mask: if provided (ZED Mini mode), only publish originally projected
        pixels — skips EDT-filled pixels which cause scattered artefacts in RViz."""
        try:
            valid = np.isfinite(pc_np).all(axis=-1)
            valid &= (pc_np[:, :, 2] > 0.1) & (pc_np[:, :, 2] < 2.0)
            if orig_mask is not None:
                valid &= orig_mask
            valid_points = pc_np[valid]

            if valid_points.shape[0] > 100:
                if valid_points.shape[0] > 50000:
                    indices = np.random.choice(valid_points.shape[0], 50000, replace=False)
                    valid_points = valid_points[indices]

                pc_msg = create_pointcloud2_msg(
                    valid_points,
                    self.cam_frame,
                    self.node.get_clock().now().to_msg()
                )
                depth_pub.publish(pc_msg)
                return True
        except Exception:
            pass
        return False

    def _publish_trunk_position(self, trunk_boxes, pc_np, image_scale, image_left_ocv, trunk_pub,
                                pts_cam=None, uv=None, use_lidar=False, trunk_cam_pub=None) -> bool:
        """Publish detected trunk position in base_link for pole obstacle update.
        Uses only the first (largest/most confident) trunk detection."""
        if not trunk_boxes:
            return False
        # Use first trunk box only
        bx1, by1, bx2, by2 = trunk_boxes[0]
        # Scale to display resolution
        x1 = int(bx1 * image_scale[0])
        y1 = int(by1 * image_scale[1])
        x2 = int(bx2 * image_scale[0])
        y2 = int(by2 * image_scale[1])
        x1 = max(0, min(x1, image_left_ocv.shape[1] - 1))
        x2 = max(0, min(x2, image_left_ocv.shape[1]))
        y1 = max(0, min(y1, image_left_ocv.shape[0] - 1))
        y2 = max(0, min(y2, image_left_ocv.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return False

        if use_lidar and pts_cam is not None and pts_cam.shape[0] > 0:
            # LiDAR path: filter projected points to trunk bounding box
            in_bbox = (
                (uv[:, 0] >= x1) & (uv[:, 0] < x2) &
                (uv[:, 1] >= y1) & (uv[:, 1] < y2)
            )
            if not in_bbox.any():
                return False
            pts_trunk = pts_cam[in_bbox]
            zs = pts_trunk[:, 2]
            idx = np.argsort(zs)
            k = max(5, int(0.2 * len(idx)))
            pts_front = pts_trunk[idx[:k]]
        else:
            # ZED depth path
            if pc_np is None:
                return False
            roi_xyz = pc_np[y1:y2, x1:x2, :]
            valid_z = np.isfinite(roi_xyz[:, :, 2]) & (roi_xyz[:, :, 2] > 0.1)
            if np.count_nonzero(valid_z) < 10:
                return False
            pts = roi_xyz[valid_z]
            zs = pts[:, 2]
            idx = np.argsort(zs)
            k = max(10, int(0.2 * len(idx)))
            pts_front = pts[idx[:k]]

        _trunk_xyz_cam = np.array([
            float(np.mean(pts_front[:, 0])),
            float(np.mean(pts_front[:, 1])),
            float(np.mean(pts_front[:, 2])),
        ], dtype=np.float64)
        # Detection gives the front surface facing the camera. Offset by trunk radius
        # along the camera→surface direction (in camera frame, camera is at origin)
        # to get the true centroid before transforming to base_link.
        from ..config import STATIC_OBSTACLES as _SO
        _trunk_radius = next((o["radius"] for o in _SO if o["name"] == "trunk"), 0.02)
        _cam_dist = float(np.linalg.norm(_trunk_xyz_cam))
        if _cam_dist > 1e-6:
            _trunk_xyz_cam = _trunk_xyz_cam + _trunk_radius * (_trunk_xyz_cam / _cam_dist)
        if trunk_cam_pub is not None:
            cam_msg = PointStamped()
            cam_msg.header.frame_id = self.cam_frame
            cam_msg.header.stamp = self._image_stamp_msg()
            cam_msg.point.x = float(_trunk_xyz_cam[0])
            cam_msg.point.y = float(_trunk_xyz_cam[1])
            cam_msg.point.z = float(_trunk_xyz_cam[2])
            trunk_cam_pub.publish(cam_msg)
        try:
            if self._cached_tf_base is not None:
                _R_b, _t_b = self._cached_tf_base
                _xyz_b = _R_b @ _trunk_xyz_cam + _t_b
                pt_base = PointStamped()
                pt_base.header.frame_id = "base_link"
                pt_base.header.stamp = self._image_stamp_msg()
                pt_base.point.x = float(_xyz_b[0])
                pt_base.point.y = float(_xyz_b[1])
                if X_FORWARD_Y_LATERAL:
                    pt_base.point.x -= TRUNK_DEPTH_OFFSET
                else:
                    pt_base.point.y += TRUNK_DEPTH_OFFSET
                pt_base.point.z = float(_xyz_b[2])
            else:
                point_msg = PointStamped()
                point_msg.header.frame_id = self.cam_frame
                point_msg.header.stamp = self._image_stamp_msg()
                point_msg.point.x = _trunk_xyz_cam[0]
                point_msg.point.y = _trunk_xyz_cam[1]
                point_msg.point.z = _trunk_xyz_cam[2]
                pt_base = self.tf_buffer.transform(
                    point_msg, "base_link", timeout=rclpyDuration(seconds=0.005))
                if X_FORWARD_Y_LATERAL:
                    pt_base.point.x -= TRUNK_DEPTH_OFFSET
                else:
                    pt_base.point.y += TRUNK_DEPTH_OFFSET
            trunk_pub.publish(pt_base)
            return True
        except Exception:
            return False

    def _process_objects(
        self,
        objects_or_dets,
        pc_np,
        image_left_ocv: np.ndarray,
        image_scale: List[float],
        display_resolution,
        intrinsics: Dict[str, float],
        pts_cam: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        use_lidar: bool = False,
        use_zed_mini: bool = False,
        use_raw_detections: bool = False,
    ) -> tuple:
        """Process detected objects and extract 3D information.
        use_lidar=True  — sparse LiDAR depth, iterate raw YOLO CustomMaskObjectData list.
        use_zed_mini=True — dense ZED Mini depth (heatmap path), iterate raw YOLO list.
        default — ZED stereo depth (heatmap path), iterate sl.Objects container.
        Returns (targets, rejected_targets, viz_only)."""
        targets = []
        rejected_targets = []
        viz_only = []  # kept for API compat but always empty now
        object_times_ms = []

        # use_zed_mini also iterates raw YOLO dets (ZED One has no object-detection API)
        obj_list = (
            objects_or_dets
            if (use_lidar or use_zed_mini or use_raw_detections)
            else objects_or_dets.object_list
        )
        for o in obj_list:
            bb = o.bounding_box_2d
            x1 = int(bb[0][0] * image_scale[0])
            y1 = int(bb[0][1] * image_scale[1])
            x2 = int(bb[2][0] * image_scale[0])
            y2 = int(bb[2][1] * image_scale[1])

            x1 = max(0, min(x1, image_left_ocv.shape[1] - 1))
            x2 = max(0, min(x2, image_left_ocv.shape[1]))
            y1 = max(0, min(y1, image_left_ocv.shape[0] - 1))
            y2 = max(0, min(y2, image_left_ocv.shape[0]))

            def mark_reject(reason: str):
                rejected_targets.append({"bb": (x1, y1, x2, y2), "reason": reason})

            # Choose mask source
            mask_mat = None
            if hasattr(o, "mask") and o.mask is not None:
                if isinstance(o.mask, np.ndarray) or o.mask.is_init():
                    mask_mat = o.mask
            if mask_mat is None and hasattr(o, "box_mask") and o.box_mask is not None:
                if isinstance(o.box_mask, np.ndarray) or o.box_mask.is_init():
                    mask_mat = o.box_mask

            if mask_mat is None:
                mark_reject("No mask")
                continue

            if x2 <= x1 or y2 <= y1:
                mark_reject("Invalid box")
                continue

            try:
                _object_t0 = time()
                target = self._extract_target_3d(
                    o, mask_mat, pc_np, x1, y1, x2, y2,
                    display_resolution, intrinsics, mark_reject,
                    pts_cam=pts_cam, uv=uv,
                    # use_zed_mini: raw dets but dense heatmap depth (use_lidar=False)
                    use_lidar=(use_lidar and not use_zed_mini),
                    use_zed_mini=use_zed_mini,
                )
                if target is not None:
                    targets.append(target)
                object_times_ms.append((time() - _object_t0) * 1000.0)
            except Exception as e:
                object_times_ms.append((time() - _object_t0) * 1000.0)
                print(f"[WARN] 3D extraction failed: {e}")
                mark_reject("3D extraction failed")

        self._last_object_processing_ms = object_times_ms
        self._last_extracted_target_count = len(targets)
        return targets, rejected_targets, viz_only

    def _extract_target_3d(
        self,
        obj, mask_mat, pc_np,
        x1: int, y1: int, x2: int, y2: int,
        display_resolution, intrinsics: Dict[str, float],
        mark_reject,
        pts_cam: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        use_lidar: bool = False,
        use_zed_mini: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Extract 3D information from a detected object."""
        depth_diag = None
        mask_local = mask_mat if isinstance(mask_mat, np.ndarray) else mask_mat.get_data()
        if mask_local.ndim == 3:
            mask_local = mask_local[:, :, 0]

        w_roi = x2 - x1
        h_roi = y2 - y1

        mask_resized = cv2.resize(mask_local, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
        mask_bool = mask_clean > 0

        # Ellipse fit for orientation
        t_short_axis, long_axis_2d, t_angle, ellipse_confidence = self._fit_ellipse(mask_clean)

        in_mask_uv: Optional[np.ndarray] = None  # ROI-local UV for depth viz
        if use_lidar and pts_cam is not None and pts_cam.shape[0] > 0:
            # ── LiDAR depth path ─────────────────────────────────────────
            # Filter projected LiDAR points to padded bounding box
            # Padding compensates for small T_CAM_LIDAR calibration error
            pad = max(20, int(0.15 * max(x2 - x1, y2 - y1)))  # 15% of bbox size, min 20px
            in_bbox = (
                (uv[:, 0] >= x1 - pad) & (uv[:, 0] < x2 + pad) &
                (uv[:, 1] >= y1 - pad) & (uv[:, 1] < y2 + pad)
            )
            if not in_bbox.any():
                mark_reject("No LiDAR pts in bbox")
                return None
            # Shift UV to ROI-local coordinates for mask lookup (account for pad offset)
            uv_local = uv[in_bbox].copy()
            uv_local[:, 0] -= x1
            uv_local[:, 1] -= y1
            pts_bbox = pts_cam[in_bbox]
            # Inline mask filtering to also capture UV coords for depth viz
            mask_bin = mask_resized > 0
            col_m = np.clip(np.round(uv_local[:, 0]).astype(int), 0, mask_bin.shape[1] - 1)
            row_m = np.clip(np.round(uv_local[:, 1]).astype(int), 0, mask_bin.shape[0] - 1)
            inside_m = mask_bin[row_m, col_m]
            in_mask_pts = pts_bbox[inside_m]
            in_mask_uv = uv_local[inside_m]  # ROI-local pixel coords of LiDAR hits
            if in_mask_pts.shape[0] < 5:
                mark_reject("Too few LiDAR pts in mask")
                return None
            # Front 20% of points by depth
            zs_l = in_mask_pts[:, 2]
            k_l = max(5, int(0.2 * len(zs_l)))
            idx_l = np.argpartition(zs_l, k_l - 1)[:k_l]
            pts_front = in_mask_pts[idx_l]
            Xc = float(np.median(pts_front[:, 0]))
            Yc = float(np.median(pts_front[:, 1]))
            Zc = float(np.median(pts_front[:, 2]))

            depth_std = float(np.std(pts_front[:, 2]))
            vis_ratio = min(1.0, in_mask_pts.shape[0] / max(1, int(0.1 * mask_bool.sum())))
            vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)
            if (not np.isfinite(Zc)
                    or not FRUIT_CAMERA_Z_MIN <= Zc <= FRUIT_CAMERA_Z_MAX):
                mark_reject(
                    f"Fruit depth {Zc:.2f}m outside "
                    f"{FRUIT_CAMERA_Z_MIN:.2f}-{FRUIT_CAMERA_Z_MAX:.2f}m"
                )
                return None

            heatmap = t_best_point = t_best_dir2d = t_best_point_3d = scored_3d_pts = surface_normal = None

            gap_target = {"bb": (x1, y1, x2, y2), "Zc": Zc,
                          "between_branches": False, "gap_angle_cam": 0.0}
            obj_confidence = float(getattr(obj, "probability", 0.5))

        elif use_zed_mini:
            # ── ZED Mini direct point query (no EDT fill artefacts) ───────
            # Query raw projected points by UV — avoids scatter/fill bleed-in
            # at depth discontinuities (fruit edge vs background).
            with self._mini_pts_lock:
                buf = list(self._mini_pts_buffer)

            if not buf:
                mark_reject("No depth data")
                return None

            # Pick the buffered frame closest in time to the current ZED One frame
            zed_one_ts = self._latest_zed_one_ts
            best = min(buf, key=lambda f: abs(f[2] - zed_one_ts))
            pts_all, uv_all, mini_ts = best
            if pts_all.shape[0] == 0:
                mark_reject("No depth data")
                return None
            _depth_age = abs(zed_one_ts - mini_ts)
            if _depth_age > 0.12:
                mark_reject(f"RGB/depth desync ({_depth_age*1000:.0f}ms)")
                return None

            # Filter to bbox
            u = uv_all[:, 0]
            v = uv_all[:, 1]
            in_bbox = (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
            pts_bbox = pts_all[in_bbox]
            uv_bbox  = uv_all[in_bbox]

            if pts_bbox.shape[0] < 10:
                mark_reject("Too few depth pts")
                return None

            # Apply segmentation mask
            u_rel = np.clip((uv_bbox[:, 0] - x1).astype(np.int32), 0, mask_bool.shape[1] - 1)
            v_rel = np.clip((uv_bbox[:, 1] - y1).astype(np.int32), 0, mask_bool.shape[0] - 1)
            in_mask = mask_bool[v_rel, u_rel]
            pts = pts_bbox[in_mask]
            uv_in_mask = uv_bbox[in_mask]

            # A small inter-camera projection error can put the nearby fruit
            # points just outside the segmentation mask while leaving distant
            # background inside it. Build a conservative fallback from the
            # central 60% of the detection box and select its nearest coherent
            # depth layer. This repairs mild mask/depth misalignment without
            # accepting arbitrary points from the padded surroundings.
            _bw = max(1.0, float(x2 - x1))
            _bh = max(1.0, float(y2 - y1))
            _central = (
                (uv_bbox[:, 0] >= x1 + 0.20 * _bw)
                & (uv_bbox[:, 0] <= x2 - 0.20 * _bw)
                & (uv_bbox[:, 1] >= y1 + 0.20 * _bh)
                & (uv_bbox[:, 1] <= y2 - 0.20 * _bh)
            )
            _fallback_pts = pts_bbox[_central]
            _fallback_uv = uv_bbox[_central]
            if _fallback_pts.shape[0] < 10:
                _fallback_pts = pts_bbox
                _fallback_uv = uv_bbox

            _use_fallback = pts.shape[0] < 10
            if not _use_fallback and _fallback_pts.shape[0] >= 10:
                _mask_p5 = float(np.percentile(pts[:, 2], 5))
                _fallback_p5 = float(np.percentile(_fallback_pts[:, 2], 5))
                # A foreground layer at least 20cm nearer than the mask layer is
                # strong evidence that projected fruit depth missed the mask.
                _use_fallback = _fallback_p5 + 0.20 < _mask_p5

            if _use_fallback:
                if _fallback_pts.shape[0] < 10:
                    mark_reject("Too few foreground depth pts")
                    return None
                pts = _fallback_pts
                uv_in_mask = _fallback_uv
                _now = time()
                if _now - getattr(self, "_last_depth_recovery_log_t", 0.0) > 1.0:
                    self._last_depth_recovery_log_t = _now
                    self.node.get_logger().warn(
                        "[DEPTH_RECOVERY] Segmentation/depth mismatch; "
                        "using nearest central foreground layer")

            # Depth: use ZED Mini pts (in ZED One frame) for Z only.
            # Keep in-mask points within a window of the nearest valid point;
            # anything past it is background bleeding through the mask.
            #
            # This window was a flat 0.10 -- about 9x the depth extent of the
            # object it is meant to isolate. A date of this cultivar is 20-25mm
            # across, so its visible cap spans only ~11mm plus a few mm of
            # neural-depth noise.
            #
            # 452 field frames (~/.ros/log, 2026-09-10) say the flat cap caught
            # the gross case and nothing else. 13% of frames carry points
            # 1.5-2.2m behind the date -- sky/canopy seen through the mask --
            # and every one of those was correctly excluded. But on the 392
            # frames where the cap excluded nothing at all:
            #
            #   in-mask spread (p95-p5)   median   9mm  <- healthy: cap + noise
            #                             p90     26mm
            #                             max     49mm  (52/392 span 25-49mm)
            #
            # On those wide frames the median sits ~21mm behind p5 where the
            # sphere geometry says it should sit ~8mm (0.707*r), i.e. Zc reads
            # roughly 10-15mm too far back. That is larger than the
            # surface->centre correction applied further down, so the window was
            # the dominant error term, not the correction it was masking.
            #
            # Size the window to the fruit instead. estimate_fruit_radius()
            # clamps to [8mm, 18mm] (see FRUIT_RADIUS_MIN/MAX), so the window
            # stays inside [25mm, 54mm] and a merged or oversized bbox cannot
            # reopen it. z_min_anchor is the measured front surface, which is a
            # good enough provisional depth to size the radius from.
            zs  = pts[:, 2]
            z_min_anchor = float(np.percentile(zs, 5))
            _r_provisional = estimate_fruit_radius(
                {"bb": (x1, y1, x2, y2), "Zc": z_min_anchor},
                fx=intrinsics["fx"])
            # ~r of visible cap + headroom for neural-depth noise. The floor
            # keeps small or partially occluded dates from getting a window so
            # tight that the real cap falls outside it.
            fruit_depth_range = max(0.025, 3.0 * _r_provisional)
            in_fruit = zs <= (z_min_anchor + fruit_depth_range)
            if in_fruit.sum() >= 10:
                pts_front = pts[in_fruit]
                uv_front = uv_in_mask[in_fruit]
            else:
                k_front = max(10, int(0.2 * len(zs)))
                front_idx = np.argpartition(zs, k_front - 1)[:k_front]
                pts_front = pts[front_idx]
                uv_front = uv_in_mask[front_idx]

            depth_std = float(np.std(pts_front[:, 2]))
            # Don't hard-reject — let depth_quality score handle noisy early frames.

            Zc = float(np.median(pts_front[:, 2]))

            _p5, _p50, _p95 = (
                float(p) for p in np.percentile(zs, [5, 50, 95]))
            _near = int(np.count_nonzero(
                zs <= (z_min_anchor + fruit_depth_range)))
            _far = int(zs.shape[0] - _near)
            # Points inside the bbox but OUTSIDE the mask. Answers "is the
            # date's depth landing next to the mask rather than on it?" -- a
            # question worth having when alignment is suspect, but it is pure
            # diagnostics and it used to run on every detection of every frame.
            # DEBUG_DEPTH_SAMPLING throttled only the printing, not this. The
            # -1 placeholders match the fallback tuple built further down.
            if DEBUG_DEPTH_SAMPLING:
                _out = pts_bbox[~in_mask]
                _out_near = (
                    int(np.count_nonzero(
                        _out[:, 2] <= (z_min_anchor + fruit_depth_range)))
                    if _out.shape[0] else 0)
                _out_p5 = (
                    float(np.percentile(_out[:, 2], 5))
                    if _out.shape[0] else -1.0)
                _out_p50 = (
                    float(np.percentile(_out[:, 2], 50))
                    if _out.shape[0] else -1.0)
                _out_count = int(_out.shape[0])
            else:
                _out_near, _out_count, _out_p5, _out_p50 = -1, -1, -1.0, -1.0
            depth_diag = (
                _p5, _p50, _p95, _near, _far,
                _out_near, _out_count, _out_p5, _out_p50)

            # X,Y from the front-depth points themselves. Same point set as
            # uv_front, so this keeps the original intent -- median over the
            # front layer only, so bunch/background mask leakage cannot pull the
            # goal inside the bunch when the date is directly in front.
            #
            # These points are already exact camera-frame coordinates, so taking
            # their median directly is both cheaper and correct. Back-projecting
            # median(uv_front) through the pinhole model was not: uv_front comes
            # from project_lidar_to_image, which projects WITH the distortion
            # coefficients (cv2.projectPoints), while the pinhole inverse below
            # ignored them. That round-trip injected the full radial distortion
            # error at the fruit's image position -- a few mm near the edges.
            Xc = float(np.median(pts_front[:, 0]))
            Yc = float(np.median(pts_front[:, 1]))

            if DEBUG_DEPTH_SAMPLING:
                _now = time()
                if _now - getattr(self, "_last_depth_dbg_t", 0.0) > 0.5:
                    self._last_depth_dbg_t = _now
                    # near-depth points in the bbox but OUTSIDE the mask -> if high while
                    # in-mask near count is ~0, the date depth is landing off the mask
                    # (extrinsic/warp misalignment) rather than being absent (sensor).
                    # Base-frame position via the same cam->base TF used for the goal.
                    # If this jumps with arm pose while cam Zc stays stable -> hand-eye/TF,
                    # not depth.
                    _base_str = ""
                    if getattr(self, "_cached_tf_base", None) is not None:
                        _Rb, _tb = self._cached_tf_base
                        _pb = _Rb @ np.array([Xc, Yc, Zc], dtype=np.float64) + _tb
                        _base_str = f" base=[{_pb[0]:.3f},{_pb[1]:.3f},{_pb[2]:.3f}]"
                    self.node.get_logger().info(
                        f"[DEPTH_DBG] bbox=({x1},{y1},{x2},{y2}) in_mask={int(zs.shape[0])}pts "
                        f"z(m) p5/p50/p95={_p5:.3f}/{_p50:.3f}/{_p95:.3f} "
                        f"win={fruit_depth_range*1000:.0f}mm(r={_r_provisional*1000:.1f}mm) "
                        f"near={_near} far={_far} -> Zc={Zc:.3f} | "
                        f"near_but_OUTSIDE_mask={_out_near}/{_out_count}"
                        f" | cam=[{Xc:.3f},{Yc:.3f},{Zc:.3f}]{_base_str}")

            if not np.isfinite(Zc):
                mark_reject("Invalid fruit depth")
                return None
            if not FRUIT_CAMERA_Z_MIN <= Zc <= FRUIT_CAMERA_Z_MAX:
                mark_reject(
                    f"Fruit depth {Zc:.2f}m outside "
                    f"{FRUIT_CAMERA_Z_MIN:.2f}-{FRUIT_CAMERA_Z_MAX:.2f}m"
                )
                # Surface this one. A date rejected here never reaches
                # /vision/all_fruit_poses, so downstream it is indistinguishable
                # from "YOLO saw nothing" -- both show as no_candidate in
                # FINAL_INFLIGHT and as a reacquire timeout. Those have
                # completely different fixes: this one is the window being too
                # tight, the other is a detection problem. Throttled to 1/s.
                _now_rej = time()
                if _now_rej - getattr(self, "_last_zrange_warn_t", 0.0) > 1.0:
                    self._last_zrange_warn_t = _now_rej
                    self.node.get_logger().warn(
                        f"[DEPTH_RANGE_REJECT] date at {Zc:.3f}m is outside "
                        f"{FRUIT_CAMERA_Z_MIN:.2f}-{FRUIT_CAMERA_Z_MAX:.2f}m "
                        f"({'TOO CLOSE' if Zc < FRUIT_CAMERA_Z_MIN else 'too far'})"
                        " -- detected but discarded")
                return None

            # Heatmap + vis_ratio — use EDT depth map so vis_ratio matches stereo mode.
            # Position/z_std come from raw points above; EDT is for visualization only.
            # Key snapped to 16px grid — small YOLO bbox jitter no longer causes cache misses.
            target_key = (x1 // 16 * 16, y1 // 16 * 16, x2 // 16 * 16, y2 // 16 * 16)
            if pc_np is not None:
                roi_xyz = pc_np[y1:y2, x1:x2, :]
                valid   = np.isfinite(roi_xyz[:, :, 2]) & mask_bool
                # vis_ratio: same formula as stereo path so scoring is comparable
                vis_mask_e = cv2.erode(mask_clean, np.ones((3, 3), np.uint8), iterations=1) > 0
                mask_pixels = np.count_nonzero(vis_mask_e)
                vis_ratio = (
                    float(np.count_nonzero(valid & vis_mask_e)) / float(mask_pixels)
                    if mask_pixels > 0 else 0.0
                )
                vis_ratio = max(0.0, min(vis_ratio, 1.0))

                # Only compute heatmap for the previous best fruit — heatmap is only
                # consumed by _process_best_target, so computing it for every detection
                # wastes N× Sobel passes per heatmap frame.
                _prev = self.best_target_prev
                _is_prev_best = (
                    _prev is not None and
                    abs(_prev.get("Xc", 1e9) - Xc) < 0.05 and
                    abs(_prev.get("Yc", 1e9) - Yc) < 0.05 and
                    abs(_prev.get("Zc", 1e9) - Zc) < 0.05
                )
                if _is_prev_best and target_key not in self._cached_heatmaps:
                    # Compute once per fruit — reuse cache every subsequent frame
                    heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = self._compute_heatmap(roi_xyz, valid, mask_clean)
                    self._cached_heatmaps[target_key] = (heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal)
                else:
                    cached = self._cached_heatmaps.get(target_key)
                    if cached is not None:
                        heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = cached
                    else:
                        heatmap = t_best_point = t_best_dir2d = t_best_point_3d = scored_3d_pts = surface_normal = None
            else:
                vis_ratio = min(1.0, float(pts.shape[0]) / max(1.0, float(np.count_nonzero(mask_bool))))
                heatmap = t_best_point = t_best_dir2d = t_best_point_3d = scored_3d_pts = surface_normal = None

            vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)

            gap_target = {"bb": (x1, y1, x2, y2), "Zc": Zc}
            if self.target_lock_active or self._show_gap_debug:
                self._detect_branch_gap(pc_np, gap_target)
            else:
                gap_target["between_branches"] = False
                gap_target["gap_angle_cam"] = 0.0

            in_mask_uv = None
            if hasattr(obj, "probability"):
                obj_confidence = float(obj.probability)
            else:
                obj_confidence = getattr(obj, "confidence", 50.0) / 100.0

        else:
            # ── ZED stereo depth path (original) ─────────────────────────
            if pc_np is None:
                mark_reject("No depth data")
                return None
            roi_xyz = pc_np[y1:y2, x1:x2, :]
            valid = np.isfinite(roi_xyz[:, :, 2]) & mask_bool

            # Visibility ratio
            vis_mask = cv2.erode(mask_clean, np.ones((3, 3), np.uint8), iterations=1) > 0
            mask_pixels = np.count_nonzero(vis_mask)
            vis_ratio = (
                float(np.count_nonzero(valid & vis_mask)) / float(mask_pixels)
                if mask_pixels > 0 else 0.0
            )
            vis_ratio = max(0.0, min(vis_ratio, 1.0))

            if np.count_nonzero(valid) < 10:
                mark_reject("Too few depth pts")
                return None

            pts = roi_xyz[valid]
            zs = pts[:, 2]
            k = max(10, int(0.2 * len(zs)))
            idx = np.argpartition(zs, k - 1)[:k]
            pts_front = pts[idx]

            depth_std = np.std(pts_front[:, 2])
            Xc = float(np.mean(pts_front[:, 0]))
            Yc = float(np.mean(pts_front[:, 1]))
            Zc = float(np.mean(pts_front[:, 2]))

            # Diagnostic-only ring around the segmentation mask. If many valid
            # pixels just outside the mask are consistently nearer than the
            # selected in-mask surface, RGB/depth alignment may be excluding the
            # true date layer. Do not alter the commanded goal yet.
            # The 21x21 dilate is the expensive part and this whole block is
            # diagnostics -- its own comment says it must not alter the goal.
            # It ran per detection per frame regardless of whether anyone was
            # reading it; gate it like the ZED Mini path above.
            if DEBUG_DEPTH_SAMPLING:
                _ring = (
                    cv2.dilate(mask_bool.astype(np.uint8),
                               np.ones((21, 21), np.uint8), iterations=1) > 0
                ) & (~mask_bool)
                _ring_valid = (
                    _ring & np.isfinite(roi_xyz[:, :, 2]) &
                    (roi_xyz[:, :, 2] > 0.0)
                )
                _outside_z = roi_xyz[:, :, 2][_ring_valid]
                _outside_near = (
                    int(np.count_nonzero(_outside_z <= Zc - 0.005))
                    if _outside_z.size else 0)
                _outside_p5 = (
                    float(np.percentile(_outside_z, 5))
                    if _outside_z.size else -1.0)
                _outside_p50 = (
                    float(np.percentile(_outside_z, 50))
                    if _outside_z.size else -1.0)
                _outside_count = int(_outside_z.size)
            else:
                _outside_near, _outside_count = -1, -1
                _outside_p5, _outside_p50 = -1.0, -1.0
            _p5, _p50, _p95 = (
                float(p) for p in np.percentile(zs, [5, 50, 95]))
            _near = int(np.count_nonzero(zs <= Zc + 0.10))
            _far = int(zs.shape[0] - _near)
            depth_diag = (
                _p5, _p50, _p95, _near, _far,
                _outside_near, _outside_count,
                _outside_p5, _outside_p50,
            )

            if not np.isfinite(Zc):
                mark_reject("Invalid fruit depth")
                return None
            if not FRUIT_CAMERA_Z_MIN <= Zc <= FRUIT_CAMERA_Z_MAX:
                mark_reject(
                    f"Fruit depth {Zc:.2f}m outside "
                    f"{FRUIT_CAMERA_Z_MIN:.2f}-{FRUIT_CAMERA_Z_MAX:.2f}m"
                )
                return None

            # Only compute heatmap for the previous best fruit (same as ZED Mini path).
            target_key = (x1 // 16 * 16, y1 // 16 * 16, x2 // 16 * 16, y2 // 16 * 16)
            _prev = self.best_target_prev
            _is_prev_best = (
                _prev is not None and
                abs(_prev.get("Xc", 1e9) - Xc) < 0.05 and
                abs(_prev.get("Yc", 1e9) - Yc) < 0.05 and
                abs(_prev.get("Zc", 1e9) - Zc) < 0.05
            )
            if _is_prev_best and target_key not in self._cached_heatmaps:
                # Compute once per fruit — reuse cache every subsequent frame
                heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = self._compute_heatmap(
                    roi_xyz, valid, mask_clean
                )
                self._cached_heatmaps[target_key] = (heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal)
            else:
                cached = self._cached_heatmaps.get(target_key)
                if cached is not None:
                    heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = cached
                else:
                    heatmap = t_best_point = t_best_dir2d = t_best_point_3d = scored_3d_pts = surface_normal = None

            vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)

            # Don't hard-reject on high depth variance — include as low-quality candidate.
            # The depth_quality score component will naturally rank it lower until depth
            # stabilises over the first few frames. Hard-rejecting causes the fruit to
            # flash between "rejected" and "best" on first appearance.

            # Branch gap detection (depth ring sampling around fruit)
            gap_target = {"bb": (x1, y1, x2, y2), "Zc": Zc}
            if self.target_lock_active or self._show_gap_debug:
                self._detect_branch_gap(pc_np, gap_target)
            else:
                gap_target["between_branches"] = False
                gap_target["gap_angle_cam"] = 0.0

            if hasattr(obj, "probability"):
                obj_confidence = float(obj.probability)
            else:
                obj_confidence = getattr(obj, "confidence", 50.0) / 100.0

        # ── Surface → centre ─────────────────────────────────────────────────
        # Every branch above measured the front surface facing the camera, but
        # the grasp goal must be the fruit CENTRE. Push back along the camera ray
        # by a fraction of the estimated radius. Applied after the depth-range
        # validation above so that check keeps testing the measured surface.
        # Radius is estimated here (not from the returned dict) because the dict
        # is built further down; the inputs it needs are bb, Zc and fx.
        _fruit_radius = estimate_fruit_radius(
            {"bb": (x1, y1, x2, y2), "Zc": Zc}, fx=intrinsics["fx"])
        if FRUIT_SURFACE_TO_CENTER_FRACTION > 0.0:
            _ray = np.array([Xc, Yc, Zc], dtype=np.float64)
            _ray_n = float(np.linalg.norm(_ray))
            if _ray_n > 1e-6:
                _push = FRUIT_SURFACE_TO_CENTER_FRACTION * _fruit_radius
                Xc, Yc, Zc = (_ray + _push * (_ray / _ray_n)).tolist()

        # Transform to base_link
        point_msg = PointStamped()
        point_msg.header.frame_id = self.cam_frame
        point_msg.header.stamp = self._image_stamp_msg()
        point_msg.point.x = Xc
        point_msg.point.y = Yc
        point_msg.point.z = Zc

        try:
            _pt_cam = np.array([Xc, Yc, Zc], dtype=np.float64)

            # Use per-frame cached transform (refreshed once per loop iteration).
            if self._cached_tf_base is not None:
                _R_b, _t_b = self._cached_tf_base
                _xyz_b = _R_b @ _pt_cam + _t_b
                pt_base = PointStamped()
                pt_base.header.frame_id = "base_link"
                pt_base.header.stamp = point_msg.header.stamp
                pt_base.point.x = float(_xyz_b[0])
                pt_base.point.y = float(_xyz_b[1])
                pt_base.point.z = float(_xyz_b[2])
            else:
                pt_base = self.tf_buffer.transform(point_msg, "base_link", timeout=rclpyDuration(seconds=0.005))

            if self._cached_tf_grip is not None:
                _R_g, _t_g = self._cached_tf_grip
                _xyz_g = _R_g @ _pt_cam + _t_g
                dist = math.sqrt(float(_xyz_g[0])**2 + float(_xyz_g[1])**2 + float(_xyz_g[2])**2)
            else:
                _pt_grip = self.tf_buffer.transform(point_msg, "gripper_tip", timeout=rclpyDuration(seconds=0.005))
                dist = math.sqrt(_pt_grip.point.x**2 + _pt_grip.point.y**2 + _pt_grip.point.z**2)

            if not (0.0 <= obj_confidence <= 1.0):
                obj_confidence = 0.5

            fruit_id = hash((round(Xc, 2), round(Yc, 2), round(Zc, 2)))

            if self.tracker.is_fruit_blacklisted(fruit_id):
                mark_reject(f"Max attempts (3)")
                return None

            attempt_count = self.tracker.register_fruit(fruit_id, [Xc, Yc, Zc])

            # Every depth backend must provide diagnostics for the selected goal.
            # ZED Mini supplies full in-mask near/far counts above; LiDAR/stereo
            # fall back to statistics over the foreground points actually used.
            if depth_diag is None:
                _diag_z = np.asarray(pts_front, dtype=np.float64)[:, 2]
                _p5, _p50, _p95 = (
                    float(p) for p in np.percentile(_diag_z, [5, 50, 95]))
                depth_diag = (
                    _p5, _p50, _p95,
                    int(_diag_z.shape[0]), 0,
                    -1, -1, -1.0, -1.0,
                )

            return {
                "Xc": Xc, "Yc": Yc, "Zc": Zc,
                "bb": (x1, y1, x2, y2),
                # Live focal length, so estimate_fruit_radius() never has to
                # guess it. Every call site used to omit fx and silently get 700.
                "fx": float(intrinsics["fx"]),
                "img_width": display_resolution.width,
                "img_height": display_resolution.height,
                "mask_resized": mask_resized,
                "pt_base": pt_base,
                "dist": dist,
                "quat": None,
                "approach_axis": None,
                "pca_stable": None,
                "reason": "",
                "pts_front": pts_front,
                "z_std": depth_std,
                "vis_ratio": vis_ratio,
                "vis_quality": vis_quality,
                "confidence": obj_confidence,
                "short_axis_cam": t_short_axis,
                "long_axis_2d": long_axis_2d,
                "ellipse_angle": t_angle,
                "ellipse_confidence": ellipse_confidence,
                "heatmap": heatmap,
                "best_dir2d": t_best_dir2d,
                "best_point2d": t_best_point,
                "best_point_3d": t_best_point_3d,
                "scored_3d_points": scored_3d_pts,
                "surface_normal": surface_normal,
                "score": 0.0,
                "score_components": {},
                "depth_diag": depth_diag,
                "fruit_id": fruit_id,
                "attempt_count": attempt_count,
                "between_branches": gap_target.get("between_branches", False),
                "gap_angle_cam": gap_target.get("gap_angle_cam", 0.0),
                "lidar_depth_uv": in_mask_uv,
                "lidar_depths": in_mask_pts[:, 2] if in_mask_uv is not None and in_mask_pts.shape[0] > 0 else None,
            }
        except Exception as e:
            print(f"[WARN] TF transform failed: {e}")
            mark_reject("TF transform failed")
            return None

    def _detect_branch_gap(self, pc_np: np.ndarray, target: dict) -> None:
        """Sample depth in a ring around the fruit to detect branch gaps.

        Sets target["between_branches"] (bool) and target["gap_angle_cam"] (radians).
        The gap angle is in image space (0 = right, pi/2 = down).
        """
        if pc_np is None:
            target["between_branches"] = False
            target["gap_angle_cam"] = 0.0
            return

        x1, y1, x2, y2 = target["bb"]
        cx_img = (x1 + x2) / 2.0
        cy_img = (y1 + y2) / 2.0
        fruit_z = target["Zc"]

        bbox_r = max(x2 - x1, y2 - y1) / 2.0
        ring_r = bbox_r * 1.5

        H, W = pc_np.shape[:2]
        n_samples = 24
        branch_threshold = 0.03  # within 3cm of fruit depth = branch

        blocked = []
        for i in range(n_samples):
            angle = i * (2 * math.pi / n_samples)
            u = int(cx_img + ring_r * math.cos(angle))
            v = int(cy_img + ring_r * math.sin(angle))

            if 0 <= u < W and 0 <= v < H:
                z = pc_np[v, u, 2]
                if np.isfinite(z) and z <= fruit_z + branch_threshold:
                    blocked.append(True)
                else:
                    blocked.append(False)
            else:
                blocked.append(False)

        target["_ring_debug"] = {
            "cx": cx_img,
            "cy": cy_img,
            "r": ring_r,
            "blocked": blocked,
            "n_samples": n_samples,
            "gap_angle": 0.0,
            "detected": False,
        }

        # Count blocked sectors and find widest clear gap (circular scan)
        n_blocked = sum(blocked)
        if n_blocked < 3 or n_blocked > n_samples - 3:
            # Too few or too many blocked = not a between-branches pattern
            target["between_branches"] = False
            target["gap_angle_cam"] = 0.0
            return

        # Find longest run of consecutive clear (False) sectors
        best_start = 0
        best_len = 0
        cur_start = 0
        cur_len = 0
        # Double the array for circular wrap-around
        doubled = blocked + blocked
        for i in range(len(doubled)):
            if not doubled[i]:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0

        # Cap best_len to n_samples (circular wrap)
        best_len = min(best_len, n_samples)

        if best_len < 3:
            # No significant clear gap
            target["between_branches"] = False
            target["gap_angle_cam"] = 0.0
            return

        # Gap midpoint angle
        mid_idx = (best_start + best_len // 2) % n_samples
        gap_angle = mid_idx * (2 * math.pi / n_samples)

        target["between_branches"] = True
        target["gap_angle_cam"] = float(gap_angle)
        target["_ring_debug"]["gap_angle"] = gap_angle
        target["_ring_debug"]["detected"] = True

    def _fit_ellipse(self, mask_clean: np.ndarray) -> tuple:
        """Fit ellipse to mask and extract short axis."""
        t_short_axis = None
        long_axis_2d = None
        t_angle = None
        elongation_confidence = 0.0

        try:
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) >= 20:
                    ellipse = cv2.fitEllipse(cnt)
                    (_, _), (axis_a, axis_b), angle_deg = ellipse
                    # OpenCV's angle follows the first returned diameter. Make
                    # it explicitly describe the fruit's major/long axis.
                    if axis_b > axis_a:
                        angle_deg += 90.0
                    major = max(float(axis_a), float(axis_b))
                    minor = min(float(axis_a), float(axis_b))
                    # 0 for circular, approaching 1 for a strongly elongated
                    # fruit. Unlike an axis-aligned bbox this is rotation invariant.
                    elongation_confidence = float(np.clip(
                        1.0 - minor / max(major, 1e-6), 0.0, 1.0))
                    angle_deg = ((angle_deg + 90.0) % 180.0) - 90.0
                    theta = math.radians(angle_deg)
                    long_dir_img = np.array([math.cos(theta), math.sin(theta)], dtype=float)
                    short_dir_cam = np.array([-long_dir_img[1], long_dir_img[0], 0.0], dtype=float)
                    n_short = np.linalg.norm(short_dir_cam)
                    if n_short > 1e-6:
                        short_dir_cam /= n_short

                    t_short_axis = short_dir_cam
                    long_axis_2d = long_dir_img
                    t_angle = float(angle_deg)
        except Exception as e:
            print(f"[WARN] Ellipse axis extraction failed: {e}")

        return t_short_axis, long_axis_2d, t_angle, elongation_confidence

    def _compute_heatmap(self, roi_xyz: np.ndarray, valid: np.ndarray, mask_clean: np.ndarray) -> tuple:
        """Compute depth heatmap and find best point."""
        heatmap = None
        t_best_point = None
        t_best_dir2d = None
        t_best_point_3d = None
        scored_3d_points = None  # (N, 4) array: x, y, z, score

        depth_vals = roi_xyz[:, :, 2][valid]
        if depth_vals.size > 0:
            z_lo, z_hi = np.percentile(depth_vals, [5.0, 90.0])
            if z_hi <= z_lo:
                z_hi = z_lo + 1e-3
            score = np.zeros_like(roi_xyz[:, :, 2], dtype=np.float32)
            score[valid] = (z_hi - roi_xyz[:, :, 2][valid]) / (z_hi - z_lo)
            score = np.clip(score, 0.0, 1.0)
            score_u8 = (score * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)

            if heatmap is not None and mask_clean is not None:
                ys, xs = np.nonzero(mask_clean)
                if len(xs) > 0:
                    scores = heatmap[ys, xs, 2].astype(float)
                    idx_max = int(np.argmax(scores))
                    peak_y = int(ys[idx_max])
                    peak_x = int(xs[idx_max])
                    t_best_point = np.array([peak_x, peak_y], dtype=float)

                    peak_xyz = roi_xyz[peak_y, peak_x, :]
                    if np.all(np.isfinite(peak_xyz)):
                        t_best_point_3d = peak_xyz.copy()

                    cx = mask_clean.shape[1] / 2.0
                    cy = mask_clean.shape[0] / 2.0
                    dir_vec = np.array([peak_x - cx, peak_y - cy], dtype=float)
                    n_dir = np.linalg.norm(dir_vec)
                    if n_dir > 1e-6:
                        dir_vec /= n_dir
                    else:
                        dir_vec = np.array([1.0, 0.0], dtype=float)
                    t_best_dir2d = dir_vec

                    # Collect scored 3D points for RViz marker (front surface only)
                    valid_in_mask = np.zeros_like(valid)
                    valid_in_mask[ys, xs] = valid[ys, xs]
                    mask_ys, mask_xs = np.nonzero(valid_in_mask)
                    if len(mask_ys) > 0:
                        pts_3d = roi_xyz[mask_ys, mask_xs, :]
                        pts_scores = score[mask_ys, mask_xs]
                        finite_mask = np.all(np.isfinite(pts_3d), axis=1) & (pts_3d[:, 2] > 0.05)
                        pts_3d = pts_3d[finite_mask]
                        pts_scores = pts_scores[finite_mask]
                        # Keep only front 30% by depth to avoid background leakage
                        if len(pts_3d) > 0:
                            z_thresh = np.percentile(pts_3d[:, 2], 30)
                            front_mask = pts_3d[:, 2] <= z_thresh
                            pts_3d = pts_3d[front_mask]
                            pts_scores = pts_scores[front_mask]
                        if len(pts_3d) > 100:
                            indices = np.linspace(0, len(pts_3d) - 1, 100, dtype=int)
                            pts_3d = pts_3d[indices]
                            pts_scores = pts_scores[indices]
                        if len(pts_3d) > 0:
                            scored_3d_points = np.column_stack([pts_3d, pts_scores])

        # Compute surface normal from depth gradients (Sobel)
        surface_normal = None
        if valid.any() and mask_clean is not None:
            combined = valid & mask_clean.astype(bool)
            if combined.any():
                x_ch = roi_xyz[:, :, 0].astype(np.float32)
                y_ch = roi_xyz[:, :, 1].astype(np.float32)
                z_ch = roi_xyz[:, :, 2].astype(np.float32)

                dx_du = cv2.Sobel(x_ch, cv2.CV_32F, 1, 0, ksize=5)
                dy_du = cv2.Sobel(y_ch, cv2.CV_32F, 1, 0, ksize=5)
                dz_du = cv2.Sobel(z_ch, cv2.CV_32F, 1, 0, ksize=5)

                dx_dv = cv2.Sobel(x_ch, cv2.CV_32F, 0, 1, ksize=5)
                dy_dv = cv2.Sobel(y_ch, cv2.CV_32F, 0, 1, ksize=5)
                dz_dv = cv2.Sobel(z_ch, cv2.CV_32F, 0, 1, ksize=5)

                # Cross product dP/du x dP/dv = surface normal per pixel
                # Suppress NaN warnings — NaNs from invalid depth are masked out by `combined` below
                with np.errstate(invalid='ignore'):
                    nx = dy_du * dz_dv - dz_du * dy_dv
                    ny = dz_du * dx_dv - dx_du * dz_dv
                    nz = dx_du * dy_dv - dy_du * dx_dv

                norms = np.sqrt(nx**2 + ny**2 + nz**2)
                good = combined & (norms > 1e-6)

                if good.any():
                    nx_g = nx[good] / norms[good]
                    ny_g = ny[good] / norms[good]
                    nz_g = nz[good] / norms[good]

                    # Flip normals to point toward camera (negative Z)
                    flip = nz_g > 0
                    nx_g[flip] *= -1
                    ny_g[flip] *= -1
                    nz_g[flip] *= -1

                    avg_normal = np.array([nx_g.mean(), ny_g.mean(), nz_g.mean()])
                    n_len = np.linalg.norm(avg_normal)
                    if n_len > 1e-6:
                        surface_normal = avg_normal / n_len

        return heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_points, surface_normal

    def _select_best_fruit(self, targets: List[Dict[str, Any]]) -> Optional[int]:
        """Select the best fruit based on scoring system."""
        # Clear hysteresis when exclusion list changes (forces fresh best selection)
        cur_excl_count = len(self.excluded_positions)
        if cur_excl_count != self._prev_excluded_count:
            self._prev_excluded_count = cur_excl_count
            if cur_excl_count > 0 and self.best_target_prev is not None:
                prev_pt = self.best_target_prev.get("pt_base")
                if prev_pt is not None:
                    prev_pos = [prev_pt.point.x, prev_pt.point.y, prev_pt.point.z]
                    if any(math.dist(prev_pos, ep) < 0.08 for ep in self.excluded_positions):
                        self.best_target_prev = None

        prev_pt_base = None
        if self.best_target_prev is not None:
            prev_pt_base = np.array([
                self.best_target_prev["pt_base"].point.x,
                self.best_target_prev["pt_base"].point.y,
                self.best_target_prev["pt_base"].point.z,
            ], dtype=float)

        best_idx = None
        best_score = -1.0

        for i, t in enumerate(targets):
            score_result = compute_fruit_score(t, prev_pt_base)
            t["score"] = score_result["total_score"]
            t["score_components"] = score_result["components"]

            # Skip targets near excluded positions (multi-subscribe)
            if self.excluded_positions:
                pt = t.get("pt_base")
                if pt is not None:
                    t_pos = [pt.point.x, pt.point.y, pt.point.z]
                    dists = [math.dist(t_pos, ep) for ep in self.excluded_positions]
                    if any(d < 0.08 for d in dists):
                        t["excluded"] = True
                        continue

            if score_result["total_score"] > best_score:
                best_score = score_result["total_score"]
                best_idx = i

        # Hysteresis (skip excluded targets)
        if self.best_target_prev is not None and best_idx is not None:
            prev_pt = self.best_target_prev.get("pt_base")
            if prev_pt is not None:
                prev_pos = [prev_pt.point.x, prev_pt.point.y, prev_pt.point.z]
                for i, t in enumerate(targets):
                    if t.get("excluded"):
                        continue
                    t_pt = t["pt_base"]
                    t_pos = [t_pt.point.x, t_pt.point.y, t_pt.point.z]
                    if math.dist(prev_pos, t_pos) < BEST_REUSE_THRESH:
                        prev_current_score = targets[i]["score"]
                        if best_idx != i and best_score < prev_current_score + SWITCH_THRESHOLD:
                            best_idx = i
                            best_score = prev_current_score
                        break

        # Target lock override
        if self.target_lock_active and self.target_lock_position is not None:
            lock_best_idx = None
            lock_best_dist = float('inf')
            for i, t in enumerate(targets):
                pt = t["pt_base"]
                t_pos = [pt.point.x, pt.point.y, pt.point.z]
                dist = math.dist(t_pos, self.target_lock_position)
                if dist < TARGET_LOCK_RADIUS and dist < lock_best_dist:
                    lock_best_dist = dist
                    lock_best_idx = i

            if lock_best_idx is not None:
                best_idx = lock_best_idx
                pt = targets[lock_best_idx]["pt_base"]
                self.target_lock_position[0] = pt.point.x
                self.target_lock_position[1] = pt.point.y
                self.target_lock_position[2] = pt.point.z

        if best_idx is not None:
            self.best_target_prev = targets[best_idx]

        # Build top-3 candidate indices by score (skip excluded targets)
        scored = [(i, targets[i].get("score", 0.0)) for i in range(len(targets))
                  if not targets[i].get("excluded")]
        scored.sort(key=lambda x: x[1], reverse=True)
        top3 = [idx for idx, _ in scored[:3]]
        # Store on each target its rank (1-based) if in top 3
        for t in targets:
            t.pop("candidate_rank", None)
        for rank, idx in enumerate(top3):
            targets[idx]["candidate_rank"] = rank + 1

        return best_idx

    def _record_grasp_candidates(
        self, target: Dict[str, Any], contacts: Dict[str, Any],
        intrinsics: Dict[str, float],
    ) -> None:
        """Retain a target-associated pre-motion Phase-3 evaluation."""
        candidates = contacts.get("candidates", [])
        point = target.get("pt_base")
        if not candidates or point is None:
            return
        fruit_id = int(target.get("fruit_id", -1))
        winner_deg = float(candidates[0]["psi_deg"])
        winner_score = float(candidates[0]["total_score"])
        self._grasp_candidate_history.append(
            (fruit_id, winner_deg, winner_score, time()))
        recent = [
            (angle, score) for fid, angle, score, stamp in self._grasp_candidate_history
            if fid == fruit_id and time() - stamp <= 1.5
        ][-5:]
        period_deg = 120.0
        unwrapped = np.asarray([
            winner_deg + ((angle - winner_deg + 60.0) % period_deg - 60.0)
            for angle, _score in recent
        ], dtype=float)
        weights = np.asarray([
            max(1e-3, score) for _angle, score in recent
        ], dtype=float)
        median_deg = float(np.median(unwrapped)) if len(unwrapped) else winner_deg
        inlier_mask = np.abs(unwrapped - median_deg) <= 15.0
        inlier_count = int(np.count_nonzero(inlier_mask))
        rejected_count = int(len(unwrapped) - inlier_count)
        if inlier_count:
            consensus_unwrapped = float(np.average(
                unwrapped[inlier_mask], weights=weights[inlier_mask]))
            consensus_psi_deg = consensus_unwrapped % period_deg
            inlier_values = unwrapped[inlier_mask]
            spread_deg = float(np.max(inlier_values) - np.min(inlier_values))
        else:
            consensus_psi_deg = winner_deg
            spread_deg = float("inf")
        consensus_ratio = inlier_count / max(1, len(unwrapped))
        latest_consensus_delta_deg = abs(
            (winner_deg - consensus_psi_deg + 60.0) % period_deg - 60.0)
        stable = (
            len(unwrapped) >= 4
            and inlier_count >= 4
            and consensus_ratio >= 0.80
            and spread_deg <= 10.0
            and latest_consensus_delta_deg <= 10.0
        )
        history_angles = [float(angle) for angle, _score in recent]

        # Project gripper local +X (the zero-angle finger-pattern reference)
        # into this target's image tangent plane. _cached_tf_grip maps camera
        # vectors into gripper_tip, so its transpose maps gripper axes to camera.
        current_psi_deg = None
        proposed_delta_deg = None
        if self._cached_tf_grip is not None:
            rotation_gripper_from_camera = self._cached_tf_grip[0]
            reference_camera = rotation_gripper_from_camera.T @ np.array(
                [1.0, 0.0, 0.0], dtype=float)
            target_camera = np.array([
                float(target.get("Xc", 0.0)),
                float(target.get("Yc", 0.0)),
                float(target.get("Zc", 0.0)),
            ], dtype=float)
            z_cam = float(target_camera[2])
            if z_cam > 0.05:
                du = float(intrinsics["fx"]) * (
                    reference_camera[0] * z_cam
                    - target_camera[0] * reference_camera[2]) / (z_cam * z_cam)
                dv = float(intrinsics["fy"]) * (
                    reference_camera[1] * z_cam
                    - target_camera[1] * reference_camera[2]) / (z_cam * z_cam)
                if math.hypot(du, dv) > 1e-6:
                    current_psi_deg = math.degrees(math.atan2(dv, du)) % period_deg
                    proposed_delta_deg = (
                        (consensus_psi_deg - current_psi_deg + 60.0)
                        % period_deg - 60.0)

        best_score = float(candidates[0]["total_score"])
        near_best = [
            candidate for candidate in candidates
            if best_score - float(candidate["total_score"]) <= 0.01
        ]
        band_angles = [
            winner_deg + ((float(candidate["psi_deg"]) - winner_deg + 60.0) % period_deg - 60.0)
            for candidate in near_best
        ]
        best_safe_candidate = None
        best_safe_delta_deg = None
        if current_psi_deg is not None:
            safe_candidates = []
            for candidate in candidates:
                candidate_delta_deg = (
                    (float(candidate["psi_deg"]) - current_psi_deg + 60.0)
                    % period_deg - 60.0)
                if abs(candidate_delta_deg) <= 35.0:
                    safe_candidates.append((candidate, candidate_delta_deg))
            if safe_candidates:
                best_safe_candidate, best_safe_delta_deg = max(
                    safe_candidates,
                    key=lambda item: float(item[0]["total_score"]),
                )
        self._latest_grasp_candidate_record = {
            "fruit_id": fruit_id,
            "stamp": time(),
            "position": [point.point.x, point.point.y, point.point.z],
            "candidates": candidates,
            "stable": stable,
            "stable_frames": len(unwrapped),
            "spread_deg": spread_deg,
            "winner_history_deg": history_angles,
            "consensus_psi_deg": consensus_psi_deg,
            "consensus_inliers": inlier_count,
            "consensus_rejected": rejected_count,
            "consensus_ratio": consensus_ratio,
            "latest_consensus_delta_deg": latest_consensus_delta_deg,
            "band_min_deg": min(band_angles),
            "band_max_deg": max(band_angles),
            "current_psi_deg": current_psi_deg,
            "proposed_delta_deg": proposed_delta_deg,
            "best_safe_candidate": best_safe_candidate,
            "best_safe_delta_deg": best_safe_delta_deg,
        }

    def _publish_locked_grasp_candidates(self, lock_position: List[float]) -> None:
        """Freeze and publish the candidate set associated with a new target lock."""
        record = self._latest_grasp_candidate_record
        if record is None:
            self.node.get_logger().debug(
                "[GRASP_CANDIDATES] no pre-lock candidate evaluation available")
            return
        age_s = time() - float(record["stamp"])
        distance_m = math.dist(lock_position, record["position"])
        if age_s > 2.0 or distance_m > TARGET_LOCK_RADIUS:
            self.node.get_logger().debug(
                f"[GRASP_CANDIDATES] rejected stale/mismatched pre-lock set "
                f"target_id={record['fruit_id']} age={age_s:.2f}s "
                f"distance={distance_m*1000:.1f}mm")
            return

        compact = []
        for candidate in record["candidates"]:
            finger_scores = candidate["point_scores"]
            compact.append(
                f"#{candidate['rank']} psi={candidate['psi_deg']:.1f}deg "
                f"F=[{finger_scores[0]:.2f},{finger_scores[1]:.2f},"
                f"{finger_scores[2]:.2f}] min={candidate['minimum_contact_score']:.2f} "
                f"mean={candidate['mean_contact_score']:.2f} "
                f"sym={candidate['symmetry_score']:.2f} "
                f"center={candidate['centering_score']:.2f} "
                f"sep={candidate['separation_score']:.2f} "
                f"Q={candidate['total_score']:.3f} "
                f"valid={candidate['geometry_valid']}")
        spread_text = (
            f"{record['spread_deg']:.1f}" if math.isfinite(record["spread_deg"])
            else "inf"
        )
        current_text = (
            f"{record['current_psi_deg']:.1f}deg"
            if record["current_psi_deg"] is not None else "unavailable"
        )
        delta_text = (
            f"{record['proposed_delta_deg']:+.1f}deg"
            if record["proposed_delta_deg"] is not None else "unavailable"
        )
        band_width = float(record["band_max_deg"] - record["band_min_deg"])
        selected_minimum = float(record["candidates"][0]["minimum_contact_score"])
        unambiguous = bool(
            record["stable"]
            and band_width <= 20.0
            and record["proposed_delta_deg"] is not None
            and selected_minimum >= 0.75
            and abs(record["proposed_delta_deg"]) <= 35.0
            and record["latest_consensus_delta_deg"] <= 10.0
        )
        history_text = ",".join(
            f"{angle:.1f}" for angle in record["winner_history_deg"])
        safe_candidate = record.get("best_safe_candidate")
        safe_delta = record.get("best_safe_delta_deg")
        safe_candidate_available = False
        safe_score_loss = 1.0
        if safe_candidate is not None and safe_delta is not None:
            safe_scores = safe_candidate["point_scores"]
            safe_score_loss = max(
                0.0,
                float(record["candidates"][0]["total_score"])
                - float(safe_candidate["total_score"]),
            )
            safe_candidate_available = bool(
                record["stable"]
                and band_width <= 20.0
                and bool(safe_candidate["geometry_valid"])
                and float(safe_candidate["minimum_contact_score"]) >= 0.75
                and safe_score_loss <= 0.05
            )
            safe_text = (
                f"safe_candidate_psi={safe_candidate['psi_deg']:.1f}deg "
                f"safe_delta={safe_delta:+.1f}deg "
                f"safe_F=[{safe_scores[0]:.2f},{safe_scores[1]:.2f},"
                f"{safe_scores[2]:.2f}] "
                f"safe_min={safe_candidate['minimum_contact_score']:.2f} "
                f"safe_Q={safe_candidate['total_score']:.3f} "
                f"score_loss={safe_score_loss:.3f} "
                f"safe_decision={'SAFE_CANDIDATE_AVAILABLE' if safe_candidate_available else 'NO_SAFE_CANDIDATE'}"
            )
        else:
            safe_text = (
                "safe_candidate_psi=unavailable safe_delta=unavailable "
                "safe_decision=NO_SAFE_CANDIDATE"
            )
        log_candidate_details = bool(self.node.get_parameter(
            "vision.log_grasp_candidate_details").value)
        detail_text = (
            f" count={len(record['candidates'])} | " + " | ".join(compact)
            if log_candidate_details else
            f" count={len(record['candidates'])} details=SUPPRESSED"
        )
        summary = (
            f"[GRASP_CANDIDATES] evaluation_only frozen_pre_motion=True "
            f"target_id={record['fruit_id']} age={age_s:.2f}s "
            f"match={distance_m*1000:.1f}mm psi_frame=image_roi "
            f"stable={record['stable']} frames={record['stable_frames']} "
            f"winner_history=[{history_text}]deg "
            f"consensus={record['consensus_psi_deg']:.1f}deg "
            f"inliers={record['consensus_inliers']} "
            f"rejected={record['consensus_rejected']} "
            f"consensus_ratio={record['consensus_ratio']:.2f} "
            f"latest_consensus_delta="
            f"{record['latest_consensus_delta_deg']:.1f}deg "
            f"inlier_spread={spread_text}deg preferred_band="
            f"[{record['band_min_deg']:.1f},{record['band_max_deg']:.1f}]deg "
            f"band_width={band_width:.1f}deg current_psi={current_text} "
            f"winner_psi={record['candidates'][0]['psi_deg']:.1f}deg "
            f"proposed_local_delta={delta_text} "
            f"decision={'YAW_PROPOSED' if unambiguous else 'YAW_AMBIGUOUS'} "
            f"{safe_text}" + detail_text
        )
        self.node.get_logger().debug(summary)
        message = StdString()
        message.data = summary
        self.grasp_candidates_pub.publish(message)
        selection = Float32MultiArray()
        selection.data = [
            1.0 if safe_candidate_available else 0.0,
            float(safe_delta) if safe_delta is not None else 0.0,
            float(safe_candidate["psi_deg"]) if safe_candidate is not None else 0.0,
            float(safe_candidate["total_score"]) if safe_candidate is not None else 0.0,
            float(safe_candidate["minimum_contact_score"]) if safe_candidate is not None else 0.0,
            float(safe_score_loss) if safe_candidate is not None else 1.0,
            float(record["position"][0]),
            float(record["position"][1]),
            float(record["position"][2]),
        ]
        self.grasp_candidate_selection_pub.publish(selection)

    def _process_best_target(
        self,
        targets: List[Dict[str, Any]],
        best_idx: int,
        intrinsics: Dict[str, float],
        bunch_boxes: Optional[List[Dict[str, Any]]] = None,
        pc_np: Optional[np.ndarray] = None,
    ) -> None:
        """Process the best target and publish goal."""
        t_best = targets[best_idx]

        # Predict three fingertip contact regions for the selected date only.
        # This is visual/diagnostic evidence; it does not change the goal pose or
        # command the gripper until field validation establishes safe thresholds.
        if SHOW_FINGER_CONTACTS:
            x1, y1, x2, y2 = t_best["bb"]
            contact_key = (
                int(t_best.get("fruit_id", 0)),
                x1 // 8, y1 // 8, x2 // 8, y2 // 8,
            )
            contacts = self._contact_cache.get(contact_key)
            if contacts is None:
                roi_xyz = None
                if (
                    pc_np is not None
                    and pc_np.ndim == 3
                    and y2 <= pc_np.shape[0]
                    and x2 <= pc_np.shape[1]
                    and y2 > y1
                    and x2 > x1
                ):
                    roi_xyz = pc_np[y1:y2, x1:x2, :3]
                contacts = estimate_three_finger_contacts(
                    t_best.get("mask_resized"),
                    roi_xyz=roi_xyz,
                    target_depth=float(t_best.get("Zc", 0.0)),
                    focal_length_px=0.5 * (
                        float(intrinsics["fx"]) + float(intrinsics["fy"])),
                    fingertip_radius_m=FINGERTIP_CONTACT_RADIUS_M,
                    radial_fraction=FINGER_CONTACT_RADIAL_FRACTION,
                    rotation_samples=FINGER_CONTACT_ROTATION_SAMPLES,
                    minimum_score=FINGER_CONTACT_MIN_SCORE,
                )
                if len(self._contact_cache) >= 32:
                    self._contact_cache.pop(next(iter(self._contact_cache)))
                self._contact_cache[contact_key] = contacts
            t_best["finger_contacts"] = contacts
            self._record_grasp_candidates(t_best, contacts, intrinsics)

        # Publish the camera-frame depth evidence for the exact target selected
        # for /external_goal_pose. The motion node caches this and prints it only
        # when an operator accepts a goal.
        _diag = t_best.get("depth_diag")
        if _diag is not None:
            _depth_msg = Float32MultiArray()
            _depth_msg.data = [
                float(t_best["Xc"]), float(t_best["Yc"]), float(t_best["Zc"]),
                *[float(v) for v in _diag],
            ]
            self.depth_diag_pub.publish(_depth_msg)

        # Use per-frame cached TF — no new lookup needed
        _R_tf: Optional[np.ndarray] = None
        _t_tf: Optional[np.ndarray] = None
        if self._cached_tf_base is not None:
            _R_tf, _t_tf = self._cached_tf_base

        # Publish the depth-supported heatmap peak as the selected fruit's 3D
        # aim point. Motion applies its own temporal/proximity validation.
        _tip_cam = t_best.get("best_point_3d")
        if _tip_cam is not None and _R_tf is not None and _t_tf is not None:
            _tip_cam = np.asarray(_tip_cam, dtype=float)
            if _tip_cam.shape == (3,) and np.all(np.isfinite(_tip_cam)):
                _tip_base = _R_tf @ _tip_cam + _t_tf
                _tip_msg = PointStamped()
                _tip_msg.header.frame_id = "base_link"
                _tip_msg.header.stamp = self.node.get_clock().now().to_msg()
                _tip_msg.point.x = float(_tip_base[0])
                _tip_msg.point.y = float(_tip_base[1])
                _tip_msg.point.z = float(_tip_base[2])
                self.date_tip_pub.publish(_tip_msg)

        # Smooth best heatmap point
        best_pt = t_best.get("best_point2d")
        if best_pt is not None:
            if self.prev_heat_point is None:
                sm_pt = best_pt.copy()
            else:
                sm_pt = 0.7 * self.prev_heat_point + 0.3 * best_pt
            self.prev_heat_point = sm_pt.copy()
            t_best["best_point2d_smooth"] = sm_pt
        else:
            self.prev_heat_point = None

        # Compute approach direction
        pt_base = t_best["pt_base"]
        dir_msg = None
        raw_pt = t_best.get("best_point2d")

        if raw_pt is not None:
            x1, y1, x2, y2 = t_best["bb"]
            roi_w = x2 - x1
            roi_h = y2 - y1
            cx = roi_w / 2.0
            cy = roi_h / 2.0
            dx = float(raw_pt[0]) - cx
            dy = float(raw_pt[1]) - cy

            min_offset = 0.1 * min(roi_w, roi_h)
            offset_mag = math.hypot(dx, dy)

            if offset_mag > min_offset:
                best_pt_3d = t_best.get("best_point_3d")
                centroid_3d = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]])
                sn = t_best.get("surface_normal")

                # Compute peak direction (heatmap closest point - centroid)
                peak_dir = None
                if best_pt_3d is not None:
                    peak_dir = best_pt_3d - centroid_3d
                    n_peak = np.linalg.norm(peak_dir)
                    if n_peak > 1e-6:
                        peak_dir = peak_dir / n_peak
                    else:
                        peak_dir = None

                # Blend: 70% surface normal + 30% heatmap peak
                if sn is not None and peak_dir is not None:
                    heatmap_dir = 0.7 * sn + 0.3 * peak_dir
                    n_blend = np.linalg.norm(heatmap_dir)
                    if n_blend > 1e-6:
                        heatmap_dir = heatmap_dir / n_blend
                    else:
                        heatmap_dir = sn.copy()
                elif sn is not None:
                    heatmap_dir = sn.copy()
                elif peak_dir is not None:
                    heatmap_dir = peak_dir
                else:
                    heatmap_dir = np.array([dx / offset_mag, dy / offset_mag, 0.0], dtype=float)

                # Only run collision direction when target is locked (robot committed to grasp).
                # Idle/scanning frames skip the vectorized ray-sphere math entirely.
                if self.target_lock_active:
                    collision_result = compute_collision_free_direction(t_best, targets, best_idx, heatmap_dir)
                else:
                    collision_result = {"direction": heatmap_dir, "clearance": float('inf'),
                                        "is_collision_free": True, "num_blocked": 0}
                dir_cam = collision_result["direction"]

                t_best["clearance"] = collision_result["clearance"]
                t_best["is_collision_free"] = collision_result["is_collision_free"]
                t_best["heatmap_dir_cam"] = heatmap_dir.copy()
                t_best["approach_dir_cam"] = dir_cam.copy()

                if _R_tf is not None:
                    dir_raw = _R_tf @ dir_cam
                else:
                    dir_raw = dir_cam.copy()

                if self.prev_direction_base is not None:
                    if float(np.dot(self.prev_direction_base, dir_raw)) < 0:
                        dir_raw = -dir_raw

                self.direction_history.append(dir_raw.copy())

            if len(self.direction_history) > 0:
                weights = np.arange(1, len(self.direction_history) + 1, dtype=float)
                stacked = np.vstack(list(self.direction_history))
                dir_avg = (stacked * weights[:, None]).sum(axis=0) / weights.sum()

                n_avg = np.linalg.norm(dir_avg)
                if n_avg > 1e-6:
                    dir_avg /= n_avg
                else:
                    dir_avg = np.array([1.0, 0.0, 0.0])

                self.prev_direction_base = dir_avg.copy()
                t_best["approach_dir_base"] = dir_avg.copy()

                dir_msg = Vector3Stamped()
                dir_msg.header.frame_id = "base_link"
                dir_msg.header.stamp = self.node.get_clock().now().to_msg()
                dir_msg.vector.x = float(dir_avg[0])
                dir_msg.vector.y = float(dir_avg[1])
                dir_msg.vector.z = float(dir_avg[2])

                up = np.array([0.0, 0.0, 1.0])
                approach_axis = np.cross(up, dir_avg)
                n_ax = np.linalg.norm(approach_axis)
                if n_ax > 1e-6:
                    approach_axis /= n_ax
                else:
                    approach_axis = np.array([1.0, 0.0, 0.0])

                q = quat_align_x_to_axis(approach_axis)
                t_best["quat"] = q
                t_best["approach_axis"] = approach_axis

        if t_best.get("quat") is None:
            t_best["quat"] = (1.0, 0.0, 0.0, 0.0)
            t_best["approach_axis"] = np.array([1.0, 0.0, 0.0])

        # Smooth position
        pt_vec = np.array([pt_base.point.x, pt_base.point.y, pt_base.point.z], dtype=float)

        if len(self.best_history) > 0:
            last_pt = self.best_history[-1]
            jump_dist = np.linalg.norm(pt_vec - last_pt)
            if jump_dist > 0.15:
                self.best_history.clear()
                self.best_target_prev = None

        self.best_history.append(pt_vec)
        if len(self.best_history) >= 2:
            weights = np.arange(1, len(self.best_history) + 1, dtype=float)
            stacked = np.vstack(self.best_history)
            pt_smooth = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
        else:
            pt_smooth = pt_vec.copy()

        pt_x, pt_y, pt_z = pt_smooth

        if pt_z <= Z_MAX:
            goal = PoseStamped()
            goal.header = pt_base.header
            # This timestamp identifies a newly computed perception sample.
            # The 50 Hz publishing timer deliberately preserves it so consumers
            # can distinguish new samples from repeats of the same sample.
            goal.header.stamp = self.node.get_clock().now().to_msg()
            goal.pose.position.x = float(pt_x)
            goal.pose.position.y = float(pt_y)
            goal.pose.position.z = float(pt_z)

            q = t_best["quat"]
            goal.pose.orientation.w = q[0]
            goal.pose.orientation.x = q[1]
            goal.pose.orientation.y = q[2]
            goal.pose.orientation.z = q[3]

            with self.pub_lock:
                self.latest_goal_msg = goal
                self.latest_dir_msg = dir_msg
                self._latest_goal_update_time = time()

            # Publish estimated fruit radius for adaptive gripper
            radius = estimate_fruit_radius(t_best, fx=intrinsics["fx"])
            radius_msg = Float32()
            radius_msg.data = float(radius)
            self.radius_pub.publish(radius_msg)

            # Publish normalised bounding-box centre [cx_norm, cy_norm] in [0,1].
            # cx_norm: 0=left edge, 1=right edge of image
            # cy_norm: 0=top edge,  1=bottom edge of image
            # Used by main node to classify approach direction from image-space position.
            _bx1, _by1, _bx2, _by2 = t_best["bb"]
            _iw = float(getattr(self, '_disp_w', 0))
            _ih = float(getattr(self, '_disp_h', 0))
            if _iw > 0 and _ih > 0:
                _cx_norm = ((_bx1 + _bx2) / 2.0) / _iw
                _cy_norm = ((_by1 + _by2) / 2.0) / _ih
                _bunch_rel_x = -1.0
                _bunch_rel_y = -1.0
                if bunch_boxes:
                    _b = bunch_boxes[0]
                    _poly = _b.get("polygon")
                    if _poly is not None and len(_poly) > 2:
                        _poly_np = np.asarray(_poly, dtype=np.float32)
                        _bxs = _poly_np[:, 0]
                        _bys = _poly_np[:, 1]
                        _b_x1, _b_x2 = float(np.min(_bxs)), float(np.max(_bxs))
                        _b_y1, _b_y2 = float(np.min(_bys)), float(np.max(_bys))
                    else:
                        _b_x1, _b_y1, _b_x2, _b_y2 = _b["bb"]
                        _b_x1, _b_y1, _b_x2, _b_y2 = float(_b_x1), float(_b_y1), float(_b_x2), float(_b_y2)
                    _fruit_cx = float((_bx1 + _bx2) / 2.0)
                    _fruit_cy = float((_by1 + _by2) / 2.0)
                    if _b_x2 > _b_x1 + 1.0:
                        _bunch_rel_x = float(np.clip((_fruit_cx - _b_x1) / (_b_x2 - _b_x1), 0.0, 1.0))
                    if _b_y2 > _b_y1 + 1.0:
                        _bunch_rel_y = float(np.clip((_fruit_cy - _b_y1) / (_b_y2 - _b_y1), 0.0, 1.0))
                _bbox_msg = Float32MultiArray()
                _bbox_msg.data = [float(_cx_norm), float(_cy_norm), float(_bunch_rel_x), float(_bunch_rel_y)]
                self.bbox_norm_pub.publish(_bbox_msg)

            # Publish branch gap info for 2-finger mode
            # Debug mode may analyze an unlocked target for visualization, but
            # only a real target lock may command the two-finger grasp behavior.
            between_branches = (
                self.target_lock_active
                and t_best.get("between_branches", False)
            )
            gap_angle_cam = t_best.get("gap_angle_cam", 0.0)
            gap_angle_base = gap_angle_cam
            if between_branches and _R_tf is not None:
                gap_dir_cam = np.array([
                    math.cos(gap_angle_cam),
                    math.sin(gap_angle_cam),
                    0.0
                ], dtype=float)
                gap_dir_base = _R_tf @ gap_dir_cam
                gap_angle_base = float(math.atan2(gap_dir_base[2], gap_dir_base[0]))

            contacts = t_best.get("finger_contacts") or {}
            contact_valid = bool(contacts.get("valid", False))
            contact_score = float(contacts.get("score", 0.0))
            contact_angle_cam = math.radians(float(contacts.get("rotation_deg", 0.0)))
            contact_angle_base = contact_angle_cam
            if contact_valid and _R_tf is not None:
                contact_dir_cam = np.array([
                    math.cos(contact_angle_cam),
                    math.sin(contact_angle_cam),
                    0.0,
                ], dtype=float)
                contact_dir_base = _R_tf @ contact_dir_cam
                # Old robot approaches along -Y, so tool roll lies in the X-Z plane.
                contact_angle_base = float(math.atan2(
                    contact_dir_base[2], contact_dir_base[0]))

            # Ellipse fitting is optional. A target may be valid even when the
            # fit returned None, so never let missing orientation metadata kill
            # the long-running perception thread.
            ellipse_angle = t_best.get("ellipse_angle")
            ellipse_confidence = t_best.get("ellipse_confidence")
            try:
                ellipse_angle = float(ellipse_angle)
            except (TypeError, ValueError):
                ellipse_angle = 90.0
            if not math.isfinite(ellipse_angle):
                ellipse_angle = 90.0
            try:
                ellipse_confidence = float(ellipse_confidence)
            except (TypeError, ValueError):
                ellipse_confidence = 0.0
            if not math.isfinite(ellipse_confidence):
                ellipse_confidence = 0.0

            gap_msg = Float32MultiArray()
            gap_msg.data = [
                1.0 if between_branches else 0.0,
                float(gap_angle_base),
                1.0 if contact_valid else 0.0,
                float(contact_angle_base),
                contact_score,
                # Signed major-axis deviation from image vertical. Vertical is
                # zero; 180-degree axis ambiguity is removed by wrapping ±90.
                float(math.radians(((((ellipse_angle - 90.0) + 90.0)
                                      % 180.0) - 90.0))),
                ellipse_confidence,
            ]
            self.gap_info_pub.publish(gap_msg)

            # Publish score components for GUI bar chart
            score_components = t_best.get("score_components")
            if score_components:
                import json as _json
                from std_msgs.msg import String as _Str
                sm = _Str()
                sm.data = _json.dumps({k: round(float(v), 3) for k, v in score_components.items()})
                self.score_pub.publish(sm)

            # Publish heatmap 3D data for goal marker (consumed on subscribe)
            scored_pts = t_best.get("scored_3d_points")
            if scored_pts is not None and _R_tf is not None and _t_tf is not None:
                # Vectorised transform: (N,3) @ R.T + t  — no Python loop needed
                pts_cam = scored_pts[:, :3].astype(np.float64)
                pts_base = pts_cam @ _R_tf.T + _t_tf
                flat = np.column_stack([pts_base, scored_pts[:, 3]]).flatten().tolist()
                hm_msg = Float32MultiArray()
                hm_msg.data = flat
                self.heatmap_data_pub.publish(hm_msg)
        else:
            self.best_history.clear()
