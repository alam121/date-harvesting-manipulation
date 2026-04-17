"""Main VisionNode class for date fruit detection."""

import math
from collections import deque
from threading import Lock, Thread
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
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import PointCloud2, Image as ROSImage
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 - Required for transform registration

from .config import (
    CAM_FRAME, Z_MAX,
    BEST_REUSE_THRESH, SWITCH_THRESHOLD, TARGET_LOCK_RADIUS,
    TRUNK_Y_OFFSET,
    USE_LIDAR, LIDAR_TOPIC, LIDAR_Z_MIN, LIDAR_Z_MAX, T_CAM_LIDAR,
    ZEDXONE_IMAGE_TOPIC, ZEDXONE_WIDTH, ZEDXONE_HEIGHT,
    ZEDXONE_FX, ZEDXONE_FY, ZEDXONE_CX, ZEDXONE_CY, ZEDXONE_DIST,
    ZEDMINI_SERIAL, ZEDMINI_DEPTH_FPS,
    ZEDMINI_DEPTH_Z_MIN, ZEDMINI_DEPTH_Z_MAX, ZEDMINI_MAX_POINTS, T_CAM_ZEDMINI,
)
from ..perception_lidar import (
    parse_pointcloud2, project_lidar_to_image,
    lidar_pts_in_mask, centroid_from_lidar_pts,
)
from .math_utils import unit_vector, quat_rotate_vec, quat_align_x_to_axis
from .ros_utils import wait_for_transform, create_pointcloud2_msg
from .zed_utils import apply_zed_one_settings, apply_zed_mini_settings, apply_zed_stereo_settings
apply_zed_camera_settings = apply_zed_one_settings  # used by older call sites below
from .tracking import FruitTracker
from .scoring import compute_fruit_score, compute_collision_free_direction
from .yolo_thread import YoloThread
from .visualization import VisionVisualizer


class VisionNode:
    """ROS2 node for date fruit detection using ZED camera and YOLO."""

    def __init__(self, args):
        self.args = args
        self.exit_signal = False

        # State
        self.best_target_prev: Optional[Dict[str, Any]] = None
        self.prev_heat_point: Optional[np.ndarray] = None
        self.prev_direction_base: Optional[np.ndarray] = None
        self.direction_history = deque(maxlen=15)
        self.best_history = deque(maxlen=3)

        # Heatmap throttling — only recompute every N frames
        self._heatmap_frame_count = 0
        self._heatmap_interval = 3  # recompute every 3rd frame
        self._cached_heatmaps = {}  # key: target index → (heatmap, best_point, best_dir2d, best_point_3d)


        # Target lock
        self.target_lock_position: Optional[List[float]] = None
        self.target_lock_active = False

        # Exclusion zones (for multi-subscribe: skip already-accepted fruits)
        self.excluded_positions: List[List[float]] = []
        self._prev_excluded_count = 0  # track changes to clear hysteresis

        # Publishing state
        self.latest_goal_msg: Optional[PoseStamped] = None
        self.latest_dir_msg: Optional[Vector3Stamped] = None
        self.pub_lock = Lock()

        # Sparse point cloud (used when --use_lidar)
        self._latest_cloud: Optional[np.ndarray] = None
        self._cloud_lock = Lock()

        # Dense depth map (H×W×3 XYZ) at display resolution (used when --use_zed_mini)
        self._latest_depth_map: Optional[np.ndarray] = None
        # Boolean mask of originally projected pixels (True = real data, False = EDT fill)
        self._latest_depth_map_orig: Optional[np.ndarray] = None
        self._depth_map_lock = Lock()

        # Raw ZED Mini points in ZED Mini frame + UV in ZED One image (no EDT fill).
        # Used by _extract_target_3d for direct point queries — avoids scatter/fill artefacts.
        self._latest_mini_pts: Optional[tuple] = None  # (pts_mini Nx3, uv_one Nx2)
        self._mini_pts_lock = Lock()

        # ZED X One image from ROS topic (used when --use_lidar)
        self._latest_ros_image: Optional[np.ndarray] = None
        self._ros_image_lock = Lock()

        # ZED X Mini (depth camera for --use_zed_mini mode)
        self._zed_mini = None

        # Components
        self.tracker = FruitTracker()
        self.yolo_thread: Optional[YoloThread] = None
        self.visualizer: Optional[VisionVisualizer] = None

        # ROS2
        self.node = None
        self.executor = None
        self.tf_buffer = None

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
        self.executor = MultiThreadedExecutor(num_threads=4)

        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        goal_pub = self.node.create_publisher(PoseStamped, "/external_goal_pose", fast_qos)
        dir_pub = self.node.create_publisher(Vector3Stamped, "/datefruit_direction", 10)
        depth_pub = self.node.create_publisher(PointCloud2, "/zed_depth_pointcloud", 10)
        trunk_pub = self.node.create_publisher(PointStamped, "/trunk_position", 10)
        self.radius_pub = self.node.create_publisher(Float32, "/fruit_radius", 10)
        self.gap_info_pub = self.node.create_publisher(Float32MultiArray, "/datefruit_gap_info", 10)
        self.image_pub = self.node.create_publisher(ROSImage, "/vision/display", 10)
        self.heatmap_data_pub = self.node.create_publisher(Float32MultiArray, "/vision/heatmap_3d_data", 10)
        self.cv_bridge = CvBridge()

        depth_frame_count = [0]

        # Timer-based publishing callback (50Hz)
        def publish_timer_cb():
            with self.pub_lock:
                if self.latest_goal_msg is not None:
                    self.latest_goal_msg.header.stamp = self.node.get_clock().now().to_msg()
                    goal_pub.publish(self.latest_goal_msg)
                if self.latest_dir_msg is not None:
                    self.latest_dir_msg.header.stamp = self.node.get_clock().now().to_msg()
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

        self.node.create_timer(5.0, self.tracker.cleanup_old_fruit_ids)

        # Camera refresh command — restart the entire process.
        # ZED X Mini cannot reopen in-process (driver doesn't release cleanly);
        # the only reliable refresh is a full process restart via exec().
        def camera_cmd_cb(msg):
            cmd = msg.data.strip()
            if cmd == "refresh":
                self.node.get_logger().info("Camera refresh requested — restarting process...")
                self.exit_signal = True
                self._refresh_requested = True
                # Shut down executor from a separate thread — calling shutdown()
                # from inside a callback that runs on this executor would deadlock.
                _exec = self.executor
                Thread(target=_exec.shutdown, daemon=True).start()

        from std_msgs.msg import String as StdString
        self.node.create_subscription(StdString, "/camera_command", camera_cmd_cb, 10)

        # TF listener
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self.node)

        # Wait for TFs
        required_tfs = [("base_link", CAM_FRAME), ("gripper_tip", CAM_FRAME)]
        for target, source in required_tfs:
            wait_for_transform(self.tf_buffer, target, source, self.node, timeout=5.0)

        use_lidar   = getattr(self.args, "use_lidar",    False)
        use_zed_mini = getattr(self.args, "use_zed_mini", False)
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

            if use_lidar:
                def _cloud_cb(msg):
                    pts = parse_pointcloud2(msg)
                    with self._cloud_lock:
                        self._latest_cloud = pts
                self.node.create_subscription(PointCloud2, LIDAR_TOPIC, _cloud_cb, fast_qos)
                self.node.get_logger().info(f"[LiDAR] subscribed to {LIDAR_TOPIC}")
            else:
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
                    _dbg_frame = 0
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
                                # pts_mini: ZED Mini-frame 3D coords
                                # uv_one:   corresponding pixel positions in ZED One image
                                _raw_mini_pts = pts_mini
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

                        # ── Periodic diagnostics ─────────────────────────────
                        _dbg_frame += 1
                        if _dbg_frame % 30 == 0:
                            _valid_after = np.isfinite(depth_map[:, :, 2]).sum()
                            _total = disp_h * disp_w
                            _zs = depth_map[:, :, 2][np.isfinite(depth_map[:, :, 2])]
                            _z_med = float(np.median(_zs)) if _zs.size > 0 else float('nan')
                            print(f"[ZedMini] scatter={n_valid}/{_total} "
                                  f"filled={_valid_after}/{_total} "
                                  f"({100*_valid_after/_total:.0f}%) "
                                  f"median_Z={_z_med:.3f}m")

                        with self._depth_map_lock:
                            self._latest_depth_map = depth_map
                            self._latest_depth_map_orig = orig_valid
                        with self._mini_pts_lock:
                            self._latest_mini_pts = (_raw_mini_pts, _raw_mini_uv)
                _pending_depth_thread = _zed_mini_depth_thread
        else:
            # ── ZED stereo path ─────────────────────────────────────────────
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
        display_resolution = sl.Resolution(disp_w, disp_h)
        image_left_ocv = np.full((disp_h, disp_w, 4), [245, 239, 239, 255], np.uint8)
        image_scale = [disp_w / cam_w, disp_h / cam_h]
        display_scale = 1.0
        image_left = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters() if not use_mono_depth else None
        objects = sl.Objects() if not use_mono_depth else None
        point_cloud = sl.Mat() if not use_mono_depth else None

        self.visualizer = VisionVisualizer(intrinsics, image_scale, display_scale)

        # Start ZED Mini depth warp thread now that all closure variables are defined
        if _pending_depth_thread is not None:
            self.node.get_logger().info("[ZedMini] depth warp thread starting")
            Thread(target=_pending_depth_thread, daemon=True).start()

        last_viz = 0.0
        loop_fps = 0.0

        def perception_loop():
            nonlocal last_viz, loop_fps, zed
            t_prev = time()

            while not self.exit_signal:
                grab_status = zed.grab() if use_mono_depth else zed.grab(runtime_params)
                if grab_status != sl.ERROR_CODE.SUCCESS:
                    self.exit_signal = True
                    break

                t_now = time()
                loop_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
                t_prev = t_now

                # Get full-res image for YOLO
                if use_mono_depth:
                    zed.retrieve_image(image_left)  # CameraOne: mono, no VIEW arg
                else:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT)
                self.yolo_thread.set_image(image_left.get_data())

                if not self.yolo_thread.dets_ready.is_set():
                    continue
                self.yolo_thread.dets_ready.clear()

                # Ingest YOLO fruit detections into ZED (trunk handled separately)
                current_dets = self.yolo_thread.get_detections()
                if current_dets is None:
                    continue
                trunk_boxes = self.yolo_thread.get_trunk_boxes()
                bunch_boxes = self.yolo_thread.get_bunch_boxes()

                # Get display-resolution image
                if use_mono_depth:
                    zed.retrieve_image(image_left,
                                       resolution=sl.Resolution(disp_w, disp_h))  # CameraOne: no VIEW, no MEM
                else:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU,
                                       sl.Resolution(disp_w, disp_h))
                np.copyto(image_left_ocv, image_left.get_data())

                if use_zed_mini:
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
                    depth_frame_count[0] += 1
                    if pc_np is not None and depth_frame_count[0] % 5 == 0:
                        self._publish_depth_cloud(pc_np, depth_pub, orig_mask=pc_np_orig)
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
                    zed.ingest_custom_mask_objects(current_dets)
                    zed.retrieve_custom_objects(objects, obj_runtime_param)
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU,
                                        sl.Resolution(disp_w, disp_h))
                    pc_np = point_cloud.get_data()[:, :, :3]
                    depth_frame_count[0] += 1
                    if depth_frame_count[0] % 5 == 0:
                        self._publish_depth_cloud(pc_np, depth_pub)

                # Debug: print detection counts every 30 frames
                if self._heatmap_frame_count % 30 == 0:
                    n_dets = len(current_dets) if use_mono_depth else len(getattr(current_dets, 'object_list', []))
                    n_depth_pts = pts_cam_l.shape[0] if use_lidar else 0
                    rej_list = rejected_targets if 'rejected_targets' in dir() else []
                    from collections import Counter
                    rej_reasons = Counter(r.get("reason", "?") for r in rej_list)
                    print(f"[DBG] dets={n_dets} depth_pts={n_depth_pts} targets={len(targets) if 'targets' in dir() else '?'} rejected={dict(rej_reasons)}")

                # Process detected objects
                self._heatmap_frame_count += 1
                if self._heatmap_frame_count % (self._heatmap_interval * 10) == 0:
                    self._cached_heatmaps.clear()  # prevent stale cache buildup
                targets, rejected_targets, viz_only = self._process_objects(
                    current_dets if use_mono_depth else objects,
                    pc_np, image_left_ocv, image_scale, display_resolution, intrinsics,
                    pts_cam=pts_cam_l, uv=uv_l,
                    use_lidar=use_lidar,       # True only for actual LiDAR
                    use_zed_mini=use_zed_mini, # raw dets + dense depth heatmap
                )

                # Temporal stabilization
                targets = self.tracker.stabilize_detections(targets)

                # Publish trunk position for pole obstacle (uses YOLO trunk boxes directly)
                self._publish_trunk_position(
                    trunk_boxes, pc_np, image_scale, image_left_ocv, trunk_pub,
                    pts_cam=pts_cam_l, uv=uv_l, use_lidar=use_lidar,
                )

                # Select best fruit
                best_idx = self._select_best_fruit(targets)

                # Compute approach direction and publish
                if best_idx is not None:
                    self._process_best_target(targets, best_idx, intrinsics)
                elif self.excluded_positions:
                    # All visible targets are excluded — stop publishing stale position
                    with self.pub_lock:
                        self.latest_goal_msg = None

                # Build viz entries for trunk and bunch bboxes (display only)
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

                # Visualization
                now = time()
                if (now - last_viz) >= 0.1:
                    display_image = self.visualizer.render_frame(
                        image_left_ocv, targets, rejected_targets,
                        best_idx, self.yolo_thread.net_fps, loop_fps,
                        viz_only=trunk_viz,
                        lidar_uv=uv_l if use_lidar else None,
                        lidar_pts_cam=pts_cam_l if use_lidar else None,
                    )
                    # Publish to RViz Image display
                    try:
                        # ZED produces BGRA (4-channel); convert to BGR for ROS
                        if len(display_image.shape) == 3 and display_image.shape[2] == 4:
                            pub_image = cv2.cvtColor(display_image, cv2.COLOR_BGRA2BGR)
                        else:
                            pub_image = display_image
                        img_msg = self.cv_bridge.cv2_to_imgmsg(pub_image, encoding="bgr8")
                        img_msg.header.stamp = self.node.get_clock().now().to_msg()
                        self.image_pub.publish(img_msg)
                    except Exception as e:
                        if not getattr(self, '_img_pub_err_logged', False):
                            print(f"[WARN] Failed to publish vision image: {e}")
                            self._img_pub_err_logged = True
                    last_viz = now

        perception_thread = Thread(target=perception_loop, daemon=True)
        perception_thread.start()

        try:
            self.executor.add_node(self.node)
            self.executor.spin()
        finally:
            self.exit_signal = True
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
            rclpy.shutdown()
            if getattr(self, '_refresh_requested', False):
                import os, sys
                print("[Vision] Restarting process for camera refresh...")
                sleep(4.0)  # let ZED/Argus driver release fully before exec
                os.execv(sys.executable, [sys.executable] + sys.argv)

    def _init_zed_and_yolo(self):
        """Initialize ZED camera then start YOLO thread. Returns zed or None on failure.
        ZED must be fully open before YOLO TRT loads — they share the GPU and
        simultaneous TRT initialization causes a segfault."""
        use_lidar    = getattr(self.args, "use_lidar",    False)
        use_zed_mini = getattr(self.args, "use_zed_mini", False)
        use_mono_depth = use_lidar or use_zed_mini

        if use_mono_depth:
            # ZED X One Mono — CameraOne API (pyzed.sl)
            print("Initializing ZED X One Mono (detection camera)...")
            zed = sl.CameraOne()
            init_params = sl.InitParametersOne()
            init_params.camera_resolution = sl.RESOLUTION.HD1080  # 1920x1080
            init_params.camera_fps = 30
            init_params.coordinate_units = sl.UNIT.METER
            init_params.sdk_verbose = 1
            init_params.enable_hdr = False

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
            apply_zed_camera_settings(zed)

            if use_zed_mini:
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
                init_mini.depth_mode = sl.DEPTH_MODE.NEURAL  # higher accuracy than NEURAL_LIGHT
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

        else:
            # ZED stereo — standard Camera API
            input_type = sl.InputType()
            if self.args.svo:
                input_type.set_from_svo_file(self.args.svo)
            zed = sl.Camera()
            init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
            init_params.depth_minimum_distance = 0.15
            init_params.depth_maximum_distance = 50.0

            print("Initializing Camera...")
            status = zed.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                return None
            print("Camera Initialized")
            apply_zed_stereo_settings(zed)
            zed.enable_positional_tracking(sl.PositionalTrackingParameters())
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
        )
        Thread(target=self.yolo_thread.run, daemon=True).start()
        return zed

    def _publish_depth_cloud(self, pc_np: np.ndarray, depth_pub,
                             orig_mask: Optional[np.ndarray] = None) -> None:
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
                    CAM_FRAME,
                    self.node.get_clock().now().to_msg()
                )
                depth_pub.publish(pc_msg)
        except Exception:
            pass

    def _publish_trunk_position(self, trunk_boxes, pc_np, image_scale, image_left_ocv, trunk_pub,
                                pts_cam=None, uv=None, use_lidar=False) -> None:
        """Publish detected trunk position in base_link for pole obstacle update.
        Uses only the first (largest/most confident) trunk detection."""
        if not trunk_boxes:
            return
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
            return

        if use_lidar and pts_cam is not None and pts_cam.shape[0] > 0:
            # LiDAR path: filter projected points to trunk bounding box
            in_bbox = (
                (uv[:, 0] >= x1) & (uv[:, 0] < x2) &
                (uv[:, 1] >= y1) & (uv[:, 1] < y2)
            )
            if not in_bbox.any():
                return
            pts_trunk = pts_cam[in_bbox]
            zs = pts_trunk[:, 2]
            idx = np.argsort(zs)
            k = max(5, int(0.2 * len(idx)))
            pts_front = pts_trunk[idx[:k]]
        else:
            # ZED depth path
            if pc_np is None:
                return
            roi_xyz = pc_np[y1:y2, x1:x2, :]
            valid_z = np.isfinite(roi_xyz[:, :, 2]) & (roi_xyz[:, :, 2] > 0.1)
            if np.count_nonzero(valid_z) < 10:
                return
            pts = roi_xyz[valid_z]
            zs = pts[:, 2]
            idx = np.argsort(zs)
            k = max(10, int(0.2 * len(idx)))
            pts_front = pts[idx[:k]]

        point_msg = PointStamped()
        point_msg.header.frame_id = CAM_FRAME
        point_msg.header.stamp = rclpyTime().to_msg()
        point_msg.point.x = float(np.mean(pts_front[:, 0]))
        point_msg.point.y = float(np.mean(pts_front[:, 1]))
        point_msg.point.z = float(np.mean(pts_front[:, 2]))
        try:
            pt_base = self.tf_buffer.transform(
                point_msg, "base_link", timeout=rclpyDuration(seconds=0.1)
            )
            pt_base.point.y += TRUNK_Y_OFFSET
            trunk_pub.publish(pt_base)
        except Exception:
            pass

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
    ) -> tuple:
        """Process detected objects and extract 3D information.
        use_lidar=True  — sparse LiDAR depth, iterate raw YOLO CustomMaskObjectData list.
        use_zed_mini=True — dense ZED Mini depth (heatmap path), iterate raw YOLO list.
        default — ZED stereo depth (heatmap path), iterate sl.Objects container.
        Returns (targets, rejected_targets, viz_only)."""
        targets = []
        rejected_targets = []
        viz_only = []  # kept for API compat but always empty now

        # use_zed_mini also iterates raw YOLO dets (ZED One has no object-detection API)
        obj_list = objects_or_dets if (use_lidar or use_zed_mini) else objects_or_dets.object_list
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
            if hasattr(o, "mask") and o.mask is not None and o.mask.is_init():
                mask_mat = o.mask
            elif hasattr(o, "box_mask") and o.box_mask is not None and o.box_mask.is_init():
                mask_mat = o.box_mask

            if mask_mat is None:
                mark_reject("No mask")
                continue

            if x2 <= x1 or y2 <= y1:
                mark_reject("Invalid box")
                continue

            try:
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
            except Exception as e:
                print(f"[WARN] 3D extraction failed: {e}")
                mark_reject("3D extraction failed")

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
        mask_local = mask_mat.get_data()
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
        t_short_axis, long_axis_2d, t_angle = self._fit_ellipse(mask_clean)

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
            idx_l = np.argsort(zs_l)
            k_l = max(5, int(0.2 * len(idx_l)))
            pts_front = in_mask_pts[idx_l[:k_l]]
            Xc = float(np.median(pts_front[:, 0]))
            Yc = float(np.median(pts_front[:, 1]))
            Zc = float(np.median(pts_front[:, 2]))
            depth_std = float(np.std(pts_front[:, 2]))
            vis_ratio = min(1.0, in_mask_pts.shape[0] / max(1, int(0.1 * mask_bool.sum())))
            vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)
            if not np.isfinite(Zc) or Zc <= 0.0 or Zc > LIDAR_Z_MAX:
                mark_reject("Z out of range")
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
                mini_pts_data = self._latest_mini_pts

            if mini_pts_data is None or mini_pts_data[0].shape[0] == 0:
                mark_reject("No depth data")
                return None

            pts_all, uv_all = mini_pts_data  # pts: ZED Mini frame, uv: ZED One pixels

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

            if pts.shape[0] < 10:
                mark_reject("Too few depth pts")
                return None

            # Depth statistics — pure measured points, no interpolation.
            # ZED Mini stereo produces mixed pixels at fruit/background edges,
            # so cluster around the median depth before computing std.
            zs  = pts[:, 2]
            idx = np.argsort(zs)
            k   = max(10, int(0.2 * len(idx)))
            z_med = float(np.median(zs))
            in_window = np.abs(zs - z_med) <= 0.08  # ±8 cm isolates fruit from edge noise
            pts_front = pts[in_window] if in_window.sum() >= 10 else pts[idx[:k]]

            depth_std = float(np.std(pts_front[:, 2]))
            if depth_std > 0.05:
                mark_reject("Depth variance")
                return None

            vis_ratio = min(1.0, float(pts.shape[0]) / max(1.0, float(np.count_nonzero(mask_bool))))
            vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)

            Xc = float(np.mean(pts_front[:, 0]))
            Yc = float(np.mean(pts_front[:, 1]))
            Zc = float(np.mean(pts_front[:, 2]))

            if not np.isfinite(Zc) or Zc <= 0.0 or Zc > 5.0:
                mark_reject("Z out of range")
                return None

            # Heatmap — use EDT depth map (visualization quality, not used for position)
            target_key = (x1, y1, x2, y2)
            if pc_np is not None:
                roi_xyz = pc_np[y1:y2, x1:x2, :]
                valid   = np.isfinite(roi_xyz[:, :, 2]) & mask_bool
                if self._heatmap_frame_count % self._heatmap_interval == 0:
                    heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = self._compute_heatmap(roi_xyz, valid, mask_clean)
                    self._cached_heatmaps[target_key] = (heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal)
                else:
                    cached = self._cached_heatmaps.get(target_key)
                    if cached is not None:
                        heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = cached
                    else:
                        heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = self._compute_heatmap(roi_xyz, valid, mask_clean)
                        self._cached_heatmaps[target_key] = (heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal)
            else:
                heatmap = t_best_point = t_best_dir2d = t_best_point_3d = scored_3d_pts = surface_normal = None

            # Depth diagnostic
            if getattr(self, '_depth_dbg_count', 0) % 30 == 0:
                print(f"[Depth/mini] cam XYZ=({Xc:.3f},{Yc:.3f},{Zc:.3f}) "
                      f"n_bbox={pts_bbox.shape[0]} n_mask={pts.shape[0]} n_front={len(pts_front)} "
                      f"z_std={depth_std:.4f} bbox=({x1},{y1},{x2},{y2})")
            self._depth_dbg_count = getattr(self, '_depth_dbg_count', 0) + 1

            gap_target = {"bb": (x1, y1, x2, y2), "Zc": Zc}
            self._detect_branch_gap(pc_np, gap_target)

            in_mask_uv = None
            obj_confidence = getattr(obj, "confidence", 50.0) / 100.0

        else:
            # ── ZED stereo depth path (original) ─────────────────────────
            if pc_np is None:
                mark_reject("No depth data")
                return None
            roi_xyz = pc_np[y1:y2, x1:x2, :]
            valid = np.isfinite(roi_xyz[:, :, 2]) & mask_bool

            # Throttle heatmap computation — reuse cached on non-compute frames
            target_key = (x1, y1, x2, y2)
            if self._heatmap_frame_count % self._heatmap_interval == 0:
                heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = self._compute_heatmap(
                    roi_xyz, valid, mask_clean
                )
                self._cached_heatmaps[target_key] = (heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal)
            else:
                cached = self._cached_heatmaps.get(target_key)
                if cached is not None:
                    heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = cached
                else:
                    heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal = self._compute_heatmap(
                        roi_xyz, valid, mask_clean
                    )
                    self._cached_heatmaps[target_key] = (heatmap, t_best_point, t_best_dir2d, t_best_point_3d, scored_3d_pts, surface_normal)

            # Visibility ratio
            vis_mask = cv2.erode(mask_clean, np.ones((3, 3), np.uint8), iterations=1) > 0
            mask_pixels = np.count_nonzero(vis_mask)
            vis_ratio = (
                float(np.count_nonzero(valid & vis_mask)) / float(mask_pixels)
                if mask_pixels > 0 else 0.0
            )
            vis_ratio = max(0.0, min(vis_ratio, 1.0))

            if np.count_nonzero(valid) < 30:
                mark_reject("Too few depth pts")
                return None

            pts = roi_xyz[valid]
            zs = pts[:, 2]
            idx = np.argsort(zs)
            k = max(10, int(0.2 * len(idx)))
            pts_front = pts[idx[:k]]

            depth_std = np.std(pts_front[:, 2])
            vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)

            Z_std = float(np.std(pts_front[:, 2]))
            if Z_std > 0.05:
                mark_reject("Depth variance")
                return None

            Xc = float(np.mean(pts_front[:, 0]))
            Yc = float(np.mean(pts_front[:, 1]))
            Zc = float(np.mean(pts_front[:, 2]))

            if not np.isfinite(Zc) or Zc <= 0.0 or Zc > 5.0:
                mark_reject("Z out of range")
                return None

            # Depth diagnostic — printed periodically to monitor accuracy
            if getattr(self, '_depth_dbg_count', 0) % 30 == 0:
                print(f"[Depth] cam XYZ=({Xc:.3f},{Yc:.3f},{Zc:.3f}) "
                      f"n_valid={np.count_nonzero(valid)} n_pts={len(pts)} n_front={len(pts_front)} "
                      f"z_std={depth_std:.4f} bbox=({x1},{y1},{x2},{y2})")
            self._depth_dbg_count = getattr(self, '_depth_dbg_count', 0) + 1

            # Branch gap detection (depth ring sampling around fruit)
            gap_target = {"bb": (x1, y1, x2, y2), "Zc": Zc}
            self._detect_branch_gap(pc_np, gap_target)

            obj_confidence = getattr(obj, "confidence", 50.0) / 100.0

        # Transform to base_link
        point_msg = PointStamped()
        point_msg.header.frame_id = CAM_FRAME
        point_msg.header.stamp = rclpyTime().to_msg()
        point_msg.point.x = Xc
        point_msg.point.y = Yc
        point_msg.point.z = Zc

        try:
            pt_base = self.tf_buffer.transform(point_msg, "base_link", timeout=rclpyDuration(seconds=0.2))
            pt_grip = self.tf_buffer.transform(point_msg, "gripper_tip", timeout=rclpyDuration(seconds=0.2))

            dist = math.sqrt(pt_grip.point.x ** 2 + pt_grip.point.y ** 2 + pt_grip.point.z ** 2)

            if not (0.0 <= obj_confidence <= 1.0):
                obj_confidence = 0.5

            fruit_id = hash((round(Xc, 2), round(Yc, 2), round(Zc, 2)))

            if self.tracker.is_fruit_blacklisted(fruit_id):
                mark_reject(f"Max attempts (3)")
                return None

            attempt_count = self.tracker.register_fruit(fruit_id, [Xc, Yc, Zc])

            return {
                "Xc": Xc, "Yc": Yc, "Zc": Zc,
                "bb": (x1, y1, x2, y2),
                "img_width": display_resolution.width,
                "img_height": display_resolution.height,
                "mask_resized": mask_resized,
                "pt_base": pt_base,
                "pt_grip": pt_grip,
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
                "heatmap": heatmap,
                "best_dir2d": t_best_dir2d,
                "best_point2d": t_best_point,
                "best_point_3d": t_best_point_3d,
                "scored_3d_points": scored_3d_pts,
                "surface_normal": surface_normal,
                "score": 0.0,
                "score_components": {},
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

    def _fit_ellipse(self, mask_clean: np.ndarray) -> tuple:
        """Fit ellipse to mask and extract short axis."""
        t_short_axis = None
        long_axis_2d = None
        t_angle = None

        try:
            contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) >= 20:
                    ellipse = cv2.fitEllipse(cnt)
                    (_, _), (_, _), angle_deg = ellipse

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

        return t_short_axis, long_axis_2d, t_angle

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

        if self.excluded_positions:
            n_excl = sum(1 for t in targets if t.get("excluded"))
            print(f"[EXCL] {n_excl}/{len(targets)} excluded, best_idx={best_idx}")

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

    def _process_best_target(
        self,
        targets: List[Dict[str, Any]],
        best_idx: int,
        intrinsics: Dict[str, float]
    ) -> None:
        """Process the best target and publish goal."""
        t_best = targets[best_idx]

        # Cache TF lookup
        cached_q_tf = None
        try:
            T = self.tf_buffer.lookup_transform("base_link", CAM_FRAME, rclpyTime())
            cached_q_tf = (
                T.transform.rotation.w,
                T.transform.rotation.x,
                T.transform.rotation.y,
                T.transform.rotation.z,
            )
        except Exception:
            pass

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

                collision_result = compute_collision_free_direction(t_best, targets, best_idx, heatmap_dir)
                dir_cam = collision_result["direction"]

                t_best["clearance"] = collision_result["clearance"]
                t_best["is_collision_free"] = collision_result["is_collision_free"]
                t_best["heatmap_dir_cam"] = heatmap_dir.copy()
                t_best["approach_dir_cam"] = dir_cam.copy()

                if cached_q_tf is not None:
                    dir_raw = np.array(quat_rotate_vec(cached_q_tf, dir_cam), dtype=float)
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

            # Publish estimated fruit radius for adaptive gripper
            from .scoring import estimate_fruit_radius
            radius = estimate_fruit_radius(t_best)
            radius_msg = Float32()
            radius_msg.data = float(radius)
            self.radius_pub.publish(radius_msg)

            # Publish branch gap info for 2-finger mode
            between_branches = t_best.get("between_branches", False)
            gap_angle_cam = t_best.get("gap_angle_cam", 0.0)
            gap_angle_base = gap_angle_cam
            if between_branches and cached_q_tf is not None:
                gap_dir_cam = np.array([
                    math.cos(gap_angle_cam),
                    math.sin(gap_angle_cam),
                    0.0
                ], dtype=float)
                gap_dir_base = np.array(quat_rotate_vec(cached_q_tf, gap_dir_cam), dtype=float)
                gap_angle_base = float(math.atan2(gap_dir_base[2], gap_dir_base[0]))

            gap_msg = Float32MultiArray()
            gap_msg.data = [1.0 if between_branches else 0.0, float(gap_angle_base)]
            self.gap_info_pub.publish(gap_msg)

            # Publish heatmap 3D data for goal marker (consumed on subscribe)
            scored_pts = t_best.get("scored_3d_points")
            if scored_pts is not None and cached_q_tf is not None:
                try:
                    T = self.tf_buffer.lookup_transform("base_link", CAM_FRAME, rclpyTime())
                    t_vec = np.array([
                        T.transform.translation.x,
                        T.transform.translation.y,
                        T.transform.translation.z,
                    ])
                except Exception:
                    t_vec = None

                if t_vec is not None:
                    # Transform all points to base_link, store as flat array
                    # Format: [x0,y0,z0,s0, x1,y1,z1,s1, ...]
                    flat = []
                    for row in scored_pts:
                        pt_cam = row[:3]
                        pt_base = np.array(quat_rotate_vec(cached_q_tf, pt_cam), dtype=float) + t_vec
                        flat.extend([float(pt_base[0]), float(pt_base[1]), float(pt_base[2]), float(row[3])])
                    hm_msg = Float32MultiArray()
                    hm_msg.data = flat
                    self.heatmap_data_pub.publish(hm_msg)
        else:
            self.best_history.clear()
