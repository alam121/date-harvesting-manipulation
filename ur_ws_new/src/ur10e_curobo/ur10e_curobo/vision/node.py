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
)
from .math_utils import unit_vector, quat_rotate_vec, quat_align_x_to_axis
from .ros_utils import wait_for_transform, create_pointcloud2_msg
from .zed_utils import apply_zed_camera_settings
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
        """Main execution loop."""
        # ROS2 setup
        rclpy.init()
        self.node = rclpy.create_node("zed_date_detector_ros")
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)

        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        goal_pub = self.node.create_publisher(PoseStamped, "/external_goal_pose", fast_qos)
        dir_pub = self.node.create_publisher(Vector3Stamped, "/datefruit_direction", 10)
        depth_pub = self.node.create_publisher(PointCloud2, "/zed_depth_pointcloud", fast_qos)
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

        # Camera refresh command
        def camera_cmd_cb(msg):
            cmd = msg.data.strip()
            if cmd == "refresh":
                self.node.get_logger().info("Camera refresh requested — reinitializing ZED...")
                self.exit_signal = True  # will restart the main loop

        from std_msgs.msg import String as StdString
        self.node.create_subscription(StdString, "/camera_command", camera_cmd_cb, 10)

        # TF listener
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self.node)

        # Wait for TFs
        required_tfs = [("base_link", CAM_FRAME), ("gripper_tip", CAM_FRAME)]
        for target, source in required_tfs:
            wait_for_transform(self.tf_buffer, target, source, self.node, timeout=5.0)

        # YOLO thread
        self.yolo_thread = YoloThread(
            weights=self.args.weights,
            img_size=self.args.img_size,
            conf_thres=self.args.conf_thres,
        )
        yolo_thread = Thread(target=self.yolo_thread.run, daemon=True)
        yolo_thread.start()

        # ZED init
        input_type = sl.InputType()
        if self.args.svo:
            input_type.set_from_svo_file(self.args.svo)

        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
        init_params.depth_minimum_distance = 0.15
        init_params.depth_maximum_distance = 50.0

        print("Initializing Camera...")
        zed = sl.Camera()
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            rclpy.shutdown()
            return
        print("Camera Initialized")
        apply_zed_camera_settings(zed)

        zed.enable_positional_tracking(sl.PositionalTrackingParameters())

        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        zed.enable_object_detection(obj_param)

        camera_infos = zed.get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution
        left_cam = camera_infos.camera_configuration.calibration_parameters.left_cam
        intrinsics = {
            "fx": float(left_cam.fx),
            "fy": float(left_cam.fy),
            "cx": float(left_cam.cx),
            "cy": float(left_cam.cy),
        }

        image_left = sl.Mat()
        runtime_params = sl.RuntimeParameters()
        obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
        objects = sl.Objects()
        point_cloud = sl.Mat()

        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        image_left_ocv = np.full(
            (display_resolution.height, display_resolution.width, 4),
            [245, 239, 239, 255],
            np.uint8,
        )
        image_scale = [
            display_resolution.width / camera_res.width,
            display_resolution.height / camera_res.height,
        ]
        display_scale = 0.6

        self.visualizer = VisionVisualizer(intrinsics, image_scale, display_scale)

        last_viz = 0.0
        loop_fps = 0.0

        def perception_loop():
            nonlocal last_viz, loop_fps
            t_prev = time()

            while not self.exit_signal:
                if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    self.exit_signal = True
                    break

                t_now = time()
                loop_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
                t_prev = t_now

                # Get image for YOLO
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

                zed.ingest_custom_mask_objects(current_dets)
                zed.retrieve_custom_objects(objects, obj_runtime_param)

                # Get image for display
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())

                # Retrieve XYZ map
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU, display_resolution)
                pc_np = point_cloud.get_data()[:, :, :3]

                # Publish depth point cloud
                depth_frame_count[0] += 1
                if depth_frame_count[0] % 5 == 0:
                    self._publish_depth_cloud(pc_np, depth_pub)

                # Process detected objects
                self._heatmap_frame_count += 1
                if self._heatmap_frame_count % (self._heatmap_interval * 10) == 0:
                    self._cached_heatmaps.clear()  # prevent stale cache buildup
                targets, rejected_targets, viz_only = self._process_objects(
                    objects, pc_np, image_left_ocv, image_scale, display_resolution, intrinsics
                )

                # Temporal stabilization
                targets = self.tracker.stabilize_detections(targets)

                # Publish trunk position for pole obstacle (uses YOLO trunk boxes directly)
                self._publish_trunk_position(trunk_boxes, pc_np, image_scale, image_left_ocv, trunk_pub)

                # Select best fruit
                best_idx = self._select_best_fruit(targets)

                # Compute approach direction and publish
                if best_idx is not None:
                    self._process_best_target(targets, best_idx, intrinsics)
                elif self.excluded_positions:
                    # All visible targets are excluded — stop publishing stale position
                    with self.pub_lock:
                        self.latest_goal_msg = None

                # Build viz entries for trunk bboxes (display only)
                trunk_viz = []
                for bx1, by1, bx2, by2 in trunk_boxes:
                    tx1 = max(0, min(int(bx1 * image_scale[0]), image_left_ocv.shape[1] - 1))
                    ty1 = max(0, min(int(by1 * image_scale[1]), image_left_ocv.shape[0] - 1))
                    tx2 = max(0, min(int(bx2 * image_scale[0]), image_left_ocv.shape[1]))
                    ty2 = max(0, min(int(by2 * image_scale[1]), image_left_ocv.shape[0]))
                    trunk_viz.append({"bb": (tx1, ty1, tx2, ty2), "class": "trunk", "conf": 1.0})

                # Visualization
                now = time()
                if (now - last_viz) >= 0.1:
                    display_image = self.visualizer.render_frame(
                        image_left_ocv, targets, rejected_targets,
                        best_idx, self.yolo_thread.net_fps, loop_fps,
                        viz_only=trunk_viz,
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
            rclpy.spin(self.node)
        finally:
            self.exit_signal = True
            self.yolo_thread.stop()
            perception_thread.join(timeout=1.0)
            zed.close()
            rclpy.shutdown()

    def _publish_depth_cloud(self, pc_np: np.ndarray, depth_pub) -> None:
        """Publish depth point cloud for voxel obstacle avoidance."""
        try:
            valid = np.isfinite(pc_np).all(axis=-1)
            valid &= (pc_np[:, :, 2] > 0.1) & (pc_np[:, :, 2] < 2.0)
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

    def _publish_trunk_position(self, trunk_boxes, pc_np, image_scale, image_left_ocv, trunk_pub) -> None:
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
        objects,
        pc_np: np.ndarray,
        image_left_ocv: np.ndarray,
        image_scale: List[float],
        display_resolution,
        intrinsics: Dict[str, float],
    ) -> tuple:
        """Process detected objects and extract 3D information.
        All objects from ZED are fruit (trunk is filtered at YOLO level).
        Returns (targets, rejected_targets, viz_only)."""
        targets = []
        rejected_targets = []
        viz_only = []  # kept for API compat but always empty now

        for o in objects.object_list:
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
                    display_resolution, intrinsics, mark_reject
                )
                if target is not None:
                    targets.append(target)
            except Exception as e:
                print(f"[WARN] 3D extraction failed: {e}")
                mark_reject("3D extraction failed")

        return targets, rejected_targets, viz_only

    def _extract_target_3d(
        self,
        obj, mask_mat, pc_np: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        display_resolution, intrinsics: Dict[str, float],
        mark_reject
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

        # 3D point extraction
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

        # Branch gap detection (depth ring sampling around fruit)
        gap_target = {"bb": (x1, y1, x2, y2), "Zc": Zc}
        self._detect_branch_gap(pc_np, gap_target)

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

            obj_confidence = getattr(obj, "confidence", 0.5) / 100.0
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

                    # Collect scored 3D points for RViz marker (downsample to ~100)
                    valid_in_mask = np.zeros_like(valid)
                    valid_in_mask[ys, xs] = valid[ys, xs]
                    mask_ys, mask_xs = np.nonzero(valid_in_mask)
                    if len(mask_ys) > 0:
                        pts_3d = roi_xyz[mask_ys, mask_xs, :]
                        pts_scores = score[mask_ys, mask_xs]
                        finite_mask = np.all(np.isfinite(pts_3d), axis=1)
                        pts_3d = pts_3d[finite_mask]
                        pts_scores = pts_scores[finite_mask]
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
