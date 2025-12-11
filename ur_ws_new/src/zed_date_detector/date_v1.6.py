#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep, time
from typing import List
from collections import deque
import ogl_viewer.viewer as gl
import tf2_geometry_msgs
import cv_viewer.tracking_viewer as cv_viewer

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

import math
from scipy.spatial.transform import Rotation as R

# ============================================================
# Globals
# ============================================================
lock = Lock()
run_signal = False
exit_signal = False
ANGLE_REF = np.array([0.0, 0.0, 1.0])  # <-- ADD HERE
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None

net_fps = 0.0
loop_fps = 0.0
prev_axis = None
prev_heat_point = None
last_dir_label = None
last_dir_count = 0
last_dir_published = None
yolo_masks = []
yolo_classes = []
yolo_scores = []
yolo_lock = Lock()

# Rendering knobs to save CPU (publishing unaffected)
DRAW_ONLY_BEST = False
SHOW_REJECTED = False
SKIP_DRAW = False

# --- Persistent BEST fruit tracking ---
best_target_prev = None          # store previous best fruit
BEST_REUSE_THRESH = 0.05         # 5 cm positional tolerance in base_link

# --- Track-by-detection smoothing ---
best_history = deque(maxlen=3)   # shorter window for snappier response


# ============================================================
# Utility Functions
# ============================================================
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_mul(q, r):
    w, x, y, z = q
    W, X, Y, Z = r
    return (
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    )

def quaternion_to_degrees(q):
    """Return rotation angle (in degrees) represented by quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    # angle = 2 * acos(w)
    angle_rad = 2.0 * math.acos(max(min(w, 1.0), -1.0))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def quat_rotate_vec(q, v3):
    """Rotate vector v3 by quaternion q."""
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (q[0], -q[1], -q[2], -q[3])
    return quat_mul(quat_mul(q, qv), qi)[1:]

def signed_angle_between(v1, v2, up=np.array([0,0,1])):
    """
    Returns signed angle in degrees between v1 and v2.
    Uses `up` vector to determine sign (positive/negative).
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = math.degrees(math.acos(dot))

    # compute sign using cross product direction
    cross = np.cross(v1, v2)
    sign = np.sign(np.dot(cross, up))  # positive if rotation follows `up`

    return angle * sign

def quat_align_axis_to_axis(src_axis, dst_axis, up_hint=(0,0,1)):
    """
    Build quaternion that rotates src_axis → dst_axis.
    """
    src = _unit(np.array(src_axis, float))
    dst = _unit(np.array(dst_axis, float))

    v = np.cross(src, dst)
    c = np.dot(src, dst)

    if np.linalg.norm(v) < 1e-8:
        # axes are parallel or antiparallel
        if c > 0:
            return (1,0,0,0)  # no rotation
        else:
            # 180° rotation around any perpendicular axis
            perp = np.cross(src, up_hint)
            perp = _unit(perp)
            return (0, perp[0], perp[1], perp[2])

    s = math.sqrt((1 + c) * 2)
    invs = 1.0 / s

    qx = v[0] * invs
    qy = v[1] * invs
    qz = v[2] * invs
    qw = s * 0.5
    return (qw, qx, qy, qz)


def normalize_grasp_angle(angle_deg):
    """Map angle to symmetric range where 180° equals 0°."""
    # Wrap angle to [-180, 180]
    angle = (angle_deg + 180) % 360 - 180
    # Flip 180° symmetry: treat angle and angle+180 as same
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


def optimize_grasp_orientation_for_dense_bunch(
    preferred_axis_cam, target_fruit, all_fruits, z_std_threshold=0.03
):
    """
    Determine optimal grasp orientation axis to avoid neighbors.

    The ellipse short axis provides the BASE target orientation.
    For dense bunches, we test ±45° adjustments around this base to avoid neighbors.

    Returns the optimal AXIS DIRECTION (not angle) in camera frame.
    The main loop will transform this to base frame and compute rotation from current orientation.

    Args:
        preferred_axis_cam: Ellipse short axis [x, y, z] in camera frame (target orientation)
        target_fruit: Dict with target fruit info (must have "Xc", "Yc", "Zc")
        all_fruits: List of all detected fruits
        z_std_threshold: Threshold for considering bunch as dense (default: 0.03m)

    Returns:
        Optimized axis direction [x, y, z] in camera frame
        For sparse: ellipse axis (no adjustment)
        For dense: ellipse axis rotated by ±45° adjustment to avoid neighbors
    """
    z_std = target_fruit.get("z_std", 0.0)

    # If no ellipse axis available, can't compute orientation
    if preferred_axis_cam is None:
        print(f"[ORIENT] No ellipse axis available")
        return None

    # Get target position in camera frame
    target_pos = np.array([
        target_fruit["Xc"],
        target_fruit["Yc"],
        target_fruit["Zc"]
    ], dtype=float)

    # Get gripper approach direction (from camera to fruit, in camera frame)
    approach_dir = target_pos / max(np.linalg.norm(target_pos), 1e-6)

    # Ellipse short axis is the BASE target orientation for gripper width
    # Normalize it
    ellipse_axis = np.array(preferred_axis_cam, dtype=float)
    ellipse_axis = ellipse_axis / max(np.linalg.norm(ellipse_axis), 1e-6)

    # Project ellipse axis onto plane perpendicular to approach direction
    # (gripper width must be perpendicular to approach)
    ellipse_axis_proj = ellipse_axis - np.dot(ellipse_axis, approach_dir) * approach_dir
    ellipse_axis_proj = ellipse_axis_proj / max(np.linalg.norm(ellipse_axis_proj), 1e-6)

    # For SPARSE bunches: use ellipse orientation as-is (no adjustment)
    if z_std < z_std_threshold:
        print(f"[ORIENT] Sparse bunch (z_std={z_std:.4f}), using ellipse axis")
        return ellipse_axis_proj  # Return the axis direction in camera frame

    # For DENSE bunches: test ±45° adjustments around ellipse orientation
    best_angle = 0  # relative to ellipse axis
    min_collision_score = float('inf')

    for angle_offset_deg in [-45, -30, -15, 0, 15, 30, 45]:
        angle_rad = math.radians(angle_offset_deg)

        # Rotate ellipse axis by offset around approach axis (Rodrigues' formula)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        gripper_width_dir = (
            ellipse_axis_proj * cos_a +
            np.cross(approach_dir, ellipse_axis_proj) * sin_a +
            approach_dir * np.dot(approach_dir, ellipse_axis_proj) * (1 - cos_a)
        )

        # Count neighbors in gripper width direction (±30° cone on both sides)
        collision_score = 0.0
        for other in all_fruits:
            if other == target_fruit:
                continue

            try:
                other_pos = np.array([other["Xc"], other["Yc"], other["Zc"]], dtype=float)
                vec = other_pos - target_pos
                dist = np.linalg.norm(vec)

                if dist < 0.01:  # skip if too close (likely same fruit)
                    continue

                if dist > 0.15:  # ignore far fruits (>15cm)
                    continue

                vec_norm = vec / dist

                # Check if neighbor is in gripper width direction (±30° cone)
                dot_width = abs(float(np.dot(vec_norm, gripper_width_dir)))
                if dot_width > math.cos(math.radians(30)):  # within ±30° of width axis
                    # Closer neighbors are worse (weighted by inverse distance)
                    collision_score += 1.0 / max(dist, 0.02)

            except (KeyError, TypeError, ValueError):
                # Skip fruits with missing/invalid position data
                continue

        # Track best orientation (minimum collisions)
        if collision_score < min_collision_score:
            min_collision_score = collision_score
            best_angle = angle_offset_deg

    # Compute the optimized axis by rotating ellipse axis by best_angle
    angle_rad = math.radians(best_angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    optimized_axis = (
        ellipse_axis_proj * cos_a +
        np.cross(approach_dir, ellipse_axis_proj) * sin_a +
        approach_dir * np.dot(approach_dir, ellipse_axis_proj) * (1 - cos_a)
    )
    optimized_axis = optimized_axis / max(np.linalg.norm(optimized_axis), 1e-6)

    if best_angle == 0:
        print(f"[ORIENT] Dense bunch (z_std={z_std:.4f}), using ellipse axis (no adjustment, score={min_collision_score:.2f})")
    else:
        print(f"[ORIENT] Dense bunch (z_std={z_std:.4f}), ellipse axis + {best_angle:+d}° adjustment (score={min_collision_score:.2f})")

    return optimized_axis  # Return the rotated axis direction in camera frame


def wait_for_transform(tf_buffer, target_frame, source_frame, node, timeout=5.0):
    """Spin until the requested TF is available or timeout."""
    start = time()
    while rclpy.ok() and (time() - start) < timeout:
        if tf_buffer.can_transform(target_frame, source_frame, rclpyTime()):
            return True
        rclpy.spin_once(node, timeout_sec=0.05)
        sleep(0.05)
    print(f"[WARN] TF {source_frame}->{target_frame} unavailable after {timeout:.1f}s")
    return False


def xywh2abcd_(xywh: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 2), dtype=np.float32)
    x_min = xywh[0] - 0.5 * xywh[2]
    x_max = xywh[0] + 0.5 * xywh[2]
    y_min = xywh[1] - 0.5 * xywh[3]
    y_max = xywh[1] + 0.5 * xywh[3]
    out[0] = [x_min, y_min]
    out[1] = [x_max, y_min]
    out[2] = [x_max, y_max]
    out[3] = [x_min, y_max]
    return out


def project_point_to_image(pt_cam, intr, scale):
    """Project a 3D point in the camera frame to display pixel coords."""
    X, Y, Z = pt_cam
    if Z <= 0 or not np.isfinite(Z):
        return None
    u = intr["fx"] * X / Z + intr["cx"]
    v = intr["fy"] * Y / Z + intr["cy"]
    return (int(u * scale[0]), int(v * scale[1]))


def detections_to_custom_masks_(dets) -> List[sl.CustomMaskObjectData]:
    global sl_mats
    output = []
    sl_mats = []
    H, W = dets.orig_shape
    for di in range(len(dets.boxes)):
        obj = sl.CustomMaskObjectData()
        xywh = dets.boxes.xywh[di].cpu().numpy().astype(np.float32)
        abcd = xywh2abcd_(xywh)
        abcd[:, 0] = np.clip(abcd[:, 0], 0, W - 1)
        abcd[:, 1] = np.clip(abcd[:, 1], 0, H - 1)
        obj.bounding_box_2d = abcd
        obj.label = int(dets.boxes.cls[di].item())
        obj.probability = float(dets.boxes.conf[di].item())
        obj.is_grounded = False

        if dets.masks is not None and dets.masks.data is not None:
            m = dets.masks.data[di].cpu().numpy()
            mask_bin = (m * 255).astype(np.uint8)
            x_min = int(abcd[0, 0])
            y_min = int(abcd[0, 1])
            x_max = int(abcd[2, 0])
            y_max = int(abcd[2, 1])
            mask_roi = mask_bin[y_min: y_max + 1, x_min: x_max + 1]
            if not mask_roi.flags.c_contiguous:
                mask_roi = np.ascontiguousarray(mask_roi)
            sl_mat = sl.Mat(
                width=mask_roi.shape[1],
                height=mask_roi.shape[0],
                mat_type=sl.MAT_TYPE.U8_C1,
                memory_type=sl.MEM.CPU,
            )
            np.copyto(sl_mat.get_data(), mask_roi)
            sl_mats.append(sl_mat)
            obj.box_mask = sl_mat

        output.append(obj)
    return output


# ============================================================
# YOLO Thread
# ============================================================
def torch_thread_(weights: str, img_size: int, conf_thres: float = 0.2) -> None:
    global image_net, exit_signal, run_signal, detections, net_fps
    print("Initializing Network...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YOLO; GPU not available.")
    device = torch.device("cuda")
    model = YOLO(weights)
    model.to(device).eval()
    print("Network Initialized...")
    while not exit_signal:
        if run_signal:
            with lock:
                img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            t0 = time()
            det = model.predict(
                img,
                save=False,
                retina_masks=True,
                imgsz=img_size,
                conf=conf_thres,
                device=device,
                verbose=False,
            )[0]
            dt = time() - t0
            net_fps = (1.0 / dt) if dt > 0 else 0.0
            with lock:
                detections = detections_to_custom_masks_(det)
            run_signal = False
        sleep(0.005)


# ============================================================
# Main
# ============================================================
def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections, loop_fps, best_target_prev, prev_axis, prev_heat_point
    global last_dir_label, last_dir_count, last_dir_published

    # --- ROS2 setup ---
    rclpy.init()
    node = rclpy.create_node("zed_date_detector_ros")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    fast_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )
    point_pub = node.create_publisher(PointStamped, "/datefruit_3d_point", 10)
    goal_pub = node.create_publisher(PoseStamped, "/external_goal_pose", fast_qos)
    dir_pub = node.create_publisher(String, "/datefruit_direction", 10)

    # TF listener
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    cam_frame = "zed2_left_camera_frame"

    # Wait for camera frame TF to be available before proceeding
    required_tfs = [
        ("base_link", cam_frame),
        ("gripper_tip", cam_frame),
    ]
    for target, source in required_tfs:
        wait_for_transform(tf_buffer, target, source, node, timeout=5.0)

    # --- YOLO thread ---
    capture_thread = Thread(
        target=torch_thread_,
        kwargs={
            "weights": args.weights,
            "img_size": args.img_size,
            "conf_thres": args.conf_thres,
        },
        daemon=True,
    )
    capture_thread.start()

    # --- ZED init ---
    input_type = sl.InputType()
    if args.svo:
        input_type.set_from_svo_file(args.svo)

    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_maximum_distance = 50.0

    print("Initializing Camera...")
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        rclpy.shutdown()
        return
    print("Camera Initialized")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

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

    point_cloud_res = sl.Resolution(
        min(camera_res.width, 720), min(camera_res.height, 404)
    )

    image_left = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    cam_w_pose = sl.Pose()
    objects = sl.Objects()
    point_cloud = sl.Mat()  # XYZ map from ZED

    display_resolution = sl.Resolution(
        min(camera_res.width, 1280), min(camera_res.height, 720)
    )
    image_left_ocv = np.full(
        (display_resolution.height, display_resolution.width, 4),
        [245, 239, 239, 255],
        np.uint8,
    )
    image_scale = [
        display_resolution.width / camera_res.width,
        display_resolution.height / camera_res.height,
    ]
    display_scale = 0.6  # shrink window display without affecting computations

    t_prev = time()
    Z_MAX = 1.34  # limit in base_link frame

    # --- Main Loop ---
    try:
        while not exit_signal:
            # Let TF listener process incoming /tf
            rclpy.spin_once(node, timeout_sec=0.0)

            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                exit_signal = True
                break

            t_now = time()
            loop_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
            t_prev = t_now

            # Get image for YOLO
            with lock:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                image_net = image_left.get_data()
                run_signal = True

            # Wait for YOLO thread
            while run_signal and not exit_signal:
                sleep(0.001)

            # Ingest YOLO detections into ZED
            with lock:
                current_dets = detections
            if current_dets is not None:
                zed.ingest_custom_mask_objects(current_dets)

            zed.retrieve_custom_objects(objects, obj_runtime_param)

            # Get image for display
            zed.retrieve_image(
                image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution
            )
            np.copyto(image_left_ocv, image_left.get_data())

            # Retrieve XYZ map (3D point cloud) in camera frame
            zed.retrieve_measure(
                point_cloud, sl.MEASURE.XYZ, sl.MEM.CPU, display_resolution
            )
            pc_np = point_cloud.get_data()[:, :, :3]  # H x W x 3

            # --------------------------------------------------
            # Process detected objects → compute 3D + store
            # --------------------------------------------------
            # each target:
            # { "Xc","Yc","Zc","bb","mask_resized","pt_base","pt_grip",
            #   "dist","z_std","vis_ratio","vis_quality",
            #   "long_axis_cam","long_axis_2d","ellipse_angle", ... }


            targets = []
            rejected_targets = []

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
                elif (
                    hasattr(o, "box_mask")
                    and o.box_mask is not None
                    and o.box_mask.is_init()
                ):
                    mask_mat = o.box_mask

                if mask_mat is None:
                    mark_reject("No mask")
                    continue

                if x2 <= x1 or y2 <= y1:
                    mark_reject("Invalid box")
                    continue

                try:
                    mask_local = mask_mat.get_data()
                    if mask_local.ndim == 3:
                        mask_local = mask_local[:, :, 0]

                    w_roi = x2 - x1
                    h_roi = y2 - y1

                    # Resize mask to display ROI size
                    mask_resized = cv2.resize(
                        mask_local,
                        (w_roi, h_roi),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    # Morphological cleanup: close small holes then open to remove specks
                    kernel = np.ones((5, 5), np.uint8)
                    mask_clean = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
                    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)
                    mask_bool = mask_clean > 0

                    # ----------------------------------------------------------
                    # 2D Ellipse Fit → SHORT AXIS extraction
                    # ----------------------------------------------------------
                    t_short_axis = None
                    long_axis_2d = None
                    t_angle = None
                    t_best_dir2d = None
                    t_best_point = None
                    try:
                        contours, _ = cv2.findContours(
                            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        if contours:
                            cnt = max(contours, key=cv2.contourArea)
                            if len(cnt) >= 20:
                                ellipse = cv2.fitEllipse(cnt)
                                (xc2d, yc2d), (major, minor), angle_deg = ellipse

                                theta = math.radians(angle_deg)
                                # LONG axis in image plane (for reference only)
                                long_dir_img = np.array(
                                    [math.cos(theta), math.sin(theta)], dtype=float
                                )
                                # SHORT axis = perpendicular to long
                                short_dir_cam = np.array(
                                    [-long_dir_img[1], long_dir_img[0], 0.0],
                                    dtype=float,
                                )
                                n_short = np.linalg.norm(short_dir_cam)
                                if n_short > 1e-6:
                                    short_dir_cam /= n_short

                                t_short_axis = short_dir_cam
                                long_axis_2d = long_dir_img
                                t_angle = float(angle_deg)
                    except Exception as e:
                        print("[WARN] Ellipse axis extraction failed:", e)
                        t_short_axis = None
                        long_axis_2d = None
                        t_angle = None

                    # Take 3D points from XYZ map only where mask is true
                    roi_xyz = pc_np[y1:y2, x1:x2, :]  # H x W x 3
                    valid = np.isfinite(roi_xyz[:, :, 2])
                    valid &= mask_bool
                    heatmap = None
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

                        
                        # ----------------------------------------------------------
                        # Compute best point/direction from peak heatmap score
                        # ----------------------------------------------------------
                        if heatmap is not None and mask_clean is not None:
                            ys, xs = np.nonzero(mask_clean)
                            if len(xs) > 0:
                                scores = heatmap[ys, xs, 2].astype(float)  # use RED channel
                                idx_max = int(np.argmax(scores))
                                peak_y = int(ys[idx_max])
                                peak_x = int(xs[idx_max])
                                t_best_point = np.array([peak_x, peak_y], dtype=float)
                                cx = mask_clean.shape[1] / 2.0
                                cy = mask_clean.shape[0] / 2.0
                                dir_vec = np.array([peak_x - cx, peak_y - cy], dtype=float)
                                n_dir = np.linalg.norm(dir_vec)
                                if n_dir > 1e-6:
                                    dir_vec /= n_dir
                                else:
                                    dir_vec = np.array([1.0, 0.0], dtype=float)
                                t_best_dir2d = dir_vec

                    # More forgiving visibility: erode mask for ratio so edge holes hurt less
                    vis_mask = cv2.erode(
                        mask_clean,
                        np.ones((3, 3), np.uint8),
                        iterations=1,
                    ) > 0
                    mask_pixels = np.count_nonzero(vis_mask)
                    vis_ratio = (
                        float(np.count_nonzero(valid & vis_mask)) / float(mask_pixels)
                        if mask_pixels > 0
                        else 0.0
                    )
                    vis_ratio = max(0.0, min(vis_ratio, 1.0))

                    if np.count_nonzero(valid) < 30:
                        # Not enough 3D points to trust
                        mark_reject("Too few depth pts")
                        continue

                    pts = roi_xyz[valid]  # N x 3
                    zs = pts[:, 2]

                    # Closest 20% points (= front surface)
                    idx = np.argsort(zs)
                    k = max(10, int(0.2 * len(idx)))
                    pts_front = pts[idx[:k]]

                    depth_std = np.std(pts_front[:, 2])
                    vis_quality = (vis_ratio ** 2) * np.exp(
                        - (depth_std / 0.015) ** 2
                    )

                    Z_std = float(np.std(pts_front[:, 2]))
                    if Z_std > 0.05:  # >5 cm variance → unreliable
                        mark_reject("Depth variance")
                        continue

                    Xc = float(np.mean(pts_front[:, 0]))
                    Yc = float(np.mean(pts_front[:, 1]))
                    Zc = float(np.mean(pts_front[:, 2]))

                    if not np.isfinite(Zc) or Zc <= 0.0 or Zc > 5.0:
                        mark_reject("Z out of range")
                        continue

                    # Transform this fruit’s 3D point to base_link and gripper_tip
                    point_msg = PointStamped()
                    point_msg.header.frame_id = cam_frame
                    point_msg.header.stamp = rclpyTime().to_msg()
                    point_msg.point.x = Xc
                    point_msg.point.y = Yc
                    point_msg.point.z = Zc

                    try:
                        pt_base = tf_buffer.transform(
                            point_msg,
                            "base_link",
                            timeout=rclpyDuration(seconds=0.2),
                        )

                        pt_grip = tf_buffer.transform(
                            point_msg,
                            "gripper_tip",
                            timeout=rclpyDuration(seconds=0.2),
                        )

                        # 3D distance from gripper tip
                        dist = math.sqrt(
                            pt_grip.point.x ** 2
                            + pt_grip.point.y ** 2
                            + pt_grip.point.z ** 2
                        )

                        targets.append(
                            {
                                "Xc": Xc,
                                "Yc": Yc,
                                "Zc": Zc,
                                "bb": (x1, y1, x2, y2),
                                "mask_resized": mask_resized,
                                "pt_base": pt_base,
                                "pt_grip": pt_grip,
                                "dist": dist,
                                "quat": None,
                                "approach_axis": None,
                                "pca_stable": None,  # kept for compatibility in drawing
                                "reason": "",
                                "pts_front": pts_front,
                                "z_std": depth_std,
                                "vis_ratio": vis_ratio,
                                "vis_quality": vis_quality,
                                "short_axis_cam": t_short_axis,
                                "long_axis_2d": long_axis_2d,
                                "ellipse_angle": t_angle,
                                "heatmap": heatmap,
                                "best_dir2d": t_best_dir2d,
                                "best_point2d": t_best_point,
                            }
                        )

                    except Exception as e:
                        print(f"[WARN] TF transform failed: {e}")
                        mark_reject("TF transform failed")
                        continue

                except Exception as e:
                    print(f"[WARN] 3D extraction failed: {e}")
                    mark_reject("3D extraction failed")
                    continue

            # --------------------------------------------------
            # Hard-Sticky BEST Fruit Selection (with Occlusion Awareness)
            # --------------------------------------------------

            # 1. Pick best accessible fruit THIS FRAME (not just closest!)
            # Scoring factors:
            #   - Distance to gripper (lower = better)
            #   - Occlusion (depth variance, higher = worse)
            #   - Edge proximity (near image border = worse)
            #   - Visibility quality (higher = better)
            frame_best_idx = None
            frame_best_score = float("inf")

            for i, t in enumerate(targets):
                # Base distance to gripper
                dist = t["dist"]

                # Occlusion penalty: depth variance indicates overlapping fruits
                # Higher z_std = more buried/occluded = harder to grasp
                z_std = t.get("z_std", 0.0)
                occlusion_penalty = z_std * 15.0  # scale factor tuned for meters

                # Edge proximity penalty: fruits near image edges are often cut off
                bb = t["bb"]
                x1, y1, x2, y2 = bb
                edge_dist_left = x1
                edge_dist_top = y1
                edge_dist_right = image_left_ocv.shape[1] - x2
                edge_dist_bottom = image_left_ocv.shape[0] - y2
                edge_dist = min(edge_dist_left, edge_dist_top, edge_dist_right, edge_dist_bottom)
                # Penalty if within 50 pixels of edge
                edge_penalty = max(0, 50 - edge_dist) * 0.02

                # Visibility quality bonus: rewards clean, well-visible fruits
                vis_quality = t.get("vis_quality", 0.0)
                vis_bonus = -vis_quality * 0.3  # negative because lower score = better

                # Combined score (lower = better fruit to pick)
                score = dist + occlusion_penalty + edge_penalty + vis_bonus

                # Debug: store score for visualization
                t["accessibility_score"] = score

                if score < frame_best_score:
                    frame_best_score = score
                    frame_best_idx = i

            best_idx = frame_best_idx  # default

            # 2. Strong Reuse (sticky lock-on)
            # We ONLY switch if previous best is completely lost.
            if best_target_prev is not None:
                prev_pt = np.array(
                    [
                        best_target_prev["pt_base"].point.x,
                        best_target_prev["pt_base"].point.y,
                        best_target_prev["pt_base"].point.z,
                    ],
                    dtype=float,
                )

                reuse_idx = None
                min_dist = float("inf")

                # Try to find the *same* fruit again by proximity
                for i, t in enumerate(targets):
                    cur_pt = np.array(
                        [
                            t["pt_base"].point.x,
                            t["pt_base"].point.y,
                            t["pt_base"].point.z,
                        ],
                        dtype=float,
                    )
                    d = np.linalg.norm(cur_pt - prev_pt)

                    if d < BEST_REUSE_THRESH:  # e.g. 5 cm
                        if d < min_dist:
                            min_dist = d
                            reuse_idx = i

                # If we found a matching fruit → ALWAYS reuse it
                if reuse_idx is not None:
                    best_idx = reuse_idx
                # else: previous best is considered LOST → we must switch to frame_best_idx

            # 3. Persistent state update
            if best_idx is not None:
                best_target_prev = targets[best_idx]

            # --------------------------------------------------
            # Orientation Optimization: Compute ROTATION ANGLE (not absolute orientation)
            # --------------------------------------------------
            if best_idx is not None:
                t_best = targets[best_idx]
                short_cam = t_best.get("short_axis_cam")

                # Optimize orientation to avoid neighbor fruits in dense bunches
                # This returns the optimized AXIS DIRECTION in camera frame
                optimized_axis_cam = optimize_grasp_orientation_for_dense_bunch(
                    short_cam, t_best, targets, z_std_threshold=0.03
                )

                # Store optimized axis in target (will be used later to compute rotation)
                t_best["optimized_axis_cam"] = optimized_axis_cam

                # Smooth best heatmap point to reduce jitter for arrows
                best_pt = t_best.get("best_point2d")
                if best_pt is not None:
                    if prev_heat_point is None:
                        sm_pt = best_pt.copy()
                    else:
                        sm_pt = 0.7 * prev_heat_point + 0.3 * best_pt
                    prev_heat_point = sm_pt.copy()
                    t_best["best_point2d_smooth"] = sm_pt
                else:
                    prev_heat_point = None

                # Compute relative rotation from current gripper orientation to target orientation
                # rotation_angle_deg = how much to rotate FROM current TO target (ellipse-based)
                try:
                    optimized_axis_cam = t_best.get("optimized_axis_cam")

                    if optimized_axis_cam is not None:
                        # Get current gripper orientation from TF (base_link -> gripper_tip)
                        transform = tf_buffer.lookup_transform(
                            "base_link", "gripper_tip", rclpyTime()
                        )
                        current_quat = transform.transform.rotation
                        current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
                        current_matrix = current_rot.as_matrix()

                        # Transform optimized axis from camera frame to base frame
                        # Create a pose at camera origin + axis direction
                        axis_point_cam = PointStamped()
                        axis_point_cam.header.frame_id = cam_frame
                        axis_point_cam.header.stamp = rclpyTime().to_msg()
                        axis_point_cam.point.x = optimized_axis_cam[0]
                        axis_point_cam.point.y = optimized_axis_cam[1]
                        axis_point_cam.point.z = optimized_axis_cam[2]

                        # Also transform camera origin
                        origin_cam = PointStamped()
                        origin_cam.header.frame_id = cam_frame
                        origin_cam.header.stamp = rclpyTime().to_msg()
                        origin_cam.point.x = 0.0
                        origin_cam.point.y = 0.0
                        origin_cam.point.z = 0.0

                        axis_point_base = tf_buffer.transform(axis_point_cam, "base_link")
                        origin_base = tf_buffer.transform(origin_cam, "base_link")

                        # Axis direction in base frame = (axis_point - origin)
                        target_axis_base = np.array([
                            axis_point_base.point.x - origin_base.point.x,
                            axis_point_base.point.y - origin_base.point.y,
                            axis_point_base.point.z - origin_base.point.z
                        ])
                        target_axis_base /= max(np.linalg.norm(target_axis_base), 1e-6)

                        # Current gripper Y-axis (width direction) in base frame
                        gripper_y_current = current_matrix[:, 1]  # Y column

                        # Current gripper Z-axis (approach direction) in base frame
                        gripper_z_current = current_matrix[:, 2]  # Z column

                        # Project target axis onto plane perpendicular to approach
                        target_proj = target_axis_base - np.dot(target_axis_base, gripper_z_current) * gripper_z_current
                        target_proj /= max(np.linalg.norm(target_proj), 1e-6)

                        # Project current gripper Y onto same plane
                        gripper_y_proj = gripper_y_current - np.dot(gripper_y_current, gripper_z_current) * gripper_z_current
                        gripper_y_proj /= max(np.linalg.norm(gripper_y_proj), 1e-6)

                        # Compute angle between projections
                        dot = np.clip(np.dot(gripper_y_proj, target_proj), -1.0, 1.0)
                        angle_rad = math.acos(dot)

                        # Determine sign using cross product
                        cross = np.cross(gripper_y_proj, target_proj)
                        if np.dot(cross, gripper_z_current) < 0:
                            angle_rad = -angle_rad

                        relative_angle_deg = normalize_grasp_angle(math.degrees(angle_rad))
                        t_best["rotation_angle_deg"] = relative_angle_deg
                        print(f"[ORIENT] Relative rotation: {relative_angle_deg:+.1f}° (current→target)")
                    else:
                        # No optimized axis available, keep current orientation
                        t_best["rotation_angle_deg"] = 0.0
                        print(f"[ORIENT] No target axis, rotation=0° (keep current)")

                except Exception as e:
                    print(f"[ORIENT ERROR] Failed to compute relative rotation: {e}")
                    import traceback
                    traceback.print_exc()
                    t_best["rotation_angle_deg"] = 0.0

                # Compute approach axis for VISUALIZATION ONLY (direction from camera to fruit)
                fruit_pos = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]], dtype=float)
                approach_distance = max(np.linalg.norm(fruit_pos), 1e-6)
                approach_axis_vis = fruit_pos / approach_distance  # Normalized direction to fruit
                t_best["approach_axis"] = approach_axis_vis

            # --------------------------------------------------
            # Publish goal for BEST fruit only (before rendering to minimize latency)
            if best_idx is not None:
                t_best = targets[best_idx]
                pt_base = t_best["pt_base"]
                q = t_best["quat"]
                dir_msg = None
                dir_vec = t_best.get("best_dir2d")
                if dir_vec is not None and len(dir_vec) >= 2:
                    vx = float(dir_vec[0])
                    vy = float(dir_vec[1])
                    n = math.hypot(vx, vy)
                    if n > 1e-6:
                        vx /= n
                        vy /= n
                    if abs(vx) < 0.35:  # widened center band to reduce flicker
                        direction_label = "center"
                    elif vx > 0:
                        direction_label = "right"
                    else:
                        direction_label = "left"
                    # Hysteresis: require persistence over frames to change label
                    if direction_label == last_dir_label:
                        last_dir_count += 1
                    else:
                        last_dir_label = direction_label
                        last_dir_count = 1
                    effective_label = last_dir_published or direction_label
                    if last_dir_count >= 3:
                        effective_label = direction_label
                        last_dir_published = direction_label
                    dir_msg = String()
                    dir_msg.data = effective_label

                # Track-by-detection smoothing: weighted avg of recent centroids
                pt_vec = np.array(
                    [pt_base.point.x, pt_base.point.y, pt_base.point.z], dtype=float
                )
                best_history.append(pt_vec)
                if len(best_history) >= 2:
                    weights = np.arange(
                        1, len(best_history) + 1, dtype=float
                    )  # newer frames weigh more
                    stacked = np.vstack(best_history)
                    pt_smooth = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
                else:
                    pt_smooth = pt_vec.copy()
                pt_x, pt_y, pt_z = pt_smooth

                try:
                    if pt_z <= Z_MAX:
                        goal = PoseStamped()
                        goal.header = pt_base.header

                        # direct fruit position
                        goal.pose.position.x = float(pt_x)
                        goal.pose.position.y = float(pt_y)
                        goal.pose.position.z = float(pt_z)

                        # ROTATION ANGLE (relative to current orientation)
                        # Encode as quaternion: rotation around gripper approach axis
                        # Robot will use THIS to rotate from current orientation
                        rotation_deg = t_best.get("rotation_angle_deg", 0.0)
                        rotation_rad = math.radians(rotation_deg)

                        # Quaternion for rotation around Z-axis (wrist rotation)
                        # Robot will extract this angle and apply it relative to current pose
                        half_angle = rotation_rad / 2.0
                        goal.pose.orientation.w = math.cos(half_angle)
                        goal.pose.orientation.x = 0.0
                        goal.pose.orientation.y = 0.0
                        goal.pose.orientation.z = math.sin(half_angle)

                        goal_pub.publish(goal)

                        pt_base_smoothed = PointStamped()
                        pt_base_smoothed.header = pt_base.header
                        pt_base_smoothed.point.x = float(pt_x)
                        pt_base_smoothed.point.y = float(pt_y)
                        pt_base_smoothed.point.z = float(pt_z)
                        point_pub.publish(pt_base_smoothed)
                        if dir_msg is not None:
                            dir_pub.publish(dir_msg)

                except Exception as e:
                    print(f"[WARN] Publish failed: {e}")
            else:
                best_history.clear()

            # --------------------------------------------------
            # Draw all targets (centroids, boxes, masks)
            # Best fruit centroid = BLUE dot + “BEST”
            # Others = RED dots
            # --------------------------------------------------
            if not SKIP_DRAW:
                # Gray out filtered detections so we can see what was rejected
                if SHOW_REJECTED:
                    for rej in rejected_targets:
                        x1, y1, x2, y2 = rej["bb"]
                        if x2 > x1 and y2 > y1:
                            roi = image_left_ocv[y1:y2, x1:x2]
                            if roi.size:
                                gray_patch = np.full_like(roi, 128)
                                image_left_ocv[y1:y2, x1:x2] = cv2.addWeighted(
                                    gray_patch, 0.45, roi, 0.55, 0.0
                                )
                        cv2.rectangle(
                            image_left_ocv,
                            (x1, y1),
                            (x2, y2),
                            (150, 150, 150, 255),
                            2,
                        )
                        if rej.get("reason"):
                            cv2.putText(
                                image_left_ocv,
                                f"REJECT: {rej['reason']}",
                                (x1 + 6, y1 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (180, 180, 180, 255),
                                1,
                                cv2.LINE_AA,
                            )

                for i, t in enumerate(targets):
                    if DRAW_ONLY_BEST and (best_idx is not None) and i != best_idx:
                        continue
                    x1, y1, x2, y2 = t["bb"]
                    mask_resized = t["mask_resized"]
                    Xc = t["Xc"]
                    Yc = t["Yc"]
                    Zc = t["Zc"]

                    # Overlay mask
                    try:
                        h_roi, w_roi = mask_resized.shape
                        colored_mask = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                        colored_mask[:, :, 1] = mask_resized  # green
                        colored_mask[:, :, 3] = mask_resized  # alpha-like

                        roi = image_left_ocv[y1:y2, x1:x2]
                        blended = cv2.addWeighted(colored_mask, 0.45, roi, 0.55, 0.0)
                        image_left_ocv[y1:y2, x1:x2] = blended
                        # Heatmap overlay for BEST target
                        if i == best_idx:
                            heatmap = t.get("heatmap")
                            vis_r = float(t.get("vis_ratio", 0.0))
                            if heatmap is not None:
                                hm = heatmap
                                if hm.shape[:2] != (h_roi, w_roi):
                                    hm = cv2.resize(
                                        hm, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST
                                    )
                                if hm.shape[2] == 3:
                                    hm_rgba = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                                    hm_rgba[:, :, :3] = hm
                                    alpha = int(80 + 150 * max(0.0, min(vis_r, 1.0)))
                                    hm_rgba[:, :, 3] = alpha
                                else:
                                    hm_rgba = hm
                                roi_best = image_left_ocv[y1:y2, x1:x2]
                                image_left_ocv[y1:y2, x1:x2] = cv2.addWeighted(
                                    hm_rgba, 0.6, roi_best, 0.4, 0.0
                                )
                    except Exception as e:
                        print(f"[WARN] Mask overlay failed: {e}")

                    # Draw bounding box
                    cv2.rectangle(
                        image_left_ocv,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0, 255),
                        2,
                    )

                    # 2D centroid
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Color: best fruit = BLUE, others = RED
                    if i == best_idx:
                        color = (255, 0, 0, 255)  # blue
                        radius = 7
                    else:
                        color = (0, 0, 255, 255)  # red
                        radius = 5

                    cv2.circle(image_left_ocv, (cx, cy), radius, color, -1)

                    # VISUALIZE LONG-AXIS-BASED APPROACH (2D ARROW) — BEST ONLY
                    if i == best_idx:
                        # Get rotation angle for visualization
                        rotation_deg = t.get("rotation_angle_deg", 0.0)

                        axis_dir = t.get("approach_axis")
                        if axis_dir is not None:
                            vx, vy, vz = axis_dir

                            # Use only XY for visualization
                            n = math.sqrt(vx * vx + vy * vy)
                            if n < 1e-6:
                                vx, vy = 1.0, 0.0
                            else:
                                vx /= n
                                vy /= n

                            L = 60
                            ax2 = int(cx + vx * L)
                            ay2 = int(cy + vy * L)
                            ax1 = int(cx - vx * L)
                            ay1 = int(cy - vy * L)

                            # Color the axis based on rotation amount
                            if abs(rotation_deg) < 1:
                                axis_color = (0, 255, 255, 255)  # cyan (no rotation)
                            elif abs(rotation_deg) <= 15:
                                axis_color = (0, 255, 200, 255)  # yellow-cyan
                            elif abs(rotation_deg) <= 30:
                                axis_color = (0, 165, 255, 255)  # orange
                            else:
                                axis_color = (0, 100, 255, 255)  # red-orange

                            cv2.arrowedLine(
                                image_left_ocv,
                                (cx, cy),
                                (ax2, ay2),
                                axis_color,
                                2,
                                tipLength=0.25,
                            )
                            cv2.line(
                                image_left_ocv,
                                (cx, cy),
                                (ax1, ay1),
                                axis_color,
                                2,
                            )

                            # Draw rotation arc to visualize rotation angle
                            if abs(rotation_deg) > 1:
                                # Draw a small arc showing rotation direction and magnitude
                                arc_radius = 25
                                # Current angle in image coordinates
                                current_angle_deg = math.degrees(math.atan2(vy, vx))

                                # Draw arc from current angle to rotated angle
                                start_angle = int(current_angle_deg)
                                end_angle = int(current_angle_deg + rotation_deg)

                                # Color based on rotation direction
                                arc_color = (0, 255, 0, 255) if rotation_deg > 0 else (255, 0, 255, 255)  # green for +, magenta for -

                                cv2.ellipse(
                                    image_left_ocv,
                                    (cx, cy),
                                    (arc_radius, arc_radius),
                                    0,  # angle
                                    -start_angle,  # negative because cv2 Y-axis is flipped
                                    -end_angle,
                                    arc_color,
                                    2
                                )

                                # Add text label showing rotation near the arrow
                                rot_text_x = ax2 + 10
                                rot_text_y = ay2 - 10
                                cv2.putText(
                                    image_left_ocv,
                                    f"{rotation_deg:+.0f}deg",
                                    (rot_text_x, rot_text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    arc_color,
                                    2,
                                    cv2.LINE_AA,
                                )

                        # Also show best heatmap peak direction (orange) toward highest score
                        peak_pt = t.get("best_point2d_smooth")
                        if peak_pt is None:
                            peak_pt = t.get("best_point2d")
                        if peak_pt is not None:
                            dest_x = int(round(x1 + peak_pt[0]))
                            dest_y = int(round(y1 + peak_pt[1]))
                            # keep destination in frame
                            dest_x = max(0, min(dest_x, image_left_ocv.shape[1] - 1))
                            dest_y = max(0, min(dest_y, image_left_ocv.shape[0] - 1))

                            dx = float(dest_x - cx)
                            dy = float(dest_y - cy)
                            n = math.hypot(dx, dy)
                            if n < 1e-3:
                                dx, dy, n = 1.0, 0.0, 1.0
                            dx /= n
                            dy /= n
                            # start outside the box, along the opposite direction
                            L_out = max(w_roi, h_roi) + 10.0
                            start_x = int(round(dest_x - dx * L_out))
                            start_y = int(round(dest_y - dy * L_out))
                            start_x = max(0, min(start_x, image_left_ocv.shape[1] - 1))
                            start_y = max(0, min(start_y, image_left_ocv.shape[0] - 1))
                            axis_color = (0, 165, 255, 255)  # orange
                            cv2.arrowedLine(
                                image_left_ocv,
                                (start_x, start_y),
                                (dest_x, dest_y),
                                axis_color,
                                2,
                                tipLength=0.25,
                            )

                    # Draw 3D text near centroid
                    cv2.putText(
                        image_left_ocv,
                        f"X:{Xc:.2f} Y:{Yc:.2f} Z:{Zc:.2f}",
                        (cx + 10, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # BEST label + stats
                    if i == best_idx:
                        cv2.putText(
                            image_left_ocv,
                            "BEST",
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        # Depth quality + visibility stats
                        z_std = t.get("z_std", 0.0)
                        vis_ratio = t.get("vis_ratio", 0.0) * 100.0
                        cv2.putText(
                            image_left_ocv,
                            f"Zstd:{z_std:.3f}m Vis:{vis_ratio:.0f}%",
                            (cx + 10, cy + 36),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                        # Distance to gripper
                        dist_grip = t.get("dist", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"dist:{dist_grip:.3f}m",
                            (cx + 10, cy + 52),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                        # ROTATION ANGLE (relative to current orientation)
                        rotation_deg = t.get("rotation_angle_deg", 0.0)
                        # Color: green if 0°, yellow if small, orange/red if large rotation
                        if abs(rotation_deg) < 1:
                            rot_color = (0, 255, 0, 255)  # green (no rotation)
                        elif abs(rotation_deg) <= 15:
                            rot_color = (0, 255, 255, 255)  # yellow (small)
                        elif abs(rotation_deg) <= 30:
                            rot_color = (0, 165, 255, 255)  # orange (medium)
                        else:
                            rot_color = (0, 0, 255, 255)  # red (large)


                        vis_quality = t.get("vis_quality", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"VisQ:{vis_quality:.2f}",
                            (cx + 10, cy + 84),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        # NEW: Show accessibility score
                        acc_score = t.get("accessibility_score", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"Score:{acc_score:.2f}",
                            (cx + 10, cy + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0, 255),  # green
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        # Show score for non-best fruits too (in gray)
                        acc_score = t.get("accessibility_score", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"{acc_score:.1f}",
                            (cx + 5, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (150, 150, 150, 255),  # gray
                            1,
                            cv2.LINE_AA,
                        )
                        # Heatmap overlay handled earlier in mask drawing

                # HUD
                cv2.putText(
                    image_left_ocv,
                    f"YOLO FPS: {net_fps:.1f}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image_left_ocv,
                    f"Loop FPS: {loop_fps:.1f}",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Resize the displayed window to keep it smaller on screen
                display_image = cv2.resize(
                    image_left_ocv,
                    (
                        int(image_left_ocv.shape[1] * display_scale),
                        int(image_left_ocv.shape[0] * display_scale),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow("ZED | Dense-bunch 3D Position", display_image)
                key = cv2.waitKey(1)
                if key in (27, ord("q"), ord("Q")):
                    exit_signal = True

    finally:
        exit_signal = True
        zed.close()
        spin_thread.join(timeout=1.0)
        rclpy.shutdown()


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="model.pt path")
    parser.add_argument("--svo", type=str, default=None, help="optional SVO file")
    parser.add_argument(
        "--img_size", type=int, default=512, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.4, help="confidence threshold"
    )
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
