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

# Global variables
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None

net_fps = 0.0
loop_fps = 0.0
prev_heat_point = None
last_dir_label = None
last_dir_count = 0
last_dir_published = None

# Rendering knobs to save CPU (publishing unaffected)
DRAW_ONLY_BEST = False
SHOW_REJECTED = False
SKIP_DRAW = False

# --- Persistent BEST fruit tracking ---
best_target_prev = None
BEST_REUSE_THRESH = 0.05  # 5 cm positional tolerance in base_link

# --- Track-by-detection smoothing ---
best_history = deque(maxlen=3)


# Utility Functions
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_mul(q, r):
    qw, qx, qy, qz = q
    rw, rx, ry, rz = r
    return (
        qw * rw - qx * rx - qy * ry - qz * rz,
        qw * rx + qx * rw + qy * rz - qz * ry,
        qw * ry - qx * rz + qy * rw + qz * rx,
        qw * rz + qx * ry - qy * rx + qz * rw,
    )

def quaternion_to_degrees(q):
    """Return rotation angle (in degrees) represented by quaternion q=(w,x,y,z)."""
    w = q[0]
    angle_rad = 2.0 * math.acos(max(min(w, 1.0), -1.0))
    return math.degrees(angle_rad)

def quat_rotate_vec(q, v3):
    """Rotate vector v3 by quaternion q."""
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (q[0], -q[1], -q[2], -q[3])
    return quat_mul(quat_mul(q, qv), qi)[1:]

def signed_angle_between(v1, v2, up=np.array([0, 0, 1])):
    """Returns signed angle in degrees between v1 and v2."""
    v1_norm = _unit(v1)
    v2_norm = _unit(v2)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = math.degrees(math.acos(dot))
    cross = np.cross(v1_norm, v2_norm)
    sign = np.sign(np.dot(cross, up))
    return angle * sign

def quat_align_axis_to_axis(src_axis, dst_axis, up_hint=(0, 0, 1)):
    """Build quaternion that rotates src_axis → dst_axis."""
    src = _unit(np.array(src_axis, float))
    dst = _unit(np.array(dst_axis, float))
    v = np.cross(src, dst)
    c = np.dot(src, dst)

    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return (1, 0, 0, 0)
        perp = _unit(np.cross(src, up_hint))
        return (0, perp[0], perp[1], perp[2])

    s = math.sqrt((1 + c) * 2)
    invs = 1.0 / s
    return (s * 0.5, v[0] * invs, v[1] * invs, v[2] * invs)


def normalize_grasp_angle(angle_deg):
    """Map angle to symmetric range where 180° equals 0°."""
    angle = (angle_deg + 180) % 360 - 180
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


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


# YOLO Detection Thread
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


# Main Function
def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections, loop_fps, best_target_prev, prev_heat_point
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
    TransformListener(tf_buffer, node)
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
    #  ZED frame : positive Y pointing down, X pointing right, and Z pointing away from the camera.
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

    image_left = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    objects = sl.Objects()
    point_cloud = sl.Mat()

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

            # Process detected objects
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
                    kernel = np.ones((5, 5), np.uint8)
                    mask_clean = cv2.morphologyEx(
                        cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel),
                        cv2.MORPH_OPEN, kernel
                    )
                    mask_bool = mask_clean > 0

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
                                _, _, angle_deg = cv2.fitEllipse(cnt)
                                theta = math.radians(angle_deg)
                                long_dir_img = np.array([math.cos(theta), math.sin(theta)], dtype=float)
                                short_dir_cam = np.array([-long_dir_img[1], long_dir_img[0], 0.0], dtype=float)
                                short_dir_cam /= max(np.linalg.norm(short_dir_cam), 1e-6)

                                t_short_axis = short_dir_cam
                                long_axis_2d = long_dir_img
                                t_angle = float(angle_deg)
                    except Exception as e:
                        print("[WARN] Ellipse axis extraction failed:", e)

                    roi_xyz = pc_np[y1:y2, x1:x2, :]
                    valid = np.isfinite(roi_xyz[:, :, 2]) & mask_bool
                    depth_vals = roi_xyz[:, :, 2][valid]
                    heatmap = None

                    if depth_vals.size > 0:
                        z_lo, z_hi = np.percentile(depth_vals, [5.0, 90.0])
                        z_hi = max(z_hi, z_lo + 1e-3)
                        score = np.zeros_like(roi_xyz[:, :, 2], dtype=np.float32)
                        score[valid] = (z_hi - roi_xyz[:, :, 2][valid]) / (z_hi - z_lo)
                        score_u8 = (np.clip(score, 0.0, 1.0) * 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)

                        ys, xs = np.nonzero(mask_clean)
                        if len(xs) > 0:
                            scores = heatmap[ys, xs, 2].astype(float)
                            idx_max = int(np.argmax(scores))
                            peak_y = int(ys[idx_max])
                            peak_x = int(xs[idx_max])
                            t_best_point = np.array([peak_x, peak_y], dtype=float)
                            cx = mask_clean.shape[1] / 2.0
                            cy = mask_clean.shape[0] / 2.0
                            dir_vec = np.array([peak_x - cx, peak_y - cy], dtype=float)
                            n_dir = np.linalg.norm(dir_vec)
                            t_best_dir2d = dir_vec / n_dir if n_dir > 1e-6 else np.array([1.0, 0.0], dtype=float)

                    vis_mask = cv2.erode(mask_clean, np.ones((3, 3), np.uint8), iterations=1) > 0
                    mask_pixels = np.count_nonzero(vis_mask)
                    vis_ratio = (
                        float(np.count_nonzero(valid & vis_mask)) / float(mask_pixels)
                        if mask_pixels > 0 else 0.0
                    )
                    vis_ratio = np.clip(vis_ratio, 0.0, 1.0)

                    if np.count_nonzero(valid) < 30:
                        mark_reject("Too few depth pts")
                        continue

                    pts = roi_xyz[valid]
                    zs = pts[:, 2]

                    idx = np.argsort(zs)
                    k = max(10, int(0.2 * len(idx)))
                    pts_front = pts[idx[:k]]

                    depth_std = np.std(pts_front[:, 2])
                    vis_quality = (vis_ratio ** 2) * np.exp(-(depth_std / 0.015) ** 2)

                    if depth_std > 0.05:
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

            # Best fruit selection with occlusion awareness
            frame_best_idx = None
            frame_best_score = float("inf")

            for i, t in enumerate(targets):
                dist = t["dist"]
                z_std = t.get("z_std", 0.0)
                occlusion_penalty = z_std * 15.0

                bb = t["bb"]
                x1, y1, x2, y2 = bb
                edge_dist = min(x1, y1, image_left_ocv.shape[1] - x2, image_left_ocv.shape[0] - y2)
                edge_penalty = max(0, 50 - edge_dist) * 0.02

                vis_quality = t.get("vis_quality", 0.0)
                vis_bonus = -vis_quality * 0.3

                score = dist + occlusion_penalty + edge_penalty + vis_bonus
                t["accessibility_score"] = score

                if score < frame_best_score:
                    frame_best_score = score
                    frame_best_idx = i

            best_idx = frame_best_idx
            if best_target_prev is not None:
                prev_pt = np.array([
                    best_target_prev["pt_base"].point.x,
                    best_target_prev["pt_base"].point.y,
                    best_target_prev["pt_base"].point.z
                ], dtype=float)

                reuse_idx = None
                min_dist = float("inf")

                for i, t in enumerate(targets):
                    cur_pt = np.array([
                        t["pt_base"].point.x,
                        t["pt_base"].point.y,
                        t["pt_base"].point.z
                    ], dtype=float)
                    d = np.linalg.norm(cur_pt - prev_pt)

                    if d < BEST_REUSE_THRESH and d < min_dist:
                        min_dist = d
                        reuse_idx = i

                if reuse_idx is not None:
                    best_idx = reuse_idx

            if best_idx is not None:
                best_target_prev = targets[best_idx]

            # Orientation optimization
            if best_idx is not None:
                t_best = targets[best_idx]
                short_cam = t_best.get("short_axis_cam")

                # USE RAW SHORT AXIS (no optimization)
                raw_axis_cam = t_best.get("short_axis_cam")

                if raw_axis_cam is not None:
                    t_best["optimized_axis_cam"] = raw_axis_cam
                    optimized_axis_cam = raw_axis_cam.copy()
                else:
                    t_best["optimized_axis_cam"] = None
                    optimized_axis_cam = None
                    print("[ORIENT] No short axis available")

                best_pt = t_best.get("best_point2d")
                if best_pt is not None:
                    sm_pt = best_pt.copy() if prev_heat_point is None else 0.7 * prev_heat_point + 0.3 * best_pt
                    prev_heat_point = sm_pt.copy()
                    t_best["best_point2d_smooth"] = sm_pt
                else:
                    prev_heat_point = None
                try:
                    if optimized_axis_cam is not None:
                        transform = tf_buffer.lookup_transform("base_link", "gripper_tip", rclpyTime())
                        current_quat = transform.transform.rotation
                        current_rot = R.from_quat([current_quat.x, current_quat.y, current_quat.z, current_quat.w])
                        current_matrix = current_rot.as_matrix()

                        axis_point_cam = PointStamped()
                        axis_point_cam.header.frame_id = cam_frame
                        axis_point_cam.header.stamp = rclpyTime().to_msg()
                        axis_point_cam.point.x = optimized_axis_cam[0]
                        axis_point_cam.point.y = optimized_axis_cam[1]
                        axis_point_cam.point.z = optimized_axis_cam[2]

                        origin_cam = PointStamped()
                        origin_cam.header.frame_id = cam_frame
                        origin_cam.header.stamp = rclpyTime().to_msg()
                        origin_cam.point.x = 0.0
                        origin_cam.point.y = 0.0
                        origin_cam.point.z = 0.0

                        axis_point_base = tf_buffer.transform(axis_point_cam, "base_link")
                        origin_base = tf_buffer.transform(origin_cam, "base_link")

                        target_axis_base = np.array([
                            axis_point_base.point.x - origin_base.point.x,
                            axis_point_base.point.y - origin_base.point.y,
                            axis_point_base.point.z - origin_base.point.z
                        ])
                        target_axis_base /= max(np.linalg.norm(target_axis_base), 1e-6)

                        gripper_y_current = current_matrix[:, 1]
                        gripper_z_current = current_matrix[:, 2]

                        target_proj = target_axis_base - np.dot(target_axis_base, gripper_z_current) * gripper_z_current
                        target_proj /= max(np.linalg.norm(target_proj), 1e-6)

                        gripper_y_proj = gripper_y_current - np.dot(gripper_y_current, gripper_z_current) * gripper_z_current
                        gripper_y_proj /= max(np.linalg.norm(gripper_y_proj), 1e-6)

                        dot = np.clip(np.dot(gripper_y_proj, target_proj), -1.0, 1.0)
                        angle_rad = math.acos(dot)

                        cross = np.cross(gripper_y_proj, target_proj)
                        if np.dot(cross, gripper_z_current) < 0:
                            angle_rad = -angle_rad

                        relative_angle_deg = normalize_grasp_angle(math.degrees(angle_rad))
                        t_best["rotation_angle_deg"] = relative_angle_deg
                        #print(f"[ORIENT] Relative rotation: {relative_angle_deg:+.1f}° (current→target)")
                    else:
                        t_best["rotation_angle_deg"] = 0.0
                        #print(f"[ORIENT] No target axis, rotation=0° (keep current)")

                except Exception as e:
                    print(f"[ORIENT ERROR] Failed to compute relative rotation: {e}")
                    import traceback
                    traceback.print_exc()
                    t_best["rotation_angle_deg"] = 0.0

                fruit_pos = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]], dtype=float)
                t_best["approach_axis"] = fruit_pos / max(np.linalg.norm(fruit_pos), 1e-6)

            # Publish goal for BEST fruit only
            if best_idx is not None:
                t_best = targets[best_idx]
                pt_base = t_best["pt_base"]
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

                    short_cam = t.get("short_axis_cam")
                    if short_cam is not None:
                        sx, sy, _ = short_cam  # ignore Z for drawing

                        # base point = centroid
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # scale for visibility
                        L = 50

                        # end point
                        ex = int(cx + sx * L)
                        ey = int(cy + sy * L)

                        # draw arrow (pink)
                        cv2.arrowedLine(image_left_ocv, (cx, cy), (ex, ey),
                                        (255, 0, 255), 2, tipLength=0.3)

                    # VISUALIZE LONG-AXIS-BASED APPROACH (2D ARROW) — BEST ONLY
                    if i == best_idx:
                        # Get rotation angle for visualization
                        rotation_deg = t.get("rotation_angle_deg", 0.0)

                        axis_dir = t.get("approach_axis")
                        if axis_dir is not None:
                            vx, vy = axis_dir[0], axis_dir[1]

                            # Normalize XY for visualization
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

                        # Visibility quality
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


# Entry Point
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
