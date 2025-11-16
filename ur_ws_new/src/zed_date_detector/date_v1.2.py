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
import ogl_viewer.viewer as gl
import tf2_geometry_msgs
import cv_viewer.tracking_viewer as cv_viewer

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from rclpy.executors import MultiThreadedExecutor

import math

# ============================================================
# Globals
# ============================================================
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None

net_fps = 0.0
loop_fps = 0.0

yolo_masks = []
yolo_classes = []
yolo_scores = []
yolo_lock = Lock()

# ============================================================
# Utility Functions
# ============================================================
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def zbuffer_visible_masks(masks, xyz_cam, z_min=0.10, z_max=1.60, eps=0.003, erode_px=0):
    """
    Assign each pixel to the nearest instance (z-buffer).
    Returns: visible_masks (front-only pixels per instance), vis_ratios in [0,1].
    """
    if not masks:
        return [], []

    H, W = xyz_cam.shape[:2]
    Z = xyz_cam[..., 2].copy()
    Z[~np.isfinite(Z)] = np.inf
    Z[(Z < z_min) | (Z > z_max)] = np.inf

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        masks_proc = [cv2.erode(m.astype(np.uint8), k, iterations=1) for m in masks]
    else:
        masks_proc = [m.astype(np.uint8) for m in masks]

    N = len(masks_proc)
    z_stack = np.full((N, H, W), np.inf, dtype=np.float32)
    sizes = np.zeros(N, dtype=np.int64)

    for i, mbin in enumerate(masks_proc):
        if mbin.shape != (H, W):
            mbin = cv2.resize(mbin, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (mbin == 1)
        sizes[i] = int(m.sum())
        z_slice = z_stack[i]
        z_slice[m] = Z[m]

    front_z = np.min(z_stack, axis=0)

    visible_masks = []
    vis_ratios = []
    for i in range(N):
        vis = (z_stack[i] <= (front_z + eps)) & np.isfinite(front_z)
        vis_u8 = vis.astype(np.uint8)
        visible_masks.append(vis_u8)
        kept = int(vis.sum())
        total = max(1, int(sizes[i]))
        vis_ratios.append(kept / total)

    return visible_masks, vis_ratios

def centroid_xyz_in_mask(mask_bin: np.ndarray, xyz_cam: np.ndarray, z_min=0.1, z_max=2.5, min_points=3):
    """
    Robust 3D centroid inside a binary mask using per-pixel depth (CAMERA frame).
    Returns: (cx_px, cy_px, Xm, Ym, Zm) or None if not enough valid points.
    """
    ys, xs = np.where(mask_bin == 1)
    if ys.size < min_points:
        return None

    X = xyz_cam[ys, xs, 0]
    Y = xyz_cam[ys, xs, 1]
    Z = xyz_cam[ys, xs, 2]

    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (Z > z_min) & (Z < z_max)
    if valid.sum() < min_points:
        return None

    # robust to outliers using median
    Xm = float(np.median(X[valid]))
    Ym = float(np.median(Y[valid]))
    Zm = float(np.median(Z[valid]))
    
    cx_px = float(np.mean(xs[valid]))
    cy_px = float(np.mean(ys[valid]))
    
    return (cx_px, cy_px, Xm, Ym, Zm)

def quat_mul(q, r):
    w, x, y, z = q
    W, X, Y, Z = r
    return (
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    )

def quat_rotate_vec(q, v3):
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (q[0], -q[1], -q[2], -q[3])
    return quat_mul(quat_mul(q, qv), qi)[1:]

def quat_align_x_to_axis(axis_world, up_hint=(0, 0, 1)):
    X = _unit(axis_world)
    U = _unit(up_hint)
    Y = np.cross(U, X)
    if np.linalg.norm(Y) < 1e-6:
        U = np.array([1.0, 0.0, 0.0])
        Y = np.cross(U, X)
    Y = _unit(Y)
    Z = _unit(np.cross(X, Y))

    R = np.array([[X[0], Y[0], Z[0]],
                  [X[1], Y[1], Z[1]],
                  [X[2], Y[2], Z[2]]], dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return (float(qw), float(qx), float(qy), float(qz))

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
            x_min = int(abcd[0, 0]); y_min = int(abcd[0, 1])
            x_max = int(abcd[2, 0]); y_max = int(abcd[2, 1])
            mask_roi = mask_bin[y_min:y_max+1, x_min:x_max+1]
            if not mask_roi.flags.c_contiguous:
                mask_roi = np.ascontiguousarray(mask_roi)
            sl_mat = sl.Mat(width=mask_roi.shape[1], height=mask_roi.shape[0],
                            mat_type=sl.MAT_TYPE.U8_C1, memory_type=sl.MEM.CPU)
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
    model = YOLO(weights)
    model.to('cuda').eval()
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
                verbose=False
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
    global image_net, exit_signal, run_signal, detections, loop_fps

    # --- ROS2 setup ---
    rclpy.init()
    node = rclpy.create_node('zed_date_detector_ros')
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    fast_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=DurabilityPolicy.VOLATILE,
    )
    point_pub = node.create_publisher(PointStamped, '/datefruit_3d_point', 10)
    goal_pub = node.create_publisher(PoseStamped, '/external_goal_pose', fast_qos)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # --- YOLO thread ---
    capture_thread = Thread(
        target=torch_thread_,
        kwargs={'weights': args.weights, 'img_size': args.img_size, 'conf_thres': args.conf_thres},
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
    init_params.depth_maximum_distance = 50
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

    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    image_left = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    cam_w_pose = sl.Pose()
    objects = sl.Objects()
    
    # For zbuffer visibility filtering
    xyz_full = sl.Mat()
    camera_res_full = camera_res  # full resolution XYZ

    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_left_ocv = np.full(
        (display_resolution.height, display_resolution.width, 4),
        [245, 239, 239, 255],
        np.uint8
    )
    image_scale = [
        display_resolution.width / camera_res.width,
        display_resolution.height / camera_res.height,
    ]

    cam_frame = 'zed2_left_camera_frame'
    t_prev = time()

    # --- Main Loop ---
    try:
        while viewer.is_available() and not exit_signal:
            # Let TF listener process incoming /tf
            rclpy.spin_once(node, timeout_sec=0.0)

            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                t_now = time()
                loop_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
                t_prev = t_now

                # Get image for YOLO
                with lock:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    image_net = image_left.get_data()
                    run_signal = True

                # Wait for YOLO to finish
                while run_signal and not exit_signal:
                    sleep(0.001)

                # Ingest YOLO detections into ZED
                with lock:
                    current_dets = detections
                if current_dets is not None:
                    zed.ingest_custom_mask_objects(current_dets)
                zed.retrieve_custom_objects(objects, obj_runtime_param)

                # Get image for display
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                np.copyto(image_left_ocv, image_left.get_data())

                # ===============================
                # Z-BUFFER: Visibility resolution
                # ===============================
                # Retrieve full-res XYZ for zbuffer filtering
                zed.retrieve_measure(xyz_full, sl.MEASURE.XYZ, sl.MEM.CPU)  # full camera res, CAMERA frame
                xyz_np = xyz_full.get_data()  # (H, W, 4) float32; meters in CAMERA frame

                # Get YOLO masks (shared with detection thread)
                with yolo_lock:
                    masks = list(yolo_masks)
                    labels = list(yolo_classes)
                    scores = list(yolo_scores)

                # Apply zbuffer to get front-only visible masks
                visible_masks, vis_ratios = zbuffer_visible_masks(
                    masks, xyz_np, z_min=0.10, z_max=1.60, eps=0.003, erode_px=0
                )

                # Process each detected object
                for o in objects.object_list:
                    # 1. Native ZED 3D position
                    Xc = float(o.position[0])
                    Yc = float(o.position[1])
                    Zc = float(o.position[2])

                    # 2. Publish PointStamped in camera frame
                    point_msg = PointStamped()
                    point_msg.header.frame_id = cam_frame
                    point_msg.header.stamp = rclpy.time.Time().to_msg()
                    point_msg.point.x = Xc
                    point_msg.point.y = Yc
                    point_msg.point.z = Zc

                    try:
                        pt_base = tf_buffer.transform(point_msg, 'base_link', timeout=rclpyDuration(seconds=0.2))

                        Z_MAX = 1.34
                        if pt_base.point.z > Z_MAX:
                            continue

                        # 4. Publish goal pose (identity quaternion)
                        goal = PoseStamped()
                        goal.header = pt_base.header
                        goal.pose.position.x = float(pt_base.point.x)
                        goal.pose.position.y = float(pt_base.point.y)
                        goal.pose.position.z = float(pt_base.point.z)

                        goal.pose.orientation.w = 1.0
                        goal.pose.orientation.x = 0.0
                        goal.pose.orientation.y = 0.0
                        goal.pose.orientation.z = 0.0

                        goal_pub.publish(goal)
                        point_pub.publish(pt_base)

                    except Exception as e:
                        print(f"[WARN] Transform or publish failed: {e}")
                        continue

                    # 5. Draw YOLO/ZED segmentation mask
                    # Try 'mask' first, then 'box_mask' as fallback
                    mask_mat = None
                    if hasattr(o, "mask") and o.mask is not None and o.mask.is_init():
                        mask_mat = o.mask
                    elif hasattr(o, "box_mask") and o.box_mask is not None and o.box_mask.is_init():
                        mask_mat = o.box_mask

                    if mask_mat is not None:
                        try:
                            mask_local = mask_mat.get_data()
                            if mask_local.ndim == 3:
                                mask_local = mask_local[:, :, 0]

                            bb = o.bounding_box_2d
                            x1 = int(bb[0][0] * image_scale[0])
                            y1 = int(bb[0][1] * image_scale[1])
                            x2 = int(bb[2][0] * image_scale[0])
                            y2 = int(bb[2][1] * image_scale[1])

                            x1 = max(0, min(x1, image_left_ocv.shape[1] - 1))
                            x2 = max(0, min(x2, image_left_ocv.shape[1]))
                            y1 = max(0, min(y1, image_left_ocv.shape[0] - 1))
                            y2 = max(0, min(y2, image_left_ocv.shape[0]))

                            if x2 <= x1 or y2 <= y1:
                                continue

                            w_roi = x2 - x1
                            h_roi = y2 - y1

                            mask_resized = cv2.resize(mask_local, (w_roi, h_roi))
                            # 4-channel colored mask to match RGBA image
                            colored_mask = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                            colored_mask[:, :, 1] = mask_resized  # green channel
                            colored_mask[:, :, 3] = mask_resized  # alpha-like channel

                            roi = image_left_ocv[y1:y2, x1:x2]
                            blended = cv2.addWeighted(colored_mask, 0.45, roi, 0.55, 0.0)
                            image_left_ocv[y1:y2, x1:x2] = blended
                        except Exception as e:
                            print(f"[WARN] Mask overlay failed: {e}")

                    # 6. Draw bounding box + depth text
                    bb = o.bounding_box_2d
                    x1 = int(bb[0][0] * image_scale[0])
                    y1 = int(bb[0][1] * image_scale[1])
                    x2 = int(bb[2][0] * image_scale[0])
                    y2 = int(bb[2][1] * image_scale[1])

                    cv2.rectangle(
                        image_left_ocv,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0, 255),
                        2
                    )

                    cv2.putText(
                    image_left_ocv,
                    f"X:{Xc:.2f}  Y:{Yc:.2f}  Z:{Zc:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # HUD
                cv2.putText(
                    image_left_ocv,
                    f"YOLO FPS: {net_fps:.1f}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    image_left_ocv,
                    f"Loop FPS: {loop_fps:.1f}",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow("ZED | Native 3D Position", image_left_ocv)
                key = cv2.waitKey(10)
                if key in (27, ord('q'), ord('Q')):
                    exit_signal = True
            else:
                exit_signal = True

    finally:
        viewer.exit()
        exit_signal = True
        zed.close()
        spin_thread.join(timeout=1.0)
        rclpy.shutdown()

# ============================================================
# Entry Point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model.pt path')
    parser.add_argument('--svo', type=str, default=None, help='optional SVO file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence threshold')
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
