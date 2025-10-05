#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.engine.results import Results
from rclpy.executors import MultiThreadedExecutor
import tf2_geometry_msgs

from threading import Lock, Thread
from time import sleep, time
from typing import List

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math

# =========================
# Globals & shared buffers
# =========================
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None  # keep sl.Mat ownership alive

# --- temporal smoothing params ---
AXIS_ALPHA = 0.20      # 0..1, higher = follow faster
POS_ALPHA  = 0.20
MAX_AXIS_JUMP_DEG = 30 # ignore larger angle jumps
MAX_POS_JUMP_M    = 0.06

# state (per current tracked fruit)
last_axis_base = None
last_pos_base  = None
last_pos_cam   = None   # to choose same instance again

# YOLO performance
net_fps = 0.0
loop_fps = 0.0

# Shared full-res YOLO masks (binary), classes, confidences with main loop
yolo_masks = []      # list[np.ndarray(H,W) uint8 {0,1}]
yolo_classes = []    # list[int]
yolo_scores  = []    # list[float]
yolo_lock = Lock()


# =========================
# Utilities
# =========================
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _angle_deg(u, v):
    if u is None or v is None: return 0.0
    dot = float(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def _smooth_vec(new, old, alpha):
    if old is None: return _unit(new)
    # keep to same hemisphere to avoid 180° flips
    if np.dot(new, old) < 0.0:
        new = -new
    out = (1.0 - alpha) * old + alpha * new
    return _unit(out)

def centroid_xyz_in_mask(mask_bin: np.ndarray, xyz_cam: np.ndarray, z_min=0.1, z_max=2.5):
    """
    Robust 3D centroid inside a binary mask using per-pixel depth (CAMERA frame).
    Returns: (cx_px, cy_px, Xm, Ym, Zm) or None if not enough valid points.
    """
    ys, xs = np.where(mask_bin == 1)
    if ys.size < 8:
        return None

    X = xyz_cam[ys, xs, 0]
    Y = xyz_cam[ys, xs, 1]
    Z = xyz_cam[ys, xs, 2]

    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (Z > z_min) & (Z < z_max)
    if valid.sum() < 8:
        return None

    # robust to outliers
    Xm = float(np.median(X[valid]))
    Ym = float(np.median(Y[valid]))
    Zm = float(np.median(Z[valid]))

    # 2D centroid (for visualization)
    cx = int(np.round(xs[valid].mean()))
    cy = int(np.round(ys[valid].mean()))
    return (cx, cy, Xm, Ym, Zm)


def quat_mul(q, r):
    # (w,x,y,z) ⊗ (W,X,Y,Z)
    w, x, y, z = q
    W, X, Y, Z = r
    return (
        w * W - x * X - y * Y - z * Z,
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
    )


def quat_rotate_vec(q, v3):
    # rotate v3 by quaternion q=(w,x,y,z)
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (q[0], -q[1], -q[2], -q[3])
    return quat_mul(quat_mul(q, qv), qi)[1:]


def pca_long_axis_from_mask(mask_bin, xyz_cam):
    """
    mask_bin: (H,W) uint8 {0,1}, xyz_cam: (H,W,4) float32 (CAMERA frame)
    Returns unit principal axis in CAMERA frame np.array([ax,ay,az]).
    """
    ys, xs = np.where(mask_bin == 1)
    if ys.size < 8:
        return np.array([1.0, 0.0, 0.0])
    pts = xyz_cam[ys, xs, :3]
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if pts.shape[0] < 8:
        return np.array([1.0, 0.0, 0.0])
    P = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    ax = Vt[0]
    return _unit(ax)


def quat_align_x_to_axis(axis_world, up_hint=(0, 0, 1)):
    """
    Build a quaternion (w,x,y,z) whose +X aligns with axis_world.
    up_hint is used to construct a stable frame (+Z ~ up_hint).
    """
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

    t = R[0, 0] + R[1, 1] + R[2, 2]
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


def detections_to_custom_masks_(dets: Results) -> List[sl.CustomMaskObjectData]:
    """Convert Ultralytics Results to ZED CustomMaskObjectData (for ZED tracking/viewers)."""
    global sl_mats
    output = []
    sl_mats = []

    H, W = dets.orig_shape

    for di in range(len(dets.boxes)):
        obj = sl.CustomMaskObjectData()

        # Bounding box (4x2 float32)
        xywh = dets.boxes.xywh[di].cpu().numpy().astype(np.float32)
        abcd = xywh2abcd_(xywh)
        abcd[:, 0] = np.clip(abcd[:, 0], 0, W - 1)
        abcd[:, 1] = np.clip(abcd[:, 1], 0, H - 1)
        obj.bounding_box_2d = abcd

        # Label & probability
        obj.label = int(dets.boxes.cls[di].item())
        obj.probability = float(dets.boxes.conf[di].item())
        obj.is_grounded = False

        # Mask (optional but recommended)
        if dets.masks is not None and dets.masks.data is not None:
            m = dets.masks.data[di].cpu().numpy()
            mask_bin = (m * 255).astype(np.uint8)

            x_min = int(abcd[0, 0]); y_min = int(abcd[0, 1])
            x_max = int(abcd[2, 0]); y_max = int(abcd[2, 1])
            mask_roi = mask_bin[y_min:y_max+1, x_min:x_max+1]
            if not mask_roi.flags.c_contiguous:
                mask_roi = np.ascontiguousarray(mask_roi)

            sl_mat = sl.Mat(
                width=mask_roi.shape[1],
                height=mask_roi.shape[0],
                mat_type=sl.MAT_TYPE.U8_C1,
                memory_type=sl.MEM.CPU
            )
            np.copyto(sl_mat.get_data(), mask_roi)
            sl_mats.append(sl_mat)
            obj.box_mask = sl_mat

        output.append(obj)

    return output


# =========================
# YOLO thread
# =========================
def torch_thread_(weights: str, img_size: int, conf_thres: float = 0.2, iou_thres: float = 0.45) -> None:
    """Run YOLO in a background thread; provide ZED-ingestible objects and export full-res masks."""
    global image_net, exit_signal, run_signal, detections, net_fps
    global yolo_masks, yolo_classes, yolo_scores

    print("Initializing Network...")
    model = YOLO(weights)
    model.to('cuda').eval()
    print("Network Initialized...")

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            t0 = time()
            det = model.predict(
                img,
                save=False,
                retina_masks=True,
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )[0]
            dt = time() - t0
            net_fps = (1.0 / dt) if dt > 0 else 0.0

            # For ZED tracking viewer
            detections = detections_to_custom_masks_(det)

            # Export full-size binary masks, classes, confidences
            export_masks, export_classes, export_scores = [], [], []
            if det.masks is not None and det.masks.data is not None:
                H, W = det.orig_shape
                for i in range(len(det.boxes)):
                    m = det.masks.data[i].float().cpu().numpy()   # [Hm,Wm] in 0..1
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    m_bin = (m > 0.5).astype(np.uint8)
                    if m.sum() > 5000:
                        continue
                    export_masks.append(m_bin)
                    export_classes.append(int(det.boxes.cls[i].item()))
                    export_scores.append(float(det.boxes.conf[i].item()))

            with yolo_lock:
                yolo_masks[:] = export_masks
                yolo_classes[:] = export_classes
                yolo_scores[:]  = export_scores

            lock.release()
            run_signal = False
        sleep(0.005)


# =========================
# ZED Pose helpers (viewer use)
# =========================
def _extract_T_wc(cam_pose_world: sl.Pose) -> np.ndarray:
    """Get CAMERA->WORLD 4x4 pose matrix (T_wc) robustly across ZED SDK variants."""
    try:
        T_attr = cam_pose_world.pose_data
        T_wc = T_attr() if callable(T_attr) else T_attr
        T_wc = np.array(T_wc)
        if T_wc.size == 16:
            return T_wc.reshape(4, 4).astype(np.float32)
        if T_wc.shape == (4, 4):
            return T_wc.astype(np.float32)
    except Exception:
        pass

    try:
        flat = cam_pose_world.m
        if hasattr(flat, "__len__") and len(flat) == 16:
            return np.array(flat, dtype=np.float32).reshape(4, 4)
    except Exception:
        pass

    # Fallback compose
    R_wc = None
    try:
        Robj = cam_pose_world.get_rotation_matrix()
        if hasattr(Robj, "r"):
            R_wc = np.array(Robj.r, dtype=np.float32).reshape(3, 3)
        elif hasattr(Robj, "get"):
            R_wc = np.array(Robj.get(), dtype=np.float32).reshape(3, 3)
        else:
            R_wc = np.array(Robj, dtype=np.float32).reshape(3, 3)
    except Exception:
        pass
    if R_wc is None:
        raise RuntimeError("Could not extract rotation")

    t_wc = None
    try:
        Tobj = cam_pose_world.get_translation()
        if hasattr(Tobj, "get"):
            vals = Tobj.get()
            t_wc = np.array([vals[0], vals[1], vals[2]], dtype=np.float32)
        else:
            arr = np.array(Tobj, dtype=np.float32).flatten()
            t_wc = arr[:3]
    except Exception:
        pass
    if t_wc is None:
        raise RuntimeError("Could not extract translation")

    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc
    return T_wc


def world_to_camera(point_world: np.ndarray, cam_pose_world: sl.Pose) -> np.ndarray:
    """Convert WORLD-frame 3D point to CAMERA frame: p_cam = R_cw * (p_w - t_wc)"""
    T_wc = _extract_T_wc(cam_pose_world)
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_cw = R_wc.T
    return R_cw.dot(point_world - t_wc)


# =========================
# Tier-2 visibility primitives
# =========================
def zbuffer_visible_masks(masks, xyz_cam, z_min=0.10, z_max=1.60, eps=0.003, erode_px=1):
    """
    Assign each pixel to the nearest instance (z-buffer).
    returns: visible_masks (front-only pixels per instance), vis_ratios in [0,1].
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


# =========================
# Main
# =========================
def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections, loop_fps
    global last_pos_cam, last_pos_base, last_axis_base

    # ROS2 setup
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
    goal_pub  = node.create_publisher(PoseStamped, '/external_goal_pose', fast_qos)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # YOLO thread
    capture_thread = Thread(
        target=torch_thread_,
        kwargs={'weights': args.weights, 'img_size': args.img_size, 'conf_thres': args.conf_thres}
    )
    capture_thread.start()

    # --- ZED init ---
    input_type = sl.InputType()
    if args.svo is not None:
        input_type.set_from_svo_file(args.svo)

    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_maximum_distance = 50

    print("Initializing Camera...")
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        return
    print("Camera Initialized")

    # Positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Object detection (custom masks)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True
    zed.enable_object_detection(obj_param)

    # Display setup
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    xyz_full = sl.Mat()  # full-res XYZ (CAMERA frame) for per-pixel depth

    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                             [245, 239, 239, 255], np.uint8)

    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(
        tracks_resolution,
        camera_config.fps,
        init_params.depth_maximum_distance * 1000,
        1
    )
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    cam_w_pose = sl.Pose()
    image_left_tmp = sl.Mat()
    objects = sl.Objects()

    class_names = {0: "datefruit"}  # optional
    t_loop_prev = time()

    # Camera frame name used by TF
    cam_frame = 'zed2_left_camera_frame'  # CHANGE if your TF uses a different optical frame

    # Visibility thresholds
    FULLY_VISIBLE_THRESH  = 0.90    # >=90% of mask pixels are truly front-most
    PARTIAL_VISIBLE_THRESH = 0.0   # <10% means effectively occluded (skip)
    GOOD_VIS_THRESH        = 0.0   # update state only if >= 30% front-visible

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # loop FPS
            t_now = time()
            dt_loop = t_now - t_loop_prev
            loop_fps = (1.0 / dt_loop) if dt_loop > 0 else 0.0
            t_loop_prev = t_now

            # input image for YOLO (full resolution)
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # wait for YOLO
            while run_signal and not exit_signal:
                sleep(0.001)

            # Ingest detections for ZED tracking
            lock.acquire()
            zed.ingest_custom_mask_objects(detections)
            lock.release()
            zed.retrieve_custom_objects(objects, obj_runtime_param)

            # Get camera pose in WORLD for viewer
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # Retrieve point cloud for viewer, and full-res XYZ for mask-based depth
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            zed.retrieve_measure(xyz_full, sl.MEASURE.XYZ, sl.MEM.CPU)  # full camera res, CAMERA frame
            xyz_np = xyz_full.get_data()  # (H, W, 4) float32; meters in CAMERA frame

            # Retrieve display image
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            np.copyto(image_left_ocv, image_left.get_data())

            # ===============================
            # Z-BUFFER: Visibility resolution
            # ===============================
            with yolo_lock:
                masks  = list(yolo_masks)
                labels = list(yolo_classes)
                scores = list(yolo_scores)

            visible_masks, vis_ratios = zbuffer_visible_masks(
                masks, xyz_np, z_min=0.10, z_max=1.60, eps=0.003, erode_px=1
            )

            # Build candidate picks from visibility-resolved masks
            picks = []
            for mbin_vis, vis_ratio, cls_i, conf_i in zip(visible_masks, vis_ratios, labels, scores):
                # Remove all filtering: publish every valid detection
                res = centroid_xyz_in_mask(mbin_vis, xyz_np, z_min=0.10, z_max=1.60)
                if res is None:
                    continue
                vx, vy, Xc, Yc, Zc = res
                visibility_state = "FULL" if vis_ratio >= FULLY_VISIBLE_THRESH else "PARTIAL"
                picks.append((Zc, vx, vy, Xc, Yc, Zc, cls_i, conf_i, mbin_vis, vis_ratio, visibility_state))

            # Publish ALL valid picks (no sorting, no smoothing, no tracking, no jump rejection)
            for (_, vx, vy, Xc, Yc, Zc, cls_i, conf_i, mbin_vis, vis_ratio, visibility_state) in picks:
                point_msg = PointStamped()
                point_msg.header.frame_id = cam_frame
                point_msg.header.stamp = rclpy.time.Time().to_msg()
                point_msg.point.x = Xc
                point_msg.point.y = Yc
                point_msg.point.z = Zc
                try:
                    pt_base = tf_buffer.transform(point_msg, 'base_link', timeout=rclpyDuration(seconds=0.2))
                    # Compute long-axis from mask (CAMERA frame) via PCA fit
                    axis_cam = pca_long_axis_from_mask(mbin_vis, xyz_np)
                    if axis_cam[2] < 0:
                        axis_cam = -axis_cam
                    try:
                        tf_cb = tf_buffer.lookup_transform('base_link', cam_frame, rclpy.time.Time())
                        q_cb = tf_cb.transform.rotation  # quaternion cam->base
                        axis_base_meas = np.array(
                            quat_rotate_vec((q_cb.w, q_cb.x, q_cb.y, q_cb.z), axis_cam), dtype=float
                        )
                        axis_base_meas = _unit(axis_base_meas)
                    except Exception as e:
                        print(f"TF axis rotation failed, using camera axis: {e}")
                        axis_base_meas = axis_cam
                    # Publish pose for each detection (no smoothing)
                    qw, qx, qy, qz = quat_align_x_to_axis(axis_base_meas, up_hint=(0, 0, 1))
                    goal = PoseStamped()
                    goal.header = pt_base.header
                    goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = pt_base.point.x, pt_base.point.y, pt_base.point.z
                    goal.pose.orientation.w = qw
                    goal.pose.orientation.x = qx
                    goal.pose.orientation.y = qy
                    goal.pose.orientation.z = qz
                    goal_pub.publish(goal)
                    # Also publish point
                    pt_pub = PointStamped()
                    pt_pub.header = pt_base.header
                    pt_pub.point.x, pt_pub.point.y, pt_pub.point.z = pt_base.point.x, pt_base.point.y, pt_base.point.z
                    point_pub.publish(pt_pub)
                    # Overlay debug on image
                    sx = int(vx * image_scale[0]); sy = int(vy * image_scale[1])
                    cv2.circle(image_left_ocv, (sx, sy), 4, (0, 255, 0, 255), -1)
                    cv2.putText(image_left_ocv,
                                f"{visibility_state} {vis_ratio*100:.0f}% Zc={Zc:.2f}m",
                                (sx + 6, sy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0, 255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"TF or publish failed: [{type(e).__name__}] {e}")

            # Tracking/bird's-eye view
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.CAMERA)  # for the tracking viewer
            image_track_ocv = np.zeros_like(image_track_ocv)
            track_view_generator.generate_view(
                objects, image_left_ocv, image_scale, cam_w_pose, image_track_ocv, objects.is_tracked
            )

            # FPS overlay
            cv2.putText(image_left_ocv, f"YOLO FPS: {net_fps:.1f}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image_left_ocv, f"Loop FPS: {loop_fps:.1f}", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255, 255), 2, cv2.LINE_AA)

            # Side-by-side display
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            cv2.imshow("ZED | 2D View + Bird's View", global_image)
            key = cv2.waitKey(10)
            if key in (27, ord('q'), ord('Q')):
                exit_signal = True
            if key == 105:  # 'i'
                track_view_generator.zoomIn()
            if key == 111:  # 'o'
                track_view_generator.zoomOut()
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()
    executor.shutdown()
    spin_thread.join(timeout=1.0)
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m-seg.pt', help='model.pt path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
