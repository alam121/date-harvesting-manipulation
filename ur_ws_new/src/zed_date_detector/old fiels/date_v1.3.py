#Computes per-instance mask-median depth (Z̄) and reprojects via intrinsics → PIX_CAM point.
#Also reads ZED’s internal native 3D position for the matched object → ZED_NATIVE point.
#If the 3D discrepancy is ≤ 2 cm (configurable), we trust PIX_CAM; otherwise we fallback to ZED_NATIVE.
#Publishes /datefruit_3d_point and /external_goal_pose for every valid detection (no class filter).
#Keeps YOLO → ZED ingest → ROS2 pubs → TF → viewer structure.

#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep, time
from typing import List, Tuple

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

# YOLO performance
net_fps = 0.0
loop_fps = 0.0

# Shared full-res YOLO masks (binary), classes, confidences with main loop
yolo_masks: List[np.ndarray] = []   # list[np.ndarray(H,W) uint8 {0,1}]
yolo_classes: List[int] = []        # list[int]
yolo_scores: List[float] = []       # list[float]
yolo_lock = Lock()

# =========================
# Utilities
# =========================
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def quat_align_x_to_axis(axis_world, up_hint=(0, 0, 1)):
    """
    Build a quaternion (w,x,y,z) whose +X aligns with axis_world.
    Minimal, stable frame build using up_hint ~ +Z.
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
    """Convert Ultralytics Results to ZED CustomMaskObjectData (for ZED tracking/viewers)."""
    global sl_mats
    output = []
    sl_mats = []
    H, W = dets.orig_shape
    for di in range(len(dets.boxes)):
        obj = sl.CustomMaskObjectData()
        # bbox
        xywh = dets.boxes.xywh[di].cpu().numpy().astype(np.float32)
        abcd = xywh2abcd_(xywh)
        abcd[:, 0] = np.clip(abcd[:, 0], 0, W - 1)
        abcd[:, 1] = np.clip(abcd[:, 1], 0, H - 1)
        obj.bounding_box_2d = abcd
        # label/prob
        obj.label = int(dets.boxes.cls[di].item())
        obj.probability = float(dets.boxes.conf[di].item())
        obj.is_grounded = False
        # mask ROI
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

def pca_long_axis_from_mask(mask_bin, xyz_cam):
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

def zbuffer_visible_masks(masks, xyz_cam, z_min=0.10, z_max=5.0, eps=0.003, erode_px=0):
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
    visible_masks, vis_ratios = [], []
    for i in range(N):
        vis = (z_stack[i] <= (front_z + eps)) & np.isfinite(front_z)
        vis_u8 = vis.astype(np.uint8)
        visible_masks.append(vis_u8)
        kept = int(vis.sum())
        total = max(1, int(sizes[i]))
        vis_ratios.append(kept / total)
    return visible_masks, vis_ratios

def mask_bbox(mbin: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mbin == 1)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def obj_bbox_pixels(obj) -> Tuple[int, int, int, int]:
    bb = obj.bounding_box_2d  # 4 corners A,B,C,D
    x1, y1 = bb[0][0], bb[0][1]
    x2, y2 = bb[2][0], bb[2][1]
    return int(x1), int(y1), int(x2), int(y2)

def bbox_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    denom = area_a + area_b - inter
    return float(inter) / float(denom) if denom > 0 else 0.0

def centroid_Zmean_and_reproject(mbin: np.ndarray, Zmap: np.ndarray, fx, fy, cx, cy):
    """
    - Compute 2D centroid of visible mask (mean of pixel coords).
    - Compute Zmean as median of Z over visible mask.
    - Reproject via intrinsics to CAMERA 3D.
      X = (u - cx)/fx * Z ; Y = (v - cy)/fy * Z
    Returns (u, v, X, Y, Z) or None.
    """
    ys, xs = np.where(mbin == 1)
    if ys.size < 3:
        return None
    Z = Zmap[ys, xs]
    valid = np.isfinite(Z)
    if valid.sum() < 3:
        return None
    u = float(xs[valid].mean())
    v = float(ys[valid].mean())
    Zm = float(np.median(Z[valid]))
    X = (u - cx) / fx * Zm
    Y = (v - cy) / fy * Zm
    return (int(round(u)), int(round(v))), (X, Y, Zm)

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
# Main
# =========================
def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections, loop_fps

    # --- ROS2 setup ---
    rclpy.init()
    node = rclpy.create_node('zed_date_detector_ros_hybrid')

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

    # Display / measures
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    xyz_full = sl.Mat()   # (H, W, 4): X,Y,Z,A in CAMERA
    depth_full = sl.Mat() # (H, W): depth in meters

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

    # Intrinsics (left camera)
    K = camera_config.calibration_parameters.left_cam
    fx, fy = float(K.fx), float(K.fy)
    cx, cy = float(K.cx), float(K.cy)

    cam_frame = 'zed2_left_camera_frame'  # Update if your TF uses a different frame
    HYBRID_TOL_M = 0.02                   # 2 cm tolerance between PIX_CAM and ZED_NATIVE

    t_loop_prev = time()

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

            # Measures & display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            zed.retrieve_measure(xyz_full, sl.MEASURE.XYZ, sl.MEM.CPU)     # CAMERA frame X,Y,Z
            zed.retrieve_measure(depth_full, sl.MEASURE.DEPTH, sl.MEM.CPU) # CAMERA depth Z
            xyz_np = xyz_full.get_data()                                   # (H, W, 4)
            depth_np = depth_full.get_data()                                # (H, W)

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
                masks, xyz_np, z_min=0.10, z_max=5.0, eps=0.003, erode_px=0
            )

            # Build mask entries with their bboxes
            mask_entries = []
            for mbin_vis, cls_i, conf_i in zip(visible_masks, labels, scores):
                bb_m = mask_bbox(mbin_vis)
                mask_entries.append({
                    'mask': mbin_vis,
                    'bbox': bb_m,
                    'cls': cls_i,
                    'conf': conf_i
                })

            # For each mask, find best matching ZED object (IoU on 2D bbox)
            used_obj_idx = set()
            for me in mask_entries:
                mbin = me['mask']
                bb_m = me['bbox']
                # Reproject PIX_CAM from mask Zmean + intrinsics
                pix = centroid_Zmean_and_reproject(mbin, depth_np, fx, fy, cx, cy)
                if pix is None:
                    continue
                (u, v), (Xp, Yp, Zp) = pix
                pix_cam = np.array([Xp, Yp, Zp], dtype=float)

                # find matching object
                best_j, best_iou = -1, 0.0
                for j, o in enumerate(objects.object_list):
                    if j in used_obj_idx:
                        continue
                    bb_o = obj_bbox_pixels(o)
                    iou = bbox_iou(bb_m, bb_o)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                chosen = 'PIX_CAM'
                final_cam = pix_cam.copy()
                native_cam = None

                if best_j >= 0 and best_iou >= 0.3:
                    used_obj_idx.add(best_j)
                    o = objects.object_list[best_j]
                    native_cam = np.array([float(o.position[0]),
                                           float(o.position[1]),
                                           float(o.position[2])], dtype=float)
                    # Compare
                    if np.all(np.isfinite(native_cam)):
                        d_err = np.linalg.norm(pix_cam - native_cam)
                        if d_err > HYBRID_TOL_M:
                            chosen = 'ZED_NATIVE'
                            final_cam = native_cam

                # Publish in base_link
                try:
                    point_msg = PointStamped()
                    point_msg.header.frame_id = cam_frame
                    point_msg.header.stamp = rclpy.time.Time().to_msg()
                    point_msg.point.x, point_msg.point.y, point_msg.point.z = final_cam.tolist()
                    pt_base = tf_buffer.transform(point_msg, 'base_link', timeout=rclpyDuration(seconds=0.2))

                    # Orientation from mask long axis (optional, simple PCA)
                    axis_cam = pca_long_axis_from_mask(mbin, xyz_np)
                    # Rotate to base_link
                    try:
                        tf_cb = tf_buffer.lookup_transform('base_link', cam_frame, rclpy.time.Time())
                        q = tf_cb.transform.rotation
                        # quaternion rotate axis_cam: (w,x,y,z)
                        # quick inline rotation
                        def qmul(q1, q2):
                            w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
                            return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
                                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                                    w1*z2 + x1*y2 - y1*x2 + z1*w2)
                        q_cb = (q.w, q.x, q.y, q.z)
                        q_axis = (0.0, axis_cam[0], axis_cam[1], axis_cam[2])
                        q_inv = (q_cb[0], -q_cb[1], -q_cb[2], -q_cb[3])
                        q_res = qmul(qmul(q_cb, q_axis), q_inv)
                        axis_base = _unit(np.array([q_res[1], q_res[2], q_res[3]], dtype=float))
                    except Exception:
                        axis_base = axis_cam

                    qw,qx,qy,qz = quat_align_x_to_axis(axis_base, up_hint=(0,0,1))

                    goal = PoseStamped()
                    goal.header = pt_base.header
                    goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = (
                        pt_base.point.x, pt_base.point.y, pt_base.point.z
                    )
                    goal.pose.orientation.w, goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z = (
                        qw,qx,qy,qz
                    )
                    goal_pub.publish(goal)

                    pt_pub = PointStamped()
                    pt_pub.header = pt_base.header
                    pt_pub.point.x, pt_pub.point.y, pt_pub.point.z = pt_base.point.x, pt_base.point.y, pt_base.point.z
                    point_pub.publish(pt_pub)

                    # Overlay debug on image
                    sx = int(u * image_scale[0]); sy = int(v * image_scale[1])
                    cv2.circle(image_left_ocv, (sx, sy), 4, (0, 255, 0, 255), -1)
                    if native_cam is not None and np.all(np.isfinite(native_cam)):
                        err = np.linalg.norm(pix_cam - native_cam)
                        txt = f"{chosen} Z={final_cam[2]:.2f}m (Δ={err*100:.0f}mm)"
                    else:
                        txt = f"{chosen} Z={final_cam[2]:.2f}m"
                    cv2.putText(image_left_ocv, txt, (sx + 6, max(0, sy - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0, 255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"[WARN] TF/publish failed: {e}")

            # 3D rendering viewer (point cloud + tracks)
            point_cloud.copy_to(point_cloud_render)
            viewer.updateData(point_cloud_render, objects)

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
            cv2.imshow("ZED | Hybrid 3D Position (PIX_CAM ⟷ ZED_NATIVE)", global_image)
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
    rclpy.shutdown()

# =========================
# Entry Point
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model.pt path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
