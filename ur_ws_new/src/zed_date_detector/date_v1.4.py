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
from rclpy.time import Time as rclpyTime
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
prev_axis = None
yolo_masks = []
yolo_classes = []
yolo_scores = []
yolo_lock = Lock()
# --- Persistent BEST fruit tracking ---
best_target_prev = None          # store previous best fruit
BEST_REUSE_THRESH = 0.05         # 5 cm positional tolerance in base_link

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

    R = np.array(
        [
            [X[0], Y[0], Z[0]],
            [X[1], Y[1], Z[1]],
            [X[2], Y[2], Z[2]],
        ],
        dtype=float,
    )
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
            mask_roi = mask_bin[y_min : y_max + 1, x_min : x_max + 1]
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
    model = YOLO(weights)
    model.to("cuda").eval()
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
    global image_net, exit_signal, run_signal, detections, loop_fps

    # --- ROS2 setup ---
    rclpy.init()
    node = rclpy.create_node("zed_date_detector_ros")
    executor = MultiThreadedExecutor(num_threads=2)
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
            # each target: {"Xc","Yc","Zc","bb","mask_resized","pt_base","pt_grip","dist"}
            targets = []

            for o in objects.object_list:
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
                    continue

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

                    # Resize mask to display ROI size
                    mask_resized = cv2.resize(
                        mask_local,
                        (w_roi, h_roi),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_bool = mask_resized > 0

                    # CLEAN MASK
                    kernel = np.ones((3, 3), np.uint8)
                    mask_clean = cv2.erode(mask_resized, kernel, iterations=1)
                    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

                    mask_bool = mask_clean > 0
                    # Take 3D points from XYZ map only where mask is true
                    roi_xyz = pc_np[y1:y2, x1:x2, :]  # H x W x 3
                    valid = np.isfinite(roi_xyz[:, :, 2])
                    valid &= mask_bool

                    if np.count_nonzero(valid) < 30:
                        # Not enough 3D points to trust (relaxed)
                        continue

                    pts = roi_xyz[valid]  # N x 3
                    zs = pts[:, 2]

                    # Closest 20% points (= front surface)
                    idx = np.argsort(zs)
                    k = max(10, int(0.2 * len(idx)))
                    pts_front = pts[idx[:k]]

                    Z_std = float(np.std(pts_front[:, 2]))
                    if Z_std > 0.05:  # >10 cm variance → unreliable (dense/occluded)
                        continue

                    Xc = float(np.mean(pts_front[:, 0]))
                    Yc = float(np.mean(pts_front[:, 1]))
                    Zc = float(np.mean(pts_front[:, 2]))

                    if not np.isfinite(Zc) or Zc <= 0.0 or Zc > 5.0:
                        continue

                    # Transform this fruit’s 3D point to base_link and gripper_tip
                    point_msg = PointStamped()
                    point_msg.header.frame_id = cam_frame
                    point_msg.header.stamp = rclpy.time.Time().to_msg()
                    point_msg.point.x = Xc
                    point_msg.point.y = Yc
                    point_msg.point.z = Zc

                    try:
                        pt_base = tf_buffer.transform(
                            point_msg,
                            "base_link",
                            timeout=rclpyDuration(seconds=0.2),
                        )

                        # Height limit in base_link
                        # if pt_base.point.z > Z_MAX:
                        #     continue

                        pt_grip = tf_buffer.transform(
                            point_msg,
                            "gripper_tip",
                            timeout=rclpyDuration(seconds=0.2),
                        )

                        # 3D distance from gripper tip
                        dist = math.sqrt(
                            pt_grip.point.x**2
                            + pt_grip.point.y**2
                            + pt_grip.point.z**2
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
                                "long_axis": None,
                                "approach_axis": None,
                                "pca_stable": None,
                                "reason": "",
                                "pts_front": pts_front,
                            }
                        )

                    except Exception as e:
                        print(f"[WARN] TF transform failed: {e}")
                        continue

                except Exception as e:
                    print(f"[WARN] 3D extraction failed: {e}")
                    continue

            # --------------------------------------------------
            # Hard-Sticky BEST Fruit Selection
            # --------------------------------------------------

            # 1. Pick closest fruit THIS FRAME (only used if we don't reuse previous)
            frame_best_idx = None
            frame_best_dist = float("inf")

            for i, t in enumerate(targets):
                if t["dist"] < frame_best_dist:
                    frame_best_dist = t["dist"]
                    frame_best_idx = i

            best_idx = frame_best_idx  # default


            # --------------------------------------------------
            # 2. Strong Reuse (sticky lock-on)
            # We ONLY switch if previous best is completely lost.
            # --------------------------------------------------

            global best_target_prev

            if best_target_prev is not None:

                prev_pt = np.array([
                    best_target_prev["pt_base"].point.x,
                    best_target_prev["pt_base"].point.y,
                    best_target_prev["pt_base"].point.z
                ], dtype=float)

                reuse_idx = None
                min_dist = float("inf")

                # Try to find the *same* fruit again by proximity
                for i, t in enumerate(targets):

                    cur_pt = np.array([
                        t["pt_base"].point.x,
                        t["pt_base"].point.y,
                        t["pt_base"].point.z
                    ], dtype=float)

                    d = np.linalg.norm(cur_pt - prev_pt)

                    # 🔥 MUCH STRONGER stickiness threshold
                    if d < BEST_REUSE_THRESH:      # recommended 0.04 → 4 cm
                        if d < min_dist:
                            min_dist = d
                            reuse_idx = i

                # If we found a matching fruit → ALWAYS reuse it
                if reuse_idx is not None:
                    best_idx = reuse_idx

                # ❗ If we DID NOT find the previous best at all → only then switch
                else:
                    # previous best is considered LOST → we must switch
                    best_idx = frame_best_idx


            # --------------------------------------------------
            # 3. Persistent state update
            # --------------------------------------------------
            if best_idx is not None:
                best_target_prev = targets[best_idx]


            # --------------------------------------------------
            # Run PCA only for the BEST target to reduce compute
            # --------------------------------------------------
            if best_idx is not None:
                t_best = targets[best_idx]
                pts_front = t_best.get("pts_front")

                long_axis = None
                pca_stable = False
                reason = ""
                S = np.array([0.0, 0.0, 0.0])

                if pts_front is not None and len(pts_front) > 0:
                    try:
                        pts_centered = pts_front - np.mean(pts_front, axis=0)
                        U, S, Vt = np.linalg.svd(pts_centered)
                        long_axis = Vt[0]
                        long_axis = long_axis / np.linalg.norm(long_axis)
                    except Exception:
                        long_axis = None

                    if pts_front.shape[0] < 80:
                        reason = "Not enough PCA points"
                    elif long_axis is None or S[0] < 1.5 * S[1]:
                        reason = "Fruit not elongated"
                    else:
                        pca_stable = True

                    global prev_axis
                    if "prev_axis" not in globals():
                        prev_axis = None

                    if pca_stable and prev_axis is not None and long_axis is not None:
                        dot = float(np.dot(prev_axis, long_axis))
                        if dot < -0.5:
                            long_axis = -long_axis
                            dot = -dot
                        if dot < 0.2:
                            pca_stable = False
                            reason = "Axis direction unstable"

                    if long_axis is not None:
                        prev_axis = long_axis.copy()
                    else:
                        prev_axis = None

                fruit_cam = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]], dtype=float)
                n = np.linalg.norm(fruit_cam)
                if pca_stable and long_axis is not None:
                    approach_axis = long_axis
                elif n < 1e-6:
                    approach_axis = np.array([0, 0, 1], dtype=float)
                else:
                    approach_axis = fruit_cam / n

                q = quat_align_x_to_axis(approach_axis)
                t_best.update(
                    {
                        "long_axis": long_axis,
                        "approach_axis": approach_axis,
                        "pca_stable": pca_stable,
                        "reason": reason,
                        "quat": q,
                    }
                )

            # --------------------------------------------------
            # Draw all targets (centroids, boxes, masks)
            # Best fruit centroid = BLUE dot + “BEST”
            # Others = RED dots
            # --------------------------------------------------
            for i, t in enumerate(targets):
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

                # Visualize PCA/approach axis projected to image
                # -------------------------------------------------
                # VISUALIZE PCA / APPROACH AXIS (2D IMAGE ARROW)
                # -------------------------------------------------
                axis_dir = t.get("approach_axis")
                if axis_dir is None:
                    axis_dir = t.get("long_axis")

                if axis_dir is not None:
                    ax, ay, az = axis_dir

                    # convert 3D axis → 2D direction (drop Z)
                    vx = ax
                    vy = ay

                    # normalize 2D vector
                    n = math.sqrt(vx*vx + vy*vy)
                    if n < 1e-6:
                        vx, vy = 1.0, 0.0
                    else:
                        vx /= n
                        vy /= n

                    # arrow length in pixels
                    L = 60

                    ax2 = int(cx + vx * L)
                    ay2 = int(cy + vy * L)
                    ax1 = int(cx - vx * L)
                    ay1 = int(cy - vy * L)

                    # stable PCA → yellow, unstable → orange
                    axis_color = (
                        (0, 255, 255, 255) if t.get("pca_stable") else (0, 128, 255, 255)
                    )

                    # forward arrow
                    cv2.arrowedLine(image_left_ocv, (cx, cy), (ax2, ay2), axis_color, 2, tipLength=0.25)
                    cv2.line(image_left_ocv, (cx, cy), (ax1, ay1), axis_color, 2)


                    # show instability reason
                    if not t.get("pca_stable") and t.get("reason"):
                        cv2.putText(
                            image_left_ocv,
                            t["reason"],
                            (cx + 12, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            axis_color,
                            1,
                            cv2.LINE_AA,
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

                # BEST label for the chosen fruit
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

            # --------------------------------------------------
            # Publish goal for BEST fruit only
            if best_idx is not None:
                t_best = targets[best_idx]
                pt_base = t_best["pt_base"]
                q = t_best["quat"]

                try:
                    if pt_base.point.z <= Z_MAX:
                        goal = PoseStamped()
                        goal.header = pt_base.header

                        # direct fruit position
                        goal.pose.position.x = float(pt_base.point.x)
                        goal.pose.position.y = float(pt_base.point.y)
                        goal.pose.position.z = float(pt_base.point.z)

                        # orientation
                        goal.pose.orientation.w = q[0]
                        goal.pose.orientation.x = q[1]
                        goal.pose.orientation.y = q[2]
                        goal.pose.orientation.z = q[3]

                        goal_pub.publish(goal)
                        point_pub.publish(pt_base)

                except Exception as e:
                    print(f"[WARN] Publish failed: {e}")

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
            key = cv2.waitKey(10)
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
        "--img_size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.4, help="confidence threshold"
    )
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
