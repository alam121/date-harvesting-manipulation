#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from threading import Lock, Thread, Event
from time import sleep, time
from typing import List
from collections import deque
import ogl_viewer.viewer as gl
import tf2_geometry_msgs
import cv_viewer.tracking_viewer as cv_viewer

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
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
run_event = Event()
dets_ready = Event()
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None

net_fps = 0.0
loop_fps = 0.0
prev_heat_point = None
prev_direction_base = None  # smoothed 3D direction in base_link
direction_history = deque(maxlen=10)  # sliding window for direction averaging

# --- Timer-based publishing data ---
latest_goal_msg = None
latest_dir_msg = None
pub_lock = Lock()
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
# Scoring System Weights (tune these for your application)
# ============================================================
SCORE_WEIGHTS = {
    "distance": 0.28,       # closer is better (normalized: 0-1)
    "visibility": 0.23,     # higher vis_ratio is better
    "depth_quality": 0.18,  # lower z_std is better
    "confidence": 0.13,     # YOLO detection confidence
    "ellipse": 0.10,        # bonus for valid ellipse fit (orientation reliability)
    "center_bias": 0.08,    # prefer fruits near frame center (better depth data)
}

# Distance scoring parameters
DIST_MIN = 0.10  # best possible distance (m)
DIST_MAX = 1.50  # worst acceptable distance (m)

# Depth quality parameters
Z_STD_IDEAL = 0.005   # ideal depth std (m)
Z_STD_WORST = 0.05    # worst acceptable depth std (m)

# Sticky bonus: how much to prefer the previous best fruit
STICKY_BONUS = 0.15   # added to score if this was the previous best

# Collision avoidance parameters
FRUIT_RADIUS = 0.035  # approximate radius of a date fruit (3.5cm)
APPROACH_CHECK_DIST = 0.15  # how far back to check for collisions (15cm)
NUM_CANDIDATE_DIRS = 12  # number of directions to sample


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
    """Rotate vector v3 by quaternion q."""
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (q[0], -q[1], -q[2], -q[3])
    return quat_mul(quat_mul(q, qv), qi)[1:]


def quat_align_x_to_axis(axis_world, up_hint=(0, 0, 1)):
    """
    Build quaternion that maps robot's +X axis to the given axis_world,
    with up_hint used to resolve roll.
    """
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


def compute_fruit_score(target: dict, prev_pt_base: np.ndarray = None) -> dict:
    """
    Compute a comprehensive score for a detected fruit.

    Returns dict with:
        - total_score: weighted sum of all factors (0-1, higher is better)
        - components: individual score components for debugging
    """
    components = {}

    # 1. Distance score (closer = better)
    dist = target.get("dist", float("inf"))
    if dist <= DIST_MIN:
        dist_score = 1.0
    elif dist >= DIST_MAX:
        dist_score = 0.0
    else:
        dist_score = 1.0 - (dist - DIST_MIN) / (DIST_MAX - DIST_MIN)
    components["distance"] = dist_score

    # 2. Visibility score (higher vis_ratio = better)
    vis_ratio = target.get("vis_ratio", 0.0)
    vis_score = max(0.0, min(1.0, vis_ratio))
    components["visibility"] = vis_score

    # 3. Depth quality score (lower z_std = better)
    z_std = target.get("z_std", Z_STD_WORST)
    if z_std <= Z_STD_IDEAL:
        depth_score = 1.0
    elif z_std >= Z_STD_WORST:
        depth_score = 0.0
    else:
        depth_score = 1.0 - (z_std - Z_STD_IDEAL) / (Z_STD_WORST - Z_STD_IDEAL)
    components["depth_quality"] = depth_score

    # 4. Detection confidence score
    confidence = target.get("confidence", 0.5)
    conf_score = max(0.0, min(1.0, confidence))
    components["confidence"] = conf_score

    # 5. Ellipse quality score (bonus for valid orientation)
    has_ellipse = target.get("short_axis_cam") is not None
    ellipse_score = 1.0 if has_ellipse else 0.3  # partial credit if no ellipse
    components["ellipse"] = ellipse_score

    # 6. Center bias score (fruits near frame center have better depth data)
    bb = target.get("bb")
    img_w = target.get("img_width", 1280)
    img_h = target.get("img_height", 720)
    if bb is not None:
        x1, y1, x2, y2 = bb
        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0
        dist_from_center = math.hypot(bbox_cx - img_cx, bbox_cy - img_cy)
        max_dist = math.hypot(img_cx, img_cy)
        center_score = 1.0 - (dist_from_center / max_dist) if max_dist > 0 else 0.5
    else:
        center_score = 0.5  # neutral if no bbox
    components["center_bias"] = center_score

    # Weighted sum
    total = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        total += weight * components.get(key, 0.0)

    # Sticky bonus: if this fruit matches the previous best, add bonus
    if prev_pt_base is not None and target.get("pt_base") is not None:
        cur_pt = np.array([
            target["pt_base"].point.x,
            target["pt_base"].point.y,
            target["pt_base"].point.z,
        ], dtype=float)
        dist_to_prev = np.linalg.norm(cur_pt - prev_pt_base)
        if dist_to_prev < BEST_REUSE_THRESH:
            total += STICKY_BONUS
            components["sticky_bonus"] = STICKY_BONUS

    # Clamp final score
    total = max(0.0, min(1.0 + STICKY_BONUS, total))

    return {"total_score": total, "components": components}


def ray_sphere_intersection(ray_origin, ray_dir, sphere_center, sphere_radius):
    """
    Check if a ray intersects a sphere.
    Returns (hit, t_enter) where t_enter is distance along ray to first intersection.
    Returns (False, inf) if no intersection.
    """
    oc = ray_origin - sphere_center
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4.0 * c

    if discriminant < 0:
        return False, float('inf')

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    # Return nearest positive intersection
    if t1 > 0.001:
        return True, t1
    elif t2 > 0.001:
        return True, t2
    return False, float('inf')


def compute_collision_free_direction(best_target, all_targets, best_idx, heatmap_dir=None):
    """
    Find the approach direction with most clearance from other fruits.

    Strategy:
    1. Sample directions in a hemisphere (toward camera = -Z in camera frame)
    2. For each direction, check clearance to other fruits
    3. Blend best clearance direction with heatmap direction for grasp accuracy

    Args:
        best_target: The target fruit we want to approach
        all_targets: List of all detected fruits
        best_idx: Index of best_target in all_targets
        heatmap_dir: Original heatmap-based direction (optional, for blending)

    Returns:
        dict with:
            - direction: 3D unit vector for approach direction in camera frame
            - clearance: Distance to nearest obstacle
            - is_collision_free: Whether the chosen direction is clear
    """
    centroid = np.array([best_target["Xc"], best_target["Yc"], best_target["Zc"]])

    # Collect other fruits as spheres
    other_spheres = []
    for i, t in enumerate(all_targets):
        if i == best_idx:
            continue
        other_c = np.array([t["Xc"], t["Yc"], t["Zc"]])
        other_spheres.append((other_c, FRUIT_RADIUS))

    # If no other fruits, just use heatmap direction
    if len(other_spheres) == 0:
        if heatmap_dir is not None:
            return {
                "direction": heatmap_dir,
                "clearance": float('inf'),
                "is_collision_free": True,
                "num_blocked": 0,
            }
        else:
            return {
                "direction": np.array([0.0, 0.0, -1.0]),  # default: toward camera
                "clearance": float('inf'),
                "is_collision_free": True,
                "num_blocked": 0,
            }

    # Generate candidate directions on a hemisphere (facing camera = -Z)
    candidate_dirs = []

    # Sample azimuth angles around Z axis
    for i in range(NUM_CANDIDATE_DIRS):
        azimuth = 2.0 * math.pi * i / NUM_CANDIDATE_DIRS

        # Multiple elevation angles (0 = horizontal, positive = toward camera)
        for elev_deg in [0, 20, 40, 60]:
            elev = math.radians(elev_deg)
            x = math.cos(azimuth) * math.cos(elev)
            y = math.sin(azimuth) * math.cos(elev)
            z = -math.sin(elev)  # negative Z = toward camera
            candidate_dirs.append(np.array([x, y, z], dtype=float))

    # Add straight toward camera
    candidate_dirs.append(np.array([0.0, 0.0, -1.0]))

    # Add the heatmap direction as a candidate (if available)
    if heatmap_dir is not None:
        candidate_dirs.append(heatmap_dir.copy())

    # Score each direction by clearance
    best_dir = None
    best_clearance = -1.0
    best_blocked = 0

    for d in candidate_dirs:
        d = d / (np.linalg.norm(d) + 1e-9)

        # Cast ray from centroid in this direction
        # Check for intersections with other fruit spheres
        min_clearance = float('inf')
        num_blocked = 0

        for (sphere_c, sphere_r) in other_spheres:
            hit, t_hit = ray_sphere_intersection(centroid, d, sphere_c, sphere_r)

            if hit and t_hit < APPROACH_CHECK_DIST:
                num_blocked += 1
                min_clearance = min(min_clearance, t_hit)
            elif not hit:
                # Compute closest approach distance
                # Project sphere center onto ray
                to_sphere = sphere_c - centroid
                t_closest = np.dot(to_sphere, d)
                if t_closest > 0:  # sphere is in front
                    closest_pt = centroid + t_closest * d
                    dist_to_center = np.linalg.norm(closest_pt - sphere_c)
                    clearance_at_closest = dist_to_center - sphere_r
                    if clearance_at_closest < min_clearance:
                        min_clearance = max(0.0, clearance_at_closest)

        # Prefer directions with higher clearance
        if min_clearance > best_clearance:
            best_clearance = min_clearance
            best_dir = d.copy()
            best_blocked = num_blocked

    # If heatmap direction has decent clearance, blend with it for grasp accuracy
    if heatmap_dir is not None and best_clearance > FRUIT_RADIUS:
        # Check heatmap direction clearance
        hm_clearance = float('inf')
        hm_d = heatmap_dir / (np.linalg.norm(heatmap_dir) + 1e-9)
        for (sphere_c, sphere_r) in other_spheres:
            hit, t_hit = ray_sphere_intersection(centroid, hm_d, sphere_c, sphere_r)
            if hit:
                hm_clearance = min(hm_clearance, t_hit)

        # If heatmap direction is also clear, blend toward it
        if hm_clearance > FRUIT_RADIUS * 2:
            # Blend: 60% collision-free, 40% heatmap for grasp accuracy
            blended = 0.6 * best_dir + 0.4 * hm_d
            blended = blended / (np.linalg.norm(blended) + 1e-9)
            best_dir = blended

    return {
        "direction": best_dir,
        "clearance": best_clearance,
        "is_collision_free": best_clearance > FRUIT_RADIUS,
        "num_blocked": best_blocked,
    }


def debug_print(msg: str) -> None:
    """Lightweight debug printer with flush."""
    print(f"[DEBUG] {msg}", flush=True)


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
    global image_net, exit_signal, run_event, dets_ready, detections, net_fps
    print("Initializing Network...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YOLO; GPU not available.")
    device = torch.device("cuda")
    model = YOLO(weights)
    model.to(device).eval()
    print("Network Initialized...")
    while not exit_signal:
        if run_event.is_set():
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
            run_event.clear()
            dets_ready.set()
        sleep(0.005)

# ============================================================
# Main
# ============================================================
def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_event, dets_ready, detections, loop_fps, best_target_prev, prev_heat_point, prev_direction_base, direction_history, latest_goal_msg, latest_dir_msg

    # --- ROS2 setup ---
    rclpy.init()
    node = rclpy.create_node("zed_date_detector_ros")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    fast_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )
    goal_pub = node.create_publisher(PoseStamped, "/external_goal_pose", fast_qos)
    dir_pub = node.create_publisher(Vector3Stamped, "/datefruit_direction", 10)

    # Timer-based publishing callback (50Hz)
    def publish_timer_cb():
        global latest_goal_msg, latest_dir_msg
        with pub_lock:
            if latest_goal_msg is not None:
                # Update timestamp for fresh publish
                latest_goal_msg.header.stamp = node.get_clock().now().to_msg()
                goal_pub.publish(latest_goal_msg)
            if latest_dir_msg is not None:
                latest_dir_msg.header.stamp = node.get_clock().now().to_msg()
                dir_pub.publish(latest_dir_msg)

    pub_timer = node.create_timer(0.02, publish_timer_cb)  # 50Hz

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

    Z_MAX = 1.34  # limit in base_link frame
    last_viz = 0.0

    def perception_loop():
        global exit_signal, run_event, dets_ready, image_net, detections, loop_fps, best_target_prev, prev_heat_point, prev_direction_base, latest_goal_msg, latest_dir_msg
        nonlocal last_viz
        t_prev = time()
        while not exit_signal:
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
                run_event.set()
            # Only ingest/process when YOLO thread produced new detections
            if not dets_ready.is_set():
                continue
            dets_ready.clear()

            # Ingest YOLO detections into ZED
            with lock:
                current_dets = detections
            if current_dets is None:
                continue
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
                        t_best_point_3d = None
                        if heatmap is not None and mask_clean is not None:
                            ys, xs = np.nonzero(mask_clean)
                            if len(xs) > 0:
                                scores = heatmap[ys, xs, 2].astype(float)  # use RED channel
                                idx_max = int(np.argmax(scores))
                                peak_y = int(ys[idx_max])
                                peak_x = int(xs[idx_max])
                                t_best_point = np.array([peak_x, peak_y], dtype=float)

                                # Get 3D coordinates of peak point from point cloud
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

                        # Get detection confidence from ZED object
                        obj_confidence = getattr(o, "confidence", 0.5) / 100.0  # ZED uses 0-100
                        if not (0.0 <= obj_confidence <= 1.0):
                            obj_confidence = 0.5

                        targets.append(
                            {
                                "Xc": Xc,
                                "Yc": Yc,
                                "Zc": Zc,
                                "bb": (x1, y1, x2, y2),
                                "img_width": display_resolution.width,
                                "img_height": display_resolution.height,
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
                                "confidence": obj_confidence,
                                "short_axis_cam": t_short_axis,
                                "long_axis_2d": long_axis_2d,
                                "ellipse_angle": t_angle,
                                "heatmap": heatmap,
                                "best_dir2d": t_best_dir2d,
                                "best_point2d": t_best_point,
                                "best_point_3d": t_best_point_3d,
                                "score": 0.0,  # will be computed below
                                "score_components": {},
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
            # Multi-Factor BEST Fruit Selection (Scoring System)
            # --------------------------------------------------

            # Get previous best position for sticky bonus calculation
            prev_pt_base = None
            if best_target_prev is not None:
                prev_pt_base = np.array(
                    [
                        best_target_prev["pt_base"].point.x,
                        best_target_prev["pt_base"].point.y,
                        best_target_prev["pt_base"].point.z,
                    ],
                    dtype=float,
                )

            # Compute scores for all targets
            best_idx = None
            best_score = -1.0

            for i, t in enumerate(targets):
                score_result = compute_fruit_score(t, prev_pt_base)
                t["score"] = score_result["total_score"]
                t["score_components"] = score_result["components"]

                if score_result["total_score"] > best_score:
                    best_score = score_result["total_score"]
                    best_idx = i

            # Persistent state update
            if best_idx is not None:
                best_target_prev = targets[best_idx]

            # --------------------------------------------------
            # Pre-processing for best fruit (TF cache, heatmap smoothing)
            # --------------------------------------------------
            if best_idx is not None:
                t_best = targets[best_idx]

                # Cache TF lookup once for this frame (used for direction transform)
                cached_q_tf = None
                try:
                    T = tf_buffer.lookup_transform("base_link", cam_frame, rclpyTime())
                    cached_q_tf = (
                        T.transform.rotation.w,
                        T.transform.rotation.x,
                        T.transform.rotation.y,
                        T.transform.rotation.z,
                    )
                except Exception:
                    pass

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

            # --------------------------------------------------
            # Publish goal for BEST fruit only (before rendering to minimize latency)
            if best_idx is not None:
                t_best = targets[best_idx]
                pt_base = t_best["pt_base"]
                q = t_best["quat"]
                # Compute 3D direction in base_link frame (from mask center to closest point)
                dir_msg = None
                raw_pt = t_best.get("best_point2d")

                if raw_pt is not None:
                    # Compute direction from mask center to peak point
                    x1, y1, x2, y2 = t_best["bb"]
                    roi_w = x2 - x1
                    roi_h = y2 - y1
                    cx = roi_w / 2.0
                    cy = roi_h / 2.0
                    dx = float(raw_pt[0]) - cx
                    dy = float(raw_pt[1]) - cy

                    # Magnitude check: if peak is too close to center, direction is unreliable
                    min_offset = 0.1 * min(roi_w, roi_h)
                    offset_mag = math.hypot(dx, dy)

                    if offset_mag > min_offset:
                        # Use TRUE 3D direction from centroid to peak point
                        best_pt_3d = t_best.get("best_point_3d")
                        centroid_3d = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]])

                        # First compute heatmap-based direction
                        heatmap_dir = None
                        if best_pt_3d is not None:
                            heatmap_dir = best_pt_3d - centroid_3d
                            n_dir_cam = np.linalg.norm(heatmap_dir)
                            if n_dir_cam > 1e-6:
                                heatmap_dir = heatmap_dir / n_dir_cam
                            else:
                                heatmap_dir = np.array([dx / offset_mag, dy / offset_mag, 0.0], dtype=float)
                        else:
                            heatmap_dir = np.array([dx / offset_mag, dy / offset_mag, 0.0], dtype=float)

                        # Compute collision-free direction (avoids other fruits)
                        collision_result = compute_collision_free_direction(
                            t_best, targets, best_idx, heatmap_dir
                        )
                        dir_cam = collision_result["direction"]

                        # Store collision info for visualization
                        t_best["clearance"] = collision_result["clearance"]
                        t_best["is_collision_free"] = collision_result["is_collision_free"]
                        t_best["heatmap_dir_cam"] = heatmap_dir.copy()

                        # Store the 3D direction in camera frame for visualization
                        t_best["approach_dir_cam"] = dir_cam.copy()

                        # Transform direction to base_link frame using cached TF rotation
                        if cached_q_tf is not None:
                            dir_raw = np.array(quat_rotate_vec(cached_q_tf, dir_cam), dtype=float)
                        else:
                            dir_raw = dir_cam.copy()

                        # Flip to be consistent with history
                        if prev_direction_base is not None:
                            if float(np.dot(prev_direction_base, dir_raw)) < 0:
                                dir_raw = -dir_raw

                        # Add to sliding window history
                        direction_history.append(dir_raw.copy())

                    # Compute averaged direction from history (weighted, newer = more weight)
                    if len(direction_history) > 0:
                        weights = np.arange(1, len(direction_history) + 1, dtype=float) ** 2
                        stacked = np.vstack(list(direction_history))
                        dir_avg = (stacked * weights[:, None]).sum(axis=0) / weights.sum()

                        # Normalize
                        n_avg = np.linalg.norm(dir_avg)
                        if n_avg > 1e-6:
                            dir_avg /= n_avg
                        else:
                            dir_avg = np.array([1.0, 0.0, 0.0])

                        prev_direction_base = dir_avg.copy()

                        # Store base_link direction for visualization
                        t_best["approach_dir_base"] = dir_avg.copy()

                        # Create Vector3Stamped message
                        dir_msg = Vector3Stamped()
                        dir_msg.header.frame_id = "base_link"
                        dir_msg.header.stamp = node.get_clock().now().to_msg()
                        dir_msg.vector.x = float(dir_avg[0])
                        dir_msg.vector.y = float(dir_avg[1])
                        dir_msg.vector.z = float(dir_avg[2])

                        # ---- UNIFIED: Derive orientation FROM direction ----
                        # For finger gripper: orient perpendicular to approach direction
                        # Cross with Z-up to get horizontal perpendicular axis
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

                # Fallback orientation if no valid direction computed
                if t_best.get("quat") is None:
                    t_best["quat"] = (1.0, 0.0, 0.0, 0.0)
                    t_best["approach_axis"] = np.array([1.0, 0.0, 0.0])

                # Track-by-detection smoothing: weighted avg of recent centroids
                pt_vec = np.array(
                    [pt_base.point.x, pt_base.point.y, pt_base.point.z], dtype=float
                )

                # Detect large position jump → reset tracking (new fruit)
                if len(best_history) > 0:
                    last_pt = best_history[-1]
                    jump_dist = np.linalg.norm(pt_vec - last_pt)
                    if jump_dist > 0.15:  # >15cm jump = new fruit, clear history
                        best_history.clear()
                        best_target_prev = None  # remove sticky bonus
                        #print(f"[VISION] Position jump {jump_dist:.2f}m - reset tracking")

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

                if pt_z <= Z_MAX:
                    goal = PoseStamped()
                    goal.header = pt_base.header

                    # direct fruit position
                    goal.pose.position.x = float(pt_x)
                    goal.pose.position.y = float(pt_y)
                    goal.pose.position.z = float(pt_z)

                    # orientation derived from approach direction (unified)
                    goal.pose.orientation.w = q[0]
                    goal.pose.orientation.x = q[1]
                    goal.pose.orientation.y = q[2]
                    goal.pose.orientation.z = q[3]

                    # Update global messages for timer-based publishing
                    with pub_lock:
                        latest_goal_msg = goal
                        latest_dir_msg = dir_msg
                        # debug_print(
                        #     f"New goal set: ({pt_x:.3f}, {pt_y:.3f}, {pt_z:.3f}), score {t_best.get('score', 0.0):.2f}"
                        # )
            else:
                best_history.clear()

            # --------------------------------------------------
            # Draw all targets (centroids, boxes, masks)
            # Best fruit centroid = BLUE dot + “BEST”
            # Others = RED dots
            # --------------------------------------------------
            now = time()
            if not SKIP_DRAW and (now - last_viz) >= 0.1:
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

                # Draw blocking fruits with transparent red overlay
                if best_idx is not None and len(targets) > 1:
                    t_best = targets[best_idx]
                    best_centroid = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]])
                    approach_dir = t_best.get("approach_dir_cam")

                    for i, t in enumerate(targets):
                        if i == best_idx:
                            continue

                        other_centroid = np.array([t["Xc"], t["Yc"], t["Zc"]])
                        x1o, y1o, x2o, y2o = t["bb"]

                        # Check if this fruit blocks the approach path
                        is_blocking = False
                        if approach_dir is not None:
                            # Vector from best to other fruit
                            to_other = other_centroid - best_centroid
                            dist_to_other = np.linalg.norm(to_other)

                            if dist_to_other > 0.01:
                                # Project onto approach direction
                                proj_dist = np.dot(to_other, approach_dir)

                                # Check if fruit is in front along approach direction
                                if proj_dist > 0 and proj_dist < APPROACH_CHECK_DIST:
                                    # Check perpendicular distance
                                    perp_vec = to_other - proj_dist * approach_dir
                                    perp_dist = np.linalg.norm(perp_vec)

                                    if perp_dist < FRUIT_RADIUS * 2:
                                        is_blocking = True

                        # Apply transparent overlay based on blocking status
                        if x2o > x1o and y2o > y1o:
                            roi = image_left_ocv[y1o:y2o, x1o:x2o]
                            if roi.size:
                                h_roi, w_roi = roi.shape[:2]
                                if is_blocking:
                                    # Red transparent overlay for blocking fruits
                                    red_overlay = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                                    red_overlay[:, :, 2] = 180  # Red channel
                                    red_overlay[:, :, 3] = 100  # Alpha
                                    blended = cv2.addWeighted(red_overlay, 0.4, roi, 0.6, 0.0)
                                    image_left_ocv[y1o:y2o, x1o:x2o] = blended

                                    # Draw line from best fruit to blocking fruit
                                    best_2d = project_point_to_image(best_centroid, intrinsics, image_scale)
                                    other_2d = project_point_to_image(other_centroid, intrinsics, image_scale)
                                    if best_2d and other_2d:
                                        cv2.line(image_left_ocv, best_2d, other_2d,
                                                 (0, 0, 200, 255), 2, cv2.LINE_AA)

                                    # Mark as BLOCKED
                                    cv2.putText(
                                        image_left_ocv,
                                        "BLOCKED",
                                        (x1o + 5, y1o + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (0, 0, 255, 255),
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

                            axis_color = (0, 255, 255, 255)

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

                        # VISUALIZE TRUE 3D APPROACH DIRECTION with coordinate frame
                        approach_dir_cam = t.get("approach_dir_cam")
                        if approach_dir_cam is not None:
                            centroid_3d = np.array([Xc, Yc, Zc])
                            axis_len = 0.05  # 5cm for coordinate axes
                            arrow_len_3d = 0.08  # 8cm for approach arrow

                            # Project centroid (origin)
                            origin_2d = project_point_to_image(centroid_3d, intrinsics, image_scale)

                            if origin_2d is not None:
                                # Draw RGB coordinate frame at centroid
                                # X axis (Red) - right
                                x_end_3d = centroid_3d + np.array([axis_len, 0, 0])
                                x_end_2d = project_point_to_image(x_end_3d, intrinsics, image_scale)
                                if x_end_2d:
                                    cv2.arrowedLine(image_left_ocv, origin_2d, x_end_2d,
                                                    (0, 0, 255, 255), 2, tipLength=0.3)
                                    cv2.putText(image_left_ocv, "X", x_end_2d,
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)

                                # Y axis (Green) - down
                                y_end_3d = centroid_3d + np.array([0, axis_len, 0])
                                y_end_2d = project_point_to_image(y_end_3d, intrinsics, image_scale)
                                if y_end_2d:
                                    cv2.arrowedLine(image_left_ocv, origin_2d, y_end_2d,
                                                    (0, 255, 0, 255), 2, tipLength=0.3)
                                    cv2.putText(image_left_ocv, "Y", y_end_2d,
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0, 255), 1)

                                # Z axis (Blue) - forward/depth
                                z_end_3d = centroid_3d + np.array([0, 0, axis_len])
                                z_end_2d = project_point_to_image(z_end_3d, intrinsics, image_scale)
                                if z_end_2d:
                                    cv2.arrowedLine(image_left_ocv, origin_2d, z_end_2d,
                                                    (255, 0, 0, 255), 2, tipLength=0.3)
                                    cv2.putText(image_left_ocv, "Z", z_end_2d,
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0, 255), 1)

                                # Draw approach direction arrow (Magenta, thicker)
                                dir_end_3d = centroid_3d + approach_dir_cam * arrow_len_3d
                                dir_end_2d = project_point_to_image(dir_end_3d, intrinsics, image_scale)

                                if dir_end_2d:
                                    # Draw XY projection (dashed line) to show lateral component
                                    dir_xy = approach_dir_cam.copy()
                                    dir_xy[2] = 0  # zero out Z
                                    n_xy = np.linalg.norm(dir_xy)
                                    if n_xy > 0.01:
                                        dir_xy_end_3d = centroid_3d + (dir_xy / n_xy) * arrow_len_3d * n_xy
                                        dir_xy_end_2d = project_point_to_image(dir_xy_end_3d, intrinsics, image_scale)
                                        if dir_xy_end_2d:
                                            # Dashed line for XY projection (cyan)
                                            cv2.line(image_left_ocv, origin_2d, dir_xy_end_2d,
                                                     (255, 255, 0, 255), 1, cv2.LINE_AA)
                                            # Vertical line showing Z component (white dashed)
                                            cv2.line(image_left_ocv, dir_xy_end_2d, dir_end_2d,
                                                     (255, 255, 255, 255), 1, cv2.LINE_AA)

                                    # Main approach arrow (thick magenta)
                                    cv2.arrowedLine(image_left_ocv, origin_2d, dir_end_2d,
                                                    (255, 0, 255, 255), 3, tipLength=0.25)

                                # Show camera frame direction (magenta)
                                dir_x, dir_y, dir_z = approach_dir_cam
                                cv2.putText(
                                    image_left_ocv,
                                    f"Cam: ({dir_x:.2f}, {dir_y:.2f}, {dir_z:.2f})",
                                    (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (255, 0, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

                                # Show base_link direction - THIS IS WHAT'S SENT TO ROBOT (green)
                                approach_dir_base = t.get("approach_dir_base")
                                if approach_dir_base is not None:
                                    bx, by, bz = approach_dir_base
                                    # base_link: X=forward, Y=left, Z=up
                                    x_lbl = "FWD" if bx > 0.1 else ("BACK" if bx < -0.1 else "")
                                    y_lbl = "L" if by > 0.1 else ("R" if by < -0.1 else "")
                                    z_lbl = "UP" if bz > 0.1 else ("DN" if bz < -0.1 else "")
                                    base_labels = " ".join(filter(None, [x_lbl, y_lbl, z_lbl]))
                                    if not base_labels:
                                        base_labels = "CENTER"

                                    cv2.putText(
                                        image_left_ocv,
                                        f"Robot: ({bx:.2f}, {by:.2f}, {bz:.2f})",
                                        (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (0, 255, 0, 255),  # green for robot/base_link
                                        1,
                                        cv2.LINE_AA,
                                    )
                                    cv2.putText(
                                        image_left_ocv,
                                        f">> {base_labels}",
                                        (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0, 255),  # green
                                        2,
                                        cv2.LINE_AA,
                                    )

                                # Show clearance info (collision avoidance status)
                                clearance = t.get("clearance", float('inf'))
                                is_clear = t.get("is_collision_free", True)
                                if clearance < float('inf'):
                                    clr_color = (0, 255, 0, 255) if is_clear else (0, 0, 255, 255)
                                    clr_text = f"CLEAR {clearance*100:.1f}cm" if is_clear else f"BLOCKED {clearance*100:.1f}cm"
                                    cv2.putText(
                                        image_left_ocv,
                                        clr_text,
                                        (x1, y1 - 55),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45,
                                        clr_color,
                                        2 if not is_clear else 1,
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
                        dist_grip = t.get("dist", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"dist_grip:{dist_grip:.3f}m",
                            (cx + 10, cy + 52),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 200, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        # Display total score prominently
                        total_score = t.get("score", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"SCORE: {total_score:.2f}",
                            (cx + 10, cy + 68),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0, 255),  # bright green
                            2,
                            cv2.LINE_AA,
                        )
                        # Show score breakdown
                        sc = t.get("score_components", {})
                        conf = t.get("confidence", 0.0)
                        cv2.putText(
                            image_left_ocv,
                            f"D:{sc.get('distance', 0):.2f} V:{sc.get('visibility', 0):.2f} Z:{sc.get('depth_quality', 0):.2f} C:{conf:.2f}",
                            (cx + 10, cy + 84),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (180, 180, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image_left_ocv,
                            f"E:{sc.get('ellipse', 0):.2f} Ctr:{sc.get('center_bias', 0):.2f}",
                            (cx + 10, cy + 98),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (180, 180, 255, 255),
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
            last_viz = now

    perception_thread = Thread(target=perception_loop, daemon=True)
    perception_thread.start()

    try:
        rclpy.spin(node)
    finally:
        exit_signal = True
        perception_thread.join(timeout=1.0)
        zed.close()
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
