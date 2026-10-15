"""
Trunk-based hand-eye calibration for ZED2 + UR10e (eye-in-hand).

No chessboard needed — uses the palm tree trunk as a static reference point.

Principle:
  T_base2gripper(i)  @  T_gripper2cam  @  p_trunk_cam(i)  =  p_trunk_base  (constant)

  For the correct T_gripper2cam, transforming the trunk from camera frame to
  base_link must give the same 3D point at every arm pose.
  We minimise the variance of those projected points across all poses.

Usage:
  1. Make sure the trunk is visible from multiple arm poses.
  2. Run:   python3 trunk_calibration.py
  3. The arm moves automatically, captures trunk positions, optimises,
     and overwrites hand_eye_calibration.yaml.

Tune CALIBRATION_POSES so the trunk is fully in frame at every pose.
"""

import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.time import Time as rclpyTime
from rclpy.duration import Duration as rclpyDuration
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import pyzed.sl as sl
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = Path(__file__).resolve().parent / "hand_eye_calibration.yaml"
BASE_FRAME    = "base_link"
EE_FRAME      = "tool0"
CAM_FRAME     = "zed2_left_camera_frame"
MOVE_DURATION = 6.0    # seconds per move
SETTLE_TIME   = 1.5    # seconds to wait after move
DETECT_TRIES  = 20     # number of ZED grabs to average trunk position per pose
TRUNK_FRONT_K = 0.25   # use front 25% of depth points (same as vision node)
MIN_TRUNK_PTS = 15     # minimum depth points in trunk bbox to accept sample

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]

# Poses where the trunk is visible — vary pan/tilt widely for good conditioning.
# Edit these to match your setup.
CALIBRATION_POSES = [
    [-0.10, -1.20,  1.40, -1.80, -1.57,  0.00],
    [-0.35, -1.10,  1.30, -1.75, -1.57,  0.30],
    [ 0.15, -1.10,  1.30, -1.75, -1.57, -0.30],
    [-0.10, -1.35,  1.55, -1.80, -1.57,  0.50],
    [-0.10, -1.05,  1.20, -1.75, -1.57, -0.50],
    [-0.40, -1.20,  1.40, -1.70, -1.40,  0.00],
    [ 0.20, -1.20,  1.40, -1.70, -1.70,  0.00],
    [-0.10, -1.20,  1.40, -1.80, -1.57,  0.80],
    [-0.10, -1.20,  1.40, -1.80, -1.57, -0.80],
    [-0.25, -1.30,  1.50, -1.85, -1.50,  0.40],
    [ 0.05, -1.30,  1.50, -1.85, -1.65, -0.40],
    [-0.30, -1.10,  1.25, -1.75, -1.45,  0.60],
    [ 0.10, -1.10,  1.25, -1.75, -1.70, -0.60],
    [-0.50, -1.20,  1.40, -1.80, -1.57,  0.00],
    [ 0.30, -1.20,  1.40, -1.80, -1.57,  0.00],
    [-0.10, -1.40,  1.60, -1.80, -1.57,  0.00],
    [-0.10, -1.00,  1.15, -1.75, -1.57,  0.00],
    [-0.20, -1.20,  1.40, -1.80, -1.57,  1.00],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_traj(target, current, duration=MOVE_DURATION):
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    p0 = JointTrajectoryPoint()
    p0.positions = list(current)
    p0.velocities = [0.0] * 6
    p0.time_from_start.sec = 0
    p0.time_from_start.nanosec = 0
    p1 = JointTrajectoryPoint()
    p1.positions = list(target)
    p1.velocities = [0.0] * 6
    p1.time_from_start.sec = int(duration)
    p1.time_from_start.nanosec = int((duration % 1.0) * 1e9)
    traj.points = [p0, p1]
    return traj


def get_T_base2gripper(tf_buffer, node):
    """4x4 homogeneous: base_link → tool0."""
    try:
        t = tf_buffer.lookup_transform(
            BASE_FRAME, EE_FRAME, rclpyTime(),
            timeout=rclpyDuration(seconds=2.0),
        )
    except Exception as e:
        node.get_logger().warn(f"TF failed: {e}")
        return None
    tr = t.transform.translation
    ro = t.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([ro.x, ro.y, ro.z, ro.w]).as_matrix()
    T[:3, 3]  = [tr.x, tr.y, tr.z]
    return T


def get_trunk_in_cam(zed, yolo_model, trunk_class_id, image_mat, pc_mat, runtime):
    """
    Detect the trunk with YOLO and return its 3D centroid in camera frame.
    Averages over DETECT_TRIES frames for stability.
    Returns np.array([x, y, z]) or None.
    """
    samples = []

    for _ in range(DETECT_TRIES):
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        zed.retrieve_measure(pc_mat, sl.MEASURE.XYZ)

        frame_rgba = image_mat.get_data()
        frame_rgb  = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2RGB)
        pc_np      = pc_mat.get_data()[:, :, :3]
        H, W       = frame_rgb.shape[:2]

        det = yolo_model.predict(
            frame_rgb, save=False, retina_masks=False,
            imgsz=640, conf=0.3, verbose=False,
        )[0]

        if det.boxes is None or len(det.boxes) == 0:
            continue

        for di in range(len(det.boxes)):
            if int(det.boxes.cls[di].item()) != trunk_class_id:
                continue

            xywh = det.boxes.xywh[di].cpu().numpy()
            x1 = int(np.clip(xywh[0] - xywh[2] / 2, 0, W - 1))
            y1 = int(np.clip(xywh[1] - xywh[3] / 2, 0, H - 1))
            x2 = int(np.clip(xywh[0] + xywh[2] / 2, 0, W))
            y2 = int(np.clip(xywh[1] + xywh[3] / 2, 0, H))

            roi = pc_np[y1:y2, x1:x2, :]
            valid = np.isfinite(roi[:, :, 2]) & (roi[:, :, 2] > 0.1)
            if np.count_nonzero(valid) < MIN_TRUNK_PTS:
                continue

            pts = roi[valid]
            idx = np.argsort(pts[:, 2])
            k   = max(10, int(TRUNK_FRONT_K * len(idx)))
            front = pts[idx[:k]]
            samples.append(np.median(front, axis=0))  # median is robust to noise
            break  # only use first trunk detection per frame

    if len(samples) < 5:
        return None

    # Reject outliers (> 2 std from median) then average
    arr    = np.array(samples)
    median = np.median(arr, axis=0)
    dists  = np.linalg.norm(arr - median, axis=1)
    good   = arr[dists < 2.0 * np.std(dists) + 1e-6]
    return np.mean(good, axis=0) if len(good) >= 3 else None


def rvec_tvec_to_T(params):
    """6-param vector [rvec(3), tvec(3)] → 4x4 matrix."""
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(params[:3])
    T[:3, 3]     = params[3:]
    return T


def T_to_rvec_tvec(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([rvec.flatten(), T[:3, 3]])


def objective(params, T_base2gripper_list, p_trunk_cam_list):
    """
    Variance of projected trunk positions in base_link.
    For the true calibration, all projected points coincide → variance = 0.
    """
    T_gripper2cam = rvec_tvec_to_T(params)
    projected = []
    for T_b2g, p_cam in zip(T_base2gripper_list, p_trunk_cam_list):
        p_h   = np.array([*p_cam, 1.0])
        p_base = (T_b2g @ T_gripper2cam @ p_h)[:3]
        projected.append(p_base)
    pts = np.array(projected)
    return float(np.sum(np.var(pts, axis=0)))  # sum of per-axis variance


def run_optimisation(T_b2g_list, p_cam_list, node):
    """
    Multi-start optimisation around the current calibration (if available)
    and several random initialisations.
    Returns best (T_gripper2cam, final_error).
    """
    # Build initial guesses
    inits = []

    # 1. Current YAML as warm start
    if OUTPUT_FILE.exists():
        try:
            cal = yaml.safe_load(OUTPUT_FILE.read_text())["hand_eye_calibration"]
            t_cur = np.array([cal["translation"]["x"],
                              cal["translation"]["y"],
                              cal["translation"]["z"]])
            q_cur = [cal["quaternion"]["x"], cal["quaternion"]["y"],
                     cal["quaternion"]["z"], cal["quaternion"]["w"]]
            R_cur = Rotation.from_quat(q_cur).as_matrix()
            T_cur = np.eye(4); T_cur[:3, :3] = R_cur; T_cur[:3, 3] = t_cur
            inits.append(T_to_rvec_tvec(T_cur))
            node.get_logger().info("  Using existing calibration as warm start.")
        except Exception:
            pass

    # 2. Identity + small perturbations
    inits.append(T_to_rvec_tvec(np.eye(4)))
    rng = np.random.default_rng(42)
    for _ in range(8):
        T_rand = np.eye(4)
        T_rand[:3, :3] = Rotation.from_euler(
            'xyz', rng.uniform(-0.5, 0.5, 3)).as_matrix()
        T_rand[:3, 3] = rng.uniform(-0.2, 0.2, 3)
        inits.append(T_to_rvec_tvec(T_rand))

    best_x, best_err = None, float("inf")
    for x0 in inits:
        res = minimize(
            objective, x0,
            args=(T_b2g_list, p_cam_list),
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
        )
        if res.fun < best_err:
            best_err, best_x = res.fun, res.x

    T_best = rvec_tvec_to_T(best_x)
    return T_best, best_err


def save_yaml(T, n_samples, error_mm):
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    t    = T[:3, 3]
    result = {
        "hand_eye_calibration": {
            "parent_frame": EE_FRAME,
            "child_frame":  CAM_FRAME,
            "translation":  {"x": float(t[0]),    "y": float(t[1]),    "z": float(t[2])},
            "quaternion":   {"x": float(quat[0]), "y": float(quat[1]),
                             "z": float(quat[2]), "w": float(quat[3])},
            "matrix":       T.tolist(),
            "method":       "trunk_consistency",
            "num_samples":  n_samples,
            "residual_mm":  round(error_mm, 3),
        }
    }
    OUTPUT_FILE.write_text(yaml.dump(result, default_flow_style=False))
    print(f"Saved → {OUTPUT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = Node("trunk_hand_eye_calibration")

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    # Track current joints
    current_joints = [0.0] * 6
    joint_ready    = [False]

    def js_cb(msg):
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                current_joints[i] = msg.position[msg.name.index(name)]
        joint_ready[0] = True

    node.create_subscription(JointState, "/joint_states", js_cb, 10)
    traj_pub = node.create_publisher(
        JointTrajectory,
        "/scaled_joint_trajectory_controller/joint_trajectory", 10
    )

    node.get_logger().info("Waiting for joint states...")
    deadline = time.time() + 10.0
    while not joint_ready[0] and time.time() < deadline:
        time.sleep(0.1)
    if not joint_ready[0]:
        node.get_logger().error("No joint states. Is the robot driver running?")
        rclpy.shutdown(); return

    time.sleep(1.0)

    # ── ZED ──────────────────────────────────────────────────────────────────
    zed = sl.Camera()
    init_p = sl.InitParameters()
    init_p.camera_resolution = sl.RESOLUTION.HD1080
    init_p.coordinate_units  = sl.UNIT.METER
    init_p.depth_mode        = sl.DEPTH_MODE.NEURAL_LIGHT
    init_p.depth_minimum_distance = 0.15
    if zed.open(init_p) != sl.ERROR_CODE.SUCCESS:
        node.get_logger().error("ZED open failed."); rclpy.shutdown(); return

    image_mat = sl.Mat()
    pc_mat    = sl.Mat()
    runtime   = sl.RuntimeParameters()

    # ── YOLO — load same model as vision node ─────────────────────────────────
    model_path = OUTPUT_FILE.parent.parent.parent.parent / \
                 "zed_date_detector" / "models" / "yolo_26_dates_trunk_bunch_seg.engine"
    if not model_path.exists():
        node.get_logger().error(f"Model not found: {model_path}"); zed.close(); rclpy.shutdown(); return

    node.get_logger().info(f"Loading YOLO model: {model_path}")
    yolo = YOLO(str(model_path), task='segment')
    names = getattr(yolo, 'names', {})
    trunk_class_id = next((cid for cid, n in names.items() if n.lower() == "trunk"), None)
    if trunk_class_id is None:
        node.get_logger().error(f"'trunk' class not found in model. Classes: {names}")
        zed.close(); rclpy.shutdown(); return
    node.get_logger().info(f"Model classes: {names}  |  trunk id={trunk_class_id}")

    # ── Calibration loop ──────────────────────────────────────────────────────
    T_b2g_list = []
    p_cam_list  = []

    node.get_logger().info(
        f"\n{'='*60}\n"
        f"  TRUNK-BASED HAND-EYE CALIBRATION\n"
        f"  {len(CALIBRATION_POSES)} poses planned\n"
        f"{'='*60}\n"
        f"  Ensure trunk is visible. Starting in 3s..."
    )
    time.sleep(3.0)

    for i, target in enumerate(CALIBRATION_POSES):
        node.get_logger().info(f"\n[{i+1}/{len(CALIBRATION_POSES)}] Moving to pose...")
        traj_pub.publish(build_traj(target, current_joints))
        time.sleep(MOVE_DURATION + SETTLE_TIME)

        # Get EE pose
        T_b2g = get_T_base2gripper(tf_buffer, node)
        if T_b2g is None:
            node.get_logger().warn(f"  Pose {i+1}: TF unavailable — skipping.")
            continue

        # Get trunk in camera frame
        node.get_logger().info(f"  Pose {i+1}: detecting trunk...")
        p_cam = get_trunk_in_cam(zed, yolo, trunk_class_id, image_mat, pc_mat, runtime)
        if p_cam is None:
            node.get_logger().warn(f"  Pose {i+1}: trunk not detected — skipping.")
            continue

        T_b2g_list.append(T_b2g)
        p_cam_list.append(p_cam)
        node.get_logger().info(
            f"  Pose {i+1}: trunk_cam=[{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]  "
            f"({len(T_b2g_list)} samples)"
        )

    zed.close()
    n = len(T_b2g_list)
    node.get_logger().info(f"\nCollection done — {n} valid samples.")

    if n < 4:
        node.get_logger().error(f"Need at least 4 samples (got {n}). Aborting.")
        rclpy.shutdown(); return

    # ── Optimise ──────────────────────────────────────────────────────────────
    node.get_logger().info("Running optimisation (multi-start L-BFGS-B)...")
    T_best, var_err = run_optimisation(T_b2g_list, p_cam_list, node)

    # Compute projected trunk positions for diagnostics
    projected = []
    for T_b2g, p_cam in zip(T_b2g_list, p_cam_list):
        p_h    = np.array([*p_cam, 1.0])
        p_base = (T_b2g @ T_best @ p_h)[:3]
        projected.append(p_base)
    pts       = np.array(projected)
    mean_pt   = pts.mean(axis=0)
    residuals = np.linalg.norm(pts - mean_pt, axis=1) * 1000  # mm
    rms_mm    = float(np.sqrt(np.mean(residuals**2)))

    print("\n" + "=" * 60)
    print(f"RESULT  (trunk consistency optimisation)")
    print("=" * 60)
    print(f"Samples : {n}")
    print(f"RMS residual: {rms_mm:.2f} mm  (spread of trunk in base_link)")
    print(f"Trunk base_link mean: {mean_pt}")
    print(f"\nPer-sample residuals (mm):")
    for i, r in enumerate(residuals):
        print(f"  pose {i+1:2d}: {r:.2f} mm")

    t_res = T_best[:3, 3]
    q_res = Rotation.from_matrix(T_best[:3, :3]).as_quat()
    print(f"\nT_gripper2cam:")
    print(f"  translation: x={t_res[0]:.6f}  y={t_res[1]:.6f}  z={t_res[2]:.6f}")
    print(f"  quaternion:  x={q_res[0]:.6f}  y={q_res[1]:.6f}  z={q_res[2]:.6f}  w={q_res[3]:.6f}")
    print(f"\nMatrix:\n{T_best}")

    # Drift vs existing calibration
    if OUTPUT_FILE.exists():
        try:
            old = yaml.safe_load(OUTPUT_FILE.read_text())["hand_eye_calibration"]
            old_t = np.array([old["translation"]["x"],
                               old["translation"]["y"],
                               old["translation"]["z"]])
            drift_mm = np.linalg.norm(t_res - old_t) * 1000
            print(f"\nTranslation drift from previous calibration: {drift_mm:.1f} mm")
            if drift_mm > 5.0:
                print("  WARNING: drift > 5 mm — camera mount may have moved.")
            else:
                print("  OK: within 5 mm tolerance.")
        except Exception:
            pass

    if rms_mm > 15.0:
        print(f"\nWARNING: RMS residual {rms_mm:.1f} mm is high.")
        print("  Possible causes: trunk moved, too few poses, poses not diverse enough.")
        print("  Calibration saved but review before use.")
    else:
        print(f"\nCalibration quality: good ({rms_mm:.1f} mm RMS)")

    save_yaml(T_best, n, rms_mm)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
