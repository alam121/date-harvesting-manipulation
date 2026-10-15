"""
Automated hand-eye calibration for ZED2 + UR10e (eye-in-hand).

Place the chessboard flat and stationary in the robot's workspace, then run:
    python3 auto_calibration.py

The arm will automatically move through calibration poses, detect the board,
capture samples, compute the best-method calibration, and save the YAML.

Adjust CALIBRATION_POSES below to poses where your chessboard is visible.
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
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import pyzed.sl as sl

# ── Chessboard parameters ─────────────────────────────────────────────────────
BOARD_ROWS   = 17       # inner corners per row
BOARD_COLS   = 24       # inner corners per column
SQUARE_SIZE  = 0.030    # metres

# ── TF frames ─────────────────────────────────────────────────────────────────
BASE_FRAME = "base_link"
EE_FRAME   = "tool0"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = Path(__file__).resolve().parent / "hand_eye_calibration.yaml"

# ── Motion parameters ─────────────────────────────────────────────────────────
MOVE_DURATION   = 6.0   # seconds per move (slow = safe)
SETTLE_TIME     = 1.5   # wait after move before detecting
DETECT_TIMEOUT  = 5.0   # seconds to wait for board detection per pose
SKIP_ON_MISS    = True  # skip pose silently if board not found

# ── Joint order (UR10e standard) ──────────────────────────────────────────────
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# ── Calibration poses (joint angles in radians) ───────────────────────────────
# These should give diverse views of the chessboard: vary all joints,
# especially shoulder_pan and wrist orientations.
# Tune these so the board stays fully in frame at each pose.
CALIBRATION_POSES = [
    # [pan,   lift,   elbow,  w1,     w2,     w3]
    [-0.10,  -1.20,   1.40,  -1.80,  -1.57,   0.00],   # 1  center, mid height
    [-0.35,  -1.10,   1.30,  -1.75,  -1.57,   0.30],   # 2  slight left, tilted
    [ 0.15,  -1.10,   1.30,  -1.75,  -1.57,  -0.30],   # 3  slight right, tilted
    [-0.10,  -1.35,   1.55,  -1.80,  -1.57,   0.50],   # 4  center, higher
    [-0.10,  -1.05,   1.20,  -1.75,  -1.57,  -0.50],   # 5  center, lower
    [-0.40,  -1.20,   1.40,  -1.70,  -1.40,   0.00],   # 6  left, wrist rotated
    [ 0.20,  -1.20,   1.40,  -1.70,  -1.70,   0.00],   # 7  right, wrist rotated
    [-0.10,  -1.20,   1.40,  -1.80,  -1.57,   0.80],   # 8  wrist roll +
    [-0.10,  -1.20,   1.40,  -1.80,  -1.57,  -0.80],   # 9  wrist roll -
    [-0.25,  -1.30,   1.50,  -1.85,  -1.50,   0.40],   # 10 combined
    [ 0.05,  -1.30,   1.50,  -1.85,  -1.65,  -0.40],   # 11 combined
    [-0.10,  -1.15,   1.35,  -1.90,  -1.57,   0.20],   # 12 wrist 1 change
    [-0.10,  -1.25,   1.45,  -1.70,  -1.57,  -0.20],   # 13 wrist 1 other
    [-0.30,  -1.10,   1.25,  -1.75,  -1.45,   0.60],   # 14 diverse
    [ 0.10,  -1.10,   1.25,  -1.75,  -1.70,  -0.60],   # 15 diverse
    [-0.50,  -1.20,   1.40,  -1.80,  -1.57,   0.00],   # 16 further left
    [ 0.30,  -1.20,   1.40,  -1.80,  -1.57,   0.00],   # 17 further right
    [-0.10,  -1.40,   1.60,  -1.80,  -1.57,   0.00],   # 18 arm higher
    [-0.10,  -1.00,   1.15,  -1.75,  -1.57,   0.00],   # 19 arm lower
    [-0.20,  -1.20,   1.40,  -1.80,  -1.57,   1.00],   # 20 large wrist roll
]


def build_move_traj(target_joints, current_joints, duration=MOVE_DURATION):
    """Build a 2-point JointTrajectory from current to target."""
    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    # Start point (time=0)
    p0 = JointTrajectoryPoint()
    p0.positions = list(current_joints)
    p0.velocities = [0.0] * 6
    p0.time_from_start.sec = 0
    p0.time_from_start.nanosec = 0
    traj.points.append(p0)

    # End point
    p1 = JointTrajectoryPoint()
    p1.positions = list(target_joints)
    p1.velocities = [0.0] * 6
    p1.time_from_start.sec = int(duration)
    p1.time_from_start.nanosec = int((duration % 1.0) * 1e9)
    traj.points.append(p1)

    return traj


def get_ee_pose(tf_buffer, node):
    """Return 4x4 homogeneous matrix of tool0 in base_link."""
    try:
        t = tf_buffer.lookup_transform(
            BASE_FRAME, EE_FRAME, rclpyTime(),
            timeout=rclpyDuration(seconds=2.0),
        )
    except Exception as e:
        node.get_logger().warn(f"TF lookup failed: {e}")
        return None
    trans = t.transform.translation
    rot   = t.transform.rotation
    T = np.eye(4)
    T[:3, 3]  = [trans.x, trans.y, trans.z]
    T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    return T


def detect_board(zed, image_mat, runtime, camera_matrix, dist_coeffs, objp, criteria):
    """
    Try to detect the chessboard from the ZED left image.
    Returns (R_target2cam, tvec) if found, else (None, None).
    """
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None, None

    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    frame = image_mat.get_data()[:, :, :3].copy()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, (BOARD_COLS, BOARD_ROWS),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None, None

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    if not ret:
        return None, None

    R_target2cam, _ = cv2.Rodrigues(rvec)
    return R_target2cam, tvec.reshape(3, 1)


def compute_consistency_error(R_g2b, t_g2b, R_t2c, t_t2c, T_x, n):
    """Mean AX=XB translation error across all pose pairs."""
    errs = []
    for i in range(n):
        Ti = np.eye(4)
        Ti[:3, :3] = R_g2b[i]; Ti[:3, 3] = t_g2b[i].flatten()
        Ci = np.eye(4)
        Ci[:3, :3] = R_t2c[i]; Ci[:3, 3] = t_t2c[i].flatten()
        for j in range(i + 1, n):
            Tj = np.eye(4)
            Tj[:3, :3] = R_g2b[j]; Tj[:3, 3] = t_g2b[j].flatten()
            Cj = np.eye(4)
            Cj[:3, :3] = R_t2c[j]; Cj[:3, 3] = t_t2c[j].flatten()
            A = np.linalg.inv(Ti) @ Tj
            B = Ci @ np.linalg.inv(Cj)
            errs.append(np.linalg.norm((A @ T_x)[:3, 3] - (T_x @ B)[:3, 3]))
    return float(np.mean(errs)) if errs else float("inf")


def run_calibration(R_g2b, t_g2b, R_t2c, t_t2c, n, node):
    """Try all methods, return (best_T, best_name, best_error)."""
    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    print("\n" + "=" * 60)
    print("COMPARING CALIBRATION METHODS")
    print("=" * 60)

    best_name, best_err, best_T = None, float("inf"), None
    for name, flag in methods.items():
        R_x, t_x = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        T_x = np.eye(4)
        T_x[:3, :3] = R_x
        T_x[:3, 3]  = t_x.flatten()
        err = compute_consistency_error(R_g2b, t_g2b, R_t2c, t_t2c, T_x, n)
        print(f"  {name:12s}  error: {err*1000:.2f} mm  "
              f"t=[{t_x[0,0]:.4f}, {t_x[1,0]:.4f}, {t_x[2,0]:.4f}]")
        if err < best_err:
            best_err, best_name, best_T = err, name, T_x

    print(f"\n  >>> Best: {best_name}  ({best_err*1000:.2f} mm)")
    return best_T, best_name, best_err


def save_yaml(T, name, n_samples):
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]
    t    = T[:3, 3]
    result = {
        "hand_eye_calibration": {
            "parent_frame":  EE_FRAME,
            "child_frame":   "zed2_left_camera_frame",
            "translation":   {"x": float(t[0]),    "y": float(t[1]),    "z": float(t[2])},
            "quaternion":    {"x": float(quat[0]), "y": float(quat[1]),
                              "z": float(quat[2]), "w": float(quat[3])},
            "matrix":        T.tolist(),
            "method":        name,
            "num_samples":   n_samples,
        }
    }
    OUTPUT_FILE.write_text(yaml.dump(result, default_flow_style=False))
    print(f"\nSaved → {OUTPUT_FILE}")


def main():
    rclpy.init()
    node = Node("auto_hand_eye_calibration")

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    spin_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Joint state subscriber — to know current joints before each move
    current_joints = [0.0] * 6
    joint_ready = [False]

    def js_cb(msg):
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                current_joints[i] = msg.position[idx]
        joint_ready[0] = True

    node.create_subscription(JointState, "/joint_states", js_cb, 10)

    traj_pub = node.create_publisher(
        JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10
    )

    # Wait for TF and joint states
    node.get_logger().info("Waiting for TF tree and joint states...")
    deadline = time.time() + 10.0
    while not joint_ready[0] and time.time() < deadline:
        time.sleep(0.1)
    if not joint_ready[0]:
        node.get_logger().error("No joint states received. Is the robot driver running?")
        rclpy.shutdown()
        return

    time.sleep(1.0)  # let TF settle

    # ── ZED setup ─────────────────────────────────────────────────────────────
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units  = sl.UNIT.METER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        node.get_logger().error("ZED open failed.")
        rclpy.shutdown()
        return

    cam_info  = zed.get_camera_information()
    left_cam  = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[left_cam.fx, 0, left_cam.cx],
                  [0, left_cam.fy, left_cam.cy],
                  [0, 0, 1]], dtype=np.float64)
    D = np.array(left_cam.disto[:5], dtype=np.float64)

    image_mat = sl.Mat()
    runtime   = sl.RuntimeParameters()
    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

    # ── Calibration data storage ───────────────────────────────────────────────
    R_gripper2base, t_gripper2base = [], []
    R_target2cam,   t_target2cam   = [], []

    node.get_logger().info(
        f"\n{'='*60}\n"
        f"  AUTO HAND-EYE CALIBRATION\n"
        f"  Chessboard: {BOARD_COLS}x{BOARD_ROWS}, square={SQUARE_SIZE*1000:.0f}mm\n"
        f"  {len(CALIBRATION_POSES)} poses planned\n"
        f"{'='*60}\n"
        f"  Place chessboard in workspace. Starting in 3s..."
    )
    time.sleep(3.0)

    # ── Main calibration loop ──────────────────────────────────────────────────
    for i, target_joints in enumerate(CALIBRATION_POSES):
        node.get_logger().info(f"\n[{i+1}/{len(CALIBRATION_POSES)}] Moving to calibration pose...")

        traj = build_move_traj(target_joints, current_joints, duration=MOVE_DURATION)
        traj_pub.publish(traj)

        # Wait for motion to complete
        time.sleep(MOVE_DURATION + SETTLE_TIME)

        # Get EE pose from TF
        ee_pose = get_ee_pose(tf_buffer, node)
        if ee_pose is None:
            node.get_logger().warn(f"  Pose {i+1}: TF unavailable — skipping.")
            continue

        # Try to detect board within timeout
        node.get_logger().info(f"  Pose {i+1}: detecting chessboard...")
        R_t2c, t_t2c = None, None
        t_start = time.time()
        while time.time() - t_start < DETECT_TIMEOUT:
            R_t2c, t_t2c = detect_board(zed, image_mat, runtime, K, D, objp, criteria)
            if R_t2c is not None:
                break
            time.sleep(0.1)

        if R_t2c is None:
            if SKIP_ON_MISS:
                node.get_logger().warn(f"  Pose {i+1}: chessboard not found — skipping.")
            continue

        R_gripper2base.append(ee_pose[:3, :3])
        t_gripper2base.append(ee_pose[:3, 3].reshape(3, 1))
        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)

        n = len(R_gripper2base)
        node.get_logger().info(f"  Pose {i+1}: captured  ({n} samples total)")

    zed.close()
    n = len(R_gripper2base)
    node.get_logger().info(f"\nCollection done — {n} valid samples.")

    if n < 3:
        node.get_logger().error(f"Need at least 3 samples (got {n}). Aborting.")
        rclpy.shutdown()
        return

    # ── Run calibration ────────────────────────────────────────────────────────
    best_T, best_name, best_err = run_calibration(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        n, node,
    )

    print("\n" + "=" * 60)
    print(f"RESULT — {best_name}  (mean error: {best_err*1000:.2f} mm)")
    print("=" * 60)
    print(f"Translation [m]: {best_T[:3, 3]}")
    quat = Rotation.from_matrix(best_T[:3, :3]).as_quat()
    print(f"Quaternion [x,y,z,w]: {quat}")
    print(f"Matrix:\n{best_T}")

    save_yaml(best_T, best_name, n)

    # ── Compare against existing calibration ──────────────────────────────────
    if OUTPUT_FILE.exists():
        try:
            old = yaml.safe_load(OUTPUT_FILE.read_text())["hand_eye_calibration"]
            old_t = np.array([old["translation"]["x"],
                               old["translation"]["y"],
                               old["translation"]["z"]])
            delta_t = np.linalg.norm(best_T[:3, 3] - old_t) * 1000
            print(f"\nTranslation drift from previous calibration: {delta_t:.1f} mm")
            if delta_t > 5.0:
                print("  ⚠  Drift > 5mm — consider re-running with more samples.")
            else:
                print("  ✓  Within 5mm tolerance.")
        except Exception:
            pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
