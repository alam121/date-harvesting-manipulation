"""
Hand-eye calibration: find the transform from camera to gripper (eye-in-hand).

Usage:
  1. Print the chessboard pattern (18x25 squares, 30 mm each).
  2. Place the board flat and stationary in the robot's workspace.
  3. Run this script.
  4. Move the robot to ~15-20 different poses (vary rotation and translation)
     while keeping the board visible in the camera.
  5. Press 'c' to capture a sample at each pose.
  6. Press 'q' when done collecting — the script computes the calibration.

The result (4x4 camera-to-gripper transform) is saved to a YAML file.
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
import pyzed.sl as sl
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation

# ── URDFs to update when calibration is confirmed ─────────────────────────────
URDF_FILES = [
    Path(__file__).resolve().parents[4] / "src/universal_robot/urdf/ur_macro.xacro",
    Path(__file__).resolve().parents[5] / "curobo/src/curobo/content/assets/robot/ur_description/ur10e_curobo.urdf",
]

# ── Chessboard parameters ─────────────────────────────────────────────
BOARD_ROWS = 10         # inner corners per row    (11 squares → 10 inner corners)
BOARD_COLS = 7          # inner corners per column (8 squares → 7 inner corners)
SQUARE_SIZE = 0.030     # square side length in metres (30 mm)

# ── Output path ───────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "hand_eye_calibration.yaml"

# ── TF frames ─────────────────────────────────────────────────────────
BASE_FRAME = "base_link"
EE_FRAME = "tool0"  # UR driver end-effector frame


def get_ee_pose(tf_buffer: Buffer, node: Node, timeout_sec: float = 2.0):
    """Return (4x4 homogeneous matrix) of the end-effector in the base frame."""
    try:
        t = tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=timeout_sec))
    except Exception as e:
        node.get_logger().warn(f"TF lookup failed: {e}")
        return None

    trans = t.transform.translation
    rot = t.transform.rotation
    T = np.eye(4)
    T[:3, 3] = [trans.x, trans.y, trans.z]
    T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    return T


def rotation_matrix_to_rvec(R):
    """Convert 3x3 rotation matrix to Rodrigues vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def main():
    rclpy.init()
    node = Node("hand_eye_calibration")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    # Spin in background so TF updates arrive
    from threading import Thread
    spin_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Poll until base_link→tool0 is available (UR bringup can take 10–30 s)
    node.get_logger().info("Waiting for TF tree (base_link → tool0)...")
    print("Waiting for robot driver TF... (start UR bringup if not running)")
    deadline = time.time() + 60.0
    while time.time() < deadline:
        try:
            tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME,
                                       rclpy.time.Time(),
                                       timeout=rclpy.duration.Duration(seconds=1.0))
            node.get_logger().info("TF ready.")
            break
        except Exception:
            elapsed = int(time.time() - (deadline - 60.0))
            print(f"\r  still waiting... {elapsed}s", end="", flush=True)
    else:
        print()
        node.get_logger().error("TF not available after 60 s — is the robot driver running?")
        zed.close()
        rclpy.shutdown()
        return
    print()

    # ── ZED X One Mono setup (CameraOne API) ─────────────────────────
    zed = sl.CameraOne()
    init_params = sl.InitParametersOne()
    init_params.camera_resolution = sl.RESOLUTION.QHDPLUS  # match production resolution
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.enable_hdr = True   # same setting used during normal operation

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        node.get_logger().error(f"ZED X One Mono open failed: {status}")
        return
    node.get_logger().info("ZED X One Mono opened for hand-eye calibration.")

    # CameraOne: calibration_parameters is the mono calibration directly (no .left_cam)
    cam_info = zed.get_camera_information()
    cam_params = cam_info.camera_configuration.calibration_parameters
    camera_matrix = np.array([
        [cam_params.fx, 0, cam_params.cx],
        [0, cam_params.fy, cam_params.cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.array(cam_params.disto[:5], dtype=np.float64)
    node.get_logger().info(
        f"Intrinsics: fx={cam_params.fx:.1f} fy={cam_params.fy:.1f} "
        f"cx={cam_params.cx:.1f} cy={cam_params.cy:.1f}"
    )

    image_mat = sl.Mat()

    # ── Chessboard object points ──────────────────────────────────────
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

    # ── Storage for calibration data ──────────────────────────────────
    R_gripper2base_list = []   # rotation:    gripper -> base
    t_gripper2base_list = []   # translation: gripper -> base
    R_target2cam_list = []     # rotation:    chessboard -> camera
    t_target2cam_list = []     # translation: chessboard -> camera

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    sample_count = 0

    node.get_logger().info(
        f"Hand-eye calibration ready.\n"
        f"  Chessboard: {BOARD_COLS}x{BOARD_ROWS}, square={SQUARE_SIZE*1000:.0f}mm\n"
        f"  Move the robot, press 'c' to capture, 'q' to finish.\n"
        f"  Aim for 15-20 diverse poses."
    )

    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:  # CameraOne: no RuntimeParameters arg
            continue

        zed.retrieve_image(image_mat)  # CameraOne: no VIEW arg
        frame = image_mat.get_data()[:, :, :3].copy()  # drop alpha channel (BGRA→BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find chessboard
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )

        display = frame.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners2, found)
            cv2.putText(display, "Chessboard FOUND - press 'c' to capture",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No chessboard detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, f"Samples: {sample_count}  |  'c'=capture  'q'=calibrate & quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for display
        h, w = display.shape[:2]
        scale = min(1280 / w, 720 / h)
        display = cv2.resize(display, (int(w * scale), int(h * scale)))
        cv2.imshow("Hand-Eye Calibration", display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('c') and found:
            # Get end-effector pose
            ee_pose = get_ee_pose(tf_buffer, node)
            if ee_pose is None:
                cv2.putText(display, "NO TF — is the robot driver running?",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Hand-Eye Calibration", display)
                cv2.waitKey(1500)
                continue

            # Solve chessboard pose in camera frame
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            if not ret:
                node.get_logger().warn("solvePnP failed — skipping.")
                continue

            R_target2cam, _ = cv2.Rodrigues(rvec)

            # Store poses
            R_gripper2base_list.append(ee_pose[:3, :3])
            t_gripper2base_list.append(ee_pose[:3, 3].reshape(3, 1))
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec.reshape(3, 1))

            sample_count += 1
            node.get_logger().info(f"Sample {sample_count} captured.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    zed.close()

    # ── Run hand-eye calibration ──────────────────────────────────────
    if sample_count < 10:
        node.get_logger().error(f"Need at least 10 samples, got {sample_count}. Aborting.")
        rclpy.shutdown()
        return

    # ── Rotation diversity check ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROTATION DIVERSITY CHECK")
    print("=" * 60)
    rot_vecs = [Rotation.from_matrix(R).as_rotvec() for R in R_gripper2base_list]
    max_angle_deg = 0.0
    for i in range(len(rot_vecs)):
        for j in range(i + 1, len(rot_vecs)):
            rel = Rotation.from_matrix(
                R_gripper2base_list[i].T @ R_gripper2base_list[j]
            ).magnitude()
            max_angle_deg = max(max_angle_deg, np.degrees(rel))
    print(f"  Max rotation between any two poses: {max_angle_deg:.1f}°")
    if max_angle_deg < 30.0:
        print(f"  WARNING: Max rotation {max_angle_deg:.1f}° < 30°.")
        print("  Daniilidis rotation estimate will be unreliable.")
        print("  Add poses with larger wrist_3 / wrist_1 rotations (±30°).")
        confirm = input("  Continue anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            rclpy.shutdown()
            return
    else:
        print(f"  OK — sufficient rotation diversity.")

    node.get_logger().info(f"Running hand-eye calibration with {sample_count} samples...")

    # ── Try all methods and pick the best ─────────────────────────────
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    def compute_consistency_error(T_x):
        """Mean AX=XB translation error for a candidate transform."""
        errs = []
        for i in range(sample_count):
            for j in range(i + 1, sample_count):
                A_i = np.linalg.inv(np.vstack([np.hstack([R_gripper2base_list[i], t_gripper2base_list[i]]), [0, 0, 0, 1]])) @ \
                      np.vstack([np.hstack([R_gripper2base_list[j], t_gripper2base_list[j]]), [0, 0, 0, 1]])
                B_i = np.vstack([np.hstack([R_target2cam_list[i], t_target2cam_list[i]]), [0, 0, 0, 1]]) @ \
                      np.linalg.inv(np.vstack([np.hstack([R_target2cam_list[j], t_target2cam_list[j]]), [0, 0, 0, 1]]))
                errs.append(np.linalg.norm((A_i @ T_x)[:3, 3] - (T_x @ B_i)[:3, 3]))
        return np.mean(errs)

    print("\n" + "=" * 60)
    print("COMPARING ALL METHODS")
    print("=" * 60)

    best_method_name = None
    best_error = float("inf")
    best_T = None

    for name, method in methods.items():
        try:
            R_x, t_x = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=method,
            )
            T_x = np.eye(4)
            T_x[:3, :3] = R_x
            T_x[:3, 3] = t_x.flatten()
            err = compute_consistency_error(T_x)
            if not np.isfinite(err):
                raise ValueError("non-finite error")
            print(f"  {name:12s}  mean error: {err*1000:.2f} mm  |  t=[{t_x[0,0]:.4f}, {t_x[1,0]:.4f}, {t_x[2,0]:.4f}]")
            if err < best_error:
                best_error = err
                best_method_name = name
                best_T = T_x
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")

    if best_T is None:
        print("\n  >>> ALL METHODS FAILED.")
        print("  The poses lack rotational diversity. Redo with larger wrist rotations (20-30 deg).")
        zed.close()
        rclpy.shutdown()
        return

    print(f"\n  >>> Best method: {best_method_name} ({best_error*1000:.2f} mm)")

    T_cam2gripper = best_T
    R_cam2gripper = T_cam2gripper[:3, :3]
    t_cam2gripper = T_cam2gripper[:3, 3].reshape(3, 1)

    # Also express as quaternion for ROS usage
    quat = Rotation.from_matrix(R_cam2gripper).as_quat()  # [x, y, z, w]

    # ── Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"HAND-EYE CALIBRATION RESULT — {best_method_name} (camera -> gripper)")
    print("=" * 60)
    print(f"\nTranslation [m]:")
    print(f"  x: {t_cam2gripper[0, 0]:.6f}")
    print(f"  y: {t_cam2gripper[1, 0]:.6f}")
    print(f"  z: {t_cam2gripper[2, 0]:.6f}")
    print(f"\nQuaternion [x, y, z, w]:")
    print(f"  x: {quat[0]:.6f}")
    print(f"  y: {quat[1]:.6f}")
    print(f"  z: {quat[2]:.6f}")
    print(f"  w: {quat[3]:.6f}")
    print(f"\n4x4 Transform Matrix:")
    print(T_cam2gripper)

    # ── Save to YAML ──────────────────────────────────────────────────
    rpy = Rotation.from_matrix(R_cam2gripper).as_euler("xyz")  # radians
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "hand_eye_calibration": {
            "parent_frame": EE_FRAME,
            "child_frame": "zed2_left_camera_frame",
            "translation": {
                "x": float(t_cam2gripper[0, 0]),
                "y": float(t_cam2gripper[1, 0]),
                "z": float(t_cam2gripper[2, 0]),
            },
            "quaternion": {
                "x": float(quat[0]),
                "y": float(quat[1]),
                "z": float(quat[2]),
                "w": float(quat[3]),
            },
            "matrix": T_cam2gripper.tolist(),
            "method": best_method_name,
            "num_samples": sample_count,
            "calibrated_at": timestamp,
        }
    }

    OUTPUT_FILE.write_text(yaml.dump(result, default_flow_style=False))
    node.get_logger().info(f"Calibration saved to {OUTPUT_FILE}")

    print(f"\nBest method: {best_method_name}  (mean error: {best_error*1000:.2f} mm)")
    print("=" * 60)

    # ── Ask user to confirm URDF update ───────────────────────────────
    print("\n" + "=" * 60)
    print("UPDATE URDFs?")
    print("=" * 60)
    print(f"  New transform (tool0 → zed2_left_camera_frame):")
    print(f"    xyz=\"{t_cam2gripper[0,0]:.6f} {t_cam2gripper[1,0]:.6f} {t_cam2gripper[2,0]:.6f}\"")
    print(f"    rpy=\"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}\"")
    print(f"\n  Files to update:")
    for f in URDF_FILES:
        exists = "✓" if f.exists() else "✗ NOT FOUND"
        print(f"    [{exists}] {f}")
    print()

    import sys
    sys.stdout.flush()
    # Drain any buffered newlines left over from the capture loop
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass
    confirm = input("Apply to URDFs? [y/N]: ").strip().lower()
    print(f"  (input received: {repr(confirm)})")
    if confirm == "y":
        _update_urdfs(t_cam2gripper, rpy, best_method_name, best_error, sample_count, timestamp)
    else:
        print("URDFs not updated. Values saved to YAML only.")

    rclpy.shutdown()


def _update_urdfs(t, rpy, method, error_m, n_samples, timestamp):
    """Update the tool0→camera joint origin in each URDF/xacro file."""
    xyz_str = f"{t[0,0]:.6f} {t[1,0]:.6f} {t[2,0]:.6f}"
    rpy_str = f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"
    comment = (
        f"hand-eye calibrated ({method}, {error_m*1000:.2f}mm error, "
        f"{n_samples} samples, updated {timestamp})"
    )

    # Pattern A: xyz before rpy, comment on same line
    # <origin xyz="..." rpy="..."/>   <!-- ... -->
    pat_xyz_first = re.compile(
        r'(<origin\s[^>]*xyz=")[^"]*("\s*rpy=")[^"]*("\s*/>[ \t]*)<!--.*?-->'
    )
    repl_xyz_first = rf'\g<1>{xyz_str}\g<2>{rpy_str}\g<3><!-- {comment} -->'

    # Pattern B: rpy before xyz, comment on next line
    # <origin rpy="..." xyz="..."/>
    # <!-- ... -->
    pat_rpy_first = re.compile(
        r'(<origin\s[^>]*rpy=")[^"]*("\s*xyz=")[^"]*("\s*/>)([ \t]*\n[ \t]*)<!--.*?-->'
    )
    repl_rpy_first = rf'\g<1>{rpy_str}\g<2>{xyz_str}\g<3>\g<4><!-- {comment} -->'

    joint_pattern = re.compile(
        r'(joint name="tool0_to_zed2_left_camera_frame".*?</joint>)',
        re.DOTALL,
    )

    for urdf_path in URDF_FILES:
        if not urdf_path.exists():
            print(f"  SKIP (not found): {urdf_path}")
            continue
        text = urdf_path.read_text()
        match = joint_pattern.search(text)
        if not match:
            print(f"  SKIP (joint not found): {urdf_path}")
            continue
        joint_old = match.group(1)
        joint_new = pat_xyz_first.sub(repl_xyz_first, joint_old)
        if joint_new == joint_old:
            joint_new = pat_rpy_first.sub(repl_rpy_first, joint_old)
        if joint_new == joint_old:
            print(f"  SKIP (origin pattern not matched): {urdf_path}")
            continue
        new_text = text[:match.start(1)] + joint_new + text[match.end(1):]
        urdf_path.write_text(new_text)
        print(f"  UPDATED: {urdf_path}")


if __name__ == "__main__":
    main()
