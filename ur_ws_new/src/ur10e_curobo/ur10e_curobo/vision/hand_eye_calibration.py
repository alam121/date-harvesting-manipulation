"""
Hand-eye calibration: find the transform from camera to gripper (eye-in-hand).

Usage:
  1. Generate and print the configured ChArUco board at 100% scale.
  2. Place the board flat and stationary in the robot's workspace.
  3. Run this script.
  4. Move the robot to ~15-20 different poses (vary rotation and translation)
     while keeping the board visible in the camera.
  5. Press 'c' to capture a sample at each pose.
  6. Press 'q' when done collecting — the script computes the calibration.

The result (4x4 camera-to-gripper transform) is saved to a YAML file.
"""

import argparse
import os
import re
import shutil
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

try:
    from .calibration_profiles import (
        contextual_profile_name,
        frame_from_profile,
        load_camera_profile,
        profile_path_for_name,
        save_camera_profile,
    )
except ImportError:
    from calibration_profiles import (  # type: ignore
        contextual_profile_name,
        frame_from_profile,
        load_camera_profile,
        profile_path_for_name,
        save_camera_profile,
    )

# ── URDFs to update when calibration is confirmed ─────────────────────────────
URDF_FILES = [
    Path(__file__).resolve().parents[4] / "src/universal_robot/urdf/ur_macro.xacro",
]

# ── ChArUco board parameters ──────────────────────────────────────────
# Calib.io ChArUco board: 17 x 24 physical squares.  ChArUco internal corners
# are one fewer than the square count along each axis.
BOARD_SQUARES_X = 17
BOARD_SQUARES_Y = 24
BOARD_INNER_X = BOARD_SQUARES_X - 1
BOARD_INNER_Y = BOARD_SQUARES_Y - 1
SQUARE_SIZE = 0.030
MARKER_SIZE = 0.022
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_5X5_1000
ARUCO_DICTIONARY_NAME = "DICT_5X5_1000"
MIN_CHARUCO_CORNERS = 12
BOARD_DPI = 300

# Legacy plain chessboard option: 7 × 10 inner corners (8 × 11 squares).
CHESSBOARD_INNER_X = 7
CHESSBOARD_INNER_Y = 10

# ── Output path ───────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "hand_eye_calibration.yaml"

# ── TF frames ─────────────────────────────────────────────────────────
BASE_FRAME = "base_link"
EE_FRAME = "tool0"  # UR driver end-effector frame


def create_charuco_board(
    legacy_pattern=False,
    squares_x=BOARD_SQUARES_X,
    squares_y=BOARD_SQUARES_Y,
):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        SQUARE_SIZE,
        MARKER_SIZE,
        dictionary,
    )
    board.setLegacyPattern(legacy_pattern)
    return dictionary, board


def generate_board_image(output_path: Path, target_type: str) -> None:
    """Generate a print-ready PNG whose DPI metadata preserves physical size."""
    from PIL import Image

    if target_type == "charuco":
        _, board = create_charuco_board()
        squares_x = BOARD_SQUARES_X
        squares_y = BOARD_SQUARES_Y
    else:
        board = None
        squares_x = CHESSBOARD_INNER_X + 1
        squares_y = CHESSBOARD_INNER_Y + 1

    width_mm = squares_x * SQUARE_SIZE * 1000.0
    height_mm = squares_y * SQUARE_SIZE * 1000.0
    width_px = round(width_mm / 25.4 * BOARD_DPI)
    height_px = round(height_mm / 25.4 * BOARD_DPI)
    if target_type == "charuco":
        image = board.generateImage(
            (width_px, height_px), marginSize=0, borderBits=1)
    else:
        image = np.full((height_px, width_px), 255, dtype=np.uint8)
        x_edges = np.rint(np.linspace(0, width_px, squares_x + 1)).astype(int)
        y_edges = np.rint(np.linspace(0, height_px, squares_y + 1)).astype(int)
        for row in range(squares_y):
            for col in range(squares_x):
                if (row + col) % 2 == 0:
                    image[y_edges[row]:y_edges[row + 1],
                          x_edges[col]:x_edges[col + 1]] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path, dpi=(BOARD_DPI, BOARD_DPI))
    print(f"Generated {target_type} board: {output_path}")
    print(f"  Squares: {squares_x} x {squares_y}")
    print(f"  Physical size: {width_mm:.0f} x {height_mm:.0f} mm")
    print(f"  Square: {SQUARE_SIZE * 1000:.0f} mm")
    if target_type == "charuco":
        print(f"  Marker: {MARKER_SIZE * 1000:.0f} mm")
        print(f"  Dictionary: {ARUCO_DICTIONARY_NAME}")
    print("Print at 100% / Actual Size with all page scaling disabled.")


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
    parser = argparse.ArgumentParser(description="UR10e hand-eye calibration")
    parser.add_argument(
        "--target",
        choices=("charuco", "chessboard"),
        default="charuco",
        help="calibration target type (default: charuco)",
    )
    parser.add_argument(
        "--resolution",
        choices=("qhdplus", "4k", "hd1080"),
        default="qhdplus",
        help="hand-eye capture resolution; ZED X Mini uses HD1080 (default: qhdplus)",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("zedx_mini", "zed_mini", "stereo", "lidar"),
        default=None,
        help=(
            "camera calibration profile to update: zedx_mini = Mini RGBD, "
            "zed_mini/lidar = ZED X One RGB frame"
        ),
    )
    parser.add_argument(
        "--robot-profile",
        choices=("old",),
        default=os.getenv("UR10E_ROBOT_PROFILE", "old").strip().lower(),
        help="physical robot being calibrated (default: UR10E_ROBOT_PROFILE or old)",
    )
    parser.add_argument(
        "--generate-board",
        nargs="?",
        const="AUTO",
        metavar="OUTPUT.png",
        help="generate the selected print-ready target and exit",
    )
    args = parser.parse_args()
    camera_mode = args.camera_mode or os.getenv("UR10E_CAMERA_MODE", "zedx_mini")
    if camera_mode in ("lidar", "stereo"):
        camera_mode = "zed_mini"
    camera_profile = load_camera_profile(camera_mode)
    child_frame = (
        frame_from_profile(camera_profile, "rgb_frame", "zed_mini_left_camera_frame")
        if camera_mode == "zedx_mini"
        else frame_from_profile(camera_profile, "rgb_frame", "zed2_left_camera_frame")
    )
    if args.generate_board:
        if args.generate_board == "AUTO":
            filename = (
                "charuco_7x24_30mm_22mm_dict5x5_100.png"
                if args.target == "charuco"
                else "chessboard_8x11_30mm.png"
            )
            output_path = OUTPUT_DIR / filename
        else:
            output_path = Path(args.generate_board).expanduser().resolve()
        generate_board_image(output_path, args.target)
        return

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
        rclpy.shutdown()
        return
    print()

    # ── Camera setup ─────────────────────────────────────────────────
    use_zedx_mini = camera_mode == "zedx_mini"
    if use_zedx_mini:
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 15
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        camera_label = "ZED X Mini"
        if args.resolution != "hd1080":
            node.get_logger().warn(
                f"ZED X Mini does not support {args.resolution} here; using HD1080.")
        resolution_label = "HD1080"
        hdr_label = "off"
    else:
        zed = sl.CameraOne()
        init_params = sl.InitParametersOne()
        if args.resolution == "4k":
            init_params.camera_resolution = sl.RESOLUTION.HD4K
            init_params.camera_fps = 15
            init_params.enable_hdr = False
        else:
            init_params.camera_resolution = sl.RESOLUTION.QHDPLUS
            init_params.camera_fps = 30
            init_params.enable_hdr = True
        init_params.coordinate_units = sl.UNIT.METER
        camera_label = "ZED X One Mono"
        resolution_label = args.resolution
        hdr_label = "on" if init_params.enable_hdr else "off"

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        node.get_logger().error(f"{camera_label} open failed: {status}")
        return
    node.get_logger().info(
        f"{camera_label} opened for hand-eye calibration "
        f"({resolution_label}, HDR={hdr_label}, child_frame={child_frame})."
    )

    # CameraOne: mono calibration directly. Stereo Camera: use left camera.
    cam_info = zed.get_camera_information()
    cal_params = cam_info.camera_configuration.calibration_parameters
    cam_params = cal_params.left_cam if use_zedx_mini else cal_params
    camera_matrix = np.array([
        [cam_params.fx, 0, cam_params.cx],
        [0, cam_params.fy, cam_params.cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)  # retrieve_image returns rectified frames — no distortion to pass to solvePnP
    node.get_logger().info(
        f"Intrinsics: fx={cam_params.fx:.1f} fy={cam_params.fy:.1f} "
        f"cx={cam_params.cx:.1f} cy={cam_params.cy:.1f}"
    )

    image_mat = sl.Mat()

    board = None
    detector = None
    objp = None
    criteria = None
    if args.target == "charuco":
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
        charuco_params = cv2.aruco.CharucoParameters()
        # Interpolate corners in image space. Supplying camera intrinsics here can
        # reject all corners when the SDK calibration geometry differs between
        # QHD+ and 4K, even though the marker outlines are detected correctly.
        charuco_params.minMarkers = 1
        charuco_params.tryRefineMarkers = True
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        marker_detector = cv2.aruco.ArucoDetector(
            dictionary, detector_params)
        charuco_detectors = []
        for squares_x, squares_y, orientation in (
            (BOARD_SQUARES_X, BOARD_SQUARES_Y, "portrait"),
            (BOARD_SQUARES_Y, BOARD_SQUARES_X, "landscape"),
            (BOARD_INNER_X, BOARD_INNER_Y, "inner-count fallback"),
            (BOARD_INNER_Y, BOARD_INNER_X, "rotated inner-count fallback"),
        ):
            for legacy_pattern, layout in ((False, "modern"), (True, "legacy")):
                _, candidate_board = create_charuco_board(
                    legacy_pattern=legacy_pattern,
                    squares_x=squares_x,
                    squares_y=squares_y,
                )
                charuco_detectors.append(
                    (
                        f"{squares_x}x{squares_y} {orientation} {layout}",
                        candidate_board,
                        cv2.aruco.CharucoDetector(
                            candidate_board, charuco_params, detector_params),
                    )
                )
    else:
        objp = np.zeros(
            (CHESSBOARD_INNER_X * CHESSBOARD_INNER_Y, 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:CHESSBOARD_INNER_X, 0:CHESSBOARD_INNER_Y
        ].T.reshape(-1, 2) * SQUARE_SIZE
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    # ── Storage for calibration data ──────────────────────────────────
    R_gripper2base_list = []   # rotation:    gripper -> base
    t_gripper2base_list = []   # translation: gripper -> base
    R_target2cam_list = []     # rotation:    chessboard -> camera
    t_target2cam_list = []     # translation: chessboard -> camera

    sample_count = 0

    node.get_logger().info(
        f"Hand-eye calibration ready.\n"
        + (
            f"  ChArUco: {BOARD_INNER_X}x{BOARD_INNER_Y} inner corners "
            f"({BOARD_SQUARES_X}x{BOARD_SQUARES_Y} squares), "
            f"square={SQUARE_SIZE*1000:.0f}mm, marker={MARKER_SIZE*1000:.0f}mm\n"
            f"  Dictionary: {ARUCO_DICTIONARY_NAME}\n"
            if args.target == "charuco"
            else f"  Chessboard: {CHESSBOARD_INNER_X}x{CHESSBOARD_INNER_Y} "
                 f"inner corners, square={SQUARE_SIZE*1000:.0f}mm\n"
        )
        +
        f"  Move the robot, press 'c' to capture, 'q' to finish.\n"
        f"  Aim for 15-20 diverse poses."
    )

    while True:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue

        if use_zedx_mini:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        else:
            zed.retrieve_image(image_mat)  # CameraOne: no VIEW arg
        frame = image_mat.get_data()[:, :, :3].copy()  # drop alpha channel (BGRA→BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        display = frame.copy()
        if args.target == "charuco":
            marker_corners, marker_ids, _ = marker_detector.detectMarkers(gray)
            best_detection = None
            for layout_name, candidate_board, candidate_detector in charuco_detectors:
                candidate_corners, candidate_ids, _, _ = candidate_detector.detectBoard(
                    gray,
                    markerCorners=marker_corners,
                    markerIds=marker_ids,
                )
                candidate_count = (
                    0 if candidate_ids is None else len(candidate_ids)
                )
                if best_detection is None or candidate_count > best_detection[0]:
                    best_detection = (
                        candidate_count,
                        layout_name,
                        candidate_board,
                        candidate_corners,
                        candidate_ids,
                    )
            corner_count, layout_name, active_board, charuco_corners, charuco_ids = (
                best_detection
            )
            marker_count = 0 if marker_ids is None else len(marker_ids)
            found = corner_count >= MIN_CHARUCO_CORNERS
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(
                    display, marker_corners, marker_ids)
            if charuco_ids is not None and corner_count > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    display, charuco_corners, charuco_ids)
            status_text = (
                f"ChArUco FOUND: {marker_count} markers, "
                f"{corner_count} corners ({layout_name}) - press 'c'"
                if found
                else f"Markers: {marker_count} | ChArUco corners: "
                     f"{corner_count}/{MIN_CHARUCO_CORNERS} ({layout_name})"
            )
        else:
            found, chessboard_corners = cv2.findChessboardCorners(
                gray,
                (CHESSBOARD_INNER_X, CHESSBOARD_INNER_Y),
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK,
            )
            if found:
                chessboard_corners = cv2.cornerSubPix(
                    gray, chessboard_corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(
                    display,
                    (CHESSBOARD_INNER_X, CHESSBOARD_INNER_Y),
                    chessboard_corners,
                    found,
                )
            status_text = (
                "Chessboard FOUND - press 'c'"
                if found
                else "No chessboard detected"
            )

        cv2.putText(
            display,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if found else (0, 0, 255),
            2,
        )

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

            if args.target == "charuco":
                obj_points, image_points = active_board.matchImagePoints(
                    charuco_corners, charuco_ids)
            else:
                obj_points, image_points = objp, chessboard_corners
            ret, rvec, tvec = cv2.solvePnP(
                obj_points, image_points, camera_matrix, dist_coeffs)
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
            "child_frame": child_frame,
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
            "camera_mode": camera_mode,
            "camera_profile": camera_profile.get("camera_profile"),
        }
    }

    OUTPUT_FILE.write_text(yaml.dump(result, default_flow_style=False))
    node.get_logger().info(f"Calibration saved to {OUTPUT_FILE}")

    profile_out = dict(camera_profile)
    profile_out["camera_profile"] = (
        profile_out.get("camera_profile") or "pending_context_selection")
    profile_out["mode"] = camera_mode
    profile_out["parent_frame"] = EE_FRAME
    profile_out["rgb_frame"] = child_frame
    if use_zedx_mini:
        profile_out["depth_frame"] = child_frame
    else:
        profile_out.setdefault("depth_frame", "zed_mini_left_camera_frame")
    profile_out["hand_eye"] = {
        "parent_frame": EE_FRAME,
        "child_frame": child_frame,
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
        "rpy": {
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
        },
        "matrix": T_cam2gripper.tolist(),
        "method": best_method_name,
        "mean_error_m": float(best_error),
        "num_samples": sample_count,
        "calibrated_at": timestamp,
    }
    print(f"\nBest method: {best_method_name}  (mean error: {best_error*1000:.2f} mm)")
    print("=" * 60)

    # ── Ask user to confirm URDF update ───────────────────────────────
    print("\n" + "=" * 60)
    print("UPDATE URDFs?")
    print("=" * 60)
    print(f"  Robot profile: {args.robot_profile}")
    print("  Environment/profile: selected after confirmation")
    print(f"  New transform ({EE_FRAME} → {child_frame}):")
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
        default_environment = os.getenv(
            "UR10E_ENVIRONMENT", "outdoor").strip().lower()
        if default_environment not in ("lab", "outdoor"):
            default_environment = "outdoor"
        while True:
            selected = input(
                f"Save/apply as lab or outdoor? "
                f"[lab/outdoor] (default: {default_environment}): "
            ).strip().lower()
            environment = selected or default_environment
            if environment in ("lab", "outdoor"):
                break
            print("  Please enter 'lab' or 'outdoor'.")

        target_name = contextual_profile_name(
            camera_mode,
            robot_profile=args.robot_profile,
            environment=environment,
        )
        target_path = profile_path_for_name(target_name)
        if target_path.exists():
            backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = target_path.with_suffix(
                f".yaml.backup_{backup_stamp}")
            shutil.copy2(target_path, backup_path)
            print(f"  BACKUP: {backup_path}")

        profile_out["camera_profile"] = target_name
        profile_out["robot_profile"] = args.robot_profile
        profile_out["environment"] = environment
        saved_profile = save_camera_profile(profile_out, target_path)
        node.get_logger().info(f"Camera profile saved to {saved_profile}")
        _update_urdfs(
            t_cam2gripper, rpy, best_method_name, best_error, sample_count,
            timestamp, child_frame, environment, args.robot_profile)
        print(f"  ACTIVE PROFILE: {target_name}")
        print("  Restart the full robot + vision system with:")
        print(f"    UR10E_ROBOT_PROFILE={args.robot_profile}")
        print(f"    UR10E_ENVIRONMENT={environment}")
    else:
        print(
            "Profile and URDFs not changed. The raw result remains in "
            f"{OUTPUT_FILE}.")

    rclpy.shutdown()


def _replace_environment_value(text, attribute, environment, value):
    """Replace one branch of an environment-conditional xacro attribute."""
    pattern = re.compile(
        rf'''({attribute}="\$\{{')(?P<lab>[^']*)'''
        rf'''(' if environment == 'lab' else ')'''
        rf'''(?P<outdoor>[^']*)('}}")'''
    )

    def replacement(match):
        lab_value = value if environment == "lab" else match.group("lab")
        outdoor_value = (
            value if environment == "outdoor" else match.group("outdoor"))
        return (
            f"{match.group(1)}{lab_value}{match.group(3)}"
            f"{outdoor_value}{match.group(5)}"
        )

    return pattern.sub(replacement, text, count=1)


def _replace_context_property(
    text, child_frame, attribute, robot_profile, environment, value,
):
    """Replace one robot/environment xacro hand-eye property."""
    property_name = (
        f"hand_eye_{child_frame}_{robot_profile}_{environment}_{attribute}")
    pattern = re.compile(
        rf'(<xacro:property\s+name="{re.escape(property_name)}"\s+value=")'
        rf'[^\"]*("\s*/>)'
    )
    return pattern.sub(rf'\g<1>{value}\g<2>', text, count=1)


def _update_urdfs(
    t, rpy, method, error_m, n_samples, timestamp, child_frame,
    environment, robot_profile,
):
    """Update the tool0→camera joint origin in each URDF/xacro file."""
    xyz_str = f"{t[0,0]:.6f} {t[1,0]:.6f} {t[2,0]:.6f}"
    rpy_str = f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"
    comment = (
        f"hand-eye calibrated ({method}, {error_m*1000:.2f}mm error, "
        f"{n_samples} samples, updated {timestamp})"
    )

    # Pattern A: xyz before rpy, optional trailing comment
    # <origin xyz="..." rpy="..."/>   <!-- ... -->   or without comment
    pat_xyz_first = re.compile(
        r'(<origin\s[^>]*xyz=")[^"]*("\s*rpy=")[^"]*("\s*/>[ \t]*)(<!--.*?-->)?'
    )
    repl_xyz_first = rf'\g<1>{xyz_str}\g<2>{rpy_str}\g<3><!-- {comment} -->'

    # Pattern B: rpy before xyz, optional trailing comment
    # <origin rpy="..." xyz="..."/>   <!-- ... -->   or without comment
    pat_rpy_first = re.compile(
        r'(<origin\s[^>]*rpy=")[^"]*("\s*xyz=")[^"]*("\s*/>[ \t]*)(<!--.*?-->)?'
    )
    repl_rpy_first = rf'\g<1>{rpy_str}\g<2>{xyz_str}\g<3><!-- {comment} -->'

    joint_pattern = re.compile(
        rf'(joint name="tool0_to_{re.escape(child_frame)}".*?</joint>)',
        re.DOTALL,
    )

    for urdf_path in URDF_FILES:
        if not urdf_path.exists():
            print(f"  SKIP (not found): {urdf_path}")
            continue
        # URDF comments may contain UTF-8 symbols (for example arrows or
        # degree signs).  Do not depend on the shell's ASCII locale.
        text = urdf_path.read_text(encoding="utf-8")
        match = joint_pattern.search(text)
        if not match:
            print(f"  SKIP (joint not found): {urdf_path}")
            continue
        joint_old = match.group(1)
        joint_new = joint_old

        context_property = (
            f"hand_eye_{child_frame}_{robot_profile}_{environment}_xyz")
        if context_property in joint_old:
            joint_new = _replace_context_property(
                joint_new, child_frame, "xyz", robot_profile, environment,
                xyz_str)
            joint_new = _replace_context_property(
                joint_new, child_frame, "rpy", robot_profile, environment,
                rpy_str)
        elif "environment == 'lab'" in joint_old:
            joint_new = _replace_environment_value(
                joint_new, "xyz", environment, xyz_str)
            joint_new = _replace_environment_value(
                joint_new, "rpy", environment, rpy_str)

        if joint_new == joint_old:
            joint_new = pat_xyz_first.sub(repl_xyz_first, joint_old)
        if joint_new == joint_old:
            joint_new = pat_rpy_first.sub(repl_rpy_first, joint_old)
        if joint_new == joint_old:
            print(f"  SKIP (origin pattern not matched): {urdf_path}")
            continue
        new_text = text[:match.start(1)] + joint_new + text[match.end(1):]
        urdf_path.write_text(new_text, encoding="utf-8")
        print(f"  UPDATED: {urdf_path}")


if __name__ == "__main__":
    main()
