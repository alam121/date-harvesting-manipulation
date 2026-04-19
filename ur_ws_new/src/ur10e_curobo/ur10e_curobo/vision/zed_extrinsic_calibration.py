"""
Extrinsic calibration: find T_CAM_ZEDMINI (ZED X Mini left-cam → ZED X One Mono).

Usage
-----
1.  Print (or display on a monitor) a checkerboard.
    Default: 8x6 inner corners, 30 mm squares.  Change BOARD_* below to match yours.
2.  Run this script (no ROS needed):
        python3 zed_extrinsic_calibration.py
3.  Hold the board so it is fully visible in BOTH camera windows.
4.  Press 's' to save a frame pair (aim for 15-20 pairs, different angles/distances).
5.  Press 'q' when done.  The script prints the 4x4 matrix to paste into config.py.

Tips
----
- Tilt the board at various angles (not always face-on) for a well-conditioned solve.
- Cover the full FOV of both cameras across your captures.
- Reject blurry captures — the windows turn red when corners are not found.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from scipy.spatial.transform import Rotation

# ── Checkerboard parameters ────────────────────────────────────────────────────
BOARD_ROWS   = 10      # inner corners along the long edge  (11 squares - 1)
BOARD_COLS   = 7       # inner corners along the short edge (8 squares - 1)
SQUARE_SIZE  = 0.030   # square side length in metres (30 mm)
MIN_CAPTURES = 10      # minimum valid pairs before 'q' is accepted

# ── Camera open parameters ─────────────────────────────────────────────────────
# ZED X One Mono  (detection camera — CameraOne API)
ZEDONE_RESOLUTION  = sl.RESOLUTION.QHDPLUS
ZEDONE_FPS         = 10

# ZED X Mini  (depth camera — stereo Camera API)
# Supported resolutions: HD1200, HD1080, SVGA  (HD720 is NOT supported on ZED X Mini)
ZEDMINI_SERIAL     = 0           # 0 = auto-detect; set to actual SN if needed
ZEDMINI_RESOLUTION = sl.RESOLUTION.HD1080
ZEDMINI_FPS        = 30

# Display scale — shrink images so they fit on screen
DISPLAY_SCALE = 0.40

# Output file (also printed to stdout so you can copy-paste)
OUTPUT_FILE = Path(__file__).parent / "T_CAM_ZEDMINI_result.txt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def open_zed_one() -> sl.CameraOne:
    cam = sl.CameraOne()
    p = sl.InitParametersOne()
    p.camera_resolution = ZEDONE_RESOLUTION
    p.camera_fps = ZEDONE_FPS
    p.coordinate_units = sl.UNIT.METER
    p.sdk_verbose = 0
    status = cam.open(p)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ZedOne] Failed to open: {status}")
        sys.exit(1)
    print("[ZedOne] opened")
    return cam


def open_zed_mini() -> sl.Camera:
    cam = sl.Camera()
    p = sl.InitParameters()
    if ZEDMINI_SERIAL > 0:
        p.input.set_from_serial_number(ZEDMINI_SERIAL)
    p.camera_resolution = ZEDMINI_RESOLUTION
    p.camera_fps = ZEDMINI_FPS
    p.coordinate_units = sl.UNIT.METER
    p.depth_mode = sl.DEPTH_MODE.NONE   # no depth needed during calibration grab
    p.sdk_verbose = 0
    status = cam.open(p)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ZedMini] Failed to open: {status}")
        sys.exit(1)
    print("[ZedMini] opened")
    return cam


def get_intrinsics(cam_info, is_mono: bool):
    """Return (K, dist) from ZED camera_information.
    dist is always zeros — retrieve_image returns rectified frames so solvePnP
    must not apply distortion correction a second time."""
    if is_mono:
        cal = cam_info.camera_configuration.calibration_parameters
        fx, fy = cal.fx, cal.fy
        cx, cy = cal.cx, cal.cy
    else:
        cal = cam_info.camera_configuration.calibration_parameters.left_cam
        fx, fy = cal.fx, cal.fy
        cx, cy = cal.cx, cal.cy
    K    = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return K, dist


def grab_bgr(cam, is_mono: bool):
    """Grab one frame and return a BGR numpy array, or None on failure."""
    mat = sl.Mat()
    if is_mono:
        if cam.grab() != sl.ERROR_CODE.SUCCESS:
            return None
        cam.retrieve_image(mat)
    else:
        if cam.grab() != sl.ERROR_CODE.SUCCESS:
            return None
        cam.retrieve_image(mat, sl.VIEW.LEFT)
    bgra = mat.get_data()
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def find_corners(img_bgr, board_size):
    """Return (corners, gray) or (None, gray) if not found."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    return (corners if found else None), gray


def pose_from_corners(corners, board_pts, K, dist):
    """solvePnP → 4×4 T_cam_board (board origin expressed in camera frame)."""
    ok, rvec, tvec = cv2.solvePnP(board_pts, corners, K, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


def average_transforms(T_list):
    """Average a list of 4×4 rigid transforms (rotation via quaternion averaging)."""
    trans = np.mean([T[:3, 3] for T in T_list], axis=0)
    quats = [Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_list]
    # Flip sign to same hemisphere before averaging
    ref = quats[0]
    quats = [q if np.dot(q, ref) >= 0 else -q for q in quats]
    avg_quat = np.mean(quats, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)
    R_avg = Rotation.from_quat(avg_quat).as_matrix()
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3,  3] = trans
    return T_avg


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    board_size = (BOARD_ROWS, BOARD_COLS)
    board_pts  = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    board_pts[:, :2] = np.mgrid[0:BOARD_ROWS, 0:BOARD_COLS].T.reshape(-1, 2)
    board_pts *= SQUARE_SIZE
    print(f"Checkerboard: {BOARD_ROWS}x{BOARD_COLS} corners, {SQUARE_SIZE*1000:.1f} mm squares")
    # Open cameras
    zed_one  = open_zed_one()
    zed_mini = open_zed_mini()

    # Read intrinsics
    K_one,  dist_one  = get_intrinsics(zed_one.get_camera_information(),  is_mono=True)
    K_mini, dist_mini = get_intrinsics(zed_mini.get_camera_information(), is_mono=False)
    print(f"[ZedOne]  K fx={K_one[0,0]:.1f} fy={K_one[1,1]:.1f} "
          f"cx={K_one[0,2]:.1f} cy={K_one[1,2]:.1f}")
    print(f"[ZedMini] K fx={K_mini[0,0]:.1f} fy={K_mini[1,1]:.1f} "
          f"cx={K_mini[0,2]:.1f} cy={K_mini[1,2]:.1f}")

    T_one_mini_list = []   # accumulated extrinsic estimates
    print(f"\nPress 's' to capture, 'q' to finish (need ≥ {MIN_CAPTURES} valid pairs).")
    print("Hold the checkerboard so it is FULLY visible in BOTH windows.\n")

    cv2.namedWindow("ZedOne (detection)",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("ZedMini (depth)",     cv2.WINDOW_NORMAL)

    while True:
        img_one  = grab_bgr(zed_one,  is_mono=True)
        img_mini = grab_bgr(zed_mini, is_mono=False)
        if img_one is None or img_mini is None:
            continue

        corners_one,  _ = find_corners(img_one,  board_size)
        corners_mini, _ = find_corners(img_mini, board_size)

        # Annotate live preview
        vis_one  = img_one.copy()
        vis_mini = img_mini.copy()
        found_both = corners_one is not None and corners_mini is not None

        if corners_one is not None:
            cv2.drawChessboardCorners(vis_one,  board_size, corners_one,  True)
        else:
            cv2.rectangle(vis_one,  (0, 0), (vis_one.shape[1],  vis_one.shape[0]),
                          (0, 0, 200), 8)

        if corners_mini is not None:
            cv2.drawChessboardCorners(vis_mini, board_size, corners_mini, True)
        else:
            cv2.rectangle(vis_mini, (0, 0), (vis_mini.shape[1], vis_mini.shape[0]),
                          (0, 0, 200), 8)

        n = len(T_one_mini_list)
        status_text = (f"Captured: {n}  |  Board: {'FOUND' if found_both else 'NOT FOUND'}"
                       f"  |  's'=save  'q'=compute")
        for vis in (vis_one, vis_mini):
            cv2.putText(vis, status_text, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if found_both else (0, 80, 255), 2, cv2.LINE_AA)

        h1, w1 = vis_one.shape[:2]
        h2, w2 = vis_mini.shape[:2]
        cv2.imshow("ZedOne (detection)",
                   cv2.resize(vis_one,  (int(w1 * DISPLAY_SCALE), int(h1 * DISPLAY_SCALE))))
        cv2.imshow("ZedMini (depth)",
                   cv2.resize(vis_mini, (int(w2 * DISPLAY_SCALE), int(h2 * DISPLAY_SCALE))))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if not found_both:
                print("[SKIP] Board not found in both cameras — move it into view.")
                continue
            T_one_board  = pose_from_corners(corners_one,  board_pts, K_one,  dist_one)
            T_mini_board = pose_from_corners(corners_mini, board_pts, K_mini, dist_mini)
            if T_one_board is None or T_mini_board is None:
                print("[SKIP] solvePnP failed.")
                continue
            # T_one_mini = T_one_board @ inv(T_mini_board)
            T_one_mini = T_one_board @ np.linalg.inv(T_mini_board)
            T_one_mini_list.append(T_one_mini)
            print(f"[SAVED] pair {len(T_one_mini_list):02d}  "
                  f"Z_one={T_one_board[2,3]:.3f}m  Z_mini={T_mini_board[2,3]:.3f}m")

        elif key == ord('q'):
            if len(T_one_mini_list) < MIN_CAPTURES:
                print(f"Need at least {MIN_CAPTURES} pairs (have {len(T_one_mini_list)}). "
                      "Keep capturing.")
            else:
                break

    cv2.destroyAllWindows()
    zed_one.close()
    zed_mini.close()

    if not T_one_mini_list:
        print("No pairs captured — exiting.")
        return

    # ── Compute final extrinsic ────────────────────────────────────────────────
    T_final = average_transforms(T_one_mini_list)

    # Report per-pair residuals so you can spot outliers
    print(f"\nPer-pair translation residuals (vs average):")
    for i, T in enumerate(T_one_mini_list):
        d = np.linalg.norm(T[:3, 3] - T_final[:3, 3]) * 100
        print(f"  pair {i+1:02d}: {d:.1f} mm offset from mean")

    print(f"\nT_CAM_ZEDMINI  ({len(T_one_mini_list)} pairs averaged):")
    rows = []
    for row in T_final.tolist():
        formatted = "[" + ", ".join(f"{v:12.8f}" for v in row) + "],"
        print("   ", formatted)
        rows.append(row)

    config_block = "T_CAM_ZEDMINI = [\n"
    for row in rows:
        config_block += "    [" + ", ".join(f"{v:12.8f}" for v in row) + "],\n"
    config_block += "]"

    print(f"\n--- paste into config.py ---\n{config_block}\n----------------------------")

    OUTPUT_FILE.write_text(config_block + "\n")
    print(f"Result saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
