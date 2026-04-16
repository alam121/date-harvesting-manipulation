"""
Quick depth sanity check — ZED X Mini depth projected onto ZED X One image.

Usage:
    python3 check_depth.py [expected_distance_m]

Hold a flat object at a known distance in front of both cameras.
The script shows the depth overlay and prints the median depth of the
centre region.  Press 'q' to quit.

Example:
    python3 check_depth.py 0.5     # object is 50 cm away
"""

import sys
import numpy as np
import cv2
import pyzed.sl as sl

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[4]))
from ur10e_curobo.vision.config import (
    T_CAM_ZEDMINI, ZEDMINI_DEPTH_Z_MIN, ZEDMINI_DEPTH_Z_MAX, ZEDMINI_SERIAL
)
from ur10e_curobo.perception_lidar import project_lidar_to_image

EXPECTED = float(sys.argv[1]) if len(sys.argv) > 1 else None
CENTRE_ROI = 0.15   # fraction of image used for centre-median check


def open_zed_one():
    cam = sl.CameraOne()
    p = sl.InitParametersOne()
    p.camera_resolution = sl.RESOLUTION.HD1200
    p.camera_fps = 10
    p.coordinate_units = sl.UNIT.METER
    p.enable_hdr = True
    p.sdk_verbose = 0
    assert cam.open(p) == sl.ERROR_CODE.SUCCESS, "ZED One open failed"
    print("[ZedOne] opened")
    return cam


def open_zed_mini():
    cam = sl.Camera()
    p = sl.InitParameters()
    if ZEDMINI_SERIAL > 0:
        p.input.set_from_serial_number(ZEDMINI_SERIAL)
    p.camera_resolution = sl.RESOLUTION.SVGA
    p.camera_fps = 30
    p.coordinate_units = sl.UNIT.METER
    p.depth_mode = sl.DEPTH_MODE.NEURAL
    p.depth_minimum_distance = ZEDMINI_DEPTH_Z_MIN
    p.depth_maximum_distance = ZEDMINI_DEPTH_Z_MAX
    p.sdk_verbose = 0
    assert cam.open(p) == sl.ERROR_CODE.SUCCESS, "ZED Mini open failed"
    print("[ZedMini] opened")
    return cam


def get_intrinsics(cam_info, mono):
    if mono:
        c = cam_info.camera_configuration.calibration_parameters
    else:
        c = cam_info.camera_configuration.calibration_parameters.left_cam
    K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(c.disto[:5], dtype=np.float64)
    return K, dist


def main():
    zed_one  = open_zed_one()
    zed_mini = open_zed_mini()

    K_one, dist_one = get_intrinsics(zed_one.get_camera_information(),  mono=True)
    T = np.array(T_CAM_ZEDMINI, dtype=np.float64)

    mat_one  = sl.Mat()
    mat_pc   = sl.Mat()
    scale    = 0.5

    cv2.namedWindow("Depth check", cv2.WINDOW_NORMAL)
    print("\nPress 'q' to quit.  Centre-region depth is printed every frame.")
    if EXPECTED:
        print(f"Expected depth: {EXPECTED:.3f} m\n")

    while True:
        if zed_one.grab()  != sl.ERROR_CODE.SUCCESS: continue
        if zed_mini.grab() != sl.ERROR_CODE.SUCCESS: continue

        zed_one.retrieve_image(mat_one)
        zed_mini.retrieve_measure(mat_pc, sl.MEASURE.XYZ, sl.MEM.CPU)

        img  = cv2.cvtColor(mat_one.get_data(), cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]

        pc_np = mat_pc.get_data()[:, :, :3].reshape(-1, 3).astype(np.float32)
        valid = (np.isfinite(pc_np).all(axis=1) &
                 (pc_np[:, 2] > ZEDMINI_DEPTH_Z_MIN) &
                 (pc_np[:, 2] < ZEDMINI_DEPTH_Z_MAX))
        pts = pc_np[valid]

        if pts.shape[0] > 0:
            pts_cam, uv = project_lidar_to_image(
                pts, K_one, dist_one, T, w, h,
                z_min=ZEDMINI_DEPTH_Z_MIN, z_max=ZEDMINI_DEPTH_Z_MAX,
            )

            # Colour by depth
            zs = pts_cam[:, 2]
            t  = np.clip((zs - ZEDMINI_DEPTH_Z_MIN) /
                         max(ZEDMINI_DEPTH_Z_MAX - ZEDMINI_DEPTH_Z_MIN, 1e-3), 0, 1)
            r = np.clip(255 * (1 - 2*t),        0, 255).astype(np.uint8)
            g = np.clip(255 * (1 - abs(2*t-1)), 0, 255).astype(np.uint8)
            b = np.clip(255 * (2*t - 1),        0, 255).astype(np.uint8)
            xs = uv[:, 0].astype(np.int32)
            ys = uv[:, 1].astype(np.int32)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            for i in np.where(ok)[0]:
                cv2.circle(img, (xs[i], ys[i]), 3, (int(b[i]), int(g[i]), int(r[i])), -1)

            # Centre-region median
            cx1 = int(w * (0.5 - CENTRE_ROI))
            cx2 = int(w * (0.5 + CENTRE_ROI))
            cy1 = int(h * (0.5 - CENTRE_ROI))
            cy2 = int(h * (0.5 + CENTRE_ROI))
            centre_mask = ok & (xs >= cx1) & (xs < cx2) & (ys >= cy1) & (ys < cy2)
            if centre_mask.any():
                median_z = float(np.median(pts_cam[centre_mask, 2]))
                err_str = ""
                if EXPECTED:
                    err_m  = median_z - EXPECTED
                    err_pc = 100 * err_m / EXPECTED
                    err_str = f"  error: {err_m*100:+.1f} cm ({err_pc:+.1f}%)"
                print(f"Centre depth: {median_z:.3f} m{err_str}")
                label = f"Z={median_z:.3f}m"
                cv2.putText(img, label, (w//2 - 60, h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw centre ROI box
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

        disp = cv2.resize(img, (int(w*scale), int(h*scale)))
        cv2.imshow("Depth check", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    zed_one.close()
    zed_mini.close()


if __name__ == "__main__":
    main()
