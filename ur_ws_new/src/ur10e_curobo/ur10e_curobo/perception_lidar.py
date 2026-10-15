
#!/usr/bin/env python3
"""
Perception module — ZED X One 4K (mono, image only) + Livox Mid-70 LiDAR.

Calibration convention
----------------------
  T_cam_lidar : (4,4) float64
      Transforms a point expressed in the *LiDAR* frame into the *camera* frame.
      P_cam = T_cam_lidar @ [x, y, z, 1]^T

  K : (3,3) float64   — camera intrinsic matrix  
  dist_coeffs : (5,)  — OpenCV distortion [k1,k2,p1,p2,k3]  

"""

import numpy as np
import cv2

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2_utils   # pip install sensor-msgs-py  (or ros-<distro>-sensor-msgs-py)

# ──────────────────────────────────────────────
# QoS
# ──────────────────────────────────────────────
FAST_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# ──────────────────────────────────────────────
# Generic math helpers  (unchanged from original)
# ──────────────────────────────────────────────
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ──────────────────────────────────────────────
# LiDAR ↔ image helpers  (replaces depth-map helpers)
# ──────────────────────────────────────────────

def parse_pointcloud2(msg: PointCloud2) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 message to an (N,3) float32 array [x,y,z]
    in the LiDAR frame.  NaN / inf rows are dropped.
    """
    gen = pc2_utils.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    raw = np.array(list(gen))
    if raw.ndim == 0 or raw.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    # ros2 humble returns a structured array — view as plain float32
    if raw.dtype.names:
        pts = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float32)
    else:
        pts = raw.astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def project_lidar_to_image(
    pts_lidar: np.ndarray,          # (N,3)  in lidar frame
    K: np.ndarray,                  # (3,3)
    dist_coeffs: np.ndarray,        # (5,)
    T_cam_lidar: np.ndarray,        # (4,4)  lidar → camera
    img_w: int,
    img_h: int,
    z_min: float = 0.10,
    z_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points into image coordinates.

    Returns
    -------
    pts_cam   : (M,3)  3-D points in *camera* frame (already depth-filtered)
    uv        : (M,2)  corresponding pixel coordinates  [col, row]
    """
    if pts_lidar.shape[0] == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)

    # Transform to camera frame
    ones = np.ones((pts_lidar.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts_lidar, ones])                 # (N,4)
    pts_cam = (T_cam_lidar @ pts_h.T).T[:, :3]          # (N,3)

    # Keep only points in front of the camera and within depth range
    depth = pts_cam[:, 2]
    valid = (depth > z_min) & (depth < z_max)
    pts_cam = pts_cam[valid]
    if pts_cam.shape[0] == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)

    # Project with distortion via OpenCV
    rvec = np.zeros(3, np.float32)
    tvec = np.zeros(3, np.float32)
    uv_raw, _ = cv2.projectPoints(
        pts_cam.reshape(-1, 1, 3).astype(np.float32),
        rvec, tvec, K.astype(np.float32), dist_coeffs.astype(np.float32)
    )
    uv = uv_raw.reshape(-1, 2)                           # (M,2) float

    # Keep only points whose projection lands inside the image
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    )
    return pts_cam[in_bounds], uv[in_bounds]
