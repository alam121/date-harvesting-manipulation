
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

from threading import Thread, Lock
from time import sleep
from typing import Optional
import math

import numpy as np
import cv2
import torch
import os

import pyzed.sl as sl
from ultralytics import YOLO

import rclpy
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PointStamped, PoseStamped
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


def quat_rotate_vec(q, v3):
    w, x, y, z = q
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (w, -x, -y, -z)

    def qm(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )

    return qm(qm(q, qv), qi)[1:]


def quat_align_x_to_axis(axis_world, up_hint=(0, 0, 1)):
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
                  [X[2], Y[2], Z[2]]], float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25*s; qx = (R[2,1]-R[1,2])/s; qy = (R[0,2]-R[2,0])/s; qz = (R[1,0]-R[0,1])/s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2.0
            qw = (R[2,1]-R[1,2])/s; qx = 0.25*s; qy = (R[0,1]+R[1,0])/s; qz = (R[0,2]+R[2,0])/s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2.0
            qw = (R[0,2]-R[2,0])/s; qx = (R[0,1]+R[1,0])/s; qy = 0.25*s; qz = (R[1,2]+R[2,1])/s
        else:
            s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2.0
            qw = (R[1,0]-R[0,1])/s; qx = (R[0,2]+R[2,0])/s; qy = (R[1,2]+R[2,1])/s; qz = 0.25*s
    return float(qw), float(qx), float(qy), float(qz)


# ──────────────────────────────────────────────
# LiDAR ↔ image helpers  (replaces depth-map helpers)
# ──────────────────────────────────────────────

def parse_pointcloud2(msg: PointCloud2) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 message to an (N,3) float32 array [x,y,z]
    in the LiDAR frame.  NaN / inf rows are dropped.
    """
    gen = pc2_utils.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    pts = np.array(list(gen), dtype=np.float32)          # (N,3)
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


def lidar_pts_in_mask(
    pts_cam: np.ndarray,    # (M,3)  camera-frame 3-D points
    uv: np.ndarray,         # (M,2)  pixel coords  [col, row]
    mask_bin: np.ndarray,   # (H,W)  uint8 binary mask
) -> np.ndarray:
    """
    Return the subset of pts_cam whose projected pixel falls inside mask_bin.
    """
    if pts_cam.shape[0] == 0:
        return np.empty((0, 3), np.float32)
    col = np.round(uv[:, 0]).astype(int)
    row = np.round(uv[:, 1]).astype(int)
    H, W = mask_bin.shape
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)
    inside = mask_bin[row, col] == 1
    return pts_cam[inside]


def centroid_from_lidar_pts(
    pts_cam: np.ndarray,    # (M,3)
    min_pts: int = 5,
) -> Optional[tuple[float, float, float]]:
    """Median centroid of LiDAR points belonging to a mask (camera frame)."""
    if pts_cam.shape[0] < min_pts:
        return None
    Xm = float(np.median(pts_cam[:, 0]))
    Ym = float(np.median(pts_cam[:, 1]))
    Zm = float(np.median(pts_cam[:, 2]))
    return Xm, Ym, Zm


def pca_long_axis_from_lidar_pts(
    pts_cam: np.ndarray,    # (M,3)
    min_pts: int = 8,
) -> np.ndarray:
    """PCA dominant axis of LiDAR points belonging to a mask (camera frame)."""
    if pts_cam.shape[0] < min_pts:
        return np.array([1.0, 0.0, 0.0])
    P = pts_cam - pts_cam.mean(axis=0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return _unit(Vt[0])


# Main perception   

class ZedOneYoloLidarPerception:
    """
    Background worker that:
      - grabs ZED X One frames (image only, no depth)
      - subscribes to Livox Mid-70 PointCloud2
      - runs YOLOv8-seg
      - projects LiDAR points into each YOLO mask
      - computes 3-D centroid + PCA axis from LiDAR points inside the mask
      - publishes PoseStamped on /external_goal_pose for the planner
    """

    def __init__(
        self,
        node,
        weights: str = "yolov8m-seg.pt",
        img_size: int = 640,
        conf_thres: float = 0.4,
        cam_frame: str = "zed_x_one_left_camera_frame",
        lidar_topic: str = "/livox/lidar",
        # ── calibration ──────────────────────────────
        K: Optional[np.ndarray] = None,            # (3,3) intrinsics
        dist_coeffs: Optional[np.ndarray] = None,  # (5,)  distortion
        T_cam_lidar: Optional[np.ndarray] = None,  # (4,4) extrinsics
        # ─────────────────────────────────────────────
        lidar_z_min: float = 0.10,
        lidar_z_max: float = 5.0,
        show_view: bool = False,
    ):
        self.node       = node
        self.weights    = weights
        self.img_size   = img_size
        self.conf_thres = conf_thres
        self.cam_frame  = cam_frame
        self.show_view  = show_view
        self.lidar_z_min = lidar_z_min
        self.lidar_z_max = lidar_z_max

        # Calibration 
        self.K           = K           if K           is not None else np.eye(3, dtype=np.float64)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float64)
        self.T_cam_lidar = T_cam_lidar if T_cam_lidar is not None else np.eye(4, dtype=np.float64)

        # Threading
        self._lock   = Lock()
        self._thread: Optional[Thread] = None
        self._alive  = False

        # Latest LiDAR cloud 
        self._latest_cloud: Optional[np.ndarray] = None   # (N,3) in lidar frame
        self._cloud_lock = Lock()

        # Latest target mask
        self.latest_target_mask: Optional[np.ndarray] = None
        self._mask_lock = Lock()

        # ROS publishers
        self.goal_pub  = node.create_publisher(PoseStamped,  "/external_goal_pose",  FAST_QOS)
        self.point_pub = node.create_publisher(PointStamped, "/datefruit_3d_point",  10)

        # LiDAR subscriber
        self._cloud_sub = node.create_subscription(
            PointCloud2,
            lidar_topic,
            self._cloud_callback,
            FAST_QOS,
        )
        node.get_logger().info(f"Subscribed to LiDAR topic: {lidar_topic}")

        # YOLO
        self.model = YOLO(self.weights)
        use_gpu = os.getenv("UR10E_PERCEPTION_USE_GPU", "0") == "1" and torch.cuda.is_available()
        self.model.to("cuda" if use_gpu else "cpu").eval()
        node.get_logger().info(f"ZED X One / YOLO / Livox using: {'GPU' if use_gpu else 'CPU'}")

    # LiDAR callback  
    def _cloud_callback(self, msg: PointCloud2):
        pts = parse_pointcloud2(msg)
        with self._cloud_lock:
            self._latest_cloud = pts

    # Lifecycle 
    def start(self):
        if self._alive:
            return
        self._alive = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.node.get_logger().info("ZedOneYoloLidarPerception started.")

    def stop(self):
        if not self._alive:
            return
        self._alive = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.node.get_logger().info("ZedOneYoloLidarPerception stopped.")

    # Main worker loop 
    def _loop(self):
        # ── ZED X One init (image only, no depth) ──
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD4K   # ZED X One 4K
        init_params.camera_fps        = 30
        init_params.depth_mode        = sl.DEPTH_MODE.NONE    # mono — no depth from SDK
        init_params.coordinate_units  = sl.UNIT.METER

        self.node.get_logger().info("Opening ZED X One …")
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.node.get_logger().error(f"ZED X One open failed: {repr(status)}")
            return
        self.node.get_logger().info("ZED X One opened.")

        # Pre-allocate SDK mat
        runtime_params = sl.RuntimeParameters()
        image_left     = sl.Mat()
        disp_img       = None

        while self._alive:
            #  Grab camera frame 
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                sleep(0.005)
                continue

            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            img_rgba = image_left.get_data()
            img_rgb  = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
            img_h, img_w = img_rgb.shape[:2]

            # YOLO segmentation 
            with torch.no_grad():
                det = self.model.predict(
                    img_rgb,
                    save=False,
                    retina_masks=True,
                    imgsz=self.img_size,
                    conf=self.conf_thres,
                    iou=0.45,
                    verbose=False,
                )[0]

            # Collect full-resolution binary masks
            masks, labels, scores = [], [], []
            if det.masks is not None and det.masks.data is not None:
                H, W = det.orig_shape
                for i in range(len(det.boxes)):
                    m    = det.masks.data[i].float().cpu().numpy()
                    m    = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    mbin = (m > 0.5).astype(np.uint8)
                    masks.append(mbin)
                    labels.append(int(det.boxes.cls[i].item()))
                    scores.append(float(det.boxes.conf[i].item()))

            #   Get latest LiDAR cloud & project into image 
            with self._cloud_lock:
                pts_lidar = self._latest_cloud  # (N,3) or None

            pts_cam, uv = np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)
            if pts_lidar is not None and pts_lidar.shape[0] > 0:
                pts_cam, uv = project_lidar_to_image(
                    pts_lidar, self.K, self.dist_coeffs, self.T_cam_lidar,
                    img_w, img_h,
                    z_min=self.lidar_z_min,
                    z_max=self.lidar_z_max,
                )

            #  Build combined target mask 
            if masks:
                combined = np.zeros((img_h, img_w), dtype=bool)
                for mbin in masks:
                    combined |= (mbin > 0)
                with self._mask_lock:
                    self.latest_target_mask = combined

            #  Per-detection: localise via LiDAR & publish 
            for mbin, cls_i, conf_i in zip(masks, labels, scores):

                # LiDAR points inside this mask (camera frame)
                in_mask = lidar_pts_in_mask(pts_cam, uv, mbin)

                centroid = centroid_from_lidar_pts(in_mask, min_pts=5)
                if centroid is None:
                    continue
                Xc, Yc, Zc = centroid

                # Build PointStamped in camera frame for TF2 transform
                pt_cam_msg = PointStamped()
                pt_cam_msg.header.frame_id = self.cam_frame
                pt_cam_msg.header.stamp    = rclpy.time.Time().to_msg()  # t=0 → latest TF
                pt_cam_msg.point.x = Xc
                pt_cam_msg.point.y = Yc
                pt_cam_msg.point.z = Zc

                try:
                    # Transform centroid to robot base frame
                    pt_base = self.node.tf_buffer.transform(
                        pt_cam_msg, "base_link", timeout=rclpyDuration(seconds=0.2)
                    )

                    # PCA dominant axis in camera frame → rotate to base frame
                    axis_cam  = pca_long_axis_from_lidar_pts(in_mask)
                    if axis_cam[2] < 0:
                        axis_cam = -axis_cam

                    tf_cb = self.node.tf_buffer.lookup_transform(
                        "base_link", self.cam_frame, rclpy.time.Time().to_msg()
                    )
                    q = tf_cb.transform.rotation
                    axis_base = np.array(
                        quat_rotate_vec((q.w, q.x, q.y, q.z), axis_cam), float
                    )
                    axis_base = _unit(axis_base)
                    qw, qx, qy, qz = quat_align_x_to_axis(axis_base, up_hint=(0, 0, 1))

                    # Publish goal pose
                    goal = PoseStamped()
                    goal.header = pt_base.header
                    goal.pose.position.x    = pt_base.point.x
                    goal.pose.position.y    = pt_base.point.y
                    goal.pose.position.z    = pt_base.point.z
                    goal.pose.orientation.w = qw
                    goal.pose.orientation.x = qx
                    goal.pose.orientation.y = qy
                    goal.pose.orientation.z = qz
                    self.goal_pub.publish(goal)

                    # Publish 3-D point
                    self.point_pub.publish(pt_base)

                    # Optional preview
                    if self.show_view:
                        if disp_img is None:
                            disp_img = np.zeros_like(img_rgba)
                        disp_img[:] = img_rgba
                        # Overlay projected LiDAR points that are inside the mask
                        if in_mask.shape[0] > 0:
                            in_mask_uv = uv[
                                np.where(
                                    (lidar_pts_in_mask(pts_cam, uv, mbin)[:, 0:1] ==
                                     pts_cam[:, 0:1]).all(axis=1)
                                )[0]
                            ] if False else uv  # simplified: draw all projected pts
                            for u_, v_ in uv.astype(int):
                                cv2.circle(disp_img, (u_, v_), 2, (0, 0, 255, 255), -1)
                        # Draw centroid
                        fx = self.K[0, 0]; fy = self.K[1, 1]
                        cx_ = self.K[0, 2]; cy_ = self.K[1, 2]
                        cu = int(fx * Xc / Zc + cx_)
                        cv_ = int(fy * Yc / Zc + cy_)
                        cv2.circle(disp_img, (cu, cv_), 7, (0, 255, 0, 255), -1)
                        cv2.imshow("Perception preview", disp_img)
                        cv2.waitKey(1)

                except Exception as e:
                    self.node.get_logger().warn(
                        f"Perception publish failed: {type(e).__name__}: {e}"
                    )

            sleep(0.002)

        #  Cleanup
        try:
            zed.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
