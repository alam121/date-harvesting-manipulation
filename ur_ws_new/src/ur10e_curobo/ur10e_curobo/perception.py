# ur10e_curobo/perception.py
# ruff: noqa
#!/usr/bin/env python3
from threading import Thread, Lock
from time import sleep, time
from typing import List, Optional, Tuple
import tf2_geometry_msgs

from ultralytics import YOLO

import os
import math
import numpy as np
import cv2
import torch
import pyzed.sl as sl
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration as rclpyDuration

# ----------- simple QoS (or import from your config.py) -----------
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
FAST_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# =========================
# Small helpers (trimmed)
# =========================
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def quat_rotate_vec(q, v3):
    w,x,y,z = q
    qv = (0.0, v3[0], v3[1], v3[2])
    qi = (w, -x, -y, -z)
    def qm(a,b):
        aw,ax,ay,az = a; bw,bx,by,bz = b
        return (
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        )
    return qm(qm(q, qv), qi)[1:]

def quat_align_x_to_axis(axis_world, up_hint=(0,0,1)):
    X = _unit(axis_world); U = _unit(up_hint)
    Y = np.cross(U, X)
    if np.linalg.norm(Y) < 1e-6:
        U = np.array([1.0,0.0,0.0]); Y = np.cross(U, X)
    Y = _unit(Y); Z = _unit(np.cross(X, Y))
    R = np.array([[X[0],Y[0],Z[0]],[X[1],Y[1],Z[1]],[X[2],Y[2],Z[2]]], float)
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        qw = 0.25*s; qx = (R[2,1] - R[1,2]) / s; qy = (R[0,2]-R[2,0]) / s; qz = (R[1,0]-R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1]-R[1,2]) / s; qx = 0.25*s; qy = (R[0,1]+R[1,0]) / s; qz = (R[0,2]+R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2]-R[2,0]) / s; qx = (R[0,1]+R[1,0]) / s; qy = 0.25*s; qz = (R[1,2]+R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0]-R[0,1]) / s; qx = (R[0,2]+R[2,0]) / s; qy = (R[1,2]+R[2,1]) / s; qz = 0.25*s
    return float(qw), float(qx), float(qy), float(qz)

def pca_long_axis_from_mask(mask_bin, xyz_cam):
    ys, xs = np.where(mask_bin == 1)
    if ys.size < 8: return np.array([1.0,0.0,0.0])
    pts = xyz_cam[ys, xs, :3]
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if pts.shape[0] < 8: return np.array([1.0,0.0,0.0])
    P = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return _unit(Vt[0])

def centroid_xyz_in_mask(mask_bin, xyz_cam, z_min=0.1, z_max=2.5):
    ys, xs = np.where(mask_bin == 1)
    if ys.size < 8: return None
    X = xyz_cam[ys, xs, 0]; Y = xyz_cam[ys, xs, 1]; Z = xyz_cam[ys, xs, 2]
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (Z > z_min) & (Z < z_max)
    if valid.sum() < 8: return None
    Xm = float(np.median(X[valid])); Ym = float(np.median(Y[valid])); Zm = float(np.median(Z[valid]))
    cx = int(np.round(xs[valid].mean())); cy = int(np.round(ys[valid].mean()))
    return (cx, cy, Xm, Ym, Zm)

def zbuffer_visible_masks(masks, xyz_cam, z_min=0.10, z_max=1.60, eps=0.003):
    if not masks: return [], []
    H, W = xyz_cam.shape[:2]
    Z = xyz_cam[..., 2].copy()
    Z[~np.isfinite(Z)] = np.inf
    Z[(Z < z_min) | (Z > z_max)] = np.inf
    N = len(masks)
    z_stack = np.full((N, H, W), np.inf, dtype=np.float32)
    sizes = np.zeros(N, dtype=np.int64)
    for i, mbin in enumerate(masks):
        m = (mbin.astype(np.uint8) == 1)
        sizes[i] = int(m.sum())
        z_slice = z_stack[i]; z_slice[m] = Z[m]
    front_z = np.min(z_stack, axis=0)
    visible_masks, vis_ratios = [], []
    for i in range(N):
        vis = (z_stack[i] <= (front_z + eps)) & np.isfinite(front_z)
        vis_u8 = vis.astype(np.uint8)
        visible_masks.append(vis_u8)
        kept = int(vis.sum()); total = max(1, int(sizes[i]))
        vis_ratios.append(kept / total)
    return visible_masks, vis_ratios

# =========================
# Perception integration
# =========================
class ZedYoloPerception:
    """
    Background worker that:
      - grabs ZED frames + depth
      - runs YOLOv8-seg
      - resolves visibility with a z-buffer
      - publishes PoseStamped on /external_goal_pose for your planner
    """
    def __init__(
        self,
        node,
        weights: str = "yolov8m-seg.pt",
        img_size: int = 640,
        conf_thres: float = 0.4,
        cam_frame: str = "zed2_left_camera_frame",
        show_view: bool = False,
    ):
        self.node = node
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.cam_frame = cam_frame
        self.show_view = show_view

        self._lock = Lock()
        self._thread: Optional[Thread] = None
        self._alive = False

        # publishers
        self.goal_pub = node.create_publisher(PoseStamped, "/external_goal_pose", FAST_QOS)
        self.point_pub = node.create_publisher(PointStamped, "/datefruit_3d_point", 10)

        # YOLO
        self.model = YOLO(self.weights)
        # CPU by default unless user sets env to allow GPU
        use_gpu = os.getenv("UR10E_PERCEPTION_USE_GPU", "0") == "1" and torch.cuda.is_available()
        self.model.to("cuda" if use_gpu else "cpu").eval()
        self.node.get_logger().info(f"ZED/YOLO using: {'GPU' if use_gpu else 'CPU'}")

    def start(self):
        if self._alive:
            return
        self._alive = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.node.get_logger().info("ZedYoloPerception started.")

    def stop(self):
        if not self._alive:
            return
        self._alive = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.node.get_logger().info("ZedYoloPerception stopped.")

    # -------------- main worker loop --------------
    def _loop(self):
        # ZED init
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.depth_maximum_distance = 50

        self.node.get_logger().info("Opening ZED…")
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.node.get_logger().error(f"ZED open failed: {repr(status)}")
            return
        self.node.get_logger().info("ZED opened.")

        # enable positional tracking + custom OD
        zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        zed.enable_object_detection(obj_param)

        # pre-allocate mats
        runtime_params = sl.RuntimeParameters()
        obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
        image_left = sl.Mat()
        xyz_full = sl.Mat()

        # simple display buffer if needed
        disp_img = None

        while self._alive:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                sleep(0.005); continue

            # get image for YOLO
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            img_rgba = image_left.get_data()
            img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

            # YOLO predict
            with torch.no_grad():
                det = self.model.predict(
                    img_rgb,
                    save=False,
                    retina_masks=True,
                    imgsz=self.img_size,
                    conf=self.conf_thres,
                    iou=0.45,
                    verbose=False
                )[0]

            # full-res depth (camera frame)
            zed.retrieve_measure(xyz_full, sl.MEASURE.XYZ, sl.MEM.CPU)
            xyz_np = xyz_full.get_data()  # (H,W,4), meters

            # gather masks/classes
            masks, labels, scores = [], [], []
            if det.masks is not None and det.masks.data is not None:
                H, W = det.orig_shape
                for i in range(len(det.boxes)):
                    m = det.masks.data[i].float().cpu().numpy()
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    mbin = (m > 0.5).astype(np.uint8)
                    masks.append(mbin)
                    labels.append(int(det.boxes.cls[i].item()))
                    scores.append(float(det.boxes.conf[i].item()))

            # resolve front-visible pixels
            vis_masks, vis_ratios = zbuffer_visible_masks(masks, xyz_np, z_min=0.10, z_max=1.60, eps=0.003)

            # publish each visible detection
            for mbin_vis, vis_ratio, cls_i, conf_i in zip(vis_masks, vis_ratios, labels, scores):
                c = centroid_xyz_in_mask(mbin_vis, xyz_np, z_min=0.10, z_max=1.60)
                if c is None:
                    continue
                _, _, Xc, Yc, Zc = c

                # point in camera frame
                pt_cam = PointStamped()
                pt_cam.header.frame_id = self.cam_frame
                pt_cam.header.stamp = rclpy.time.Time().to_msg()   # <-- time=0 (LATEST), not now()
                pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = Xc, Yc, Zc

                try:
                    # transform to base
                    pt_base = self.node.tf_buffer.transform(pt_cam, "base_link", timeout=rclpyDuration(seconds=0.2))
                    # axis via PCA in camera → rotate to base
                    axis_cam = pca_long_axis_from_mask(mbin_vis, xyz_np)
                    if axis_cam[2] < 0: axis_cam = -axis_cam
                    tf_cb = self.node.tf_buffer.lookup_transform("base_link", self.cam_frame, rclpy.time.Time().to_msg())
                    q = tf_cb.transform.rotation
                    axis_base = np.array(quat_rotate_vec((q.w, q.x, q.y, q.z), axis_cam), float)
                    axis_base = _unit(axis_base)
                    qw,qx,qy,qz = quat_align_x_to_axis(axis_base, up_hint=(0,0,1))

                    # publish pose (what your planner listens to)
                    goal = PoseStamped()
                    goal.header = pt_base.header
                    goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = \
                        pt_base.point.x, pt_base.point.y, pt_base.point.z
                    goal.pose.orientation.w, goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z = \
                        qw,qx,qy,qz
                    self.goal_pub.publish(goal)

                    # optional: publish point
                    self.point_pub.publish(pt_base)

                    # optional: on-screen preview
                    if self.show_view:
                        if disp_img is None:
                            disp_img = np.zeros_like(img_rgba)
                        disp_img[:] = img_rgba
                        H, W = disp_img.shape[:2]
                        cx = int(np.clip((W/2) + Xc*100, 0, W-1))
                        cy = int(np.clip((H/2) - Zc*100, 0, H-1))
                        cv2.circle(disp_img, (cx, cy), 5, (0,255,0,255), -1)
                        cv2.imshow("Perception preview", disp_img); cv2.waitKey(1)

                except Exception as e:
                    self.node.get_logger().warn(f"Perception publish failed: {type(e).__name__}: {e}")

            sleep(0.002)

        # cleanup
        try:
            zed.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
