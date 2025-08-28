#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.engine.results import Results

from threading import Lock, Thread
from time import sleep, time
from typing import List
import tf2_geometry_msgs

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration as rclpyDuration

# Globals
lock = Lock()
run_signal = False
exit_signal = False
image_net: np.ndarray = None
detections: List[sl.CustomMaskObjectData] = None
sl_mats: List[sl.Mat] = None  # keep sl.Mat ownership alive

# For old-style centroid sampling & FPS
centroids = []              # list of (cx, cy, cls_id, conf)
centroids_lock = Lock()
net_fps = 0.0               # YOLO inference FPS (updated by torch_thread_)
loop_fps = 0.0              # main loop FPS


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


def detections_to_custom_masks_(dets: Results) -> List[sl.CustomMaskObjectData]:
    """Convert Ultralytics Results to ZED CustomMaskObjectData."""
    global sl_mats
    output = []
    sl_mats = []

    H, W = dets.orig_shape

    for di in range(len(dets.boxes)):
        obj = sl.CustomMaskObjectData()

        # Bounding box (4x2 float32)
        xywh = dets.boxes.xywh[di].cpu().numpy().astype(np.float32)
        abcd = xywh2abcd_(xywh)
        abcd[:, 0] = np.clip(abcd[:, 0], 0, W - 1)
        abcd[:, 1] = np.clip(abcd[:, 1], 0, H - 1)
        obj.bounding_box_2d = abcd

        # Label & probability
        obj.label = int(dets.boxes.cls[di].item())
        obj.probability = float(dets.boxes.conf[di].item())
        obj.is_grounded = False

        # Mask (optional but recommended)
        if dets.masks is not None and dets.masks.data is not None:
            m = dets.masks.data[di].cpu().numpy()
            mask_bin = (m * 255).astype(np.uint8)

            x_min = int(abcd[0, 0]); y_min = int(abcd[0, 1])
            x_max = int(abcd[2, 0]); y_max = int(abcd[2, 1])
            mask_roi = mask_bin[y_min:y_max+1, x_min:x_max+1]
            if not mask_roi.flags.c_contiguous:
                mask_roi = np.ascontiguousarray(mask_roi)

            sl_mat = sl.Mat(
                width=mask_roi.shape[1],
                height=mask_roi.shape[0],
                mat_type=sl.MAT_TYPE.U8_C1,
                memory_type=sl.MEM.CPU
            )
            np.copyto(sl_mat.get_data(), mask_roi)
            sl_mats.append(sl_mat)
            obj.box_mask = sl_mat

        output.append(obj)

    return output


def torch_thread_(weights: str, img_size: int, conf_thres: float = 0.2, iou_thres: float = 0.45) -> None:
    """Runs YOLO in a background thread, prepares ZED-ingestible objects AND
       computes mask-centroid (cx,cy) list for old-style single-pixel XYZ sampling.
    """
    global image_net, exit_signal, run_signal, detections, net_fps, centroids
    print("Initializing Network...")
    model = YOLO(weights)
    model.to('cuda').eval()
    print("Network Initialized...")

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            t0 = time()
            det = model.predict(
                img,
                save=False,
                retina_masks=True,
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )[0]
            dt = time() - t0
            net_fps = (1.0 / dt) if dt > 0 else 0.0

            # Build ZED custom objects for tracking/viewer
            detections = detections_to_custom_masks_(det)

            # Compute full-res mask centroids for old-style depth sampling
            new_centroids = []
            if det.masks is not None and det.masks.data is not None:
                for i in range(len(det.boxes)):
                    m = det.masks.data[i].cpu().numpy()
                    m_bin = (m > 0.5).astype(np.uint8)
                    M = cv2.moments(m_bin)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cls_i = int(det.boxes.cls[i].item())
                    conf_i = float(det.boxes.conf[i].item())
                    new_centroids.append((cx, cy, cls_i, conf_i))

            with centroids_lock:
                centroids = new_centroids

            lock.release()
            run_signal = False
        sleep(0.005)


def label_to_name(o, class_names=None) -> str:
    if hasattr(o, 'raw_label'):
        try:
            return class_names.get(int(o.raw_label), str(int(o.raw_label))) if class_names else str(int(o.raw_label))
        except Exception:
            return str(o.raw_label)
    try:
        if isinstance(o.label, sl.OBJECT_CLASS):
            return o.label.name
    except Exception:
        pass
    try:
        return class_names.get(int(o.label), str(int(o.label))) if class_names else str(int(o.label))
    except Exception:
        return str(o.label)


def _extract_T_wc(cam_pose_world: sl.Pose) -> np.ndarray:
    """
    Get CAMERA->WORLD 4x4 pose matrix (T_wc) robustly across ZED SDK variants.
    Tries pose_data(), pose_data, .m, then falls back to rotation/translation getters.
    Returns np.ndarray shape (4,4), dtype=float32.
    """
    # 1) Try pose_data() or pose_data attr
    try:
        T_attr = cam_pose_world.pose_data
        T_wc = T_attr() if callable(T_attr) else T_attr
        T_wc = np.array(T_wc)  # don't force dtype yet
        if T_wc.size == 16:
            T_wc = T_wc.reshape(4, 4).astype(np.float32)
            return T_wc
        if T_wc.shape == (4, 4):
            return T_wc.astype(np.float32)
    except Exception:
        pass

    # 2) Some versions expose a flat list via .m
    try:
        flat = cam_pose_world.m  # e.g., list of 16 values
        if hasattr(flat, "__len__") and len(flat) == 16:
            T_wc = np.array(flat, dtype=np.float32).reshape(4, 4)
            return T_wc
    except Exception:
        pass

    # 3) Fallback: build from rotation & translation getters
    # Rotation (3x3)
    R_wc = None
    for getter in ("get_rotation_matrix",):
        try:
            Robj = getattr(cam_pose_world, getter)()
            # Try common fields on ZED matrix wrappers
            if hasattr(Robj, "r"):              # e.g., row-major flat list
                R_wc = np.array(Robj.r, dtype=np.float32).reshape(3, 3)
                break
            if hasattr(Robj, "get"):            # sometimes returns nested lists
                R_wc = np.array(Robj.get(), dtype=np.float32)
                if R_wc.shape == (3, 3):
                    break
            # Last resort: direct cast
            R_wc = np.array(Robj, dtype=np.float32).reshape(3, 3)
            break
        except Exception:
            continue
    if R_wc is None:
        raise RuntimeError("Could not extract rotation matrix from cam pose")

    # Translation (3,)
    t_wc = None
    for getter in ("get_translation",):
        try:
            Tobj = getattr(cam_pose_world, getter)()
            if hasattr(Tobj, "get"):            # e.g., sl.Translation
                vals = Tobj.get()               # [x, y, z]
                t_wc = np.array([vals[0], vals[1], vals[2]], dtype=np.float32)
                break
            # Last resort: direct cast
            arr = np.array(Tobj, dtype=np.float32).flatten()
            if arr.size >= 3:
                t_wc = arr[:3]
                break
        except Exception:
            continue
    if t_wc is None:
        raise RuntimeError("Could not extract translation from cam pose")

    # Assemble 4x4
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc
    return T_wc


def world_to_camera(point_world: np.ndarray, cam_pose_world: sl.Pose) -> np.ndarray:
    """
    Convert a WORLD-frame 3D point to CAMERA frame:
        p_cam = R_cw * (p_w - t_wc)  with  R_cw = R_wc^T
    """
    T_wc = _extract_T_wc(cam_pose_world)        # (4,4)
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    R_cw = R_wc.T
    return R_cw.dot(point_world - t_wc)

def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections, loop_fps

    # ROS2 setup
    rclpy.init()
    node = rclpy.create_node('zed_date_detector_ros')
    point_pub = node.create_publisher(PointStamped, '/datefruit_3d_point', 10)
    goal_pub = node.create_publisher(PoseStamped, '/external_goal_pose', 10)
    goal_sent = False
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    capture_thread = Thread(
        target=torch_thread_,
        kwargs={'weights': args.weights, 'img_size': args.img_size, 'conf_thres': args.conf_thres}
    )
    capture_thread.start()

    # --- ZED init ---
    input_type = sl.InputType()
    if args.svo is not None:
        input_type.set_from_svo_file(args.svo)

    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    # Keep Y-UP world (fine; we’ll convert to CAMERA when needed)
    init_params.depth_maximum_distance = 50

    print("Initializing Camera...")
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        return
    print("Camera Initialized")

    # Positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Object detection (custom masks)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True
    zed.enable_object_detection(obj_param)

    # Display setup
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    xyz_full = sl.Mat()  # full-res XYZ (CAMERA frame) for centroid sampling

    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4),
                             [245, 239, 239, 255], np.uint8)

    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(
        tracks_resolution,
        camera_config.fps,
        init_params.depth_maximum_distance * 1000,
        1
    )
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.CustomObjectDetectionRuntimeParameters()
    cam_w_pose = sl.Pose()
    image_left_tmp = sl.Mat()
    objects = sl.Objects()

    class_names = {0: "datefruit"}  # optional

    t_loop_prev = time()

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # loop FPS
            t_now = time()
            dt_loop = t_now - t_loop_prev
            loop_fps = (1.0 / dt_loop) if dt_loop > 0 else 0.0
            t_loop_prev = t_now

            # input image for YOLO (full resolution)
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # wait for network
            while run_signal and not exit_signal:
                sleep(0.001)

            # Ingest detections for ZED tracking
            lock.acquire()
            zed.ingest_custom_mask_objects(detections)
            lock.release()
            zed.retrieve_custom_objects(objects, obj_runtime_param)

            # Get camera pose in WORLD for WORLD->CAMERA conversion
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # Retrieve point cloud for viewer, and full-res XYZ for centroid sampling
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            zed.retrieve_measure(xyz_full, sl.MEASURE.XYZ, sl.MEM.CPU)  # full camera res, CAMERA frame
            xyz_np = xyz_full.get_data()  # (H, W, 4) float32; meters in CAMERA frame

            # Retrieve display image
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            np.copyto(image_left_ocv, image_left.get_data())

            # ---------- ONLY PIX_CAM (old-style centroid sampling) ----------
            if objects.is_new and len(objects.object_list) > 0:
                print("\nDetections this frame:")
            with centroids_lock:
                centroid_list = list(centroids)

            Hxyz, Wxyz = xyz_np.shape[:2]
            for (cx, cy, cls_i, conf_i) in centroid_list:
                if 0 <= cx < Wxyz and 0 <= cy < Hxyz:
                    Xp, Yp, Zp, _ = xyz_np[cy, cx]  # meters, CAMERA frame
                    if np.isfinite(Xp) and np.isfinite(Yp) and np.isfinite(Zp):
                        print(f"     mask-centroid[ DESIRED(mm)=[{Xp:.3f}, {Yp:.3f}, {Zp:.3f}]") 
                        
                        
                        rclpy.spin_once(node, timeout_sec=0.0)
                        # Publish to ROS2 topic
                        point_msg = PointStamped()
                        point_msg.header.frame_id = 'zed2_left_camera_frame'
                        point_msg.header.stamp = rclpy.time.Time().to_msg()
                        point_msg.point.x = float(Xp)
                        point_msg.point.y = float(Yp)
                        point_msg.point.z = float(Zp)
                        try:
                            pt_base = tf_buffer.transform(point_msg, 'base_link', timeout=rclpyDuration(seconds=0.5))
                            point_pub.publish(pt_base)
                            goal = PoseStamped()
                            goal.pose.position = pt_base.point
                            goal.pose.orientation.x = 0.0
                            goal.pose.orientation.y = 0.0
                            goal.pose.orientation.z = 0.0
                            goal.pose.orientation.w = 1.0
                            if not goal_sent:
                                goal_pub.publish(goal)
                                goal_sent = False
                                print(f"✅ Published FIRST fruit pose:\n    position = ({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, {goal.pose.position.z:.3f})\n    orientation = ({goal.pose.orientation.x:.3f}, {goal.pose.orientation.y:.3f}, {goal.pose.orientation.z:.3f}, {goal.pose.orientation.w:.3f})")
                            else:
                                print("Skipping further goal publications.")
                        except Exception as e:
                            print(f"TF or publish failed: [{type(e).__name__}] {e}\n  looking for transform '{point_msg.header.frame_id}' → 'base_link'")                       # overlay on display (scaled coords)
    
                        # overlay on display (scaled coords)
                        sx = int(cx * image_scale[0])
                        sy = int(cy * image_scale[1])
                        cv2.circle(image_left_ocv, (sx, sy), 3, (0, 255, 255, 255), -1)
                        txt_b = f"PIX_CAM=({Xp:.2f},{Yp:.2f},{Zp:.2f})m"
                        cv2.putText(image_left_ocv, txt_b, (sx + 6, sy - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255, 255), 1, cv2.LINE_AA)

            # Tracking/bird's-eye view
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.CAMERA)  # for the tracking viewer
            image_track_ocv = np.zeros_like(image_track_ocv)
            track_view_generator.generate_view(
                objects, image_left_ocv, image_scale, cam_w_pose, image_track_ocv, objects.is_tracked
            )

            # FPS overlay
            cv2.putText(image_left_ocv, f"YOLO FPS: {net_fps:.1f}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image_left_ocv, f"Loop FPS: {loop_fps:.1f}", (12, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255, 255), 2, cv2.LINE_AA)

            # 3D rendering viewer (point cloud + tracks)
            point_cloud.copy_to(point_cloud_render)
            viewer.updateData(point_cloud_render, objects)

            # Side-by-side display
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            cv2.imshow("ZED | 2D View + Bird's View", global_image)
            key = cv2.waitKey(10)
            if key in (27, ord('q'), ord('Q')):
                exit_signal = True
            if key == 105:  # 'i'
                track_view_generator.zoomIn()
            if key == 111:  # 'o'
                track_view_generator.zoomOut()
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m-seg.pt', help='model.pt path')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    args = parser.parse_args()

    with torch.no_grad():
        main_(args)
