#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from ultralytics.engine.results import Results

from threading import Lock, Thread
from time import sleep
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

# TF listener setup (must pass node)

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

        # Label & probability as plain scalars
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
    global image_net, exit_signal, run_signal, detections
    print("Initializing Network...")
    model = YOLO(weights)
    model.to('cuda').eval()
    print("Network Initialized...")

    while not exit_signal:
        if run_signal:
            lock.acquire()
            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            det = model.predict(
                img,
                save=False,
                retina_masks=True,
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                verbose=False
            )[0]
            detections = detections_to_custom_masks_(det)
            lock.release()
            run_signal = False
        sleep(0.01)


def label_to_name(o, class_names=None) -> str:
    """
    Robustly convert ZED object label to a printable string:
    - prefer o.raw_label if present (custom numeric id)
    - handle enum sl.OBJECT_CLASS (use .name)
    - fallback to int() when possible, else str()
    """
    # Some SDK versions expose raw_label for custom detectors
    if hasattr(o, 'raw_label'):
        try:
            return class_names.get(int(o.raw_label), str(int(o.raw_label))) if class_names else str(int(o.raw_label))
        except Exception:
            return str(o.raw_label)

    # Handle enum
    try:
        if isinstance(o.label, sl.OBJECT_CLASS):
            return o.label.name  # enum name
    except Exception:
        pass

    # Try int cast
    try:
        return class_names.get(int(o.label), str(int(o.label))) if class_names else str(int(o.label))
    except Exception:
        return str(o.label)


def main_(args: argparse.Namespace):
    global image_net, exit_signal, run_signal, detections, tf_listener

    # ROS2 setup
    rclpy.init()
    node = rclpy.create_node('zed_date_detector')
    point_pub = node.create_publisher(PointStamped, '/datefruit_3d_point', 10)
    goal_pub = node.create_publisher(PoseStamped, '/external_goal_pose', 10)
    goal_sent = False

    # TF listener must be initialized with node
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
    #init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
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
    # positional_tracking_parameters.set_as_static = True  # if camera is static
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

    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

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

    # Optional: class id -> name mapping (if you want nicer names)
    class_names = {0: "datefruit"}

    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # get left image for network thread
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # wait for network
            while run_signal:
                sleep(0.001)

            # ingest detections and retrieve tracked objects
            lock.acquire()
            zed.ingest_custom_mask_objects(detections)
            lock.release()
            zed.retrieve_custom_objects(objects, obj_runtime_param)

            # ------- PRINT XYZ for each detected/tracked object (meters) -------
            if objects.is_new and len(objects.object_list) > 0:
                print("\nDetections this frame:")
                for o in objects.object_list:
                    # (x,y,z) meters in WORLD coordinates
                    x = float(o.position[0])
                    y = float(o.position[1])
                    z = float(o.position[2])

                    name = label_to_name(o, class_names)

                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        print(f"  id={o.id}  class={name}  XYZ(m) = [{x:.3f}, {y:.3f}, {z:.3f}]")
                        # Publish to ROS2 topic
                        point_msg = PointStamped()
                        point_msg.header.frame_id = 'zed2_left_camera_frame'
                        point_msg.header.stamp = rclpy.time.Time().to_msg()
                        point_msg.point.x = float(x)
                        point_msg.point.y = float(y)
                        point_msg.point.z = float(z)
                        try:
                            while not tf_buffer.can_transform('base_link', 'zed2_left_camera_frame',
                                  rclpy.time.Time(), rclpyDuration(seconds=0.1)):
                                rclpy.spin_once(node, timeout_sec=0.1)  # optional; keeps things responsive
                                
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
                            print(f"TF or publish failed: [{type(e).__name__}] {e}\n  looking for transform '{point_msg.header.frame_id}' → 'base_link'")

            # ------- Display -------
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.CAMERA)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)

            # 2D composite
            np.copyto(image_left_ocv, image_left.get_data())

            # (Optional) overlay XYZ near each object bbox on the 2D view
            for o in objects.object_list:
                if len(o.bounding_box_2d) == 4:
                    x1 = int(o.bounding_box_2d[0][0] * image_scale[0])
                    y1 = int(o.bounding_box_2d[0][1] * image_scale[1])
                    try:
                        xx = float(o.position[0]); yy = float(o.position[1]); zz = float(o.position[2])
                        txt = f"({xx:.2f},{yy:.2f},{zz:.2f}) m"
                        cv2.putText(image_left_ocv, txt, (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255, 255), 1, cv2.LINE_AA)
                    except Exception:
                        pass

            # Tracking view
            track_view_generator.generate_view(objects, image_left_ocv, image_scale, cam_w_pose, image_track_ocv, objects.is_tracked)

            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            cv2.imshow("ZED | 2D View and Birds View", global_image)
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
