"""Print --nurec_pose and --wrist_camera_extrinsic for isaac_hardware_scene.py from the bag's /tf.

The reconstruction frame is the anchor left image (metric_frame.json) in ROS body axes.
For the wrist camera that is TF base_link -> zed_mini_left_camera_frame (OpenCV optical
axes, from the hand-eye calibration) at the anchor timestamp, re-expressed in body axes and
offset by where the Isaac scene spawns the robot base.

  $NUREC_ROS_PYTHON nurec_pose_from_bag.py BAG SCENE [--camera_frame F] [--robot_base_world X Y Z]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from metric_frame import R_BODY_FROM_OPTICAL, quat_to_rot, rot_to_quat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--base_frame", default="base_link")
    parser.add_argument("--camera_frame", default="zed_mini_left_camera_frame")
    parser.add_argument("--tool_frame", default="tool0")
    parser.add_argument("--camera_axes", choices=("optical", "body"), default="optical")
    parser.add_argument("--robot_base_world", type=float, nargs=3, default=(0.0, 0.0, 0.30))
    return parser.parse_args()


def load_tf(bag: Path) -> Buffer:
    buffer = Buffer(cache_time=Duration(seconds=3600))
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap" if any(bag.glob("*.mcap")) else "sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf", "/tf_static"]))
    while reader.has_next():
        topic, data, _ = reader.read_next()
        for transform in deserialize_message(data, TFMessage).transforms:
            if topic == "/tf_static":
                buffer.set_transform_static(transform, "bag")
            else:
                buffer.set_transform(transform, "bag")
    return buffer


def main():
    args = parse_args()
    calibration = json.loads((args.scene / "stereo_calibration.json").read_text())
    anchor = json.loads((args.scene / "sparse_metric_txt" / "metric_frame.json").read_text())["anchor_image"]
    stamp = calibration["left_stamps"][int(Path(anchor).stem)]

    buffer = load_tf(args.bag)
    # The first image can precede the first /tf message by a few ms; step forward to the nearest sample.
    for offset in np.arange(0.0, 0.5, 0.01):
        when = Time(sec=int(stamp + offset), nanosec=int(round(((stamp + offset) % 1) * 1e9)))
        if buffer.can_transform(args.base_frame, args.camera_frame, when):
            break
    else:
        raise SystemExit(f"No {args.base_frame} -> {args.camera_frame} TF within 0.5 s of t={stamp:.3f}")
    if offset:
        print(f"[WARN] TF sampled {1000 * offset:.0f} ms after the anchor image")
    transform = buffer.lookup_transform(args.base_frame, args.camera_frame, when).transform
    q, t = transform.rotation, transform.translation
    rotation = quat_to_rot(q.w, q.x, q.y, q.z)
    if args.camera_axes == "optical":
        rotation = rotation @ R_BODY_FROM_OPTICAL.T
    position = np.array([t.x, t.y, t.z]) + np.array(args.robot_base_world)

    mount = buffer.lookup_transform(args.tool_frame, args.camera_frame, when).transform
    mount_q, mount_t = mount.rotation, mount.translation

    print(f"[INFO] anchor {anchor} at t={stamp:.3f}: {args.base_frame} -> {args.camera_frame}")
    print("--nurec_pose " + " ".join(f"{v:.5f}" for v in (*position, *rot_to_quat(rotation))))
    print("--wrist_camera_extrinsic " + " ".join(
        f"{v:.5f}" for v in (mount_t.x, mount_t.y, mount_t.z, mount_q.w, mount_q.x, mount_q.y, mount_q.z)
    ))


if __name__ == "__main__":
    main()
