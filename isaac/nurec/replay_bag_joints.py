"""Replay the arm motion of a NuRec recording into isaac_hardware_scene.py (no robot or Jetson).

Arm joint angles are recovered from the bag's /tf with the robot URDF (joint origin removed,
remaining rotation about z), then streamed to the scene's UDP command port in "state" mode,
exactly like the digital-twin bridge does with the physical robot.

  $NUREC_ROS_PYTHON replay_bag_joints.py BAG [--urdf BAG/robot_description.urdf] [--speed 1.0] [--loop]
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

from metric_frame import quat_to_rot
from nurec_pose_from_bag import load_tf


ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--urdf", type=Path, default=None, help="Default: BAG/robot_description.urdf")
    parser.add_argument("--rate", type=float, default=50.0, help="Command rate [Hz].")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor.")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--port", type=int, default=55000, help="isaac_hardware_scene.py --command_port")
    parser.add_argument(
        "--print_start",
        action="store_true",
        help="Only print the first recorded arm joints (as --start_joints for live_detector.py --jog).",
    )
    return parser.parse_args()


def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def tf_time_range(bag: Path) -> tuple[float, float]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap" if any(bag.glob("*.mcap")) else "sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf"]))
    stamps = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        for transform in deserialize_message(data, TFMessage).transforms:
            stamps.append(transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9)
    return min(stamps), max(stamps)


def extract_urdf(bag: Path, out: Path) -> bool:
    """Save the /robot_description recorded in the bag (record_stereo.sh records it)."""
    from std_msgs.msg import String

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap" if any(bag.glob("*.mcap")) else "sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    if "/robot_description" not in {t.name for t in reader.get_all_topics_and_types()}:
        return False
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/robot_description"]))
    if not reader.has_next():
        return False
    out.write_text(deserialize_message(reader.read_next()[1], String).data)
    return True


def joint_trajectory(bag: Path, urdf: Path, rate: float) -> list[tuple[float, list[float]]]:
    robot = ET.parse(urdf).getroot()
    joints = []
    for name in ARM_JOINTS:
        joint = robot.find(f"joint[@name='{name}']")
        origin = rpy_matrix(*map(float, joint.find("origin").get("rpy").split()))
        joints.append((joint.find("parent").get("link"), joint.find("child").get("link"), origin))

    buffer = load_tf(bag)
    start, end = tf_time_range(bag)
    trajectory = []
    for t in np.arange(start, end, 1.0 / rate):
        when = Time(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
        angles = []
        for parent, child, origin in joints:
            if not buffer.can_transform(parent, child, when):
                break
            q = buffer.lookup_transform(parent, child, when).transform.rotation
            relative = origin.T @ quat_to_rot(q.w, q.x, q.y, q.z)
            angles.append(float(np.arctan2(relative[1, 0], relative[0, 0])))
        if len(angles) == len(joints):
            trajectory.append((float(t), angles))
    return trajectory


def main():
    args = parse_args()
    urdf = args.urdf or args.bag / "robot_description.urdf"
    if not urdf.exists() and not extract_urdf(args.bag, urdf):
        raise SystemExit(f"{urdf} missing and no /robot_description in the bag (see isaac/nurec/README.md)")
    trajectory = joint_trajectory(args.bag, urdf, args.rate)
    if not trajectory:
        raise SystemExit("No arm joint states could be recovered from /tf.")
    angles = np.unwrap(np.array([a for _, a in trajectory]), axis=0)
    times = np.array([t for t, _ in trajectory])
    velocities = np.gradient(angles, times, axis=0)
    if args.print_start:
        print("--start_joints " + " ".join(f"{v:.5f}" for v in angles[0]))
        return
    print(f"[INFO] {len(trajectory)} joint states over {times[-1] - times[0]:.1f} s; streaming to udp://127.0.0.1:{args.port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / args.rate
    while True:
        for angle, velocity in zip(angles, velocities):
            sock.sendto(json.dumps({
                "arm_target": angle.tolist(),
                "arm_velocity": (velocity * args.speed).tolist(),
                "arm_mode": "state",
                "gripper_target": [],
            }).encode(), ("127.0.0.1", args.port))
            time.sleep(period / args.speed)
        if not args.loop:
            break
        print("[INFO] looping")


if __name__ == "__main__":
    main()
