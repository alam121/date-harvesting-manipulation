"""ROS2 utility functions for vision module."""

from time import sleep, time

import numpy as np
import rclpy
from rclpy.time import Time as rclpyTime
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def debug_print(msg: str) -> None:
    """Lightweight debug printer with flush."""
    print(f"[DEBUG] {msg}", flush=True)


def wait_for_transform(tf_buffer, target_frame, source_frame, node, timeout=5.0):
    """Spin until the requested TF is available or timeout."""
    start = time()
    while rclpy.ok() and (time() - start) < timeout:
        if tf_buffer.can_transform(target_frame, source_frame, rclpyTime()):
            return True
        rclpy.spin_once(node, timeout_sec=0.05)
        sleep(0.05)
    print(f"[WARN] TF {source_frame}->{target_frame} unavailable after {timeout:.1f}s")
    return False


def create_pointcloud2_msg(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    Create a PointCloud2 message from numpy array of XYZ points.

    Args:
        points: (N, 3) array of XYZ points in meters
        frame_id: TF frame ID
        stamp: ROS timestamp

    Returns:
        PointCloud2 message
    """
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp

    msg.height = 1
    msg.width = points.shape[0]

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 12  # 3 floats * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True

    msg.data = points.astype(np.float32).tobytes()

    return msg
