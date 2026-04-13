"""Vision module for date fruit detection using ZED camera and YOLO."""

from .node import VisionNode
from .config import DEFAULT_WEIGHTS, DEFAULT_CONF_THRES, DEFAULT_IMG_SIZE


def main():
    """Entry point for ros2 run ur10e_curobo vision."""
    import argparse

    parser = argparse.ArgumentParser(description="Date fruit vision detection node")
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help=f"YOLO model.pt path (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument("--svo", type=str, default=None, help="Optional SVO file for playback")
    parser.add_argument(
        "--img_size", type=int, default=DEFAULT_IMG_SIZE,
        help=f"YOLO inference size in pixels (default: {DEFAULT_IMG_SIZE})"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=DEFAULT_CONF_THRES,
        help=f"YOLO confidence threshold (default: {DEFAULT_CONF_THRES})"
    )
    parser.add_argument(
        "--use_lidar", action="store_true",
        help="Use Livox LiDAR for depth instead of ZED stereo depth"
    )
    args = parser.parse_args()

    node = VisionNode(args)
    node.run()


__all__ = [
    "VisionNode",
    "main",
]
