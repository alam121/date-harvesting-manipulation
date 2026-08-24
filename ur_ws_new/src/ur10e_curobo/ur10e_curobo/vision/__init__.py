"""Vision module for date fruit detection using ZED camera and YOLO."""

from .config import DEFAULT_WEIGHTS, DEFAULT_CONF_THRES, DEFAULT_IMG_SIZE


def __getattr__(name):
    """Keep VisionNode available without loading CUDA for calibration tools."""
    if name == "VisionNode":
        from .node import VisionNode
        return VisionNode
    raise AttributeError(name)


def main():
    """Entry point for ros2 run ur10e_curobo vision."""
    import argparse
    from .node import VisionNode

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
        "--hdr", type=int, default=1,
        help="Enable ZED X One HDR at camera open, 1=on, 0=off"
    )
    parser.add_argument(
        "--use_lidar", action="store_true",
        help="Use Livox LiDAR for depth instead of ZED stereo depth"
    )
    parser.add_argument(
        "--use_zed_mini", action="store_true",
        help="Dual-camera mode: ZED X One Mono for detection, ZED X Mini for depth"
    )
    parser.add_argument(
        "--use_zedx_mini_only", action="store_true",
        help="Use ZED X Mini stereo camera for both RGB detection and depth"
    )
    parser.add_argument(
        "--raw_yolo_view", action="store_true",
        help="Display raw YOLO boxes/masks only; disable 3D processing and goal publishing"
    )
    args = parser.parse_args()

    node = VisionNode(args)
    node.run()


__all__ = [
    "VisionNode",
    "main",
]
