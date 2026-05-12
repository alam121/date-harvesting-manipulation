import os
import subprocess
import time
from datetime import datetime

from . import motions as motions_mod


def run_lidar_scan(node):
    """
    Sweep the arm through configured half-circle waypoints around the tree
    while recording /livox/lidar and /livox/imu to a ROS 2 bag file.

    Waypoints are defined in cfg.lidar_scan.scan_waypoints (config.py).
    The arm moves to the first waypoint, starts recording, sweeps through
    the remaining waypoints, stops recording, then returns home.
    """
    cfg = node.cfg.lidar_scan
    waypoints = cfg.scan_waypoints

    if not waypoints:
        node.get_logger().error("Lidar scan: no waypoints in cfg.lidar_scan.scan_waypoints")
        return

    bag_dir = os.path.expanduser(cfg.bag_dir)
    os.makedirs(bag_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.join(bag_dir, f"lidar_scan_{timestamp}")

    node.get_logger().info(
        f"Lidar scan: {len(waypoints)} waypoints, recording to {bag_path}"
    )
    node.motion_phase = "LIDAR_SCAN"
    bag_proc = None

    try:
        # Move to scan start before recording so motion artefacts are not captured
        node.get_logger().info("Lidar scan: moving to start position")
        ok = motions_mod.plan_execute_js(
            node, waypoints[0],
            label="SCAN_START",
            motion_type="scan",
            speed_factor=cfg.speed_factor,
        )
        if not ok or node.stop_requested:
            node.get_logger().warn("Lidar scan: could not reach start position, aborting")
            return

        # Start ros2 bag record
        bag_proc = subprocess.Popen(
            ["ros2", "bag", "record", "-o", bag_path,
             "/livox/lidar", "/livox/imu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        node.get_logger().info("Lidar scan: bag recording started")
        time.sleep(1.0)  # allow recorder to initialise before moving

        # Sweep through remaining waypoints
        for i, wp in enumerate(waypoints[1:], 1):
            if node.stop_requested:
                node.get_logger().info("Lidar scan: interrupted by stop request")
                break
            node.get_logger().info(
                f"Lidar scan: moving to waypoint {i}/{len(waypoints) - 1}"
            )
            motions_mod.plan_execute_js(
                node, wp,
                label=f"SCAN_{i}",
                motion_type="scan",
                speed_factor=cfg.speed_factor,
            )
            time.sleep(0.3)  # brief dwell at each waypoint for full LiDAR sweep

        node.get_logger().info(f"Lidar scan complete — bag saved at {bag_path}")

    except Exception as e:
        node.get_logger().error(f"Lidar scan error: {e}")

    finally:
        if bag_proc is not None:
            bag_proc.terminate()
            try:
                bag_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                bag_proc.kill()
        node.motion_phase = "IDLE"
        if not node.stop_requested:
            motions_mod.move_to_home_position(node)
