#!/usr/bin/env python3
import numpy as np
import argparse
from rclpy.executors import MultiThreadedExecutor
import tf2_geometry_msgs

from threading import Lock, Thread
import time
from typing import List

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import math

def main_(args: argparse.Namespace):
    # --- ROS2 setup ---
    rclpy.init()
    node = rclpy.create_node('goal_pose_tester')
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    fast_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=DurabilityPolicy.VOLATILE,
    )

    goal_pub = node.create_publisher(PoseStamped, '/external_goal_pose_tester', fast_qos)

    # --------------------------------------------------------
    # ✨ PREDEFINED GOALS (EDIT THESE)
    # --------------------------------------------------------
    goal_list = [
    [0.15094344317913055, -0.6711788773536682, 0.8939238882064819],
    [0.17208731174468994, -0.7551352882385254, 0.8808734107017517],
    [0.23739399015903473, -0.6592073440551758, 0.8997570371627808]]
    # Orientation - robot points tool straight down
    orientation = (1.0, 0.0, 0.0, 0.0)

    print("\n===================================")
    print(" PREDEFINED GOAL MODE ACTIVE")
    print(" Sending goals to /external_goal_pose")
    print("===================================\n")

    # --------------------------------------------------------
    # ✨ MAIN LOOP – cycle through predefined goals
    # --------------------------------------------------------
    try:
        idx = 0
        for i in range(len(goal_list)+10): 
        
            x, y, z = goal_list[idx]

            msg = PoseStamped()
            msg.header.stamp = rclpy.time.Time().to_msg()
            msg.header.frame_id = 'base_link'

            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)

            msg.pose.orientation.w = orientation[0]
            msg.pose.orientation.x = orientation[1]
            msg.pose.orientation.y = orientation[2]
            msg.pose.orientation.z = orientation[3]

            print(f"🟢 Sending goal #{idx+1}:  [{x:.3f}, {y:.3f}, {z:.3f}]")
            goal_pub.publish(msg)

            # Move to next goal
            idx = (idx + 1) % len(goal_list)

            # Wait before sending next (adjust)
            time.sleep(2.0)

    except KeyboardInterrupt:
        pass
    finally:
        spin_thread.join(timeout=1.0)
        rclpy.shutdown()

if __name__ == '__main__':
        
        parser = argparse.ArgumentParser(description="External Goal Pose Tester")
        args = parser.parse_args()
        main_(args)