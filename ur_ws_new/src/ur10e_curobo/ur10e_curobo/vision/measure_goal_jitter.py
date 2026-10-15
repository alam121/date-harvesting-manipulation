#!/usr/bin/env python3
"""Measure published-goal jitter for a STATIONARY date.

Localization noise is the one quantity this project has never measured. With the
arm parked and a single date fixed in place, every published /external_goal_pose
should be identical; whatever spread appears is the noise budget the grasp has to
live inside. A 20-25mm date gives roughly +/-11mm of margin, so a spread
approaching that explains intermittent misses on its own.

Usage (arm parked, date stationary, vision running):

    ros2 run ur10e_curobo vision_jitter          # if exposed as an entry point
    python3 -m ur10e_curobo.vision.measure_goal_jitter --seconds 30

Reports per-axis std dev and peak-to-peak in base_link, plus the distribution of
frame-to-frame jumps. Passive: subscribes only, never commands the robot, and
touches neither the GPU nor the planner.
"""

from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy)


class GoalJitterMonitor(Node):
    def __init__(self, topic: str, seconds: float):
        super().__init__("goal_jitter_monitor")
        self._samples: List[Tuple[float, float, float]] = []
        self._stamps: List[int] = []
        self._seconds = seconds
        self._start = self.get_clock().now()
        # Match the vision node's fast_qos (BEST_EFFORT / KEEP_LAST / VOLATILE,
        # vision/node.py:262). A RELIABLE subscriber is INCOMPATIBLE with a
        # BEST_EFFORT publisher and silently receives nothing; a BEST_EFFORT
        # subscriber accepts either. Depth here is larger than the publisher's 1
        # only to absorb scheduling hiccups on this side.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, topic, self._cb, qos)
        self.create_timer(0.25, self._tick)
        self.get_logger().info(
            f"Listening on {topic} for {seconds:.0f}s — keep the arm parked "
            "and the date still.")

    def _cb(self, msg: PoseStamped) -> None:
        # The vision node preserves header.stamp across its 50 Hz republish
        # timer, so a repeated stamp is the SAME perception sample. Counting
        # repeats would understate the true spread.
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self._stamps and stamp_ns == self._stamps[-1]:
            return
        self._stamps.append(stamp_ns)
        p = msg.pose.position
        self._samples.append((p.x, p.y, p.z))

    def _tick(self) -> None:
        elapsed = (self.get_clock().now() - self._start).nanoseconds / 1e9
        if elapsed >= self._seconds:
            self._report()
            rclpy.shutdown()

    def _report(self) -> None:
        n = len(self._samples)
        if n < 2:
            self.get_logger().error(
                f"Only {n} distinct perception sample(s) received. Is vision "
                "running and is a date detected?")
            return

        def stats(vals: List[float]) -> Tuple[float, float, float]:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            return mean, math.sqrt(var), max(vals) - min(vals)

        print("\n" + "=" * 62)
        print(f"  Goal jitter over {n} distinct perception samples")
        print("=" * 62)
        print(f"  {'axis':>6}  {'mean (m)':>10}  {'std (mm)':>9}  {'p-p (mm)':>9}")
        worst_std = 0.0
        for i, axis in enumerate("xyz"):
            mean, sd, pp = stats([s[i] for s in self._samples])
            worst_std = max(worst_std, sd)
            print(f"  {axis:>6}  {mean:10.4f}  {sd * 1000:9.2f}  {pp * 1000:9.2f}")

        jumps = [
            math.dist(self._samples[i], self._samples[i - 1])
            for i in range(1, n)
        ]
        jumps_sorted = sorted(jumps)
        p50 = jumps_sorted[len(jumps_sorted) // 2]
        p95 = jumps_sorted[int(len(jumps_sorted) * 0.95)]
        print(f"\n  frame-to-frame jump: median {p50 * 1000:.2f}mm  "
              f"p95 {p95 * 1000:.2f}mm  max {max(jumps) * 1000:.2f}mm")

        # A 20-25mm date leaves ~11mm of radial margin around the centre.
        margin_mm = 11.0
        print(f"\n  fruit radius (margin available): ~{margin_mm:.0f}mm")
        if worst_std * 1000 > margin_mm * 0.5:
            print("  VERDICT: noise is a large fraction of the margin — "
                  "localization jitter alone can explain intermittent misses.")
        else:
            print("  VERDICT: noise is small against the margin — misses are "
                  "probably NOT caused by goal jitter; look elsewhere.")
        print("=" * 62 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/external_goal_pose")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    rclpy.init()
    node = GoalJitterMonitor(args.topic, args.seconds)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._report()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
