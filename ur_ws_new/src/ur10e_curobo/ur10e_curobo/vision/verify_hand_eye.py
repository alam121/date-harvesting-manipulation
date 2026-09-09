#!/usr/bin/env python3
"""Hand-eye acceptance test: one stationary date, several arm poses.

A correct hand-eye transform reports the SAME base_link position for a
stationary fruit no matter where the arm is standing. Any spread is
calibration error, and unlike a solver's own residual it cannot be fooled by a
self-consistent but wrong solution -- which is exactly how the 2026-04-17
calibration passed (44 samples, 68.8deg diversity, all five methods within 1mm)
and still shipped a 12.5deg yaw error.

It also measures how DIFFERENT the arm poses actually were. A tight spread
proves nothing if every station had nearly the same viewing geometry, so the
test refuses to pass on poses that do not exercise the calibration.

Usage (vision running, ONE date in view, arm parked at each station):

    python3 -m ur10e_curobo.vision.verify_hand_eye --poses 4 --seconds 15

Passive: subscribes only. Never commands the robot, never touches the GPU.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy)
from rclpy.time import Time as RclpyTime
from tf2_ros import Buffer, TransformListener

try:
    from .config import CAM_FRAME
except ImportError:  # running as a plain script
    CAM_FRAME = "zed_mini_left_camera_frame"


# A 20-25mm date leaves ~11mm of radial margin. Spending more than about half
# of it on frame error alone leaves nothing for depth, grasp geometry and the
# fruit not being a perfect sphere.
PASS_MM = 5.0
MARGINAL_MM = 8.0
# Below these the stations were too alike for the result to mean anything.
MIN_TRANSLATION_SPREAD_MM = 80.0
MIN_ROTATION_SPREAD_DEG = 15.0


def quat_angle_between(a: Tuple[float, float, float, float],
                       b: Tuple[float, float, float, float]) -> float:
    """Angle in degrees between two orientations, as (x, y, z, w)."""
    dot = abs(sum(ai * bi for ai, bi in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class HandEyeVerifier(Node):
    def __init__(self, topic: str, cam_frame: str):
        super().__init__("hand_eye_verifier")
        self.cam_frame = cam_frame
        self._goals: List[Tuple[float, float, float]] = []
        self._stamps: List[int] = []
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PoseStamped, topic, self._cb, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _cb(self, msg: PoseStamped) -> None:
        # Skip republishes of the same perception sample; the vision node
        # preserves header.stamp across its 50 Hz timer.
        ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self._stamps and ns == self._stamps[-1]:
            return
        self._stamps.append(ns)
        p = msg.pose.position
        self._goals.append((p.x, p.y, p.z))

    def reset(self) -> None:
        self._goals.clear()
        self._stamps.clear()

    def collect(self, seconds: float):
        """Spin for `seconds`, then return (goal_mean, goal_std_mm, cam_pose, n)."""
        self.reset()
        end = self.get_clock().now() + Duration(seconds=seconds)
        while rclpy.ok() and self.get_clock().now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

        n = len(self._goals)
        if n < 5:
            return None, None, None, n

        mean = tuple(sum(g[i] for g in self._goals) / n for i in range(3))
        std_mm = max(
            math.sqrt(sum((g[i] - mean[i]) ** 2 for g in self._goals) / n) * 1000.0
            for i in range(3))

        cam_pose = None
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", self.cam_frame, RclpyTime(),
                timeout=Duration(seconds=0.5))
            t, r = tf.transform.translation, tf.transform.rotation
            cam_pose = ((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))
        except Exception as exc:  # noqa: BLE001 - report and continue
            self.get_logger().warn(
                f"TF base_link<-{self.cam_frame} unavailable: {exc}")
        return mean, std_mm, cam_pose, n


def report(goals, cam_poses) -> int:
    print("\n" + "=" * 70)
    print("  HAND-EYE ACCEPTANCE TEST")
    print("=" * 70)
    print(f"  {'pose':>5}  {'x (m)':>9} {'y (m)':>9} {'z (m)':>9}   {'jitter':>8}")
    for i, (g, sd) in enumerate(goals, 1):
        print(f"  {i:>5}  {g[0]:9.4f} {g[1]:9.4f} {g[2]:9.4f}   {sd:6.2f}mm")

    # Worst disagreement between any two stations about the same fruit.
    worst, pair = 0.0, (0, 0)
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            d = math.dist(goals[i][0], goals[j][0]) * 1000.0
            if d > worst:
                worst, pair = d, (i + 1, j + 1)

    # How hard did we actually push the calibration?
    t_spread = r_spread = 0.0
    valid = [c for c in cam_poses if c is not None]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            t_spread = max(t_spread, math.dist(valid[i][0], valid[j][0]) * 1000.0)
            r_spread = max(r_spread, quat_angle_between(valid[i][1], valid[j][1]))

    print("-" * 70)
    print(f"  worst disagreement : {worst:.1f}mm  (pose {pair[0]} vs {pair[1]})")
    if valid:
        print(f"  camera pose spread : {t_spread:.0f}mm translation, "
              f"{r_spread:.0f}deg rotation")
    else:
        print("  camera pose spread : UNKNOWN (no TF)")
    print("-" * 70)

    # Rotation diversity is NOT interchangeable with translation. A hand-eye
    # translation error contributes R*dt to the reported position; with the
    # wrist orientation held constant R is the same at every station, so that
    # term cancels exactly and the error is invisible no matter how far the arm
    # travels. Only a change in camera ORIENTATION exposes it. So a low rotation
    # spread is disqualifying on its own, even with plenty of translation.
    weak_rot = bool(valid) and r_spread < MIN_ROTATION_SPREAD_DEG
    weak_trans = bool(valid) and t_spread < MIN_TRANSLATION_SPREAD_MM
    if weak_rot or weak_trans:
        print("  INCONCLUSIVE — the stations did not exercise the calibration.")
        if weak_rot:
            print(f"    rotation spread {r_spread:.0f}deg "
                  f"(< {MIN_ROTATION_SPREAD_DEG:.0f}deg): a hand-eye TRANSLATION")
            print("    error cancels exactly when the wrist orientation is fixed,")
            print("    so it is entirely hidden. Any figure above is a LOWER BOUND.")
        if weak_trans:
            print(f"    translation spread {t_spread:.0f}mm "
                  f"(< {MIN_TRANSLATION_SPREAD_MM:.0f}mm): too little parallax.")
        print("  Repeat with stations that differ in WRIST ORIENTATION — rotate")
        print("  the wrist while keeping the date in view, do not just move the arm.")
        verdict = 2
    elif worst <= PASS_MM:
        print(f"  PASS — within {PASS_MM:.0f}mm. Frame error is a small part of")
        print("  the ~11mm margin a 20-25mm date allows.")
        verdict = 0
    elif worst <= MARGINAL_MM:
        print(f"  MARGINAL — {worst:.1f}mm eats most of the ~11mm margin.")
        print("  Usable, but expect intermittent misses at the edges.")
        verdict = 1
    else:
        print(f"  FAIL — {worst:.1f}mm. The same fruit is reported in different")
        print("  places depending on where the arm stands; no downstream tuning")
        print("  can compensate. Recalibrate.")
        verdict = 1

    print("\n  Reminder: this is necessary but still not sufficient. Finish with")
    print("  a physical reach test — both previous bad calibrations passed every")
    print("  statistical check and failed on the robot.")
    print("=" * 70 + "\n")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/external_goal_pose")
    ap.add_argument("--cam-frame", default=CAM_FRAME)
    ap.add_argument("--poses", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    rclpy.init()
    node = HandEyeVerifier(args.topic, args.cam_frame)
    print(f"\nHand-eye acceptance test — {args.poses} poses, "
          f"{args.seconds:.0f}s each.")
    print("Keep ONE date in view and DO NOT move it. Vary wrist orientation")
    print("between stations, not just position.\n")

    goals, cam_poses = [], []
    try:
        for i in range(1, args.poses + 1):
            input(f"  Park the arm at pose {i}/{args.poses}, then press Enter…")
            print(f"  collecting {args.seconds:.0f}s…", flush=True)
            mean, sd, cam, n = node.collect(args.seconds)
            if mean is None:
                print(f"  only {n} distinct samples — is the date detected? "
                      "Skipping this pose.\n")
                continue
            goals.append((mean, sd))
            cam_poses.append(cam)
            print(f"  pose {i}: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}] "
                  f"from {n} samples (jitter {sd:.2f}mm)\n")

        if len(goals) < 2:
            print("Need at least 2 usable poses to compare.")
            sys.exit(2)
        sys.exit(report(goals, cam_poses))
    except KeyboardInterrupt:
        if len(goals) >= 2:
            report(goals, cam_poses)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
