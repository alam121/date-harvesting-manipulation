#!/usr/bin/env python3
"""Benchmark the vision GPU-handoff handshake against a running vision node.

Mimics the motion node's pause side without needing cuRobo or the robot:
publishes "paused"/"full" cycles on /vision/mode and times how long until the
vision node confirms it has drained in-flight YOLO inference by publishing
"paused" on /vision/mode_state.

Typical use (with an SVO so frames are identical run-to-run):

    # terminal 1 — the vision pipeline replaying a recording
    UR10E_YOLO_TIMING_CSV=/tmp/yolo_timing.csv \
        ros2 run ur10e_curobo vision --svo /path/to/scene.svo

    # terminal 2 — this harness
    python3 benchmarks/bench_gpu_handshake.py --cycles 50 \
        --out /tmp/handshake_latency.csv

Each cycle records the paused->ack latency (which contains the real inference
drain time plus topic round-trip). Rows where the ack never arrives within the
timeout are the cases the old fixed 120 ms sleep was gambling on.
"""

import argparse
import csv
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HandshakeBench(Node):
    def __init__(self, cycles, ack_timeout, work_window, out_path):
        super().__init__("handshake_bench")
        self.cycles = cycles
        self.ack_timeout = ack_timeout
        self.work_window = work_window
        self.out_path = out_path

        self._mode_pub = self.create_publisher(String, "/vision/mode", 10)
        self.create_subscription(String, "/vision/mode_state", self._state_cb, 10)

        self._last_state = None
        self._paused_ack = False
        self.rows = []

    def _state_cb(self, msg):
        self._last_state = msg.data
        if msg.data == "paused":
            self._paused_ack = True

    def _publish(self, mode):
        m = String()
        m.data = mode
        self._mode_pub.publish(m)

    def _spin_until_ack(self, deadline):
        """Return latency (s) when a fresh paused ack arrives, else None on timeout."""
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.005)
            if self._paused_ack:
                return True
        return False

    def run(self):
        # Give pub/sub time to connect before the first cycle.
        settle = time.time() + 1.0
        while time.time() < settle:
            rclpy.spin_once(self, timeout_sec=0.02)

        self.get_logger().info(f"Running {self.cycles} handshake cycles...")
        for i in range(self.cycles):
            # Simulate a period of active detection so an inference is likely in
            # flight when we ask to pause (this is what the drain must handle).
            self._publish("full")
            end_work = time.time() + self.work_window
            while time.time() < end_work:
                rclpy.spin_once(self, timeout_sec=0.01)

            # Arm and command pause, then time the drain-and-ack.
            self._paused_ack = False
            t0 = time.time()
            self._publish("paused")
            acked = self._spin_until_ack(t0 + self.ack_timeout)
            latency_ms = (time.time() - t0) * 1000.0

            self.rows.append({
                "cycle": i,
                "acked": int(acked),
                "latency_ms": round(latency_ms, 3),
            })
            tag = "ack" if acked else "TIMEOUT->fallback"
            self.get_logger().info(
                f"cycle {i:3d}: {tag} latency={latency_ms:7.2f} ms")

        self._write()
        self._summary()

    def _write(self):
        with open(self.out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cycle", "acked", "latency_ms"])
            w.writeheader()
            w.writerows(self.rows)
        self.get_logger().info(f"Wrote {len(self.rows)} rows -> {self.out_path}")

    def _summary(self):
        lat = [r["latency_ms"] for r in self.rows if r["acked"]]
        n_timeout = sum(1 for r in self.rows if not r["acked"])
        if not lat:
            self.get_logger().warn("No acks received — is the vision node running?")
            return
        lat_sorted = sorted(lat)
        p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]
        over_120 = sum(1 for x in lat if x > 120.0)
        print("\n===== HANDSHAKE SUMMARY =====")
        print(f"cycles          : {len(self.rows)}")
        print(f"acked           : {len(lat)}  timeouts: {n_timeout}")
        print(f"latency mean    : {sum(lat)/len(lat):.2f} ms")
        print(f"latency median  : {p(0.50):.2f} ms")
        print(f"latency p95     : {p(0.95):.2f} ms")
        print(f"latency max     : {max(lat):.2f} ms")
        print(f"> 120 ms (old sleep would have overlapped): {over_120} / {len(lat)}")
        print("=============================\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--ack-timeout", type=float, default=0.5,
                    help="max seconds to wait for the paused ack per cycle")
    ap.add_argument("--work-window", type=float, default=0.3,
                    help="seconds of 'full' detection before each pause")
    ap.add_argument("--out", type=str, default="/tmp/handshake_latency.csv")
    args = ap.parse_args()

    rclpy.init()
    node = HandshakeBench(args.cycles, args.ack_timeout, args.work_window, args.out)
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
