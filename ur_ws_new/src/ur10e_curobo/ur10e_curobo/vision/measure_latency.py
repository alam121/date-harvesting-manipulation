#!/usr/bin/env python3
"""Where vision's delay goes: per-stage latency of /vision/fruit_observation.

Passive -- subscribes only. Run it during a stationary test or a harvest:

    python3 -m ur10e_curobo.vision.measure_latency --seconds 120

Stages, from the photo's acquisition time:
  grab+depth  camera grab and image/depth retrieval in the capture thread
  handoff     waiting in the capture queue for the vision loop
  to_yolo     loop holding the packet before submitting it (depth handshake)
  yolo        queued + inference until the result is consumed
  geometry    depth, TF, tracking, scoring until published
  transport   publish -> received here (DDS + this subscriber)
Phases follow main's /vision/mode commands: idle/discovery = "full"; in each
cycle the first "reacquire_fast" is APPROACH travel, the second is FINAL.
"""
import argparse
import json
import statistics
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

STAGES = (("grab+depth", None, "ready"), ("handoff", "ready", "got"),
          ("to_yolo", "got", "submit"), ("yolo", "submit", "infer"),
          ("geometry", "infer", "publish"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node("vision_latency_probe")
    phase = {"name": "full", "fast_count": 0}
    rows = {}

    def on_mode(msg):
        mode = msg.data.partition(":")[0]
        if mode == "full":
            phase.update(name="full", fast_count=0)
        elif mode == "reacquire_fast":
            phase["fast_count"] += 1
            phase["name"] = "approach" if phase["fast_count"] == 1 else "final"
        elif mode == "reacquire":
            phase["name"] = "reacquire"

    def on_obs(msg):
        data = json.loads(msg.data)
        lat = data.get("latency_ms")
        if not lat:
            return
        received = (node.get_clock().now().nanoseconds - int(data["stamp_ns"])) / 1e6
        row = {name: lat[b] - (lat[a] if a else 0.0) for name, a, b in STAGES}
        # Split "yolo" with the worker's own timings.
        if "predict_wall_ms" in lat:
            row["y.wait"] = lat["capture_to_start_ms"] - lat["submit"]
            row["y.predict"] = lat.get("color_ms", 0.0) + lat["predict_wall_ms"]
            row["y.masks"] = lat["decode_mask_ms"]
            # predict_wall split: TensorRT kernels vs CPU pre/post around them.
            if "inference_ms" in lat:
                row["  .trt"] = lat["inference_ms"]
                row["  .pre"] = lat["preprocess_ms"]
                row["  .post"] = lat["postprocess_ms"]
            if "post_masks_ms" in lat:
                row["   ..filter"] = lat["post_filter_ms"]
                row["   ..masks"] = lat["post_masks_ms"]
                row["   ..tail"] = lat["post_tail_ms"]
                row["#masked"] = lat["mask_rows"]
            row["y.pickup"] = (row["yolo"] - row["y.wait"] - row["y.predict"]
                               - row["y.masks"])
        row["transport"] = received - lat["publish"]
        row["TOTAL"] = received
        rows.setdefault(phase["name"], []).append(row)

    node.create_subscription(String, "/vision/mode", on_mode, 10)
    node.create_subscription(String, "/vision/fruit_observation", on_obs, 50)
    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not rows:
        print("No observations with latency_ms -- is vision running the new code?")
        return
    names = [s[0] for s in STAGES] + ["transport", "TOTAL"]
    if any("y.wait" in r for rs in rows.values() for r in rs):
        names[4:4] = ["y.wait", "y.predict", "  .pre", "  .trt", "  .post",
                      "   ..filter", "   ..masks", "   ..tail", "#masked",
                      "y.masks", "y.pickup"]
    print(f"{'phase':<10}{'n':>5}  " + "".join(f"{n:>13}" for n in names))
    print(" " * 17 + "".join(f"{'med / p95':>13}" for _ in names))
    for ph in ("full", "approach", "final", "reacquire"):
        data = rows.get(ph)
        if not data:
            continue
        cells = []
        for n in names:
            vals = sorted(r[n] for r in data if n in r) or [float("nan")]
            p95 = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
            cells.append(f"{statistics.median(vals):6.0f}/{p95:<6.0f}")
        print(f"{ph:<10}{len(data):>5}  " + "".join(f"{c:>13}" for c in cells))
    rate = {ph: len(rs) / args.seconds for ph, rs in rows.items()}
    print("\nall values in ms; observations/s overall: "
          f"{sum(rate.values()):.1f}")


if __name__ == "__main__":
    main()
