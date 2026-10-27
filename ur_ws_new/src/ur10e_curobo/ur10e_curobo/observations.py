"""Atomic perception packets shared by vision and motion (no ROS dependency)."""

from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class FruitObservation:
    stamp_ns: int
    sequence: int
    poses: tuple
    qualities: tuple

    @classmethod
    def decode(cls, payload):
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("Observation must be an object")
        if data.get("frame_id") != "base_link":
            raise ValueError("Fruit observations must use base_link")
        stamp, sequence = int(data["stamp_ns"]), int(data["sequence"])
        poses = tuple(tuple(float(v) for v in p) for p in data["poses"])
        qualities = tuple(tuple(float(v) for v in q) for q in data["qualities"])
        if stamp <= 0 or sequence < 0 or len(poses) != len(qualities):
            raise ValueError("Invalid observation metadata")
        if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in poses):
            raise ValueError("Invalid fruit positions")
        if any(len(q) != 5 or not all(math.isfinite(v) for v in q) for q in qualities):
            raise ValueError("Invalid fruit quality")
        return cls(stamp, sequence, poses, qualities)

    # 0.5 -> 0.8 (2026-10-26). Measured during FINAL, observations already
    # arrive 0.26-0.46s old (depth handoff + inference), and main reads them a
    # little later still while busy with IK -- so at 0.5 the in-flight monitor
    # and reacquire discarded most of them (frames=5 matched=0). Positions are
    # in base_link at the photo's arm pose, so age only matters if the fruit
    # itself moves (wind).
    def fresh(self, now_ns, max_age_s=0.8):
        return 0 <= now_ns - self.stamp_ns <= max_age_s * 1e9
