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

    def fresh(self, now_ns, max_age_s=0.5):
        return 0 <= now_ns - self.stamp_ns <= max_age_s * 1e9
