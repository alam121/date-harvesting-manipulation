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
    # Optional per pose: (diameter_mm, L, a, b, overlap_iou); NaN = unknown.
    # Empty when vision does not publish them or they are malformed.
    features: tuple = ()
    # Optional per pose: vision's fruit id (-1 unknown). Empty if absent.
    ids: tuple = ()

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
        try:
            features = tuple(tuple(float(v) for v in f) for f in data.get("features") or ())
            if len(features) != len(poses) or any(len(f) != 5 for f in features):
                features = ()
        except (TypeError, ValueError):
            features = ()
        try:
            ids = tuple(int(v) for v in data.get("ids") or ())
            if len(ids) != len(poses):
                ids = ()
        except (TypeError, ValueError):
            ids = ()
        return cls(stamp, sequence, poses, qualities, features, ids)

    # 0.5 -> 0.8 (2026-10-26). Measured during FINAL, observations already
    # arrive 0.26-0.46s old (depth handoff + inference), and main reads them a
    # little later still while busy with IK -- so at 0.5 the in-flight monitor
    # and reacquire discarded most of them (frames=5 matched=0). Positions are
    # in base_link at the photo's arm pose, so age only matters if the fruit
    # itself moves (wind).
    def fresh(self, now_ns, max_age_s=0.8):
        return 0 <= now_ns - self.stamp_ns <= max_age_s * 1e9
