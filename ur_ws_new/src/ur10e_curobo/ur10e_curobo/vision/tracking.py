"""Temporal tracking and fruit ID management."""

from time import time
from typing import Dict, List, Any

from .config import (
    FRUIT_ID_TIMEOUT,
    MAX_FRUIT_ATTEMPTS,
    MIN_FRAMES_TO_SHOW,
    MAX_FRAMES_TO_KEEP,
)


class FruitTracker:
    """Manages fruit ID registry and temporal stabilization."""

    def __init__(self):
        # Fruit ID tracking (prevent re-targeting same fruit)
        self.fruit_id_registry: Dict[int, dict] = {}

        # Temporal stabilization (reduce flickering)
        self.detection_history: Dict[int, dict] = {}

    def increment_fruit_attempt(self, position_xyz: List[float]) -> None:
        """
        Increment attempt count for fruit at given position.
        Call this after a grasp attempt (success or failure).
        """
        fruit_id = hash((
            round(position_xyz[0], 2),
            round(position_xyz[1], 2),
            round(position_xyz[2], 2)
        ))

        if fruit_id in self.fruit_id_registry:
            self.fruit_id_registry[fruit_id]["attempt_count"] += 1
            self.fruit_id_registry[fruit_id]["last_seen"] = time()
            attempt_count = self.fruit_id_registry[fruit_id]["attempt_count"]

            if attempt_count >= MAX_FRUIT_ATTEMPTS:
                print(f"[FRUIT_TRACK] Fruit {fruit_id} blacklisted after {attempt_count} attempts")
            else:
                print(f"[FRUIT_TRACK] Fruit {fruit_id} attempt count: {attempt_count}/{MAX_FRUIT_ATTEMPTS}")
        else:
            print(f"[FRUIT_TRACK] WARNING: Attempted to increment unknown fruit {fruit_id}")

    def cleanup_old_fruit_ids(self) -> None:
        """Remove old fruit IDs from registry (call periodically)."""
        now = time()
        to_remove = [
            fid for fid, info in self.fruit_id_registry.items()
            if now - info["last_seen"] > FRUIT_ID_TIMEOUT
        ]
        for fid in to_remove:
            del self.fruit_id_registry[fid]
        if to_remove:
            print(f"[FRUIT_TRACK] Cleaned up {len(to_remove)} old fruit IDs")

    def is_fruit_blacklisted(self, fruit_id: int) -> bool:
        """Check if a fruit ID has exceeded max attempts."""
        now_time = time()
        if fruit_id in self.fruit_id_registry:
            fruit_info = self.fruit_id_registry[fruit_id]
            # Clean up old entries
            if now_time - fruit_info["last_seen"] > FRUIT_ID_TIMEOUT:
                del self.fruit_id_registry[fruit_id]
                return False
            elif fruit_info["attempt_count"] >= MAX_FRUIT_ATTEMPTS:
                return True
        return False

    def register_fruit(self, fruit_id: int, position: List[float]) -> int:
        """Register or update a fruit in the registry. Returns attempt count."""
        now_time = time()
        if fruit_id not in self.fruit_id_registry:
            self.fruit_id_registry[fruit_id] = {
                "last_seen": now_time,
                "attempt_count": 0,
                "position": position
            }
        else:
            self.fruit_id_registry[fruit_id]["last_seen"] = now_time

        return self.fruit_id_registry[fruit_id]["attempt_count"]

    def stabilize_detections(self, current_targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply temporal filtering to reduce detection flickering.
        Only show fruits that appear consistently across multiple frames.

        Returns: list of stabilized targets (with confirmed detections only)
        """
        # Build current frame fruit IDs
        current_ids = set()
        current_data = {}

        for t in current_targets:
            fid = t["fruit_id"]
            current_ids.add(fid)
            current_data[fid] = t

        # Update history for all fruits
        all_ids = set(self.detection_history.keys()) | current_ids

        for fid in all_ids:
            if fid not in self.detection_history:
                # New detection
                self.detection_history[fid] = {
                    "frames_seen": 1 if fid in current_ids else 0,
                    "frames_missed": 0,
                    "last_data": current_data.get(fid, None)
                }
            else:
                if fid in current_ids:
                    # Still detected - increment seen counter, reset missed
                    self.detection_history[fid]["frames_seen"] += 1
                    self.detection_history[fid]["frames_missed"] = 0
                    self.detection_history[fid]["last_data"] = current_data[fid]
                else:
                    # Not detected this frame - increment missed counter
                    self.detection_history[fid]["frames_missed"] += 1

        # Build stabilized output (only include fruits that meet criteria)
        stabilized = []
        to_remove = []

        for fid, hist in self.detection_history.items():
            # Include fruit if:
            # 1. Seen in enough consecutive frames (confirmed detection)
            # 2. OR was previously confirmed and not missed for too long (persistence)

            is_confirmed = hist["frames_seen"] >= MIN_FRAMES_TO_SHOW
            is_persistent = hist["frames_missed"] > 0 and hist["frames_missed"] <= MAX_FRAMES_TO_KEEP

            if is_confirmed or is_persistent:
                if hist["last_data"] is not None:
                    stabilized.append(hist["last_data"])

            # Clean up entries that are too old
            if hist["frames_missed"] > MAX_FRAMES_TO_KEEP:
                to_remove.append(fid)

        # Remove stale entries
        for fid in to_remove:
            del self.detection_history[fid]

        return stabilized
