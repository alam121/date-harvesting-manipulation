"""Temporal tracking and fruit ID management."""

import math
import threading
from time import time
from typing import Dict, List, Any

# Nearest-neighbour association in base_link. 15mm was tight against real
# measurement jitter -- an arm moving at 0.3 m/s with 28ms of TF lag is already
# ~8mm, before depth noise -- and every miss created a duplicate registry entry.
ASSOCIATION_RADIUS_M = 0.025
# Two candidates this close in distance are treated as a tie.
AMBIGUITY_MARGIN_M = 0.003

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
        self._registry_lock = threading.RLock()
        self._next_fruit_id = 1
        self._claimed_ids = set()

    def begin_frame(self):
        with self._registry_lock:
            self._claimed_ids.clear()
            self.cleanup_old_fruit_ids()
            self._merge_duplicate_ids()

    def associate_fruit(self, position_xyz):
        """Nearest base-frame association, with at most one detection per ID."""
        with self._registry_lock:
            matches = sorted((math.dist(position_xyz, info["position"]), fid)
                             for fid, info in self.fruit_id_registry.items()
                             if fid not in self._claimed_ids)
            fruit_id = None
            if matches and matches[0][0] <= ASSOCIATION_RADIUS_M:
                # Ambiguity only matters when the two candidates are genuinely
                # DIFFERENT fruit. If the top two registry entries sit within
                # the association radius of each other they are duplicate
                # records of the same fruit, and refusing to match simply mints
                # a third -- which makes the next frame more ambiguous still.
                #
                # That is a trap with no exit: one frame of >radius jitter
                # creates the first duplicate, and from then on a completely
                # stationary date gets a fresh id every frame, never reaches
                # MIN_FRAMES_TO_SHOW, and never reaches the goal publisher.
                # Reproduced 2026-09-16 with /vision/all_fruit_poses publishing
                # an empty list at 4.8 Hz while YOLO was detecting normally.
                ambiguous = False
                if (len(matches) > 1
                        and matches[1][0] - matches[0][0] < AMBIGUITY_MARGIN_M):
                    p0 = self.fruit_id_registry[matches[0][1]]["position"]
                    p1 = self.fruit_id_registry[matches[1][1]]["position"]
                    ambiguous = math.dist(p0, p1) > ASSOCIATION_RADIUS_M
                if not ambiguous:
                    fruit_id = matches[0][1]
            if fruit_id is None:
                fruit_id = self._next_fruit_id
                self._next_fruit_id += 1
            self._claimed_ids.add(fruit_id)
            self.register_fruit(fruit_id, list(position_xyz))
            return fruit_id

    def _merge_duplicate_ids(self) -> None:
        """Drop registry entries that describe the same fruit.

        Belt and braces for the trap above: even with the ambiguity fix, a
        duplicate can still be created legitimately (a fruit seen, lost for
        >MAX_FRAMES_TO_KEEP, then re-detected). Left in place they accumulate
        and make association steadily less certain, so collapse them on the
        entry that has been seen most recently.
        """
        ids = sorted(self.fruit_id_registry,
                     key=lambda f: self.fruit_id_registry[f]["last_seen"],
                     reverse=True)
        keep = []
        for fid in ids:
            pos = self.fruit_id_registry[fid]["position"]
            if any(math.dist(pos, self.fruit_id_registry[k]["position"])
                   <= ASSOCIATION_RADIUS_M for k in keep):
                del self.fruit_id_registry[fid]
                self._claimed_ids.discard(fid)
            else:
                keep.append(fid)

    def increment_fruit_attempt(self, position_xyz: List[float]) -> None:
        """
        Increment attempt count for fruit at given position.
        Call this after a grasp attempt (success or failure).
        """
        with self._registry_lock:
            matches = [(math.dist(position_xyz, info["position"]), fid)
                       for fid, info in self.fruit_id_registry.items()]
            if not matches or min(matches)[0] > 0.015:
                return
            _, fruit_id = min(matches)

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
        with self._registry_lock:
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
        with self._registry_lock:
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
        with self._registry_lock:
            now_time = time()
            if fruit_id not in self.fruit_id_registry:
                self.fruit_id_registry[fruit_id] = {
                    "last_seen": now_time,
                    "attempt_count": 0,
                    "position": position
                }
            else:
                self.fruit_id_registry[fruit_id]["last_seen"] = now_time
                self.fruit_id_registry[fruit_id]["position"] = position

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
                    "confirmed": MIN_FRAMES_TO_SHOW <= 1,
                    "last_data": current_data.get(fid, None)
                }
            else:
                if fid in current_ids:
                    # Still detected - increment seen counter, reset missed
                    self.detection_history[fid]["frames_seen"] += 1
                    if self.detection_history[fid]["frames_seen"] >= MIN_FRAMES_TO_SHOW:
                        self.detection_history[fid]["confirmed"] = True
                    self.detection_history[fid]["frames_missed"] = 0
                    self.detection_history[fid]["last_data"] = current_data[fid]
                else:
                    # Not detected this frame - increment missed counter
                    self.detection_history[fid]["frames_missed"] += 1
                    self.detection_history[fid]["frames_seen"] = 0

        # Build stabilized output (only include fruits that meet criteria)
        stabilized = []
        to_remove = []

        for fid, hist in self.detection_history.items():
            # Include fruit if:
            # 1. Seen in enough consecutive frames (confirmed detection)
            # 2. OR was previously confirmed and not missed for too long (persistence)

            if hist["frames_missed"] > MAX_FRAMES_TO_KEEP:
                to_remove.append(fid)
                continue
            if hist["confirmed"] and hist["last_data"] is not None:
                stabilized.append(hist["last_data"])

        # Remove stale entries
        for fid in to_remove:
            del self.detection_history[fid]

        return stabilized
