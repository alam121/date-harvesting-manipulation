"""Grasp learning system — learns from force profile during closure.

Key insight: when grabbing an object, force deltas rise EARLIER during closure
(as fingers contact the object mid-closure). When closing on air, forces only
spike at the very end due to mechanical stop. The signal is WHEN forces rise,
not how much.
"""

import csv
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Optional, List

from . import grasp_learning_config as cfg


@dataclass
class GraspRecord:
    """One grasp attempt with force profile and outcome."""
    timestamp: float = 0.0
    fruit_x: float = 0.0
    fruit_y: float = 0.0
    fruit_z: float = 0.0
    # First step where any finger delta > contact threshold
    first_contact_step: int = 10
    total_steps: int = 10
    # Whether gripper stopped early (large objects only)
    stopped_early: bool = False
    closure_step: int = 10
    # Final force deltas (for logging, not primary signal)
    delta_f0: float = 0.0
    delta_f1: float = 0.0
    delta_f2: float = 0.0
    success: Optional[bool] = None


_CSV_FIELDS = [f.name for f in GraspRecord.__dataclass_fields__.values()]


class GraspLearner:
    """Learns grasp success from force profile timing.

    Tracks the step at which force first rises during closure.
    Early rise (step 3-6 of 10) = fingers hit object = likely holding something.
    Late rise (step 8-9) or no rise = mechanical stop = probably air.

    Learns a contact_step threshold via EMA:
    - Success with early contact: pull threshold later (more permissive)
    - Failure: push threshold earlier (more strict)
    """

    def __init__(self):
        self.enabled = cfg.GRASP_LEARNING_ENABLED
        self._ensure_data_dir()

        self._stats = {
            "count": 0,
            "successes": 0,
            "contact_step_threshold": cfg.DEFAULT_CONTACT_STEP_THRESHOLD,
        }
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_enough_data(self) -> bool:
        return self._stats["count"] >= cfg.MIN_SAMPLES_TO_LEARN

    def predict_grasp_success(self, first_contact_step: int,
                               total_steps: int,
                               stopped_early: bool) -> str:
        """Predict grasp quality from force profile timing.

        Args:
            first_contact_step: Step where force first exceeded contact threshold.
                                Equal to total_steps if no early contact detected.
            total_steps: Total closure steps.
            stopped_early: Whether gripper stopped before full closure.

        Returns:
            "PROPER" — early contact detected (object in gripper).
            "NO_CONTACT" — no early contact (probably air).
        """
        if stopped_early:
            return "PROPER"

        threshold = self._stats["contact_step_threshold"]
        if first_contact_step < threshold:
            return "PROPER"
        return "NO_CONTACT"

    def suggest_action(self, first_contact_step: int,
                        total_steps: int,
                        stopped_early: bool) -> str:
        """Suggest what to do based on force profile.

        Returns:
            "PROCEED" — grip looks good.
            "REGRIP" — no contact detected, try again.
        """
        prediction = self.predict_grasp_success(
            first_contact_step, total_steps, stopped_early
        )
        if prediction == "PROPER":
            return "PROCEED"
        return "REGRIP"

    def log_attempt(self, record: GraspRecord) -> None:
        """Log a completed grasp record and update the model."""
        if record.timestamp == 0.0:
            record.timestamp = time.time()

        self._append_csv(record)

        if record.success is not None:
            self._update(record)
            self._save_model()

        success_str = "SUCCESS" if record.success else "FAIL"
        count = self._stats["count"]
        rate = self._stats["successes"] / max(count, 1)
        threshold = self._stats["contact_step_threshold"]
        print(f"[LEARNER] {success_str} | "
              f"rate={rate:.0%} ({count} attempts) | "
              f"contact_step_threshold={threshold:.1f} | "
              f"first_contact={record.first_contact_step}/{record.total_steps}")

    def get_stats_summary(self) -> str:
        count = self._stats["count"]
        rate = self._stats["successes"] / max(count, 1)
        threshold = self._stats["contact_step_threshold"]
        return (f"Grasp Learning: {rate:.0%} success ({count} attempts), "
                f"contact_step_threshold={threshold:.1f}")

    # ------------------------------------------------------------------
    # Model update (EMA-based)
    # ------------------------------------------------------------------

    def _update(self, record: GraspRecord) -> None:
        self._stats["count"] += 1
        if record.success:
            self._stats["successes"] += 1

        alpha = cfg.LEARNING_RATE
        current = self._stats["contact_step_threshold"]

        if record.success:
            # Success: pull threshold toward (first_contact_step + 1)
            # This makes detection more permissive — allow later contact steps.
            target = record.first_contact_step + 1
            self._stats["contact_step_threshold"] = (
                (1 - alpha) * current + alpha * target
            )
        else:
            # Failure: push threshold earlier — need contact sooner to trust it.
            # Use (first_contact_step - 1) as target, clamped to 1.
            target = max(1, record.first_contact_step - 1)
            self._stats["contact_step_threshold"] = (
                (1 - alpha) * current + alpha * target
            )

        # Clamp to reasonable range [2, total_steps-1]
        self._stats["contact_step_threshold"] = max(
            2, min(record.total_steps - 1, self._stats["contact_step_threshold"])
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_data_dir(self) -> None:
        os.makedirs(cfg.DATA_DIR, exist_ok=True)

    def _append_csv(self, record: GraspRecord) -> None:
        file_exists = os.path.exists(cfg.GRASP_LOG_PATH)
        with open(cfg.GRASP_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            row = asdict(record)
            row["success"] = str(record.success) if record.success is not None else ""
            writer.writerow(row)

    def _save_model(self) -> None:
        with open(cfg.MODEL_PATH, "wb") as f:
            pickle.dump(self._stats, f)

    def _load_model(self) -> None:
        if os.path.exists(cfg.MODEL_PATH):
            try:
                with open(cfg.MODEL_PATH, "rb") as f:
                    self._stats = pickle.load(f)
                if "contact_step_threshold" not in self._stats:
                    print("[LEARNER] Old model format detected, resetting.")
                    self._stats = {
                        "count": 0,
                        "successes": 0,
                        "contact_step_threshold": cfg.DEFAULT_CONTACT_STEP_THRESHOLD,
                    }
                    self._save_model()
                    return
                count = self._stats["count"]
                rate = self._stats["successes"] / max(count, 1)
                threshold = self._stats["contact_step_threshold"]
                print(f"[LEARNER] Loaded: {rate:.0%} success ({count} attempts), "
                      f"contact_step_threshold={threshold:.1f}")
            except Exception as e:
                print(f"[LEARNER] Failed to load model: {e}")
                self._stats = {
                    "count": 0,
                    "successes": 0,
                    "contact_step_threshold": cfg.DEFAULT_CONTACT_STEP_THRESHOLD,
                }
