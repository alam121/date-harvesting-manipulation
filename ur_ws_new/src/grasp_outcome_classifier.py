# grasp_outcome_classifier.py
# Pure-Python template-only classifier with slip-by-stage.
from typing import List, Tuple, Callable, Optional
import time

# ---- Templates (your values) ----
OPEN           = [-0.10, -0.09, -0.09]
CLOSED_NOTHING = [-0.17, -0.21, -0.17]
PROPER         = [-0.18, -0.21, -0.18]
WEAK_L         = [-0.18, -0.21, -0.17]
WEAK_R         = [-0.17, -0.21, -0.18]

TEMPLATES: List[Tuple[str, List[float]]] = [
    ("OPEN", OPEN),
    ("CLOSED_NOTHING", CLOSED_NOTHING),
    ("PROPER", PROPER),
    ("WEAK", WEAK_L),
    ("WEAK", WEAK_R),
]
STAGE = {"OPEN": 0, "CLOSED_NOTHING": 1, "WEAK": 2, "PROPER": 3}

def _dist2(a: List[float], b: List[float]) -> float:
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

def classify_triplet(f: List[float]) -> str:
    best, bd = None, float("inf")
    for name, tpl in TEMPLATES:
        d = _dist2(f, tpl)
        if d < bd:
            best, bd = name, d
    return best  # "WEAK" covers left/right variants

class GraspOutcomeClassifier:
    """
    Minimal state machine driven by *your code* calling:
      - start_closing()
      - start_opening() [optional]
      - on_force([f0,f1,f2])
      - tick()  (e.g., from a ROS timer)

    When a decision is made, calls: on_outcome(outcome:str, end:str)
      outcome in {"GRABBED","SLIPPED","NO_GRAB"}
      end in {"OPEN","CLOSED_NOTHING","WEAK","PROPER"}
    """
    def __init__(self,
                 on_outcome: Optional[Callable[[str, str], None]] = None,
                 dead_time_thresh_s: float = 0.6,
                 hold_time_s: float = 0.5):
        
        self.on_outcome = on_outcome
        self.dead_time_thresh_s = dead_time_thresh_s
        self.hold_time_s = hold_time_s

        self.phase = "IDLE"             # IDLE | CLOSING | HOLDING
        self.last_forces = [0.0, 0.0, 0.0]
        self.last_target_time = 0.0     # we emulate "target quiet" using start_closing() time
        self.hold_start_t = 0.0
        self.max_stage_seen = -1

    # ---- drive the state from your main code ----
    def start_closing(self, now: Optional[float] = None):
        t = now or time.time()
        self.phase = "CLOSING"
        self.last_target_time = t       # start the "quiet" timer now
        self.hold_start_t = 0.0
        self.max_stage_seen = -1

    def start_opening(self):
        self.phase = "IDLE"
        self.hold_start_t = 0.0
        self.max_stage_seen = -1

    def on_force(self, forces_3: List[float]):
        """Call from your /gripper/force callback (pass first 3 channels)."""
        self.last_forces = [float(forces_3[0]), float(forces_3[1]), float(forces_3[2])]
        if self.phase in ("CLOSING", "HOLDING"):
            name = classify_triplet(self.last_forces)
            st = STAGE[name]
            if st > self.max_stage_seen:
                self.max_stage_seen = st

    def tick(self, now: Optional[float] = None):
        """Call from a fast ROS timer (e.g., 20–50 Hz)."""
        t = now or time.time()
        if self.phase == "CLOSING":
            # If we've been "closing" for long enough with no new targets fed in,
            # consider that motion ended and start holding.
            if (t - self.last_target_time) >= self.dead_time_thresh_s:
                self.phase = "HOLDING"
                self.hold_start_t = t

        if self.phase == "HOLDING":
            if (t - self.hold_start_t) >= self.hold_time_s:
                self._finalize()

        # in GraspOutcomeClassifier
    def note_target_update(self, now=None):
        self.last_target_time = (now or time.time())

    def mark_close_done(self, now=None):
        # skip dead-time heuristic and start hold *now*
        self.phase = "HOLDING"
        self.hold_start_t = (now or time.time())

    # ---- decision ----
    def _finalize(self):
        self.phase = "IDLE"
        end_name = classify_triplet(self.last_forces)
        end_stage = STAGE[end_name]

        if end_stage < self.max_stage_seen:
            label = "SLIPPED"
        elif end_stage >= STAGE["WEAK"]:
            label = "GRABBED"
        else:
            label = "NO_GRAB"

        if self.on_outcome:
            self.on_outcome(label, end_name)

        print(f"[grasp] final forces={self.last_forces} → end={end_name}")
