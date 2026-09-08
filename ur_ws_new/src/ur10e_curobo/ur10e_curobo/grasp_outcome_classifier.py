# grasp_outcome_classifier.py
# Pure-Python template-only classifier with slip-by-stage.
from typing import List, Tuple, Callable, Optional
import time

# ============================================================
# Force Templates for Date Fruit Grasping
# NOTE: Calibrate these values by running with DEBUG_FORCES=True
#       and observing the force readings for each scenario
# ============================================================
DEBUG_FORCES = False  # Enable temporarily only for force-template calibration

# ---- Templates (tune for your dates) ----
# Forces are POSITIVE for Delto gripper (motor current based)
OPEN           = [2.25, 2.55, 3.15]   # fingers open, no contact (baseline)
CLOSED_NOTHING = [2.55, 2.85, 3.15]   # closed on air (slight increase)
WEAK_L         = [3.5, 4.0, 3.5]      # weak grip, left finger light contact
WEAK_R         = [3.5, 4.0, 4.5]      # weak grip, right finger light contact
WEAK_C         = [3.0, 5.0, 3.5]      # weak grip, center finger contact
PROPER         = [5.0, 5.5, 5.0]      # solid 3-finger grip on date

TEMPLATES: List[Tuple[str, List[float]]] = [
    ("OPEN", OPEN),
    ("CLOSED_NOTHING", CLOSED_NOTHING),
    ("WEAK", WEAK_L),
    ("WEAK", WEAK_R),
    ("WEAK", WEAK_C),
    ("PROPER", PROPER),
]
STAGE = {"OPEN": 0, "CLOSED_NOTHING": 1, "WEAK": 2, "PROPER": 3}

# Minimum force difference to distinguish templates
MIN_TEMPLATE_DIST = 0.01

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
        self.contact_evidence = None
        self.close_done = False

    # ---- drive the state from your main code ----
    def start_closing(self, now: Optional[float] = None):
        t = now or time.time()
        self.phase = "CLOSING"
        self.last_target_time = t       # start the "quiet" timer now
        self.hold_start_t = 0.0
        self.max_stage_seen = -1
        self.contact_evidence = None
        self.close_done = False

    def start_opening(self):
        self.phase = "IDLE"
        self.hold_start_t = 0.0
        self.max_stage_seen = -1
        self.contact_evidence = None
        self.close_done = False

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
        # The close controller explicitly calls mark_close_done() after its
        # position-feedback trim and contact snapshot. Do not use elapsed time
        # to finalize here: a slow physical close can exceed the old timeout
        # and otherwise emit a premature result without contact evidence.

        if self.phase == "HOLDING":
            if (t - self.hold_start_t) >= self.hold_time_s:
                self._finalize()

        # in GraspOutcomeClassifier
    def note_target_update(self, now=None):
        self.last_target_time = (now or time.time())

    def mark_close_done(self, now=None):
        # skip dead-time heuristic and start hold *now*
        self.close_done = True
        self.phase = "HOLDING"
        self.hold_start_t = (now or time.time())

    def set_contact_evidence(self, evidence):
        """Attach position/current evidence from the Delto close controller."""
        self.contact_evidence = evidence if isinstance(evidence, dict) else None

    # ---- decision ----
    def _finalize(self):
        self.phase = "IDLE"
        end_name = classify_triplet(self.last_forces)
        end_stage = STAGE[end_name]

        evidence = self.contact_evidence
        if evidence and evidence.get('valid') and not evidence.get('grasp_detected'):
            # An empty closure can generate high current at the mechanical end
            # position.  Position obstruction plus current is authoritative.
            label = "NO_GRAB"
            end_name = "CLOSED_NOTHING"
        elif evidence and evidence.get('valid') and evidence.get('grasp_detected'):
            contact_count = int(evidence.get('contact_count', 0))
            label = "GRABBED"
            end_name = "PROPER" if contact_count >= 3 else "WEAK"
        elif end_stage < self.max_stage_seen:
            label = "SLIPPED"
        elif end_stage >= STAGE["WEAK"]:
            label = "GRABBED"
        else:
            label = "NO_GRAB"

        if self.on_outcome:
            self.on_outcome(label, end_name)

        # Debug output for template calibration
        if DEBUG_FORCES:
            f = self.last_forces
            print(f"[grasp] ═══════════════════════════════════════")
            print(f"[grasp] FORCES: [{f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f}]")
            print(f"[grasp] RESULT: {label} ({end_name})")
            print(f"[grasp] max_stage={self.max_stage_seen} end_stage={end_stage}")
            print(f"[grasp] ═══════════════════════════════════════")
            # Suggest template update if this looks like a valid grab
            if label == "GRABBED" and end_name == "PROPER":
                print(f"[grasp] 💡 Good grab! If grip was solid, use these as PROPER template:")
                print(f"[grasp]    PROPER = [{f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}]")
