# grasp_outcome_classifier.py
# Pure-Python template-only classifier with slip-by-stage.
from typing import List, Tuple, Callable, Optional
import time

# ============================================================
# Force Templates for Date Fruit Grasping
# NOTE: Calibrate these values by running with DEBUG_FORCES=True
#       and observing the force readings for each scenario
# ============================================================
DEBUG_FORCES = False  # Calibration done 2026-09-13; see the block below. The
                      # templates below still assume a three-finger grip --
                      # PROPER=[5.0,5.5,5.0] wants the CENTRE channel highest,
                      # which this gripper can never produce, so a perfect
                      # two-finger grip scores nearest to OPEN. Re-measure and
                      # replace all six, then set this back to False.

# ---- Templates (tune for your dates) ----
# Forces are POSITIVE for Delto gripper (motor current based)
# Re-measured 2026-09-13. Channel order is [RIGHT, CENTER, LEFT].
#
# READ THIS BEFORE "FIXING" THE ORDERING: magnitude ANTI-CORRELATES with grip
# quality on this gripper. Closing on air runs the fingers to their limit where
# they STALL, drawing maximum current. A date stops them early, before they
# stall, so a good grip draws LESS. Measured:
#
#     empty close   [59.10, 1.50, 45.90]   remaining [0.10, 0.10, 0.05]
#     empty close   [57.30, 1.80, 48.00]
#     good grip     [48.90, 5.10, 42.00]   remaining [0.26, 0.49, 0.13]
#
# The old values had PROPER ABOVE CLOSED_NOTHING, i.e. the relationship
# backwards, and were on a ~2-5 scale from when the driver used a wrong -17.5
# per-motor baseline. With the baselines actually measured the channel reads in
# tens, so every old number was obsolete twice over.
#
# The CENTER channel stays near zero because that fingertip is small and never
# reaches the fruit; it is the only channel that RISES on a grip (1.7 -> 5.1).
#
# These are the FALLBACK path. classify_outcome prefers contact evidence
# (current delta + remaining travel) whenever it is available, and that is the
# signal to trust -- it separated empty from gripping cleanly every time.
OPEN           = [0.00,  0.00, 0.00]   # measured: 161 samples, zero variance
CLOSED_NOTHING = [58.20, 1.70, 47.00]  # measured: mean of two empty closes
WEAK           = [53.50, 3.40, 44.50]  # NOT MEASURED -- interpolated midpoint.
                                       # Physically a weak grip stops only
                                       # slightly early, so it should sit
                                       # between PROPER and CLOSED_NOTHING.
                                       # Replace with a real slipping-grip
                                       # sample when one is captured.
PROPER         = [48.90, 5.10, 42.00]  # measured: ONE sample only -- widen with
                                       # more good grasps before relying on it

TEMPLATES: List[Tuple[str, List[float]]] = [
    ("OPEN", OPEN),
    ("CLOSED_NOTHING", CLOSED_NOTHING),
    # The three WEAK_L/R/C variants encoded which finger made light contact on a
    # three-finger grip. Meaningless here: the centre finger never contacts, so
    # the pattern is always right+left. Collapsed to one.
    ("WEAK", WEAK),
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
            # Was hardcoded 3. The CENTER finger's tip is small and never
            # reaches the fruit, so contact_count caps at 2 and this returned
            # WEAK for every grasp, however good -- which is why regrips fired
            # on solid grips and no tuning could be judged from the label.
            # min_contacts comes from the evidence dict and is already capped
            # to the number of load-bearing fingers.
            _need = int(evidence.get('min_contacts', 3) or 3)
            end_name = "PROPER" if contact_count >= _need else "WEAK"
        else:
            # NO CONTACT EVIDENCE. Previously this guessed from force templates.
            # It must not: measured 2026-09-13, the force signature cannot
            # discriminate a grasp on this gripper.
            #
            #   good grasp  [48.90, 5.10, 42.00]   centre blocked, 166mA
            #   good grasp  [56.40, 0.00, 54.60]   centre free,     11mA
            #   good grasp  [58.50, 0.00, 55.20]   centre free,      7mA
            #   empty close [59.10, 1.50, 45.90]
            #   empty close [57.30, 1.80, 48.00]
            #
            # The good grasps are 15-17 apart from each other but only 8-11 from
            # an empty close, so grips 2 and 3 match CLOSED_NOTHING. That is not
            # noise: the signature is BIMODAL, depending on whether the date
            # happens to sit against the small centre fingertip. Two different
            # physical geometries, one template -- unfixable by re-calibration.
            #
            # Magnitude also anti-correlates with quality (see the template
            # block above), so "more force" is not "better grip" here either.
            #
            # Slip-by-stage went with it: max_stage_seen is accumulated from the
            # same classify_triplet() matching, so a close that rises through
            # ~[30,0,30] and settles at ~[57,0,55] reads as PROPER-then-
            # CLOSED_NOTHING and would report a SLIP that never happened.
            #
            # Contact evidence (current delta + remaining travel) separated all
            # six closes cleanly and is what _finalize uses above. If it is
            # missing, say so rather than inventing an answer.
            label = "UNKNOWN"
            end_name = "UNKNOWN"
            print("[grasp] no contact evidence; outcome UNKNOWN "
                  "(force templates cannot discriminate on this gripper)")

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
