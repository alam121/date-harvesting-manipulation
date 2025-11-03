#!/usr/bin/env python3
# grasp_outcome_on_close_min_templates_slip_end.py
#
# Uses ONLY your force templates (no posture/extra thresholds):
#   OPEN            = [-0.10, -0.09, -0.09]
#   CLOSED_NOTHING  = [-0.17, -0.21, -0.17]
#   PROPER          = [-0.18, -0.21, -0.18]
#   WEAK_L          = [-0.18, -0.21, -0.17]
#   WEAK_R          = [-0.17, -0.21, -0.18]
#
# Slip rule (by stage):
#   Stage map: OPEN=0, CLOSED_NOTHING=1, WEAK=2, PROPER=3
#   Track max_stage_seen during CLOSING/HOLDING.
#   Final label:
#     if end_stage < max_stage_seen   -> SLIPPED
#     elif end_stage >= 2             -> GRABBED
#     else                            -> NO_GRAB
#
# Publishes JSON on /gripper/grasp_result:
#   {"outcome": "<GRABBED|SLIPPED|NO_GRAB>", "end": "<OPEN|CLOSED_NOTHING|WEAK|PROPER>"}

import time
import json
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray

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

def dist2(a: List[float], b: List[float]) -> float:
    return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

def classify_triplet(f: List[float]) -> str:
    best, bd = None, float("inf")
    for name, tpl in TEMPLATES:
        d = dist2(f, tpl)
        if d < bd:
            best, bd = name, d
    return best  # "WEAK" covers left/right

class GraspOutcomeOnClose(Node):
    def __init__(self):
        super().__init__('grasp_outcome_on_close_min_templates_slip_end')

        # Topics
        self.force_topic  = '/gripper/force'
        self.target_topic = '/gripper/target_joint'

        # End-of-close + hold
        self.dead_time_thresh_s = 0.6
        self.hold_time_s = 0.5

        # I/O
        self.sub_force  = self.create_subscription(Float32MultiArray, self.force_topic,  self._force_cb,  10)
        self.sub_target = self.create_subscription(Float32MultiArray, self.target_topic, self._target_cb, 10)
        self.pub_result = self.create_publisher(String, '/gripper/grasp_result', 10)

        # State
        self.phase = 'IDLE'              # IDLE | CLOSING | HOLDING
        self.last_forces = [0.0, 0.0, 0.0]
        self.last_target_time = 0.0
        self.hold_start_t = 0.0

        self.max_stage_seen = -1
        self._prev_vec = None

        # Timer
        self.create_timer(0.05, self._tick)

        self.get_logger().info("Template-only classifier with slip-by-stage and JSON outcome including 'end'.")

    # ---------- callbacks ----------
    def _target_cb(self, msg: Float32MultiArray):
        now = time.time()
        vec = list(msg.data)
        self.last_target_time = now

        # simple closing/opening trend: sum of deltas on indices 2,6,10
        delta_sum = 0.0
        if self._prev_vec is not None:
            for j in (2, 6, 10):
                delta_sum += (vec[j] - self._prev_vec[j])
        self._prev_vec = vec

        if delta_sum > 0.01 and self.phase != 'CLOSING':
            self._start_closing()
        elif delta_sum < -0.01 and self.phase != 'IDLE':
            self.phase = 'IDLE'

    def _force_cb(self, msg: Float32MultiArray):
        f = list(msg.data)[:3]
        self.last_forces = f

        if self.phase in ('CLOSING', 'HOLDING'):
            name = classify_triplet(f)
            st = STAGE[name]
            if st > self.max_stage_seen:
                self.max_stage_seen = st

    # ---------- timer ----------
    def _tick(self):
        now = time.time()
        if self.phase == 'CLOSING' and self.last_target_time > 0:
            if (now - self.last_target_time) >= self.dead_time_thresh_s:
                self.phase = 'HOLDING'
                self.hold_start_t = now

        if self.phase == 'HOLDING' and (now - self.hold_start_t) >= self.hold_time_s:
            self._finalize()

    # ---------- helpers ----------
    def _start_closing(self):
        self.phase = 'CLOSING'
        self.hold_start_t = 0.0
        self.max_stage_seen = -1
        self.get_logger().info("CLOSING started.")

    def _finalize(self):
        self.phase = 'IDLE'
        end_name = classify_triplet(self.last_forces)
        end_stage = STAGE[end_name]

        if end_stage < self.max_stage_seen:
            label = "SLIPPED"
        elif end_stage >= STAGE["WEAK"]:
            label = "GRABBED"
        else:
            label = "NO_GRAB"

        payload = json.dumps({"outcome": label, "end": end_name})
        self.pub_result.publish(String(data=payload))

        self.get_logger().info(
            f"Outcome: {label} | end={end_name}({end_stage}) | max_stage_seen={self.max_stage_seen} | "
            f"forces={['%.3f'%x for x in self.last_forces]}"
        )

def main():
    rclpy.init()
    rclpy.spin(GraspOutcomeOnClose())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
