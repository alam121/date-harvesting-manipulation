#!/usr/bin/env python3
import cv2
import numpy as np
import time
from collections import deque


class GraspVisualizer:
    """
    Live OpenCV-based dashboard for your 3-finger force classifier.
    - visualizes real-time forces
    - shows matched template
    - shows classifier phase + outcome
    """

    def __init__(self, window="Gripper Status", history_len=150):

        self.win = window
        self.history_len = history_len

        # sliding history buffer for forces (for graph)
        self.f0 = deque(maxlen=history_len)
        self.f1 = deque(maxlen=history_len)
        self.f2 = deque(maxlen=history_len)

        # text overlays
        self.current_stage = "IDLE"
        self.current_template = "?"
        self.current_outcome = ""
        self.current_forces = [0.0, 0.0, 0.0]
        self.last_update_t = 0

        # color scheme
        self.colors = {
            "OPEN": (200, 200, 200),
            "WEAK": (0, 255, 255),
            "PROPER": (0, 200, 0),
            "CLOSED_NOTHING": (50, 50, 200),
            "SLIPPED": (0, 128, 255),
            "GRABBED": (0, 255, 0),
            "NO_GRAB": (0, 0, 255),
        }

    # ---------------------------------------------------
    # UPDATE FROM ROS FORCE CALLBACK
    # ---------------------------------------------------
    def update_forces(self, f):
        """Call this every time you get /gripper/force data."""
        self.current_forces = f
        self.f0.append(f[0])
        self.f1.append(f[1])
        self.f2.append(f[2])
        self.last_update_t = time.time()

    # ---------------------------------------------------
    # UPDATE FROM CLASSIFIER
    # ---------------------------------------------------
    def update_classifier(self, phase: str, template: str):
        """Call this every tick from your classifier: phase + matched template."""
        self.current_stage = phase
        self.current_template = template

    def update_outcome(self, outcome: str):
        """Call this when classifier finalizes: GRABBED / SLIPPED / NO_GRAB"""
        self.current_outcome = outcome

    # ---------------------------------------------------
    # DRAW DASHBOARD
    # ---------------------------------------------------
    def draw(self):
        w, h = 600, 300
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # text region backgrounds
        cv2.rectangle(img, (0, 0), (w, 50), (40, 40, 40), -1)

        # HEADER TEXT
        phase_col = (0, 255, 255) if self.current_stage == "CLOSING" else (0, 200, 0)
        cv2.putText(img, f"Phase: {self.current_stage}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_col, 2)

        # CLASSIFICATION TEMPLATE
        tpl_col = self.colors.get(self.current_template, (255, 255, 255))
        cv2.putText(img, f"Template: {self.current_template}",
                    (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tpl_col, 2)

        # OUTCOME (if exists)
        if self.current_outcome:
            out_col = self.colors.get(self.current_outcome, (255, 255, 255))
            cv2.putText(img, f"Outcome: {self.current_outcome}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, out_col, 2)

        # Current forces (numbers)
        f0, f1, f2 = self.current_forces
        cv2.putText(img, f"F0: {f0:.3f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        cv2.putText(img, f"F1: {f1:.3f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.putText(img, f"F2: {f2:.3f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

        # ---------------------------------------------------
        # REAL-TIME GRAPH
        # ---------------------------------------------------
        graph_x0, graph_y0 = 150, 100
        graph_w, graph_h = 430, 180

        cv2.rectangle(img,
                      (graph_x0, graph_y0),
                      (graph_x0 + graph_w, graph_y0 + graph_h),
                      (80, 80, 80), 1)

        def draw_curve(values, color):
            if len(values) < 2:
                return
            vals = np.array(values)
            # normalize for display
            minv, maxv = -0.3, 0.1
            yrng = maxv - minv
            pts = []
            for i, v in enumerate(vals):
                x = graph_x0 + int(i / self.history_len * graph_w)
                y = graph_y0 + int((1 - (v - minv) / yrng) * graph_h)
                pts.append((x, y))
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(img, p, q, color, 2)

        draw_curve(self.f0, (200, 200, 255))
        draw_curve(self.f1, (200, 255, 200))
        draw_curve(self.f2, (255, 200, 200))

        # Render
        cv2.imshow(self.win, img)
        cv2.waitKey(1)

