#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque


class GraspVisualizerLogic:
    """
    Core drawing logic. Maintains force history and renders into a numpy image.
    Does NOT show any window by itself.
    """

    def __init__(self, width=800, height=400, history_len=300):
        self.history_len = history_len
        self.W, self.H = width, height

        # Buffers
        self.f0 = deque([0.0] * history_len, maxlen=history_len)
        self.f1 = deque([0.0] * history_len, maxlen=history_len)
        self.f2 = deque([0.0] * history_len, maxlen=history_len)
        self.smooth_forces = [0.0, 0.0, 0.0]
        self.alpha = 0.2  # smoothing factor

        # State
        self.current_outcome = ""
        self.current_stage = ""
        self.current_template = ""

        # Canvas
        self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Colors (BGR for OpenCV)
        self.C_BG = (30, 30, 30)       # background
        self.C_ZERO = (150, 150, 150)  # zero line
        self.C_F0 = (100, 100, 255)    # F0
        self.C_F1 = (100, 255, 100)    # F1
        self.C_F2 = (255, 200, 50)     # F2

    # ------------------------------------------------------------------
    # Data update
    # ------------------------------------------------------------------
    def update_forces(self, f):
        """
        f: iterable of at least 3 floats [f0, f1, f2]
        """
        if len(f) < 3:
            return
        for i in range(3):
            self.smooth_forces[i] = (
                self.alpha * f[i] + (1.0 - self.alpha) * self.smooth_forces[i]
            )
        self.f0.append(self.smooth_forces[0])
        self.f1.append(self.smooth_forces[1])
        self.f2.append(self.smooth_forces[2])

    def set_outcome(self, txt: str):
        self.current_outcome = txt or ""

    def set_classifier_info(self, phase: str, template: str):
        self.current_stage = phase or ""
        self.current_template = template or ""

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def get_frame(self):
        """
        Returns the current graph as an RGB numpy array (H, W, 3).
        """
        canvas = self.canvas
        canvas[:] = self.C_BG

        d0, d1, d2 = list(self.f0), list(self.f1), list(self.f2)
        all_data = d0 + d1 + d2

        if all_data:
            dmin, dmax = min(all_data), max(all_data)
            span = dmax - dmin

            # Smart zooming
            if span < 0.2:
                mid = 0.5 * (dmax + dmin)
                vmin, vmax = mid - 0.15, mid + 0.15
            else:
                pad = span * 0.1
                vmin, vmax = dmin - pad, dmax + pad

            span = max(vmax - vmin, 1e-6)

            def to_y(val):
                return int(self.H - ((val - vmin) / span * self.H))

            # Zero line
            zy = to_y(0.0)
            if 0 <= zy < self.H:
                cv2.line(canvas, (0, zy), (self.W, zy), self.C_ZERO, 1)

            x_step = self.W / max(self.history_len - 1, 1)
            datasets = [d0, d1, d2]
            colors = [self.C_F0, self.C_F1, self.C_F2]
            labels = ["F0", "F1", "F2"]
            final_pos = []

            for idx, data in enumerate(datasets):
                pts = []
                for j, val in enumerate(data):
                    pts.append([int(j * x_step), to_y(val)])
                if len(pts) > 1:
                    pts_arr = np.array(pts, dtype=np.int32)
                    cv2.polylines(canvas, [pts_arr], False, colors[idx], 2, cv2.LINE_AA)
                    final_pos.append(
                        {
                            "y": pts_arr[-1][1],
                            "val": data[-1],
                            "l": labels[idx],
                            "c": colors[idx],
                        }
                    )

            # Avoid label overlap
            final_pos.sort(key=lambda k: k["y"])
            for i in range(1, len(final_pos)):
                if final_pos[i]["y"] - final_pos[i - 1]["y"] < 25:
                    final_pos[i]["y"] = final_pos[i - 1]["y"] + 25

            for item in final_pos:
                txt = f"{item['l']}: {item['val']:.3f}"
                cv2.putText(
                    canvas,
                    txt,
                    (self.W - 160, item["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    item["c"],
                    1,
                    cv2.LINE_AA,
                )

        # Outcome box
        if self.current_outcome:
            col = (0, 200, 0) if "GRAB" in self.current_outcome.upper() else (0, 0, 255)
            cv2.rectangle(
                canvas, (self.W // 2 - 60, 10), (self.W // 2 + 60, 45), col, -1
            )
            cv2.putText(
                canvas,
                self.current_outcome,
                (self.W // 2 - 55, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Header: phase + template
        header_h = 32
        cv2.rectangle(canvas, (0, 0), (self.W, header_h), (40, 40, 40), -1)
        cv2.putText(
            canvas,
            f"Phase: {self.current_stage}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Template: {self.current_template}",
            (260, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Return RGB (even though we show in BGR)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class GraspVisualizer:
    """
    Legacy-compatible visualizer used by the ROS node.

    - Has update_forces(), update_classifier(), update_outcome(), draw()
    - draw() opens / updates an OpenCV window called self.win
    """

    def __init__(self, window: str = "Gripper Status", history_len: int = 300):
        self.win = window
        self.logic = GraspVisualizerLogic(width=800, height=400, history_len=history_len)

    def update_forces(self, f):
        self.logic.update_forces(f)

    def update_classifier(self, phase: str, template: str):
        self.logic.set_classifier_info(phase, template)

    def update_outcome(self, outcome: str):
        self.logic.set_outcome(outcome)

    def draw(self):
        """
        Render the current frame and show it in an OpenCV window.
        Called periodically by a ROS timer in your node.
        """
        rgb = self.logic.get_frame()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.win, bgr)
        # Small waitKey to keep window responsive
        cv2.waitKey(1)
