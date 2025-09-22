#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import matplotlib
matplotlib.use("TkAgg")  # or Qt5Agg; must be interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import threading, csv, time
from collections import deque
import numpy as np

class RealTimeForceGrapher(Node):
    def __init__(self,
                 topic='/gripper/force',
                 maxlen=12000,                 # ~10 min at 20 Hz
                 plot_hz=30.0,                 # UI refresh rate
                 window_s=30.0,                # rolling window
                 reliable=True,                # QoS setting
                 ema_alpha=0.15,               # 0=off, else EMA smoothing
                 diff_thresh=0.5,              # slip detector on dF/dt
                 record_csv=None):             # e.g. "/tmp/force.csv"
        super().__init__('real_time_force_grapher')

        # ---- Buffers ----
        self.lock = threading.Lock()
        self.t = deque(maxlen=maxlen)
        self.y = []                 # list[deque] per channel
        self.y_raw = []             # pre-EMA (optional)
        self.start_t = None
        self.paused = False
        self.window_s = float(window_s)
        self.ema_alpha = float(ema_alpha)
        self.diff_thresh = float(diff_thresh) if diff_thresh else None
        self.events = []            # list[(t, ch, 'slip')]

        # ---- CSV logging (optional) ----
        self.csv_path = record_csv
        self.csv_writer = None
        self.csv_file = None
        if self.csv_path:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t_sec", "ch0", "ch1", "ch2", "…"])

        # ---- QoS ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.sub = self.create_subscription(Float32MultiArray, topic, self.cb, qos)
        self.get_logger().info(f"Subscribing to {topic} ("
                               f"{'RELIABLE' if reliable else 'BEST_EFFORT'})")

        # ---- Plot figure with blitting ----
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 4.5))
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force")
        self.ax.grid(True, alpha=0.25)

        self.lines: list[Line2D] = []
        self.scatter = self.ax.scatter([], [], s=20, marker='x')  # event markers
        self._bg = None  # cached background for blitting

        # threshold band (optional visual aid)
        self.th_band = None
        if self.diff_thresh:
            self.th_band = self.ax.axhline(0, color='k', lw=0.5, alpha=0.2)  # baseline for reference

        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Timers: plot updater
        self.timer = self.create_timer(1.0/plot_hz, self.update_plot)
        self.get_logger().info("Ready. Keys: [p]=pause [c]=clear [s]=save PNG/CSV")

    # ---------------- ROS callback ----------------
    def cb(self, msg: Float32MultiArray):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_t is None:
            self.start_t = now
        t_rel = now - self.start_t
        data = np.asarray(msg.data, dtype=np.float64)

        if not np.all(np.isfinite(data)):
            return  # skip bad sample

        with self.lock:
            # Initialize channels on first message
            if not self.y:
                n = len(data)
                self.y = [deque(maxlen=self.t.maxlen) for _ in range(n)]
                self.y_raw = [deque(maxlen=self.t.maxlen) for _ in range(n)]
                # build line objects once
                for i in range(n):
                    line, = self.ax.plot([], [], lw=1.3, label=f'ch {i}')
                    self.lines.append(line)
                self.ax.legend(loc='upper right', ncol=min(4, n), fontsize=8)
                self.fig.canvas.draw()     # first draw to create artists
                self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)  # cache bg for blit

            if self.paused:
                return

            # Append raw & (optional) EMA
            self.t.append(t_rel)
            for i, v in enumerate(data):
                self.y_raw[i].append(v)
                if self.ema_alpha > 0 and len(self.y[i]) > 0:
                    y_prev = self.y[i][-1]
                    y_new = (1.0 - self.ema_alpha) * y_prev + self.ema_alpha * v
                else:
                    y_new = v
                self.y[i].append(y_new)

            # Simple slip/event detector on derivative
            if self.diff_thresh and len(self.t) >= 2:
                dt = self.t[-1] - self.t[-2]
                if dt > 0:
                    for i in range(len(self.y)):
                        dy = self.y[i][-1] - self.y[i][-2]
                        if abs(dy / dt) > self.diff_thresh:
                            self.events.append((self.t[-1], i, 'slope'))

            # CSV log (optional)
            if self.csv_writer:
                self.csv_writer.writerow([f"{t_rel:.6f}", *[f"{v:.6f}" for v in data]])

    # -------------- Plot update (blitting) --------------
    def update_plot(self):
        with self.lock:
            if not self.lines or not self.t:
                return

            t = np.asarray(self.t)
            t_last = t[-1]
            t_min = max(0.0, t_last - self.window_s)
            # find first index in window
            i0 = np.searchsorted(t, t_min, side='left')

            # Restore background
            if self._bg is None:
                self.fig.canvas.draw()
                self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            else:
                self.fig.canvas.restore_region(self._bg)

            # Update lines within window
            for i, line in enumerate(self.lines):
                yi = np.asarray(self.y[i])
                line.set_data(t[i0:], yi[i0:])
                self.ax.draw_artist(line)

            # Update axes limits (cheap)
            self.ax.set_xlim(t_min, t_last)
            # autoscale y only on visible segment
            y_min, y_max = np.inf, -np.inf
            for i in range(len(self.lines)):
                seg = np.asarray(self.y[i])[i0:]
                if seg.size:
                    y_min = min(y_min, np.min(seg))
                    y_max = max(y_max, np.max(seg))
            if np.isfinite(y_min) and np.isfinite(y_max):
                pad = 0.05 * max(1e-6, (y_max - y_min))
                self.ax.set_ylim(y_min - pad, y_max + pad)

            # Update event markers (in window)
            if self.events:
                et = [et for (et, ch, tag) in self.events if et >= t_min]
                ev = []
                for (et, ch, tag) in self.events:
                    if et >= t_min:
                        # y value at closest time for that channel
                        k = max(i0, np.searchsorted(t, et) - 1)
                        ev.append(self.y[ch][k] if k < len(self.y[ch]) else self.y[ch][-1])
                if et and ev:
                    offsets = np.column_stack((et, ev))
                    self.scatter.set_offsets(offsets)
                    self.scatter.set_alpha(0.8)
                    self.ax.draw_artist(self.scatter)

            # Blit
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()

    # -------------- Keys: pause/clear/save --------------
    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            self.get_logger().info(f"{'Paused' if self.paused else 'Resumed'}")
        elif event.key == 'c':
            with self.lock:
                self.t.clear()
                for dq in (self.y + self.y_raw):
                    dq.clear()
                self.events.clear()
            self.get_logger().info("Cleared buffers.")
        elif event.key == 's':
            ts = time.strftime("%Y%m%d-%H%M%S")
            png = f"force_plot_{ts}.png"
            self.fig.savefig(png, dpi=160)
            self.get_logger().info(f"Saved {png}")
            if self.csv_writer:
                self.get_logger().info(f"Logging → {self.csv_path}")

    # -------------- Clean up --------------
    def destroy_node(self):
        if self.csv_file:
            try: self.csv_file.close()
            except: pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RealTimeForceGrapher(
            topic='/gripper/force',
            maxlen=12000,
            plot_hz=30.0,
            window_s=30.0,
            reliable=True,       # set False if your publisher is BEST_EFFORT
            ema_alpha=0.15,      # 0 to disable smoothing
            diff_thresh=0.6,     # event on |dF/dt|>0.6 units/s
            record_csv=None      # e.g. "/tmp/force.csv"
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
