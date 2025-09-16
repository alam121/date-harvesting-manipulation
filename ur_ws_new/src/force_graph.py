#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import threading
from collections import deque

class RealTimeForceGrapher(Node):
    def __init__(self, maxlen=3000, plot_hz=20.0, topic='/gripper/force'):
        super().__init__('real_time_force_grapher')

        # Subscribe to force topic
        self.sub = self.create_subscription(Float32MultiArray, topic, self.callback, 10)

        # Buffers (capped history)
        self.lock = threading.Lock()
        self.times = deque(maxlen=maxlen)      # time (s since first msg)
        self.channels = None                   # list[deque], one per channel, set on first msg
        self.start_time = None

        # Plotting
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.lines = []
        self.window_s = 30.0                   # show last 30 seconds on x-axis (tweak as needed)

        # Redraw timer at fixed rate (20 Hz by default)
        self.timer = self.create_timer(1.0/plot_hz, self.update_plot)

        self.get_logger().info(f"Waiting for {topic} messages…")

    def callback(self, msg: Float32MultiArray):
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.start_time is None:
                self.start_time = now
            t = now - self.start_time
            data = list(msg.data)

            with self.lock:
                # On first message, set up per-channel deques and plot lines
                if self.channels is None:
                    self.channels = [deque(maxlen=self.times.maxlen) for _ in data]
                    for i in range(len(data)):
                        line, = self.ax.plot([], [], label=f'ch {i}')
                        self.lines.append(line)
                    self.ax.set_xlabel('Time (s)')
                    self.ax.set_ylabel('Force')
                    self.ax.legend(loc='upper right')
                    self.ax.grid(True)

                # Append new sample
                self.times.append(t)
                for i, v in enumerate(data):
                    self.channels[i].append(v)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {type(e).__name__}: {e}")

    def update_plot(self):
        try:
            with self.lock:
                if not self.lines or not self.times:
                    return
                times = list(self.times)
                t_last = times[-1]
                # Optional rolling time window on x-axis
                t_min = max(0.0, t_last - self.window_s)

                for i, line in enumerate(self.lines):
                    ys = list(self.channels[i])
                    # Slice to visible window (times and ys have same length)
                    # Find first index >= t_min
                    idx = 0
                    for k, tk in enumerate(times):
                        if tk >= t_min:
                            idx = k
                            break
                    line.set_data(times[idx:], ys[idx:])

                self.ax.set_xlim(t_min, t_last)
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=True)

            # Redraw (decoupled from message rate)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        except Exception as e:
            self.get_logger().error(f"Error in update_plot: {type(e).__name__}: {e}")

    def run(self):
        try:
            rclpy.spin(self)
        except KeyboardInterrupt:
            pass
        finally:
            plt.ioff()
            plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeForceGrapher(maxlen=3000, plot_hz=20.0, topic='/gripper/force')
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
