#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import threading

class RealTimeForceGrapher(Node):
    def __init__(self):
        super().__init__('real_time_force_grapher')

        self.sub = self.create_subscription(
            Float32MultiArray,
            '/gripper/force',
            self.callback,
            10)

        self.lock = threading.Lock()
        self.times = []
        self.values = []
        self.start_time = None

        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.lines = []

        self.get_logger().info("Waiting for /gripper/force messages…")

    def callback(self, msg: Float32MultiArray):
        try:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.start_time is None:
                self.start_time = now
            t = now - self.start_time
            data = list(msg.data)

            with self.lock:
                self.times.append(t)
                self.values.append(data)

                if not self.lines:
                    # first message: create one line per channel
                    for i in range(len(data)):
                        line, = self.ax.plot([], [], label=f'ch {i}')
                        self.lines.append(line)
                    self.ax.set_xlabel('Time (s)')
                    self.ax.set_ylabel('Force')
                    self.ax.legend(loc='upper right')
                    self.ax.grid(True)

                # transpose to channels
                times = self.times
                channels = list(zip(*self.values))
                for i, line in enumerate(self.lines):
                    line.set_data(times, channels[i])

                self.ax.relim()
                self.ax.autoscale_view()

            # redraw and process GUI events
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {type(e).__name__}: {e}")

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
    node = RealTimeForceGrapher()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

