#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time

class GripperCommandPublisher(Node):
    def __init__(self):
        super().__init__('gripper_command_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/gripper/target_joint', 10)
        self.force_sub = self.create_subscription(Float32MultiArray, '/gripper/force', self.force_callback, 10)

        self.timer = self.create_timer(0.5, self.step_closure)

        self.step = 0
        self.total_steps = 10
        self.force_values = [-0.0, -0.0, -0.0]
        self.force_threshold = -0.1
        self.active = True

        self.open_position = [
            -0.0942, -0.1500, 1.8660, -0.5062,
            -1.6318, 0.1309, 1.2753, -0.4887,
            0.3333, 0.2234, 1.8673, -0.4311
        ]
        self.closed_position = self.open_position.copy()
        self.closed_position[2] = 2.5660   # finger 1
        self.closed_position[6] = 2.3753   # finger 2
        self.closed_position[10] = 2.5673  # finger 3

        self.current_position = self.open_position.copy()
        self.get_logger().info("🔁 Starting gripper force-aware loop...")

    def step_closure(self):
        if not self.active:
            return

        if all(f <= self.force_threshold for f in self.force_values):
            self.get_logger().info(f"🛑 Force limit reached: {self.force_values}")
            self.active = False
            self.get_logger().info("⏳ Waiting 5 seconds before reopening...")
            time.sleep(5.0)
            self.reset()
            return

        alpha = self.step / self.total_steps
        for i in [2, 6, 10]:
            self.current_position[i] = (
                (1 - alpha) * self.open_position[i] + alpha * self.closed_position[i]
            )

        msg = Float32MultiArray()
        msg.data = self.current_position
        self.publisher_.publish(msg)

        self.get_logger().info(f"📉 Step {self.step}/{self.total_steps} — closing fingers...")

        self.step += 1
        if self.step > self.total_steps:
            self.step = self.total_steps  # Clamp

    def reset(self):
        self.current_position = self.open_position.copy()
        self.step = 0
        self.active = True

        msg = Float32MultiArray()
        msg.data = self.open_position
        self.publisher_.publish(msg)
        self.get_logger().info("🔄 Gripper reopened. Restarting loop...")

    def force_callback(self, msg):
        self.force_values = msg.data
        self.get_logger().info(f"📈 Force: {self.force_values}")

def main(args=None):
    rclpy.init(args=args)
    node = GripperCommandPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

