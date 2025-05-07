# delto_gripper_controller.py

import time
from std_msgs.msg import Float32MultiArray

class DeltoGripperController:
    def __init__(self, node, force_topic='/gripper/force', target_topic='/gripper/target_joint'):
        self.node = node
        self.force_data = [0.0, 0.0, 0.0]
        self.force_threshold = -0.15
        self.steps = 10
        self.step_delay = 0.5  # seconds
        self.current_step = 0

        self.open_position = [
            -0.0942, -0.1500, 2.0660, -0.5062,
            -1.6318, 0.1309, 1.2753, -0.4887,
            0.3333, 0.2234, 2.0673, -0.4311
        ]
        self.closed_position = self.open_position.copy()
        self.closed_position[2] = 2.5660
        self.closed_position[6] = 2.3753
        self.closed_position[10] = 2.5673
        self.current_position = self.open_position.copy()

        self.publisher = node.create_publisher(Float32MultiArray, target_topic, 10)
        self.subscriber = node.create_subscription(Float32MultiArray, force_topic, self.force_callback, 10)

    def force_callback(self, msg):
        self.force_data = msg.data

    def is_force_threshold_reached(self):
        return all(f <= self.force_threshold for f in self.force_data)

    def step_close(self):
        if self.current_step >= self.steps or self.is_force_threshold_reached():
            self.node.get_logger().info("✅ Gripper fully closed or force limit reached.")
            self.current_step = 0
            return False

        alpha = self.current_step / self.steps
        for i in [2, 6, 10]:
            self.current_position[i] = (
                (1 - alpha) * self.open_position[i] + alpha * self.closed_position[i]
            )

        msg = Float32MultiArray()
        msg.data = self.current_position
        self.publisher.publish(msg)
        self.node.get_logger().info(f"Step {self.current_step}/{self.steps} — closing...")
        self.current_step += 1
        return True

    def run_closure_loop(self):
        """Run the full closure loop until force is reached or max steps."""
        while self.step_close():
            time.sleep(self.step_delay)

    def open_gripper(self):
        msg = Float32MultiArray()
        msg.data = self.open_position
        self.publisher.publish(msg)
        self.node.get_logger().info("🔓 Gripper OPEN")

