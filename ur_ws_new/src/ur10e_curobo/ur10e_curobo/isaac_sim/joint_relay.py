"""Downsample physical UR joint feedback for the remote Isaac twin."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from .bridge import JOINT_ORDER


LATEST_ONLY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class IsaacJointRelay(Node):
    """Publish only the newest complete UR state at a network-friendly rate."""

    def __init__(self):
        super().__init__("isaac_joint_relay")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.latest = None
        self.publisher = self.create_publisher(
            JointState, "/isaac/physical_joint_states", LATEST_ONLY_QOS
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_cb, LATEST_ONLY_QOS
        )
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / max(rate, 1.0), self._publish_latest)
        self.get_logger().info(
            f"Relaying newest physical UR joint state at {rate:.1f} Hz."
        )

    def _joint_cb(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            return
        lookup = {name: value for name, value in zip(msg.name, msg.position)}
        if not all(name in lookup for name in JOINT_ORDER):
            return
        self.latest = msg

    def _publish_latest(self) -> None:
        if self.latest is not None:
            self.publisher.publish(self.latest)


def main() -> None:
    rclpy.init()
    node = IsaacJointRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
