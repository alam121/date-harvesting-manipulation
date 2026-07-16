import math
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


FINGER_JOINTS = {
    "f1": (2, 3),
    "f2": (6, 7),
    "f3": (10, 11),
}

ALT_POSES = {
    # Some DG-3F-M units report F3 in the last four motor slots but physically
    # curl on M1/M2 of that finger instead of M3/M4.
    "f3_alt": ((8, 9),),
    "f3_all": ((8, 9, 10, 11),),
    "close_f3_alt": ((2, 3), (6, 7), (8, 9)),
    "close_f3_all": ((2, 3), (6, 7), (8, 9, 10, 11)),
}


def _pose_for(name: str):
    pose = [0.0] * 12
    close = math.radians(100.0)
    if name == "open":
        return pose
    if name == "close":
        selected = FINGER_JOINTS.values()
    elif name in ALT_POSES:
        selected = ALT_POSES[name]
    elif name.startswith("m") and name[1:].isdigit():
        idx = int(name[1:]) - 1
        if idx < 0 or idx >= len(pose):
            raise KeyError(name)
        selected = [(idx,)]
    else:
        selected = [FINGER_JOINTS[name]]
    for joints in selected:
        for idx in joints:
            pose[idx] = close
    return pose


class DG3FMFingerTest(Node):
    def __init__(self, command: str):
        super().__init__("dg3fm_finger_test")
        self.command = command
        self.pub = self.create_publisher(Float32MultiArray, "/gripper/target_joint", 10)
        self.msg = Float32MultiArray()
        self.msg.data = _pose_for(self.command)

    def wait_for_driver(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            if self.count_subscribers("/gripper/target_joint") > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def publish_burst(self, count: int = 6, delay_s: float = 0.15):
        for _ in range(count):
            self.pub.publish(self.msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(delay_s)
        self.get_logger().info(
            f"Published DG-3F-M {self.command} x{count}: {list(self.msg.data)}"
        )


def main():
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "close"
    valid = {
        "open", "close", *FINGER_JOINTS.keys(), *ALT_POSES.keys(),
        *(f"m{i}" for i in range(1, 13)),
    }
    if command not in valid:
        print(
            "Usage: ros2 run ur10e_curobo dg3fm_finger_test "
            "[open|close|close_f3_alt|close_f3_all|f1|f2|f3|f3_alt|f3_all|m1|...|m12]"
        )
        raise SystemExit(2)

    rclpy.init()
    node = DG3FMFingerTest(command)
    if not node.wait_for_driver():
        node.get_logger().warn(
            "No /gripper/target_joint subscriber found; publishing anyway."
        )
    node.publish_burst()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
