import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from ur10e_curobo.gripper_profiles import (
    make_closed_position,
    make_open_position,
    new_gripper_open_extra_deg,
)

FINGER_JOINTS = {
    "f1": ((2, 1.0), (3, 1.0)),
    "f2": ((6, 1.0), (7, 1.0)),
    "f3": ((10, 1.0), (11, 1.0)),
}

ALT_POSES = {
    # Keep the M9/M10 candidate available for diagnostics; this unit's F3 close
    # is M11/M12 in the normal command path.
    "f3_m9_m10": (((8, -1.0), (9, -1.0)),),
    "f3_all": (((8, -1.0), (9, -1.0), (10, 1.0), (11, 1.0)),),
    "close_f3_m9_m10": (
        FINGER_JOINTS["f1"],
        FINGER_JOINTS["f2"],
        ((8, -1.0), (9, -1.0)),
    ),
    "close_f3_all": (
        FINGER_JOINTS["f1"],
        FINGER_JOINTS["f2"],
        ((8, -1.0), (9, -1.0), (10, 1.0), (11, 1.0)),
    ),
}

DIAGNOSTIC_POSES = {
    # Low-amplitude F3 tests. Use these to find the physical curl/down axis
    # without commanding a full close.
    "f3_m9_pos35": (35.0, ((8, 1.0),)),
    "f3_m9_neg35": (35.0, ((8, -1.0),)),
    "f3_m10_pos35": (35.0, ((9, 1.0),)),
    "f3_m10_neg35": (35.0, ((9, -1.0),)),
    "f3_pair_a35": (35.0, ((8, 1.0), (9, -1.0))),
    "f3_pair_b35": (35.0, ((8, -1.0), (9, 1.0))),
    "f3_pair_pos35": (35.0, ((8, 1.0), (9, 1.0))),
    "f3_pair_neg35": (35.0, ((8, -1.0), (9, -1.0))),
    "f3_m11_pos35": (35.0, ((10, 1.0),)),
    "f3_m11_neg35": (35.0, ((10, -1.0),)),
    "f3_m12_pos35": (35.0, ((11, 1.0),)),
    "f3_m12_neg35": (35.0, ((11, -1.0),)),
    "f3_pair_11_12_a35": (35.0, ((10, 1.0), (11, -1.0))),
    "f3_pair_11_12_b35": (35.0, ((10, -1.0), (11, 1.0))),
    "f3_pair_11_12_pos35": (35.0, ((10, 1.0), (11, 1.0))),
    "f3_pair_11_12_neg35": (35.0, ((10, -1.0), (11, -1.0))),
    # Focused candidates after scan narrowed F3 close to M10 or M12.
    "f3_m10_pos20": (20.0, ((9, 1.0),)),
    "f3_m10_neg20": (20.0, ((9, -1.0),)),
    "f3_m12_pos20": (20.0, ((11, 1.0),)),
    "f3_m12_neg20": (20.0, ((11, -1.0),)),
    "f3_m10_m12_pos20": (20.0, ((9, 1.0), (11, 1.0))),
    "f3_m10_m12_neg20": (20.0, ((9, -1.0), (11, -1.0))),
    "f3_m10_pos_m12_neg20": (20.0, ((9, 1.0), (11, -1.0))),
    "f3_m10_neg_m12_pos20": (20.0, ((9, -1.0), (11, 1.0))),
}


def _pose_for(name: str):
    pose = [0.0] * 12
    if name == "open":
        return make_open_position("new")
    if name == "close":
        return make_closed_position()
    if name in DIAGNOSTIC_POSES:
        deg, selected = DIAGNOSTIC_POSES[name]
        close = math.radians(deg)
        for idx, sign in selected:
            pose[idx] = sign * close
        return pose
    close = math.radians(100.0)
    if name in ALT_POSES:
        selected = ALT_POSES[name]
    elif name.startswith("m") and name[1:].isdigit():
        idx = int(name[1:]) - 1
        if idx < 0 or idx >= len(pose):
            raise KeyError(name)
        selected = [(idx,)]
    else:
        selected = [FINGER_JOINTS[name]]
    for joints in selected:
        for idx, sign in joints:
            pose[idx] = sign * close
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
            f"Published DG-3F-M {self.command} x{count} "
            f"(new open extra={new_gripper_open_extra_deg():.1f}deg): {list(self.msg.data)}"
        )


class DG3FMMotorScan(Node):
    def __init__(self, amplitude_deg: float = 30.0):
        super().__init__("dg3fm_motor_scan")
        self.amplitude_deg = float(amplitude_deg)
        self.pub = self.create_publisher(Float32MultiArray, "/gripper/target_joint", 10)
        self.last_joint_state = None
        self.create_subscription(JointState, "/gripper/joint_states", self._joint_cb, 10)

    def _joint_cb(self, msg):
        if len(msg.position) >= 12:
            self.last_joint_state = list(msg.position[:12])

    def _wait_for_driver(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            if self.count_subscribers("/gripper/target_joint") > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def _wait_for_joint_state(self, timeout_s: float = 2.0):
        deadline = time.time() + timeout_s
        while rclpy.ok() and time.time() < deadline:
            if self.last_joint_state is not None:
                return list(self.last_joint_state)
            rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def _publish_pose(self, pose, settle_s: float = 0.7):
        msg = Float32MultiArray()
        msg.data = pose
        for _ in range(4):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.08)
        deadline = time.time() + settle_s
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        if not self._wait_for_driver():
            self.get_logger().warn("No /gripper/target_joint subscriber found; scanning anyway.")

        amp = math.radians(self.amplitude_deg)
        open_pose = [0.0] * 12
        self.get_logger().info("Motor scan: opening all motors to 0 deg")
        self._publish_pose(open_pose, settle_s=1.0)
        baseline = self._wait_for_joint_state()
        if baseline is None:
            self.get_logger().warn("Motor scan: no /gripper/joint_states received; visual-only scan.")
            baseline = [0.0] * 12

        for motor_idx in range(12):
            for sign in (1.0, -1.0):
                pose = [0.0] * 12
                pose[motor_idx] = sign * amp
                label = f"M{motor_idx + 1} {'+' if sign > 0 else '-'}{self.amplitude_deg:.0f}deg"
                self.get_logger().info(f"Motor scan: testing {label}")
                before = self._wait_for_joint_state() or baseline
                self._publish_pose(pose, settle_s=0.8)
                after = self._wait_for_joint_state() or before
                deltas = [math.degrees(a - b) for a, b in zip(after, before)]
                moved = [
                    f"M{i + 1}:{delta:+.1f}deg"
                    for i, delta in enumerate(deltas)
                    if abs(delta) > 2.0
                ]
                self.get_logger().info(
                    f"Motor scan result {label}: " + (", ".join(moved) if moved else "no joint-state delta >2deg")
                )
                self._publish_pose(open_pose, settle_s=0.4)

        self.get_logger().info("Motor scan complete; gripper returned to open command.")


def main():
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "close"
    if command == "scan":
        amplitude = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        rclpy.init()
        node = DG3FMMotorScan(amplitude_deg=amplitude)
        node.run()
        node.destroy_node()
        rclpy.shutdown()
        return

    valid = {
        "open", "close", *FINGER_JOINTS.keys(), *ALT_POSES.keys(),
        *DIAGNOSTIC_POSES.keys(),
        *(f"m{i}" for i in range(1, 13)),
    }
    if command not in valid:
        print(
            "Usage: ros2 run ur10e_curobo dg3fm_finger_test "
            "[open|close|close_f3_m9_m10|close_f3_all|f1|f2|f3|f3_m9_m10|f3_all|"
            "f3_m9_pos35|f3_m9_neg35|f3_m10_pos35|f3_m10_neg35|"
            "f3_pair_a35|f3_pair_b35|f3_pair_pos35|f3_pair_neg35|"
            "f3_m11_pos35|f3_m11_neg35|f3_m12_pos35|f3_m12_neg35|"
            "f3_pair_11_12_a35|f3_pair_11_12_b35|"
            "f3_pair_11_12_pos35|f3_pair_11_12_neg35|scan [deg]|m1|...|m12]"
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
