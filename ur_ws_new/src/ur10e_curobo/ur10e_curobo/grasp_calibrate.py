"""Standalone grasp force calibration tool.

Position the robot near a date manually, then use keyboard commands to
close/open the gripper and label outcomes. Trains the force-profile grasp
learner without running the full pipeline.

Usage:  ros2 run ur10e_curobo calibrate
Keys:   c=close  o=open  y=success  n=fail  s=stats  q=quit
"""

import sys
import termios
import time
import tty

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .delto_gripper_controller import DeltoGripperController
from .grasp_learner import GraspLearner, GraspRecord


class CalibrateNode(Node):
    def __init__(self):
        super().__init__("grasp_calibrate")
        self.gripper = DeltoGripperController(
            self,
            suction=False,
            min_fingers_for_stop=2,
            steps=10,
            step_delay=0.05,
        )
        self.learner = GraspLearner()
        self._last_first_contact_step = 10
        self._last_stopped_early = False
        self._last_closure_step = 0
        self._last_deltas = [0.0, 0.0, 0.0]
        self._pending = False


def _read_key():
    """Read a single keypress (raw terminal mode)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main():
    rclpy.init()
    node = CalibrateNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("Waiting for gripper force data...")
    t0 = time.time()
    while node.gripper.force_data == [0.0, 0.0, 0.0]:
        if time.time() - t0 > 5.0:
            print("Warning: no force data after 5s — is the gripper driver running?")
            break
        time.sleep(0.1)

    print("\n=== Grasp Force Calibration (Profile Mode) ===")
    print(node.learner.get_stats_summary())
    print("\nKeys: [c]lose  [o]pen  [y]=success  [n]=fail  [s]tats  [q]uit\n")

    try:
        while True:
            key = _read_key()

            if key == "c":
                print("Closing gripper...", flush=True)
                node.gripper.run_closure_loop()

                first_contact = getattr(node.gripper, 'closure_first_contact_step', node.gripper.steps)
                stopped_early = getattr(node.gripper, 'closure_stopped_early', False)
                closure_step = getattr(node.gripper, 'closure_step_stopped', node.gripper.steps)
                deltas = getattr(node.gripper, 'closure_deltas', [0.0, 0.0, 0.0])
                profile = getattr(node.gripper, 'closure_force_profile', [])

                node._last_first_contact_step = first_contact
                node._last_stopped_early = stopped_early
                node._last_closure_step = closure_step
                node._last_deltas = deltas

                prediction = node.learner.predict_grasp_success(
                    first_contact, node.gripper.steps, stopped_early
                )
                action = node.learner.suggest_action(
                    first_contact, node.gripper.steps, stopped_early
                )
                threshold = node.learner._stats["contact_step_threshold"]
                early_str = "EARLY (contact)" if stopped_early else "FULL"

                # Show force profile — max delta per step, * marks first contact
                if profile:
                    print(f"  Profile: ", end="")
                    for i, d in enumerate(profile):
                        marker = "<" if i == first_contact else " "
                        print(f"s{i}:{max(d):.1f}{marker}", end="")
                    print()

                print(f"  Closure:  {early_str} at step {closure_step}/{node.gripper.steps}")
                print(f"  Contact:  step {first_contact}/{node.gripper.steps} "
                      f"(threshold<{threshold:.1f}) → {prediction} → {action}")
                print(f"  Label: [y]=success  [n]=fail  [o]=open (skip)")
                node._pending = True

            elif key == "o":
                print("Opening gripper...", flush=True)
                node.gripper.open_gripper()
                node._pending = False
                print("  Ready.")

            elif key == "y" and node._pending:
                record = GraspRecord(
                    first_contact_step=node._last_first_contact_step,
                    total_steps=node.gripper.steps,
                    stopped_early=node._last_stopped_early,
                    closure_step=node._last_closure_step,
                    delta_f0=node._last_deltas[0],
                    delta_f1=node._last_deltas[1],
                    delta_f2=node._last_deltas[2],
                    success=True,
                )
                node.learner.log_attempt(record)
                node._pending = False

            elif key == "n" and node._pending:
                record = GraspRecord(
                    first_contact_step=node._last_first_contact_step,
                    total_steps=node.gripper.steps,
                    stopped_early=node._last_stopped_early,
                    closure_step=node._last_closure_step,
                    delta_f0=node._last_deltas[0],
                    delta_f1=node._last_deltas[1],
                    delta_f2=node._last_deltas[2],
                    success=False,
                )
                node.learner.log_attempt(record)
                node._pending = False

            elif key == "s":
                print(f"\n  {node.learner.get_stats_summary()}\n")

            elif key == "q":
                print("\nQuitting.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.gripper.open_gripper()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
