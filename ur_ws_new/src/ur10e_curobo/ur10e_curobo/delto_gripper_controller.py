# delto_gripper_controller.py
import time
import math
import os
from std_msgs.msg import Float32MultiArray
from .gripper_profiles import (
    finger_joint_indices_for_profile,
    make_closed_position,
    make_open_position,
    new_gripper_open_extra_deg,
)

class DeltoGripperController:
    def __init__(self, node,
                 suction: bool = False,
                 min_fingers_for_stop: int = 3,
                 steps: int = 10,
                 step_delay: float = 0.2,
                 force_topic='/gripper/force',
                 target_topic='/gripper/target_joint'):

        self.node = node
        self.suction = suction  # <-- master mode switch
        self.min_fingers_for_stop = min_fingers_for_stop

        self.force_data = [0.0, 0.0, 0.0]
        self.baseline_force = [0.0, 0.0, 0.0]  # Baseline force when gripper is open

        # Contact threshold (force value indicating contact)
        # Detect contact based on CHANGE from baseline, not absolute value
        self.force_threshold = 4.0  # Higher threshold to allow full closure (change > 4.0N)

        # Motion parameters
        self.steps = steps
        self.step_delay = step_delay
        self.current_step = 0

        # FSM state
        self.state = 'IDLE'  # IDLE | OPENING | CLOSING

        self.gripper_profile = os.environ.get("UR10E_GRIPPER_PROFILE", "old").strip().lower()
        if self.gripper_profile not in ("old", "new"):
            self.node.get_logger().warn(
                f"Unknown UR10E_GRIPPER_PROFILE={self.gripper_profile!r}; using old"
            )
            self.gripper_profile = "old"

        # Primary curl joint fallback: F1=M3, F2=M7, F3=M11.
        self.finger_joint_idx = {0: 2, 1: 6, 2: 10}
        self.finger_joint_indices = finger_joint_indices_for_profile(self.gripper_profile)

        self.open_position = make_open_position(self.gripper_profile)
        self.closed_position = make_closed_position(self.gripper_profile)

        self.current_position = self.open_position.copy()
        self.current_open_alpha = 0.0
        self.node.get_logger().info(
            f"Delto gripper profile: {self.gripper_profile} "
            f"(new open extra={new_gripper_open_extra_deg():.1f}deg)"
        )

        # Freeze memory
        self.first_contact_index = None
        self.frozen_fingers = set()

        # Debounce after OPEN
        self.ignore_contacts_until = 0.0
        self.need_rearm = False
        self.last_force_stamp = 0.0

        # ROS pub/sub
        self.publisher  = node.create_publisher(Float32MultiArray, target_topic, 10)
        self.subscriber = node.create_subscription(
            Float32MultiArray, force_topic, self.force_callback, 10)

        # Schedule baseline capture after a short delay to allow force data to arrive
        import threading
        def capture_initial_baseline():
            time.sleep(1.0)  # Wait for force data to start flowing
            if self.force_data != [0.0, 0.0, 0.0]:
                self.baseline_force = self.force_data.copy()
                self.node.get_logger().debug(
                    f"Gripper initialized (baseline=[{self.baseline_force[0]:.2f}, "
                    f"{self.baseline_force[1]:.2f}, {self.baseline_force[2]:.2f}]N)"
                )
            else:
                self.node.get_logger().warn("⚠️ No force data received - baseline will be set on first close")
        threading.Thread(target=capture_initial_baseline, daemon=True).start()

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------
    def publish_position(self, position):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in position]
        self.publisher.publish(msg)

    def active_joint_degrees(self, position):
        indices = [idx for joints in self.finger_joint_indices.values() for idx in joints]
        return [round(math.degrees(float(position[idx])), 1) for idx in indices]

    def move_to_position(self, target_position, steps=None, delay=None):
        start_position = self.current_position.copy()
        steps = max(1, int(steps if steps is not None else getattr(
            self.node.cfg.gripper, "opening_steps", min(self.steps, 6))))
        delay = float(delay if delay is not None else getattr(
            self.node.cfg.gripper, "opening_step_delay_s", min(self.step_delay, 0.08)))

        for step in range(1, steps + 1):
            alpha = step / steps
            self.current_position = [
                (1.0 - alpha) * start_position[i] + alpha * target_position[i]
                for i in range(len(target_position))
            ]
            self.publish_position(self.current_position)
            if step < steps and delay > 0.0:
                time.sleep(delay)

    def hold_position(self, position, repeats=None, interval=None):
        repeats = max(1, int(repeats if repeats is not None else getattr(
            self.node.cfg.gripper, "open_hold_repeats", 8)))
        interval = float(interval if interval is not None else getattr(
            self.node.cfg.gripper, "open_hold_interval_s", 0.10))
        self.current_position = position.copy()
        for repeat in range(repeats):
            self.publish_position(self.current_position)
            if repeat < repeats - 1 and interval > 0.0:
                time.sleep(interval)

    def set_state(self, new_state):
        if new_state != self.state:
            self.state = new_state

    def channel_contact(self, i: int) -> bool:
        # Detect contact based on CHANGE from baseline, not absolute force
        f = float(self.force_data[i])
        baseline = float(self.baseline_force[i])
        delta = abs(f - baseline)
        return delta >= self.force_threshold

    def all_channels_contacted(self):
        return all(self.channel_contact(i) for i in (0, 1, 2))

    def unfreeze_all(self):
        if self.frozen_fingers:
            prev = sorted(list(self.frozen_fingers))
            self.frozen_fingers.clear()
            self.node.get_logger().debug(f"Unfroze fingers: {prev}")
        self.first_contact_index = None

    def force_callback(self, msg):
        self.force_data = list(msg.data)
        self.last_force_stamp = time.time()
        if hasattr(self.node, "classifier"):
            self.node.classifier.on_force(self.force_data[:3])

        # Only handle during closing
        if self.state != 'CLOSING':
            if not any(self.channel_contact(i) for i in (0, 1, 2)):
                self.need_rearm = False
            return

        now = time.time()
        if now < self.ignore_contacts_until or self.need_rearm:
            return  # ignore stale contact window

        # First contact (we do not freeze any finger in this unified version)
        if self.first_contact_index is None:
            for ch in (0, 1, 2):
                if self.channel_contact(ch):
                    self.first_contact_index = ch
                    break

    def is_force_threshold_reached(self):
        now = time.time()

        if now < self.ignore_contacts_until:
            return False
        if self.need_rearm:
            return False
        if self.last_force_stamp < self.ignore_contacts_until:
            return False
        if self.current_step < 2:
            return False

        return self.all_channels_contacted()

    # --------------------------------------------------------
    # CLOSING MOTION
    # --------------------------------------------------------
    def step_close(self):
        force_reached = self.is_force_threshold_reached()
        if self.current_step >= self.steps or force_reached:
            stopped_early = force_reached and self.current_step < self.steps
            reason = "contact detected" if stopped_early else "fully closed"
            current_force = list(self.force_data)  # atomic snapshot
            deltas = [abs(current_force[i] - self.baseline_force[i]) for i in range(3)]
            self.closure_stopped_early = stopped_early
            self.closure_step_stopped = self.current_step
            self.closure_deltas = list(deltas)
            self.node.get_logger().info(
                f"🔒 Gripper closed ({reason}): {self.current_step}/{self.steps} steps, "
                f"force delta=[{deltas[0]:.2f}, {deltas[1]:.2f}, {deltas[2]:.2f}]N"
            )
            self.current_step = 0
            self.set_state('IDLE')
            return False

        alpha = (self.current_step + 1) / self.steps

        # --------------------------------------------------------
        # SUCTION MODE: center-first closing (Version A behavior)
        # NON-SUCTION: all fingers close uniformly (Version B behavior)
        # --------------------------------------------------------
        if self.suction:
            phase_split = int(self.steps * 0.6)  # first 60% only center moves

            if self.current_step < phase_split:
                fingers_to_move = [1]  # ONLY center
            else:
                fingers_to_move = [0, 1, 2]  # then all close together
        else:
            fingers_to_move = [0, 1, 2]  # all fingers move every step

        for ch in fingers_to_move:
            if ch in self.frozen_fingers:
                continue
            start = getattr(self, 'close_start_position', self.open_position)
            for j_idx in self.finger_joint_indices.get(ch, (self.finger_joint_idx[ch],)):
                self.current_position[j_idx] = (
                    (1 - alpha) * start[j_idx] +
                    alpha * self.closed_position[j_idx]
                )

        msg = Float32MultiArray()
        msg.data = self.current_position
        self.publisher.publish(msg)

        self.current_step += 1
        return True

    def run_closure_loop(self):
        self.set_state('CLOSING')
        self.first_contact_index = None
        # Clear need_rearm flag and reset ignore window when starting to close
        self.need_rearm = False
        self.ignore_contacts_until = 0.0  # Clear debounce window to allow immediate force detection

        # ALWAYS capture baseline from current force before closing
        # This ensures we use the actual open state, not stale values
        self.baseline_force = self.force_data.copy()
        if getattr(self.node.cfg.planner, "log_gripper_force_profile", False):
            self.node.get_logger().info(
                f"🔒 Closing gripper (baseline=[{self.baseline_force[0]:.2f}, "
                f"{self.baseline_force[1]:.2f}, {self.baseline_force[2]:.2f}]N, "
                f"threshold={self.force_threshold}N)"
            )

        # Track closure result for learning
        self.closure_stopped_early = False
        self.closure_step_stopped = self.steps  # default: fully closed
        self.closure_first_contact_step = self.steps  # default: no early contact
        self.closure_force_profile = []  # force deltas at each step

        while self.step_close():
            # Record force profile at each step (atomic snapshot before the delay)
            current_force = list(self.force_data)
            deltas = [abs(current_force[i] - self.baseline_force[i]) for i in range(3)]
            self.closure_force_profile.append(deltas)
            time.sleep(self.step_delay)

        # Compute early contact step: first step where any finger delta > 0.5N
        self.closure_first_contact_step = len(self.closure_force_profile)  # default: no early contact
        for i, deltas in enumerate(self.closure_force_profile):
            if any(d > 0.5 for d in deltas):
                self.closure_first_contact_step = i
                break

        if getattr(self.node.cfg.planner, "log_gripper_force_profile", False):
            profile_str = " | ".join(
                f"s{i}:[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}]"
                for i, d in enumerate(self.closure_force_profile)
            )
            self.node.get_logger().info(f"🔒 Force profile: {profile_str}")
        self.node.get_logger().info(
            f"🔒 First contact: {self.closure_first_contact_step}/{len(self.closure_force_profile)}"
        )

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------
    def open_gripper(self):
        self.set_state('OPENING')
        self.current_step = 0
        self.current_open_alpha = 0.0
        self.move_to_position(self.open_position)
        self.close_start_position = self.open_position.copy()

        # The command is non-blocking, but 150ms is sufficient for release and
        # baseline capture while avoiding a fixed 300ms delay in every cycle.
        time.sleep(float(getattr(
            self.node.cfg.gripper, "open_settle_s", 0.15)))
        self.hold_position(self.open_position)

        # Capture baseline force when gripper is fully open
        self.baseline_force = self.force_data.copy()

        if getattr(self.node.cfg.planner, "log_gripper_force_profile", False):
            self.node.get_logger().info(
                f"Gripper opened fully (baseline=[{self.baseline_force[0]:.2f}, "
                f"{self.baseline_force[1]:.2f}, {self.baseline_force[2]:.2f}]N)"
            )
        self.node.get_logger().info(
            f"Gripper OPEN command active_joint_deg={self.active_joint_degrees(self.open_position)}"
        )

        # Debounce window after open
        self.ignore_contacts_until = time.time() + 0.25
        self.need_rearm = True

        self.unfreeze_all()
        self.set_state('IDLE')

    def open_gripper_to(self, alpha: float = 0.0):
        """Open gripper to a partial position.

        alpha=0.0 is the calibrated open posture, alpha=1.0 is fully closed.
        A small negative alpha allows a slight over-open beyond calibration.
        """
        alpha = max(-0.10, min(1.0, alpha))
        self.set_state('OPENING')
        self.current_step = 0
        self.current_open_alpha = alpha

        # Compute partial open position
        position = self.open_position.copy()
        for ch in [0, 1, 2]:
            for j_idx in self.finger_joint_indices.get(ch, (self.finger_joint_idx[ch],)):
                position[j_idx] = (
                    (1 - alpha) * self.open_position[j_idx] +
                    alpha * self.closed_position[j_idx]
                )

        self.move_to_position(position)
        self.close_start_position = position.copy()

        time.sleep(float(getattr(
            self.node.cfg.gripper, "open_settle_s", 0.15)))
        self.hold_position(position)
        self.baseline_force = self.force_data.copy()

        if getattr(self.node.cfg.planner, "log_gripper_force_profile", False):
            self.node.get_logger().info(
                f"Gripper opened to alpha={alpha:.2f} (baseline=[{self.baseline_force[0]:.2f}, "
                f"{self.baseline_force[1]:.2f}, {self.baseline_force[2]:.2f}]N)"
            )
        self.node.get_logger().info(
            f"Gripper OPEN alpha={alpha:.2f} command active_joint_deg={self.active_joint_degrees(position)}"
        )

        self.ignore_contacts_until = time.time() + 0.25
        self.need_rearm = True

        self.unfreeze_all()
        self.set_state('IDLE')
