# delto_gripper_controller.py
import time
from std_msgs.msg import Float32MultiArray

class DeltoGripperController:
    def __init__(self, node,
                 suction: bool = False,
                 force_topic='/gripper/force',
                 target_topic='/gripper/target_joint',
                 force_thresholds: dict = None,
                 min_fingers_for_stop: int = 2,
                 steps: int = 10,
                 step_delay: float = 0.2):

        self.node = node
        self.suction = suction  # <-- master mode switch

        self.force_data = [0.0, 0.0, 0.0]

        # Contact threshold (negative force on contact)
        # PER-FINGER thresholds for better dense bunch handling
        if force_thresholds is None:
            # Default thresholds
            self.force_thresholds = {
                0: -0.14,  # left finger (slightly softer)
                1: -0.16,  # center finger (stronger - contacts first in suction mode)
                2: -0.14,  # right finger (slightly softer)
            }
        else:
            self.force_thresholds = force_thresholds

        # Fallback single threshold for backward compatibility
        self.force_threshold = -0.15

        # Dense bunch handling: require minimum fingers touching before stopping
        # This prevents early stop when one finger hits a neighbor fruit
        self.min_fingers_for_stop = min_fingers_for_stop
        self.fingers_contacted = {0: False, 1: False, 2: False}

        # Motion parameters
        self.steps = steps
        self.step_delay = step_delay
        self.current_step = 0

        # FSM state
        self.state = 'IDLE'  # IDLE | OPENING | CLOSING

        # ----------------------------------------------------
        # SUCTION MODE → USE VERSION A joint mapping
        # NON-SUCTION MODE → USE VERSION B joint mapping
        # ----------------------------------------------------
        if self.suction:
            self.finger_joint_idx = {0: 2, 1: 7, 2: 10}  # Version A
            self.node.get_logger().info("Delto Gripper: SUCTION MODE (center joint=7)")
        else:
            self.finger_joint_idx = {0: 2, 1: 6, 2: 10}  # Version B
            self.node.get_logger().info("Delto Gripper: NON-SUCTION MODE (center joint=6)")

        # Positions (shared)
        self.open_position = [
            -0.0942, -0.1500, 2.1960, -0.5062,
            -1.6318, 0.1309, 1.7753, -0.4887,
            0.3333, 0.2234, 2.1973, -0.4311
        ]

        self.closed_position = self.open_position.copy()
        # Fingers
        self.closed_position[self.finger_joint_idx[0]] = 2.5660  # left finger
        self.closed_position[self.finger_joint_idx[1]] = 2.3753  # center finger
        self.closed_position[self.finger_joint_idx[2]] = 2.5673  # right finger

        self.current_position = self.open_position.copy()

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

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------
    def set_state(self, new_state):
        if new_state != self.state:
            self.node.get_logger().info(f"[state] {self.state} → {new_state}")
            self.state = new_state

    def channel_contact(self, i: int) -> bool:
        """Check if finger i is in contact, using per-finger threshold."""
        f = float(self.force_data[i])
        threshold = self.force_thresholds.get(i, self.force_threshold)
        if threshold < 0:
            return f <= threshold
        return abs(f) >= threshold

    def all_channels_contacted(self):
        return all(self.channel_contact(i) for i in (0, 1, 2))

    def unfreeze_all(self):
        if self.frozen_fingers:
            prev = sorted(list(self.frozen_fingers))
            self.frozen_fingers.clear()
            self.node.get_logger().info(f"🧊→🔥 Unfroze fingers: {prev}")
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
        """Check if enough fingers are in contact to stop closing.

        Dense bunch improvement: Instead of requiring ALL fingers to contact
        (which fails when one finger hits a neighbor fruit), we only require
        a minimum number of fingers (default: 2).
        """
        now = time.time()

        if now < self.ignore_contacts_until:
            return False
        if self.need_rearm:
            return False
        if self.last_force_stamp < self.ignore_contacts_until:
            return False
        if self.current_step < 2:
            return False

        # Count how many fingers are currently in contact
        contacted_count = sum(1 for i in (0, 1, 2) if self.channel_contact(i))

        # Update contact tracking
        for i in (0, 1, 2):
            self.fingers_contacted[i] = self.channel_contact(i)

        # Stop closing when minimum fingers reached
        threshold_reached = contacted_count >= self.min_fingers_for_stop

        if threshold_reached and contacted_count < 3:
            # Log when we stop with partial contact (useful for debugging dense bunches)
            contacted_fingers = [i for i in (0,1,2) if self.fingers_contacted[i]]
            self.node.get_logger().info(
                f"🔶 Partial contact: {contacted_count}/3 fingers contacted {contacted_fingers} "
                f"(forces: {[f'{self.force_data[i]:.3f}' for i in (0,1,2)]})"
            )

        return threshold_reached

    # --------------------------------------------------------
    # CLOSING MOTION
    # --------------------------------------------------------
    def step_close(self):
        if self.current_step >= self.steps or self.is_force_threshold_reached():
            self.node.get_logger().info("✅ Gripper fully closed or force limit reached.")
            self.current_step = 0
            self.set_state('IDLE')
            return False

        # --- smoothing curve (smoothstep) ---
        t = self.current_step / self.steps
        alpha = t * t * (3 - 2 * t)

        # --- slow down near final 20% ---
        if t > 0.8:
            time.sleep(self.step_delay * 1.5)
        else:
            time.sleep(self.step_delay)

        # finger selection stays the same
        if self.suction:
            phase_split = int(self.steps * 0.6)
            fingers_to_move = [1] if self.current_step < phase_split else [0, 1, 2]
        else:
            fingers_to_move = [0, 1, 2]

        for ch in fingers_to_move:
            if ch in self.frozen_fingers:
                continue
            j_idx = self.finger_joint_idx[ch]
            self.current_position[j_idx] = (
                (1 - alpha) * self.open_position[j_idx] +
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

        while self.step_close():
            time.sleep(self.step_delay)

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------
    def open_gripper(self):
        self.set_state('OPENING')
        self.current_step = 0
        self.current_position = self.open_position.copy()

        msg = Float32MultiArray()
        msg.data = self.open_position
        self.publisher.publish(msg)

        self.node.get_logger().info("🔓 Gripper OPEN")

        # Debounce window after open
        self.ignore_contacts_until = time.time() + 0.25
        self.need_rearm = True

        self.unfreeze_all()
        self.set_state('IDLE')
