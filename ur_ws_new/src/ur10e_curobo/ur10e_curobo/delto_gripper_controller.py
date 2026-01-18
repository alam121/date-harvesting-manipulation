# delto_gripper_controller.py
import time
from std_msgs.msg import Float32MultiArray

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

        # Contact threshold (force value indicating contact)
        # Finger 1 shows +0.15 when grabbed, so use 0.1 to detect abs(0.15) >= 0.1
        self.force_threshold = 0.1

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
            -0.0942, -0.1500, 2.1260, -0.5062,
            -1.6318, 0.1309, 1.6753, -0.4887,
            0.3333, 0.2234, 2.1260, -0.4311
        ]

        self.closed_position = self.open_position.copy()
        # Fingers
        self.closed_position[self.finger_joint_idx[0]] = 2.5673  # left finger
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
        f = float(self.force_data[i])
        if self.force_threshold < 0:
            return f <= self.force_threshold
        return abs(f) >= self.force_threshold

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
        if self.current_step >= self.steps or self.is_force_threshold_reached():
            self.node.get_logger().info("✅ Gripper fully closed or force limit reached.")
            self.current_step = 0
            self.set_state('IDLE')
            return False

        alpha = self.current_step / self.steps

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
