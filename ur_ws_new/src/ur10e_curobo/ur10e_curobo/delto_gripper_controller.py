# delto_gripper_controller.py
import time
from std_msgs.msg import Float32MultiArray

class DeltoGripperController:
    def __init__(self, node, force_topic='/gripper/force', target_topic='/gripper/target_joint'):
        self.node = node
        self.force_data = [0.0, 0.0, 0.0]

        # If forces go negative at contact, keep this negative (we check f <= threshold).
        # If positive-on-contact, set positive and we'll check abs(f) >= threshold.
        self.force_threshold = -0.15

        self.steps = 10
        self.step_delay = 0.2  # seconds
        self.current_step = 0

        # Simple state machine
        self.state = 'IDLE'  # 'IDLE' | 'OPENING' | 'CLOSING'

        # Joint mapping: left=2, center=6, right=10
        self.finger_joint_idx = {0: 2, 1: 6, 2: 10}

        self.open_position = [
            -0.0942, -0.1500, 2.0660, -0.5062,
            -1.6318, 0.1309, 1.6753, -0.4887,
            0.3333, 0.2234, 2.0673, -0.4311
        ]
        self.closed_position = self.open_position.copy()
        self.closed_position[self.finger_joint_idx[0]] = 2.5660  # left
        self.closed_position[self.finger_joint_idx[1]] = 2.3753  # center
        self.closed_position[self.finger_joint_idx[2]] = 2.5673  # right
        self.current_position = self.open_position.copy()

        # Freeze state
        self.first_contact_index = None  # which channel hit first (0/1/2)
        self.frozen_fingers = set()      # channels frozen from further closing

        # Debounce / re-arm after OPEN to avoid "stale contact" on re-close
        self.ignore_contacts_until = 0.0  # time until which to ignore contact after opening
        self.need_rearm = False           # require a clean non-contact frame before re-arming
        self.last_force_stamp = 0.0       # timestamp of last force packet

        # ROS I/O
        self.publisher  = node.create_publisher(Float32MultiArray, target_topic, 10)
        self.subscriber = node.create_subscription(Float32MultiArray, force_topic, self.force_callback, 10)

    # ---------- Helpers ----------
    def set_state(self, new_state: str):
        if new_state != self.state:
            self.node.get_logger().info(f"[state] {self.state} → {new_state}")
            self.state = new_state

    def channel_contact(self, i: int) -> bool:
        f = float(self.force_data[i])
        if self.force_threshold < 0:
            return f <= self.force_threshold
        else:
            return abs(f) >= self.force_threshold

    def all_channels_contacted(self) -> bool:
        return all(self.channel_contact(i) for i in (0, 1, 2))

    def unfreeze_all(self):
        """Clear any frozen fingers and first-contact memory."""
        if self.frozen_fingers:
            prev = sorted(list(self.frozen_fingers))
            self.frozen_fingers.clear()
            self.node.get_logger().info(f"🧊→🔥 Unfroze fingers: {prev} (ready for next cycle).")
        self.first_contact_index = None

    def set_close_debounce(self, seconds: float = 0.25):
        """Optional: tune the post-open ignore window."""
        self.ignore_contacts_until = time.time() + max(0.0, float(seconds))
        self.need_rearm = True

    # ---------- ROS Callbacks ----------
    def force_callback(self, msg):
        self.force_data = list(msg.data)
        self.last_force_stamp = time.time()

        # If we're not closing, watch for a clean non-contact to re-arm
        if self.state != 'CLOSING':
            if not any(self.channel_contact(i) for i in (0, 1, 2)):
                self.need_rearm = False
            return

        # ---- Debounce during early re-close ----
        now = time.time()
        if (now < self.ignore_contacts_until) or self.need_rearm:
            return  # ignore stale contact until window passes AND we've seen a clean non-contact

        # Establish "first contact" only after gate
        if self.first_contact_index is None:
            c0 = self.channel_contact(0)
            c1 = self.channel_contact(1)
            c2 = self.channel_contact(2)

            # # If center (1) contacts first while sides haven't, freeze center (joint 6)
            # if c1 and (not c0) and (not c2):
            #     self.first_contact_index = 1
            #     self.frozen_fingers.add(1)
            #     j_idx = self.finger_joint_idx[1]
            #     frozen_val = self.current_position[j_idx]
            #     self.current_position[j_idx] = frozen_val
            #     self.node.get_logger().info(
            #         f"🧊 Freezing center finger (ch1, joint {j_idx}) at {frozen_val:.4f} — ch1 hit first (CLOSING)."
            #     )
            # elif c0 or c2:
            #     # Not freezing side fingers in this policy; just remember who hit first
            #     self.first_contact_index = 0 if c0 else 2

    # ---------- Motion ----------
    def is_force_threshold_reached(self):
        """Stop when all three have contacted, but only after debounce/re-arm conditions."""
        now = time.time()

        # 1) Ignore contact during the post-open window
        if now < self.ignore_contacts_until:
            return False

        # 2) Require at least one clean non-contact frame after OPEN before re-arming
        if self.need_rearm:
            return False

        # 3) Ensure at least one fresh packet after the window
        if self.last_force_stamp < self.ignore_contacts_until:
            return False

        # 4) Avoid "step-0/1" accidental triggers at start of motion
        if self.current_step < 2:
            return False

        # 5) Real check
        return self.all_channels_contacted()

    def step_close(self):
        if self.current_step >= self.steps or self.is_force_threshold_reached():
            self.node.get_logger().info("✅ Gripper fully closed or force limit reached on all fingers.")
            self.current_step = 0
            # We're done closing
            self.set_state('IDLE')
            return False

        alpha = self.current_step / self.steps

        # Move each finger unless it is frozen
        for ch in (0, 1, 2):
            j_idx = self.finger_joint_idx[ch]
            if ch in self.frozen_fingers:
                continue  # keep current (frozen) value
            self.current_position[j_idx] = (
                (1 - alpha) * self.open_position[j_idx] + alpha * self.closed_position[j_idx]
            )

        msg = Float32MultiArray()
        msg.data = self.current_position
        self.publisher.publish(msg)

        # NEW: tell the classifier we’re still actively closing
        if hasattr(self.node, "classifier"):
            self.node.classifier.note_target_update()

        self.node.get_logger().info(
            f"Step {self.current_step}/{self.steps} — closing... (frozen: {sorted(list(self.frozen_fingers))})"
        )
        self.current_step += 1
        return True

    def run_closure_loop(self):
        """Run the full closure loop until all fingers contact or max steps."""
        # Start closing
        self.set_state('CLOSING')
        self.first_contact_index = None  # ensure fresh cycle
        while self.step_close():
            time.sleep(self.step_delay)

    def open_gripper(self):
        """Open and clear any frozen fingers so next cycle works normally."""
        self.set_state('OPENING')
        self.current_step = 0
        self.current_position = self.open_position.copy()

        # Publish open first (so it moves)
        msg = Float32MultiArray()
        msg.data = self.open_position
        self.publisher.publish(msg)
        self.node.get_logger().info("🔓 Gripper OPEN")

        # Start ignore window and require re-arm; tune 0.15–0.35s if needed
        self.ignore_contacts_until = time.time() + 0.25
        self.need_rearm = True

        # Now remove any freezes and reset first-contact memory
        self.unfreeze_all()
        self.set_state('IDLE')
