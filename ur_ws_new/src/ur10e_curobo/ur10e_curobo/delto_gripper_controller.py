# delto_gripper_controller.py
import time
import math
import os
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from .gripper_profiles import (
    finger_joint_indices_for_profile,
    make_closed_position,
    make_envelop_closed_position,
    make_envelop_open_position,
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
        self.actual_joint_position = None
        self.actual_joint_effort = None
        self.actual_joint_stamp = 0.0
        self.close_baseline_effort = None
        self.contact_evidence = None

        # Contact threshold (force value indicating contact)
        # Detect contact based on CHANGE from baseline, not absolute value
        self.force_threshold = 4.0  # Higher threshold to allow full closure (change > 4.0N)

        # Motion parameters
        self.steps = steps
        self.step_delay = step_delay
        self.current_step = 0

        # FSM state
        self.state = 'IDLE'  # IDLE | OPENING | CLOSING

        self.gripper_profile = os.environ.get("UR10E_GRIPPER_PROFILE", "new").strip().lower()
        if self.gripper_profile not in ("old", "new"):
            self.node.get_logger().warn(
                f"Unknown UR10E_GRIPPER_PROFILE={self.gripper_profile!r}; using new"
            )
            self.gripper_profile = "new"

        # Primary curl joint fallback: F1=M3, F2=M7, F3=M11.
        self.finger_joint_idx = {0: 2, 1: 6, 2: 10}
        self.finger_joint_indices = finger_joint_indices_for_profile(self.gripper_profile)

        self.open_position = make_open_position(self.gripper_profile)
        self.closed_position = make_closed_position(self.gripper_profile)
        self.normal_open_position = self.open_position.copy()
        self.normal_closed_position = self.closed_position.copy()

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
        self.holding_control_publisher = node.create_publisher(
            Int16MultiArray, '/gripper/holding_control', 10)
        # Closure runs synchronously from a GUI/service callback. The force
        # subscriber must use a separate re-entrant group so MultiThreadedExecutor
        # can continue updating force_data during that loop.
        self.force_callback_group = ReentrantCallbackGroup()
        self.subscriber = node.create_subscription(
            Float32MultiArray, force_topic, self.force_callback, 10,
            callback_group=self.force_callback_group)
        self.joint_state_subscriber = node.create_subscription(
            JointState, '/gripper/joint_states', self.joint_state_callback, 10,
            callback_group=self.force_callback_group)

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

    def configure_grasp_mode(self, mode: str):
        """Select exact NORMAL or measured ENVELOP open/closed postures."""
        normalized = str(mode or "NORMAL").strip().upper()
        if normalized == "ENVELOP":
            self.open_position = make_envelop_open_position()
            self.closed_position = make_envelop_closed_position()
        else:
            normalized = "NORMAL"
            self.open_position = self.normal_open_position.copy()
            self.closed_position = self.normal_closed_position.copy()
        self.active_grasp_mode = normalized

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------
    def publish_position(self, position):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in position]
        self.publisher.publish(msg)

    def _main_closing_joint_indices(self):
        indices = []
        for finger in range(3):
            finger_indices = list(range(finger * 4, finger * 4 + 4))
            indices.append(max(
                finger_indices,
                key=lambda idx: abs(
                    self.closed_position[idx] - self.open_position[idx])))
        return indices

    def _set_holding_force(self, active: bool):
        # Always send release before opening, even if holding was disabled at
        # runtime after a previous close had activated it.
        if active and not bool(getattr(
                self.node.cfg.gripper, "holding_force_enabled", True)):
            return
        mode = 6 if getattr(self, "active_grasp_mode", "NORMAL") == "ENVELOP" else 1
        force = max(0, min(200, int(getattr(
            self.node.cfg.gripper, "holding_force_command", 50))))
        free_indices = set(self._main_closing_joint_indices())
        hold_mask = [0 if idx in free_indices else 1 for idx in range(12)]
        msg = Int16MultiArray()
        msg.data = [1 if active else 0, mode, force, *hold_mask]
        self.holding_control_publisher.publish(msg)
        if active:
            self.node.get_logger().info(
                "[GRIPPER_HOLD] requested=ON "
                f"mode={mode} force={force * 0.1:.1f}N "
                f"free_motors={[idx + 1 for idx in sorted(free_indices)]}")
            time.sleep(max(0.0, float(getattr(
                self.node.cfg.gripper, "holding_force_settle_s", 0.15))))
        else:
            time.sleep(max(0.0, float(getattr(
                self.node.cfg.gripper, "holding_release_settle_s", 0.05))))

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

    def joint_state_callback(self, msg):
        """Keep one consistently ordered snapshot of real Delto feedback."""
        expected = [
            'F1M1', 'F1M2', 'F1M3', 'F1M4',
            'F2M1', 'F2M2', 'F2M3', 'F2M4',
            'F3M1', 'F3M2', 'F3M3', 'F3M4',
        ]
        if len(msg.position) < 12 or len(msg.effort) < 12:
            return
        if msg.name:
            by_name = {name: i for i, name in enumerate(msg.name)}
            if not all(name in by_name for name in expected):
                return
            indices = [by_name[name] for name in expected]
        else:
            indices = list(range(12))
        self.actual_joint_position = [float(msg.position[i]) for i in indices]
        self.actual_joint_effort = [float(msg.effort[i]) for i in indices]
        self.actual_joint_stamp = time.time()

    def evaluate_nontactile_contact(self):
        """Classify per-finger obstruction using position plus motor current.

        The primary closing motor is derived from the calibrated open/closed
        travel for each finger. A real date both prevents that joint from
        reaching the empty-close posture and raises current in the finger.
        Requiring both signals rejects motor transients and hard-close current.
        """
        cfg = self.node.cfg.gripper
        max_age = float(getattr(cfg, 'nontactile_feedback_max_age_s', 0.30))
        if (self.actual_joint_position is None
                or self.actual_joint_effort is None
                or self.close_baseline_effort is None
                or time.time() - self.actual_joint_stamp > max_age):
            return {
                'valid': False, 'reason': 'STALE_OR_MISSING_JOINT_FEEDBACK',
                'contacts': [False, False, False], 'contact_count': 0,
            }

        current_thresholds = list(getattr(
            cfg, 'nontactile_current_delta_a', [0.220, 0.120, 0.220]))
        remaining_thresholds = list(getattr(
            cfg, 'nontactile_remaining_fraction', [0.75, 0.55, 0.75]))
        contacts, current_deltas, remaining, tracking_joints = [], [], [], []
        for finger in range(3):
            finger_indices = list(range(finger * 4, finger * 4 + 4))
            joint_idx = max(
                finger_indices,
                key=lambda idx: abs(
                    self.closed_position[idx] - self.open_position[idx]))
            tracking_joints.append(joint_idx + 1)
            current_delta = max(
                abs(self.actual_joint_effort[idx]
                    - self.close_baseline_effort[idx])
                for idx in finger_indices)
            span = abs(self.closed_position[joint_idx] - self.open_position[joint_idx])
            remaining_fraction = (
                abs(self.closed_position[joint_idx]
                    - self.actual_joint_position[joint_idx]) / max(span, 1e-6))
            current_deltas.append(current_delta)
            remaining.append(remaining_fraction)
            contacts.append(
                current_delta >= float(current_thresholds[finger])
                and remaining_fraction >= float(remaining_thresholds[finger]))

        min_contacts = int(getattr(
            cfg, 'nontactile_min_contact_fingers', self.min_fingers_for_stop))
        count = sum(contacts)
        return {
            'valid': True,
            'reason': 'CONTACT' if count >= min_contacts else 'EMPTY_CLOSE',
            'contacts': contacts,
            'contact_count': count,
            'min_contacts': min_contacts,
            'current_delta_a': current_deltas,
            'remaining_fraction': remaining,
            'tracking_joints': tracking_joints,
            'grasp_detected': count >= min_contacts,
        }

    def is_force_threshold_reached(self):
        if not bool(getattr(
                self.node.cfg.gripper, 'stop_closing_on_force', False)):
            return False
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

        start = getattr(self, 'close_start_position', self.open_position)
        if (getattr(self, "active_grasp_mode", "NORMAL") == "ENVELOP"
                or not bool(getattr(
                    self.node.cfg.gripper, "stop_closing_on_force", False))):
            # A measured closed posture contains meaningful M1..M12 values.
            # When early force stopping is disabled, command the complete
            # calibrated posture so "10/10" really targets the saved physical
            # close rather than leaving M1/M2 at their open values.
            for j_idx in range(len(self.closed_position)):
                self.current_position[j_idx] = (
                    (1 - alpha) * start[j_idx] +
                    alpha * self.closed_position[j_idx]
                )
        else:
            for ch in fingers_to_move:
                if ch in self.frozen_fingers:
                    continue
                for j_idx in self.finger_joint_indices.get(
                        ch, (self.finger_joint_idx[ch],)):
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
        self.close_baseline_effort = (
            list(self.actual_joint_effort)
            if self.actual_joint_effort is not None else None)
        self.contact_evidence = None
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
            # Allow the hardware and force subscriber to respond to this step
            # before sampling it. Sampling immediately after publish records the
            # previous command and can produce an all-zero calibration profile.
            time.sleep(self.step_delay)
            current_force = list(self.force_data)
            deltas = [abs(current_force[i] - self.baseline_force[i]) for i in range(3)]
            self.closure_force_profile.append(deltas)

        # The DG-3F-M can retain several degrees of steady-state error on a
        # loaded/coupled finger even though the final target register was
        # written successfully.  When force stopping is disabled, use bounded
        # feedback trim so the *measured* joints reproduce the calibrated
        # closed posture.  The correction is deliberately capped; it must not
        # turn a bad calibration or blocked finger into an unlimited command.
        self._trim_closed_pose_from_feedback()

        # set_position() is asynchronous.  At the final command the real
        # fingers are still travelling, so an immediate snapshot contains
        # near-open positions and near-zero current.  Use the configured
        # post-close hold here, before classifying, and do not repeat it in the
        # harvesting sequence.
        time.sleep(max(0.0, float(getattr(
            self.node.cfg.planner, 'grasp_post_close_settle_s', 0.50))))

        # Classify from the completed position-controlled close before enabling
        # firmware holding. Otherwise the holding-current rise can be mistaken
        # for fruit contact.
        if bool(getattr(
                self.node.cfg.gripper, 'nontactile_contact_validation', True)):
            self.contact_evidence = self.evaluate_nontactile_contact()
            evidence = self.contact_evidence
            if evidence.get('valid'):
                currents = evidence['current_delta_a']
                remaining = evidence['remaining_fraction']
                self.node.get_logger().info(
                    '[GRIPPER_CONTACT] '
                    f"result={evidence['reason']} contacts={evidence['contact_count']}/3 "
                    f"finger_contact={evidence['contacts']} "
                    f"tracking_joints={evidence['tracking_joints']} "
                    f"current_delta_mA={[round(v * 1000.0, 1) for v in currents]} "
                    f"remaining={[round(v, 2) for v in remaining]}")
            else:
                self.node.get_logger().warn(
                    f"[GRIPPER_CONTACT] result=INCONCLUSIVE reason={evidence['reason']}")

        # Keep the calibrated shape joints fixed while the primary closing
        # joint of each finger applies the configured firmware grasp force.
        # This stays active through reverse/pull and is released by OPEN.
        self._set_holding_force(True)

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
        if not getattr(self.node.cfg.planner, "concise_console_logs", False):
            self.node.get_logger().info(
                f"🔒 First contact: "
                f"{self.closure_first_contact_step}/{len(self.closure_force_profile)}"
            )

    def _trim_closed_pose_from_feedback(self):
        cfg = self.node.cfg.gripper
        if (not bool(getattr(cfg, "close_feedback_trim_enabled", True))
                or bool(getattr(cfg, "stop_closing_on_force", False))):
            return

        iterations = max(0, int(getattr(
            cfg, "close_feedback_trim_iterations", 4)))
        tolerance = max(0.0, float(getattr(
            cfg, "close_feedback_trim_tolerance_rad", 0.0175)))
        max_correction = max(0.0, float(getattr(
            cfg, "close_feedback_trim_max_correction_rad", 0.1745)))
        settle_s = max(0.05, float(getattr(
            cfg, "close_feedback_trim_settle_s", 0.12)))
        desired = list(self.closed_position)
        final_error = None

        for _ in range(iterations):
            time.sleep(settle_s)
            if (self.actual_joint_position is None
                    or time.time() - self.actual_joint_stamp > 0.30):
                return
            errors = [
                desired[i] - self.actual_joint_position[i]
                for i in range(len(desired))
            ]
            final_error = errors
            if max(abs(error) for error in errors) <= tolerance:
                return
            compensated = [
                desired[i] + max(-max_correction, min(max_correction, errors[i]))
                for i in range(len(desired))
            ]
            self.publish_position(compensated)

        time.sleep(settle_s)
        if (self.actual_joint_position is not None
                and time.time() - self.actual_joint_stamp <= 0.30):
            final_error = [
                desired[i] - self.actual_joint_position[i]
                for i in range(len(desired))
            ]
        if final_error is not None:
            worst = max(range(len(final_error)), key=lambda i: abs(final_error[i]))
            if abs(final_error[worst]) > tolerance:
                self.node.get_logger().warn(
                    "[GRIPPER_CLOSE_TRACKING] calibrated close not fully reached "
                    f"worst=M{worst + 1} error={math.degrees(final_error[worst]):+.1f}deg "
                    f"tolerance={math.degrees(tolerance):.1f}deg "
                    f"max_trim={math.degrees(max_correction):.1f}deg")

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------
    def open_gripper(self):
        self._set_holding_force(False)
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
        if not getattr(self.node.cfg.planner, "concise_console_logs", False):
            self.node.get_logger().info(
                f"Gripper OPEN command "
                f"active_joint_deg={self.active_joint_degrees(self.open_position)}"
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
        self._set_holding_force(False)
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
