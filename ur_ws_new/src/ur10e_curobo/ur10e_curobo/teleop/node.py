"""Joystick teleop node for UR10e control."""

import sys
from threading import Lock
from time import sleep, time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool

from .config import (
    AXIS_X, AXIS_Y, AXIS_Z, AXIS_YAW,
    INVERT_X, INVERT_Y, INVERT_Z, INVERT_YAW,
    BUTTON_ENABLE,  # <-- deadman (hold to move)
    BUTTON_GRIPPER_OPEN, BUTTON_GRIPPER_CLOSE,
    BUTTON_Z_UP, BUTTON_Z_DOWN,
    BUTTON_HOME, BUTTON_STOP, BUTTON_SPEED_UP, BUTTON_SPEED_DOWN,
    DEADZONE,
    MAX_LINEAR_VELOCITY, MAX_ANGULAR_VELOCITY,
    MIN_VELOCITY_SCALE, MAX_VELOCITY_SCALE, VELOCITY_SCALE_STEP,
    SMOOTHING_ALPHA, UPDATE_RATE_HZ, DEFAULT_VELOCITY_SCALE,
)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class JoystickTeleop(Node):
    """ROS2 node for joystick-based teleop control of UR10e."""

    def __init__(self):
        super().__init__("joystick_teleop")

        # State
        self.velocity_scale = DEFAULT_VELOCITY_SCALE
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0
        self.smoothed_z = 0.0
        self.smoothed_yaw = 0.0
        self.running = True
        self.joystick = None
        self.lock = Lock()

        # Button state tracking (for edge detection)
        self.prev_buttons = {}

        # Publishers - use default RELIABLE QoS to match MotionExecutor subscription
        self.teleop_pub = self.create_publisher(Twist, "/teleop_delta", 10)
        self.cmd_pub = self.create_publisher(String, "/ui_command", 10)
        self.estop_pub = self.create_publisher(Bool, "/emergency_stop", 10)

        # Initialize pygame and joystick
        if not PYGAME_AVAILABLE:
            self.get_logger().error("pygame not installed! Install with: pip install pygame")
            raise RuntimeError("pygame not installed")

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            self.get_logger().error("No joystick found! Please connect a joystick.")
            raise RuntimeError("No joystick found")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.get_logger().info(f"Connected to: {self.joystick.get_name()}")
        self.get_logger().info(f"  Axes: {self.joystick.get_numaxes()}")
        self.get_logger().info(f"  Buttons: {self.joystick.get_numbuttons()}")

        self._print_controls()

    def _print_controls(self):
        """Print control mapping to console."""
        name = self.joystick.get_name().lower()
        print("\n" + "=" * 55)
        print("JOYSTICK TELEOP CONTROLS - " + self.joystick.get_name())
        print("=" * 55)

        if "extreme" in name or "3d pro" in name:
            print("Trigger (hold): Enable motion (deadman)")
            print("Stick X/Y:       Move robot X/Y (left/right, forward/back)")
            print("Twist:           Rotate (yaw)")
            print("Top buttons:     Move Z UP/DOWN (hold)")
            print("Thumb buttons:   Open/close gripper")
            print("Speed buttons:   Speed UP/DOWN")
            print("Home button:     Go to home position")
            print("STOP button:     EMERGENCY STOP")
        else:
            print(f"Axes: X={AXIS_X}, Y={AXIS_Y}, Z={AXIS_Z}, Yaw={AXIS_YAW}")
            print(f"Deadman enable: Button {BUTTON_ENABLE}")
            print(f"Open gripper:   Button {BUTTON_GRIPPER_OPEN}")
            print(f"Close gripper:  Button {BUTTON_GRIPPER_CLOSE}")
            print(f"Z up/down:      Buttons {BUTTON_Z_UP}/{BUTTON_Z_DOWN}")
            print(f"Home:           Button {BUTTON_HOME}")
            print(f"E-stop:         Button {BUTTON_STOP}")
            print(f"Speed up/down:  Buttons {BUTTON_SPEED_UP}/{BUTTON_SPEED_DOWN}")

        print("=" * 55)
        print(f"Current speed: {self.velocity_scale:.0%}")
        print("=" * 55 + "\n")

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick axis value."""
        if abs(value) < DEADZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)

    def _smooth(self, current: float, target: float) -> float:
        """Apply exponential smoothing."""
        return current + SMOOTHING_ALPHA * (target - current)

    def _button_pressed(self, button: int) -> bool:
        """Check if button was just pressed (edge detection)."""
        current = self.joystick.get_button(button) if button < self.joystick.get_numbuttons() else False
        prev = self.prev_buttons.get(button, False)
        self.prev_buttons[button] = current
        return current and not prev

    def _send_command(self, cmd: str):
        """Send a command to the robot."""
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Command: {cmd}")

    def _send_estop(self):
        """Send emergency stop."""
        msg = Bool()
        msg.data = True
        self.estop_pub.publish(msg)
        self.get_logger().warn("EMERGENCY STOP!")

    def run(self):
        """Main loop - read joystick and publish teleop commands."""
        period = 1.0 / UPDATE_RATE_HZ
        last_status_time = time()

        self.get_logger().info("Joystick teleop running. Press Ctrl+C to exit.")

        while self.running and rclpy.ok():
            loop_start = time()

            # Required for joystick updates
            pygame.event.pump()

            # Deadman enable (hold to move)
            enable = (
                self.joystick.get_button(BUTTON_ENABLE)
                if BUTTON_ENABLE < self.joystick.get_numbuttons()
                else False
            )

            # Read joystick axes with deadzone
            raw_x = self._apply_deadzone(self.joystick.get_axis(AXIS_X)) * INVERT_X
            raw_y = self._apply_deadzone(self.joystick.get_axis(AXIS_Y)) * INVERT_Y

            # Z: axis if available, otherwise use buttons (recommended for Extreme 3D Pro)
            if AXIS_Z >= 0 and AXIS_Z < self.joystick.get_numaxes():
                raw_z = self._apply_deadzone(self.joystick.get_axis(AXIS_Z)) * INVERT_Z
            else:
                z_up = (
                    self.joystick.get_button(BUTTON_Z_UP)
                    if BUTTON_Z_UP < self.joystick.get_numbuttons()
                    else False
                )
                z_down = (
                    self.joystick.get_button(BUTTON_Z_DOWN)
                    if BUTTON_Z_DOWN < self.joystick.get_numbuttons()
                    else False
                )
                raw_z = float(z_up) - float(z_down)  # +1 for up, -1 for down

            # Yaw axis (optional)
            if AXIS_YAW < self.joystick.get_numaxes():
                raw_yaw = self._apply_deadzone(self.joystick.get_axis(AXIS_YAW)) * INVERT_YAW
            else:
                raw_yaw = 0.0

            # Apply smoothing only when enabled; otherwise decay to zero quickly
            if enable:
                self.smoothed_x = self._smooth(self.smoothed_x, raw_x)
                self.smoothed_y = self._smooth(self.smoothed_y, raw_y)
                self.smoothed_z = self._smooth(self.smoothed_z, raw_z)
                self.smoothed_yaw = self._smooth(self.smoothed_yaw, raw_yaw)
            else:
                # Force stop + prevent "jump" when re-enabling
                self.smoothed_x = 0.0
                self.smoothed_y = 0.0
                self.smoothed_z = 0.0
                self.smoothed_yaw = 0.0

            # Scale to velocity (Twist convention: m/s and rad/s)
            vel_x = self.smoothed_x * MAX_LINEAR_VELOCITY * self.velocity_scale
            vel_y = self.smoothed_y * MAX_LINEAR_VELOCITY * self.velocity_scale
            vel_z = self.smoothed_z * MAX_LINEAR_VELOCITY * self.velocity_scale
            vel_yaw = self.smoothed_yaw * MAX_ANGULAR_VELOCITY * self.velocity_scale

            # Always publish: zero when not enabled (fail-safe)
            twist = Twist()
            if enable:
                twist.linear.x = vel_x
                twist.linear.y = vel_y
                twist.linear.z = vel_z
                twist.angular.z = vel_yaw
            self.teleop_pub.publish(twist)

            has_movement = enable and (
                abs(twist.linear.x) > 1e-6 or abs(twist.linear.y) > 1e-6 or
                abs(twist.linear.z) > 1e-6 or abs(twist.angular.z) > 1e-6
            )

            # Handle button presses (edge-detected)
            if self._button_pressed(BUTTON_GRIPPER_OPEN):
                self._send_command("open")

            if self._button_pressed(BUTTON_GRIPPER_CLOSE):
                self._send_command("close")

            if self._button_pressed(BUTTON_HOME):
                self._send_command("home")

            if self._button_pressed(BUTTON_STOP):
                self._send_estop()

            if self._button_pressed(BUTTON_SPEED_UP):
                self.velocity_scale = min(MAX_VELOCITY_SCALE, self.velocity_scale + VELOCITY_SCALE_STEP)
                print(f"Speed: {self.velocity_scale:.0%}")

            if self._button_pressed(BUTTON_SPEED_DOWN):
                self.velocity_scale = max(MIN_VELOCITY_SCALE, self.velocity_scale - VELOCITY_SCALE_STEP)
                print(f"Speed: {self.velocity_scale:.0%}")

            # Periodic status print (1 Hz)
            now = time()
            if now - last_status_time > 1.0:
                pressed_btns = [i for i in range(self.joystick.get_numbuttons()) if self.joystick.get_button(i)]
                btn_str = f" BTN:{pressed_btns}" if pressed_btns else ""
                stick_str = f"Stick[X={raw_x:+.2f} Y={raw_y:+.2f} Z={raw_z:+.2f} Yaw={raw_yaw:+.2f}]"
                gate_str = "EN" if enable else "DIS"
                if has_movement:
                    print(f"{gate_str} | {stick_str} | Vel: X={vel_x*1000:.2f} Y={vel_y*1000:.2f} Z={vel_z*1000:.2f}mm Yaw={vel_yaw*1000:.2f}mrad{btn_str}")
                else:
                    print(f"{gate_str} | {stick_str} | IDLE{btn_str}")
                last_status_time = now

            # Spin ROS once
            rclpy.spin_once(self, timeout_sec=0)

            # Maintain update rate
            elapsed = time() - loop_start
            if elapsed < period:
                sleep(period - elapsed)

    def stop(self):
        """Stop the teleop node."""
        self.running = False
        # Send zero velocity (extra safety)
        self.teleop_pub.publish(Twist())
        if self.joystick:
            self.joystick.quit()
        pygame.quit()


def main():
    """Entry point."""
    rclpy.init()
    node = JoystickTeleop()

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
