# ur10e_curobo/managers/ui_manager.py
"""User interface management for UR10e cuRobo node."""

import json
import threading
import time
from typing import TYPE_CHECKING
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

from ..utils import read_key
from .. import markers as markers_mod
from .. import motions as motions_mod
from .. import goals as goals_mod
from .. import gripper as gripper_mod

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .state_manager import StateManager
    from .motion_executor import MotionExecutor


class UIManager:
    """Handles keyboard input, GUI commands, and user interaction."""

    def __init__(
        self,
        node: Node,
        config: "ConfigManager",
        state: "StateManager",
        motion: "MotionExecutor",
    ):
        self._node = node
        self._config = config
        self._state = state
        self._motion = motion

        # Publishers
        self.velocity_scale_pub = None
        self.goal_info_pub = None

        # Keyboard thread
        self.keyboard_thread = None

    def initialize(self) -> None:
        """Initialize UI subscriptions, publishers, and keyboard thread."""
        # Publishers
        self.velocity_scale_pub = self._node.create_publisher(
            Float32, "/velocity_scale", 10
        )
        self.goal_info_pub = self._node.create_publisher(
            String, "/goal_info", 10
        )

        # GUI command subscription
        self._node.create_subscription(
            String, "/ui_command", self._ui_command_cb, 10
        )

        # Timer to publish goal info periodically
        self._node.create_timer(0.2, self._publish_goal_info)  # 5Hz

        # Start keyboard thread
        self.keyboard_thread = threading.Thread(
            target=self._wait_for_key_press, daemon=True
        )
        self.keyboard_thread.start()

        self._node.get_logger().info("UIManager initialized")

    # ============ GUI Command Handler ============

    def _ui_command_cb(self, msg: String) -> None:
        """Handle commands from the GUI."""
        cmd = msg.data.strip()
        self._node.get_logger().info(f"UI command received: {cmd}")

        # Run blocking motion commands in separate thread to avoid blocking ROS callbacks
        def run_home():
            if not self._motion._motion_lock.acquire(blocking=False):
                self._node.get_logger().warn("Motion already in progress, ignoring HOME command")
                return
            try:
                motions_mod.move_to_home_position(self._node)
            finally:
                self._motion._motion_lock.release()

        def run_dropoff():
            if not self._motion._motion_lock.acquire(blocking=False):
                self._node.get_logger().warn("Motion already in progress, ignoring DROPOFF command")
                return
            try:
                motions_mod.move_to_dropoff_position(self._node)
                gripper_mod.control_gripper(self._node, 'OPEN')
            finally:
                self._motion._motion_lock.release()

        def run_execute():
            if not self._motion._motion_lock.acquire(blocking=False):
                self._node.get_logger().warn("Motion already in progress, ignoring EXECUTE command")
                return
            try:
                if self._node.goal_poses:
                    self._prep_and_execute()
            finally:
                self._motion._motion_lock.release()

        if cmd == "home":
            threading.Thread(target=run_home, daemon=True).start()
        elif cmd == "dropoff":
            threading.Thread(target=run_dropoff, daemon=True).start()
        elif cmd == "execute":
            threading.Thread(target=run_execute, daemon=True).start()
        elif cmd == "clear":
            self._node.goal_poses.clear()
            self._node.get_logger().info("Goals cleared")
        elif cmd == "open":
            gripper_mod.control_gripper(self._node, 'OPEN')
        elif cmd == "close":
            gripper_mod.control_gripper(self._node, 'CLOSE')
        elif cmd == "stop":
            self._state.stop_requested = True
            self._motion.publish_stop_trajectory()
            self._node.goal_poses.clear()
        elif cmd.startswith("capture "):
            try:
                duration = float(cmd.split()[1])
                self._node.start_goal_capture(duration)
            except (ValueError, IndexError):
                self._node.start_goal_capture(10.0)
        elif cmd == "capture_stop":
            self._node.stop_goal_capture()
        elif cmd.startswith("set_speeds "):
            self._handle_set_speeds(cmd)
        elif cmd.startswith("set_velocity_scale "):
            self._handle_set_velocity_scale(cmd)
        elif cmd == "debug_world":
            self._motion.debug_print_world()
        else:
            self._node.get_logger().warn(f"Unknown UI command: {cmd}")

    def _handle_set_speeds(self, cmd: str) -> None:
        """Handle set_speeds command."""
        try:
            parts = cmd.split()
            self._config.cfg.planner.speed_home = float(parts[1])
            self._config.cfg.planner.speed_dropoff = float(parts[2])
            self._config.cfg.planner.speed_approach = float(parts[3])
            if len(parts) > 4:
                self._config.cfg.planner.speed_predropoff = float(parts[4])
            self._node.get_logger().info(
                f"Speeds updated: home={parts[1]}, dropoff={parts[2]}, "
                f"approach={parts[3]}, predropoff={self._config.cfg.planner.speed_predropoff}"
            )
        except (ValueError, IndexError) as e:
            self._node.get_logger().warn(f"Invalid set_speeds format: {e}")

    def _handle_set_velocity_scale(self, cmd: str) -> None:
        """Handle set_velocity_scale command."""
        try:
            scale = float(cmd.split()[1])
            scale = max(0.1, min(scale, 10.0))  # Clamp between 0.1 and 10.0
            self._config.cfg.planner.global_speed_multiplier = scale
            self._config.speed_scale = scale
            self._node.get_logger().info(f"Velocity scale set to {scale}")
            # Publish updated scale
            scale_msg = Float32()
            scale_msg.data = scale
            self.velocity_scale_pub.publish(scale_msg)
        except (ValueError, IndexError) as e:
            self._node.get_logger().warn(f"Invalid set_velocity_scale format: {e}")

    # ============ Goal Info Publisher ============

    def _publish_goal_info(self) -> None:
        """Publish goal information for GUI consumption."""
        goal_poses = self._node.goal_poses

        # Convert ThreadSafeGoalList to a list safely
        if hasattr(goal_poses, "data"):
            safe_goals = list(goal_poses.data)
        elif hasattr(goal_poses, "_list"):
            safe_goals = list(goal_poses._list)
        elif hasattr(goal_poses, "get_list"):
            safe_goals = list(goal_poses.get_list())
        else:
            try:
                safe_goals = [g for g in goal_poses]
            except Exception:
                safe_goals = []

        latest_goal = self._node.latest_goal_pose
        best_goal = self._node.best_goal_xyz

        msg_data = {
            "goal_count": len(safe_goals),
            "goals": [[round(v, 4) for v in g[:3]] for g in safe_goals[:5]],
            "latest_goal": (
                [round(v, 4) for v in latest_goal[:3]] if latest_goal else None
            ),
            "best_goal_xyz": (
                [round(v, 4) for v in best_goal] if best_goal else None
            ),
            "capture_active": self._node.goal_capture_active,
            "capture_count": getattr(self._node, "goal_capture_count", 0),
            "velocity_scale": self._config.cfg.planner.global_speed_multiplier,
            "speed_home": self._config.cfg.planner.speed_home,
            "speed_dropoff": self._config.cfg.planner.speed_dropoff,
            "speed_approach": self._config.cfg.planner.speed_approach,
            "speed_predropoff": self._config.cfg.planner.speed_predropoff,
        }
        msg = String()
        msg.data = json.dumps(msg_data)
        self.goal_info_pub.publish(msg)

    # ============ Keyboard UI ============

    def _wait_for_key_press(self) -> None:
        """Keyboard input loop running in daemon thread."""
        def ts():
            return time.strftime("%H:%M:%S")

        print(f"[{ts()}] Keys: y=save marker, m=manual, s=subscribe, p=capture, "
              "n=execute, o=open, c=close, h=home, d=dropoff, k=stop, z=teleop-toggle, q=quit")

        last_key = None
        while self._state.running:
            key = read_key()
            if key is None:
                continue

            print(f"[{ts()}] key='{key}'")

            if key == 'y':
                self._handle_key_save_marker(ts)
            elif key == 'm':
                self._handle_key_manual_goal(ts)
            elif key == 's':
                self._handle_key_subscribe(ts)
            elif key == 'p':
                self._handle_key_capture(ts)
            elif key == 'n':
                self._handle_key_execute(ts)
            elif key == 'o':
                print(f"[{ts()}] gripper -> OPEN")
                gripper_mod.control_gripper(self._node, 'OPEN')
            elif key == 'c':
                print(f"[{ts()}] gripper -> CLOSE")
                gripper_mod.control_gripper(self._node, 'CLOSE')
                print(f"[{ts()}] post-close micro-motions done")
            elif key == 'h':
                print(f"[{ts()}] going HOME...")
                motions_mod.move_to_home_position(self._node)
                print(f"[{ts()}] reached HOME (or attempted)")
            elif key == 'd':
                print(f"[{ts()}] going to DROPOFF...")
                motions_mod.move_to_dropoff_position(self._node)
                gripper_mod.control_gripper(self._node, 'OPEN')
                print(f"[{ts()}] at drop-off; gripper opened")
            elif key == 'k':
                print(f"[{ts()}] STOP requested -> publishing hold trajectory")
                self._state.stop_requested = True
                self._motion.publish_stop_trajectory()
                self._node.goal_poses.clear()
            elif key == 'z':
                self._handle_key_teleop_toggle(ts)
            elif key == 'q':
                print(f"[{ts()}] quitting...")
                self._state.running = False
                self._state.stop_requested = True
                rclpy.shutdown()
                break
            elif key == 'f':
                print("finding current joint positions...")
                print(self._state.current_joint_positions)
                print("finding current end-effector pose...")
                print(self._motion.get_end_effector_pose())
            elif key == 't':
                print("Update dynamic obstacle position:")
                pos = self._motion.obstacles.ask_user_position("dyn_sphere")
                if pos is not None:
                    self._motion.obstacles.update_pose("dyn_sphere", pos)
                    self._motion.obstacles.print_world()
            else:
                if key != last_key:
                    print(f"[{ts()}] NOTE: key '{key}' has no action. valid keys: y m s p n o c h d k q")
            last_key = key

    def _handle_key_save_marker(self, ts) -> None:
        """Handle 'y' key - save marker pose as goal."""
        if self._state.latest_marker_pose:
            from ..goals import pose_to_vec7
            g = pose_to_vec7(self._state.latest_marker_pose)
            self._node.goal_poses.append(g)
            print(f"[{ts()}] saved goal #{len(self._node.goal_poses)} from marker: {g}")
            markers_mod.publish_goal_marker(self._node, g[:3])
        else:
            print(f"[{ts()}] WARN: no latest_marker_pose yet; press 'y' again after moving the interactive marker in RViz.")

    def _handle_key_manual_goal(self, ts) -> None:
        """Handle 'm' key - manual goal entry."""
        print(f"[{ts()}] manual goal entry requested...")
        curr_pose = self._motion.get_end_effector_pose()
        print(f"  current pose: {curr_pose}")
        self._manual_goal()
        print(f"[{ts()}] manual goal entry done. total goals={len(self._node.goal_poses)}")

    def _handle_key_subscribe(self, ts) -> None:
        """Handle 's' key - subscribe to goal pose."""
        print(f"[{ts()}] subscribing to /external_goal_pose...")
        goals_mod.subscribe_to_goal_pose(self._node)
        self._node.goal_capture_active = False
        print(f"[{ts()}] subscribe called. waiting for external goal...")

    def _handle_key_capture(self, ts) -> None:
        """Handle 'p' key - start goal capture."""
        print(f"[{ts()}] starting timed goal capture (10s)...")
        self._node.start_goal_capture(10.0)
        print(f"[{ts()}] capture armed. current collected={getattr(self._node, 'goal_capture_count', 0)}")

    def _handle_key_execute(self, ts) -> None:
        """Handle 'n' key - execute stored goals."""
        if self._node.goal_poses:
            print(f"[{ts()}] executing {len(self._node.goal_poses)} stored goal(s)...")
            self._prep_and_execute()
            print(f"[{ts()}] execute finished. remaining goals={len(self._node.goal_poses)}")
        else:
            print(f"[{ts()}] INFO: no goals to execute. add with 'y', 'm', or 's'.")

    def _handle_key_teleop_toggle(self, ts) -> None:
        """Handle 'z' key - toggle teleop."""
        self._motion.teleop_enabled = not self._motion.teleop_enabled
        if self._motion.teleop_enabled:
            self._motion.latest_teleop_twist = None
            self._motion.last_teleop_time = 0.0
            print(f"[{ts()}] TELEOP enabled. Waiting for incoming teleop deltas...")
        else:
            print(f"[{ts()}] TELEOP disabled.")

    def _manual_goal(self) -> None:
        """Manual goal entry via console input."""
        cur = self._motion.get_end_effector_pose()
        try:
            x = float(input("Enter X: "))
            y = float(input("Enter Y: "))
            z = float(input("Enter Z: "))
        except ValueError:
            print("Invalid input.")
            return
        goal = [x, y, z] + (cur[3:] if cur else [1.0, 0.0, 0.0, 0.0])
        self._node.goal_poses.append(goal)
        markers_mod.publish_goal_marker(self._node, goal[:3])
        print("Manual goal saved.")

    def _prep_and_execute(self) -> None:
        """Prepare and execute stored goals."""
        if self._node.goal_capture_active:
            self._node.stop_goal_capture()
        if hasattr(self._node, 'goal_pose_sub'):
            self._node.destroy_subscription(self._node.goal_pose_sub)
            del self._node.goal_pose_sub

        self._node.get_logger().info("Executing stored goals...")
        goals_mod.plan_and_execute(self._node)
        self._node.get_logger().info("Done.")

    def shutdown(self) -> None:
        """Clean shutdown of UI manager."""
        # Keyboard thread will exit when running becomes False
        pass
