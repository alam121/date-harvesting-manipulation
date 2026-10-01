"""Learn and safely apply a target-relative grasp skill from one demonstration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Optional

import rclpy
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from ur_dashboard_msgs.srv import IsProgramRunning

from .bridge import JOINT_ORDER


LATEST_ONLY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

ROBOT_PROGRAM_TOPIC = "/io_and_status_controller/robot_program_running"

ROBOT_STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _finite_vector(values, length: int) -> Optional[list[float]]:
    if not isinstance(values, list) or len(values) != length:
        return None
    result = [float(value) for value in values]
    return result if all(math.isfinite(value) for value in result) else None


def _load_jsonl(path: Path) -> list[dict]:
    frames = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if item.get("record_type", "frame") == "frame":
                frames.append(item)
    if not frames:
        raise ValueError("demonstration contains no frames")
    return frames


def _episode_metadata(path: Path) -> dict:
    metadata_path = path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise ValueError(f"episode metadata is missing: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def learn_skill(
    demo_path: Path,
    output_path: Path,
    allow_non_success: bool = False,
) -> dict:
    metadata = _episode_metadata(demo_path)
    if metadata.get("success") is not True and not allow_non_success:
        raise ValueError(
            "one-shot skill requires an episode labelled success; use "
            "--allow-non-success only for debugging"
        )
    frames = _load_jsonl(demo_path)
    labelled_frames = [
        frame
        for frame in frames
        if isinstance(frame.get("controller"), dict)
        and isinstance(frame.get("perception", {}).get("target_pose"), dict)
        and isinstance(frame.get("ee_pose"), dict)
    ]
    if not labelled_frames:
        raise ValueError(
            "no frames contain controller, target, and tool0 pose data; "
            "record a new episode with the current hardware_demo_recorder"
        )
    final_frames = [
        frame
        for frame in labelled_frames
        if str(frame["controller"].get("motion_phase", "")).upper() == "FINAL"
    ]
    if not final_frames:
        raise ValueError("episode has no controller motion_phase=FINAL frames")
    closure_frame = final_frames[-1]
    controller = closure_frame["controller"]
    offsets = controller.get("grasp_mode_offsets_m", {})
    normal = _finite_vector(offsets.get("normal"), 2)
    envelop = _finite_vector(offsets.get("envelop"), 2)
    if normal is None or envelop is None:
        raise ValueError("FINAL frame does not contain valid grasp-mode offsets")
    if any(abs(value) > 0.05 for value in normal + envelop):
        raise ValueError("recorded grasp offset exceeds the controller's 50 mm bound")

    grasp_attempt = closure_frame.get("events", {}).get("grasp_attempt")
    target = None
    target_source = "perception_final_frame"
    if isinstance(grasp_attempt, dict):
        target = _finite_vector(grasp_attempt.get("position"), 3)
        if target is not None:
            target_source = "fruit_grasp_attempt"
    if target is None:
        target = _finite_vector(
            closure_frame["perception"]["target_pose"].get("position"), 3
        )
    ee_position = _finite_vector(closure_frame["ee_pose"].get("position"), 3)
    ee_orientation = _finite_vector(
        closure_frame["ee_pose"].get("orientation_xyzw"), 4
    )
    if target is None or ee_position is None or ee_orientation is None:
        raise ValueError("FINAL frame target or tool0 pose is invalid")

    skill = {
        "schema_version": 1,
        "skill_type": "guarded_target_relative_grasp",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_demo": str(demo_path.resolve()),
        "source_episode_id": metadata.get("episode_id"),
        "source_outcome": metadata.get("outcome"),
        "source_sync": metadata.get("sync"),
        "grasp_mode": str(controller.get("active_goal_grasp_mode", "NORMAL")).upper(),
        "goal_classification": controller.get("goal_classification"),
        "grasp_mode_offsets_m": {"normal": normal, "envelop": envelop},
        "demonstrated_target_b": target,
        "demonstrated_target_source": target_source,
        "demonstrated_tool0_b": {
            "position": ee_position,
            "orientation_xyzw": ee_orientation,
        },
        "target_to_tool0_translation_b": [
            ee - goal for ee, goal in zip(ee_position, target)
        ],
        "safety": {
            "max_target_shift_m": 0.03,
            "max_runtime_offset_change_m": 0.005,
            "max_joint_velocity_rad_s": 0.02,
        },
    }
    if skill["grasp_mode"] not in ("NORMAL", "ENVELOP"):
        raise ValueError(f"unsupported demonstrated grasp mode: {skill['grasp_mode']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(skill, indent=2) + "\n", encoding="utf-8")
    return skill


class SkillApplicator(Node):
    """Validate live state and publish grasp settings without starting motion."""

    def __init__(self, skill: dict):
        super().__init__("one_shot_grasp_applicator")
        self.skill = skill
        self.target = None
        self.target_time = 0.0
        self.controller = None
        self.controller_time = 0.0
        self.joint_velocities = None
        self.joint_time = 0.0
        self.robot_running = False
        self.running_time = 0.0
        self.dashboard_time = 0.0
        self.dynamic_state_time = 0.0
        self.dashboard_future = None
        self.next_dashboard_query = 0.0
        self.command_pub = self.create_publisher(String, "/ui_command", 10)
        self.dashboard_client = self.create_client(
            IsProgramRunning, "/dashboard_client/program_running"
        )
        self.create_subscription(
            PoseStamped, "/external_goal_pose", self._target_cb, LATEST_ONLY_QOS
        )
        self.create_subscription(String, "/goal_info", self._controller_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joints_cb, 10)
        self.create_subscription(
            DynamicJointState,
            "/dynamic_joint_states",
            self._dynamic_joints_cb,
            10,
        )
        self.create_subscription(
            Bool,
            ROBOT_PROGRAM_TOPIC,
            self._running_cb,
            ROBOT_STATUS_QOS,
        )

    def _target_cb(self, msg: PoseStamped) -> None:
        self.target = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
        self.target_time = time.monotonic()

    def _controller_cb(self, msg: String) -> None:
        try:
            self.controller = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.controller_time = time.monotonic()

    def _joints_cb(self, msg: JointState) -> None:
        if len(msg.velocity) != len(msg.name):
            return
        lookup = {name: float(value) for name, value in zip(msg.name, msg.velocity)}
        if all(name in lookup for name in JOINT_ORDER):
            self.joint_velocities = [lookup[name] for name in JOINT_ORDER]
            self.joint_time = time.monotonic()

    def _running_cb(self, msg: Bool) -> None:
        self.robot_running = bool(msg.data)
        self.running_time = time.monotonic()

    def _dynamic_joints_cb(self, msg: DynamicJointState) -> None:
        for joint_name, interfaces in zip(msg.joint_names, msg.interface_values):
            if not joint_name.endswith("gpio"):
                continue
            lookup = dict(zip(interfaces.interface_names, interfaces.values))
            if "program_running" not in lookup:
                continue
            value = float(lookup["program_running"])
            if not math.isfinite(value):
                return
            now = time.monotonic()
            self.robot_running = value > 0.5
            self.running_time = now
            self.dynamic_state_time = now
            return

    def _poll_dashboard_status(self) -> None:
        now = time.monotonic()
        if self.dashboard_future is not None:
            if not self.dashboard_future.done():
                return
            try:
                response = self.dashboard_future.result()
            except Exception:
                response = None
            self.dashboard_future = None
            if response is not None and response.success:
                self.robot_running = bool(response.program_running)
                self.running_time = now
                self.dashboard_time = now
            self.next_dashboard_query = now + 0.5
            return
        if now >= self.next_dashboard_query and self.dashboard_client.service_is_ready():
            self.dashboard_future = self.dashboard_client.call_async(
                IsProgramRunning.Request()
            )

    def validation_errors(
        self,
        max_target_shift_m: float,
        max_offset_change_m: float,
        max_joint_velocity: float,
    ) -> list[str]:
        now = time.monotonic()
        self._poll_dashboard_status()
        status_source_available = (
            self.count_publishers(ROBOT_PROGRAM_TOPIC) > 0
            or now - self.dashboard_time <= 1.5
            or now - self.dynamic_state_time <= 1.0
        )
        errors = []
        program_status_available = self.running_time > 0.0 and status_source_available
        if program_status_available and not self.robot_running:
            errors.append("robot program is not running")
        if self.target is None or now - self.target_time > 0.25:
            errors.append("camera target is missing or older than 0.25 s")
        if self.controller is None or now - self.controller_time > 1.0:
            errors.append("controller status is missing or stale")
        elif str(self.controller.get("motion_phase", "")).upper() != "IDLE":
            errors.append(
                f"controller phase is {self.controller.get('motion_phase')}, not IDLE"
            )
        if self.joint_velocities is None or now - self.joint_time > 0.5:
            errors.append("joint velocities are missing or stale")
        elif max(abs(value) for value in self.joint_velocities) > max_joint_velocity:
            errors.append("physical robot is moving")

        if self.target is not None:
            demonstrated = self.skill["demonstrated_target_b"]
            shift = math.dist(self.target, demonstrated)
            if shift > max_target_shift_m:
                errors.append(
                    f"target shifted {shift * 1000:.1f} mm; limit is "
                    f"{max_target_shift_m * 1000:.1f} mm"
                )
        if self.controller is not None:
            current = self.controller.get("grasp_mode_offsets_m", {})
            learned = self.skill["grasp_mode_offsets_m"]
            for mode in ("normal", "envelop"):
                current_values = _finite_vector(current.get(mode), 2)
                if current_values is None:
                    errors.append(f"live controller has no valid {mode} offsets")
                    continue
                for index, (new, old) in enumerate(
                    zip(learned[mode], current_values)
                ):
                    if abs(new - old) > max_offset_change_m:
                        axis = "depth" if index == 0 else "z"
                        errors.append(
                            f"{mode} {axis} change is {(new-old)*1000:+.1f} mm; "
                            f"limit is {max_offset_change_m*1000:.1f} mm"
                        )
        return errors

    def publish_skill(self) -> bool:
        mode = self.skill["grasp_mode"]
        offsets = self.skill["grasp_mode_offsets_m"]
        values = offsets["normal"] + offsets["envelop"]
        commands = [
            f"set_grasp_mode {mode}",
            "set_grasp_mode_offsets " + " ".join(f"{value:.6f}" for value in values),
        ]
        for command in commands:
            for _ in range(3):
                self.command_pub.publish(String(data=command))
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(0.05)
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._controller_matches_skill():
                return True
        return False

    def _controller_matches_skill(self) -> bool:
        if self.controller is None:
            return False
        if str(self.controller.get("grasp_mode", "")).upper() != self.skill["grasp_mode"]:
            return False
        current = self.controller.get("grasp_mode_offsets_m", {})
        learned = self.skill["grasp_mode_offsets_m"]
        for mode in ("normal", "envelop"):
            values = _finite_vector(current.get(mode), 2)
            if values is None:
                return False
            if any(abs(actual - expected) > 1e-6 for actual, expected in zip(values, learned[mode])):
                return False
        return True


def _learn_command(args) -> int:
    try:
        skill = learn_skill(
            args.demo.expanduser().resolve(),
            args.out.expanduser().resolve(),
            args.allow_non_success,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ONE SHOT] Learn failed: {exc}")
        return 2
    print(f"[ONE SHOT] Skill saved: {args.out}")
    print(f"[ONE SHOT] Mode: {skill['grasp_mode']}")
    print(f"[ONE SHOT] Learned offsets: {skill['grasp_mode_offsets_m']}")
    print(
        "[ONE SHOT] Target-to-tool0 translation: "
        f"{skill['target_to_tool0_translation_b']}"
    )
    return 0


def _apply_command(args) -> int:
    try:
        skill = json.loads(args.skill.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ONE SHOT] Could not load skill: {exc}")
        return 2
    if skill.get("skill_type") != "guarded_target_relative_grasp":
        print("[ONE SHOT] Unsupported skill file")
        return 2

    safety = skill.get("safety", {})
    max_target_shift = (
        args.max_target_shift
        if args.max_target_shift is not None
        else float(safety.get("max_target_shift_m", 0.03))
    )
    max_offset_change = (
        args.max_offset_change
        if args.max_offset_change is not None
        else float(safety.get("max_runtime_offset_change_m", 0.005))
    )
    max_joint_velocity = float(safety.get("max_joint_velocity_rad_s", 0.02))

    rclpy.init()
    node = SkillApplicator(skill)
    deadline = time.monotonic() + args.wait_timeout
    errors = ["waiting for live state"]
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            errors = node.validation_errors(
                max_target_shift, max_offset_change, max_joint_velocity
            )
            if not errors:
                break
        if errors:
            print("[ONE SHOT] Safety validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 3
        shift = math.dist(node.target, skill["demonstrated_target_b"])
        print("[ONE SHOT] Safety validation passed")
        if node.running_time == 0.0:
            print(
                "[ONE SHOT] Program-state value unavailable on PolyScopeX; "
                "fresh stationary joints and controller IDLE were verified"
            )
        print(f"[ONE SHOT] Live target shift: {shift * 1000:.1f} mm")
        print(f"[ONE SHOT] Proposed mode: {skill['grasp_mode']}")
        print(f"[ONE SHOT] Proposed offsets: {skill['grasp_mode_offsets_m']}")
        if not args.execute:
            print("[ONE SHOT] DRY RUN: no controller settings were changed")
            return 0
        if not node.publish_skill():
            print("[ONE SHOT] Controller did not confirm the learned settings")
            return 4
        print("[ONE SHOT] Skill applied; it is effective from the next grasp onward")
        print("[ONE SHOT] Use the normal GUI harvest command with operator supervision")
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    learn = commands.add_parser("learn", help="extract a skill from one episode")
    learn.add_argument("--demo", type=Path, required=True)
    learn.add_argument(
        "--out", type=Path, default=Path("skills/one_shot/grasp_skill.json")
    )
    learn.add_argument("--allow-non-success", action="store_true")

    apply = commands.add_parser("apply", help="validate and optionally apply a skill")
    apply.add_argument("--skill", type=Path, required=True)
    apply.add_argument("--execute", action="store_true")
    apply.add_argument("--wait-timeout", type=float, default=10.0)
    apply.add_argument("--max-target-shift", type=float)
    apply.add_argument("--max-offset-change", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    code = _learn_command(args) if args.command == "learn" else _apply_command(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
