"""Record aligned physical-robot and Isaac digital-twin demonstrations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory

from .bridge import JOINT_ORDER, TRAJECTORY_TOPIC


LATEST_ONLY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def _ordered_joint_values(
    msg: JointState, names: Sequence[str]
) -> Optional[tuple[list[float], list[float]]]:
    if len(msg.name) != len(msg.position):
        return None
    positions = {name: float(value) for name, value in zip(msg.name, msg.position)}
    if not all(name in positions for name in names):
        return None
    velocities = {name: float(value) for name, value in zip(msg.name, msg.velocity)}
    return (
        [positions[name] for name in names],
        [velocities.get(name, 0.0) for name in names],
    )


def _joint_state_dict(msg: JointState) -> dict:
    return {
        "names": list(msg.name),
        "position": [float(value) for value in msg.position],
        "velocity": [float(value) for value in msg.velocity],
        "effort": [float(value) for value in msg.effort],
    }


def _point_dict(msg: PointStamped) -> dict:
    return {
        "frame_id": msg.header.frame_id,
        "position": [float(msg.point.x), float(msg.point.y), float(msg.point.z)],
    }


def _duration_s(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


class TwinRecorder(Node):
    """Sample the latest values from the physical and simulated pipelines."""

    def __init__(
        self,
        output_path: Path,
        rate_hz: float,
        duration_s: float,
        episode_id: str,
        notes: str,
        record_images: bool,
    ):
        super().__init__("isaac_hardware_demo_recorder")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.metadata_path = output_path.with_suffix(".meta.json")
        self.output_file = output_path.open("w", encoding="utf-8")
        self.episode_id = episode_id
        self.notes = notes
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.start_monotonic = time.monotonic()
        self.duration_s = duration_s
        self.finished = False
        self.frames = 0
        self.sync_samples = 0
        self.sync_mean_abs_sum = 0.0
        self.sync_peak_abs = 0.0
        self.sync_age_sum = 0.0
        self.topic_times: dict[str, float] = {}
        self.topic_counts: dict[str, int] = defaultdict(int)

        self.physical_joints = None
        self.isaac_joints = None
        self.gripper_joints = None
        self.sync_metrics = None
        self.target_pose = None
        self.gripper_target = None
        self.gripper_force = None
        self.robot_running = None
        self.all_fruits = None
        self.fruit_radius = None
        self.fruit_bbox = None
        self.fruit_direction = None
        self.gap_info = None
        self.grasp_candidate = None
        self.vision_score = None
        self.camera_status = None
        self.fingertip_verification = None
        self.grasp_attempt = None
        self.grasp_attempt_sequence = 0
        self.grasp_result = None
        self.controller_info = None
        self.trajectory = None
        self.trajectory_start = 0.0
        self.trajectory_sequence = 0
        self.last_ee_pose = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.record_images = record_images
        self.image_count = 0
        self.latest_image = None
        self.image_error_reported = False
        self.image_dir = output_path.parent / f"{output_path.stem}_images"
        self.cv_bridge = None
        self.cv2 = None
        if self.record_images:
            from cv_bridge import CvBridge
            import cv2

            self.image_dir.mkdir(parents=True, exist_ok=True)
            self.cv_bridge = CvBridge()
            self.cv2 = cv2

        self._create_subscriptions()
        self.create_timer(1.0 / max(rate_hz, 1.0), self._record_frame)
        self.get_logger().info(f"Recording aligned hardware demo to: {output_path}")
        if self.record_images:
            self.get_logger().info(f"Recording camera frames to: {self.image_dir}")

    def _create_subscriptions(self) -> None:
        self.create_subscription(
            JointState, "/isaac/physical_joint_states", self._physical_cb, LATEST_ONLY_QOS
        )
        self.create_subscription(
            JointState, "/isaac/joint_states", self._isaac_cb, LATEST_ONLY_QOS
        )
        self.create_subscription(
            JointState, "/gripper/joint_states", self._gripper_joints_cb, LATEST_ONLY_QOS
        )
        self.create_subscription(Float32MultiArray, "/isaac/joint_sync_error", self._sync_cb, 20)
        self.create_subscription(PoseStamped, "/external_goal_pose", self._target_cb, LATEST_ONLY_QOS)
        self.create_subscription(Float32MultiArray, "/gripper/target_joint", self._gripper_target_cb, 10)
        self.create_subscription(Float32MultiArray, "/gripper/force", self._gripper_force_cb, 10)
        self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self._running_cb, 10)
        self.create_subscription(JointTrajectory, TRAJECTORY_TOPIC, self._trajectory_cb, 10)
        self.create_subscription(Float32MultiArray, "/vision/all_fruit_poses", self._all_fruits_cb, LATEST_ONLY_QOS)
        self.create_subscription(Float32, "/fruit_radius", self._fruit_radius_cb, 10)
        self.create_subscription(Float32MultiArray, "/fruit_image_bbox_norm", self._bbox_cb, LATEST_ONLY_QOS)
        self.create_subscription(Vector3Stamped, "/datefruit_direction", self._direction_cb, LATEST_ONLY_QOS)
        self.create_subscription(Float32MultiArray, "/datefruit_gap_info", self._gap_cb, LATEST_ONLY_QOS)
        self.create_subscription(Float32MultiArray, "/vision/grasp_candidate_selection", self._candidate_cb, LATEST_ONLY_QOS)
        self.create_subscription(String, "/vision/fruit_score", self._vision_score_cb, 10)
        self.create_subscription(String, "/camera_status", self._camera_status_cb, 10)
        self.create_subscription(String, "/vision/fingertip_verification", self._fingertip_cb, 10)
        self.create_subscription(PointStamped, "/fruit_grasp_attempt", self._grasp_attempt_cb, 10)
        self.create_subscription(String, "/gripper/grasp_result", self._grasp_result_cb, 10)
        self.create_subscription(String, "/goal_info", self._goal_info_cb, 10)
        if self.record_images:
            self.create_subscription(Image, "/vision/raw", self._image_cb, LATEST_ONLY_QOS)

    def _mark(self, topic: str) -> None:
        self.topic_times[topic] = time.monotonic()
        self.topic_counts[topic] += 1

    def _physical_cb(self, msg: JointState) -> None:
        values = _ordered_joint_values(msg, JOINT_ORDER)
        if values is not None:
            self.physical_joints = values
            self._mark("physical_joints")

    def _isaac_cb(self, msg: JointState) -> None:
        values = _ordered_joint_values(msg, JOINT_ORDER)
        if values is not None:
            self.isaac_joints = values
            self._mark("isaac_joints")

    def _gripper_joints_cb(self, msg: JointState) -> None:
        self.gripper_joints = _joint_state_dict(msg)
        self._mark("gripper_joints")

    def _sync_cb(self, msg: Float32MultiArray) -> None:
        self.sync_metrics = [float(value) for value in msg.data]
        self._mark("sync")

    def _target_cb(self, msg: PoseStamped) -> None:
        pose = msg.pose
        self.target_pose = {
            "frame_id": msg.header.frame_id,
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation_xyzw": [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        }
        self._mark("target_pose")

    def _gripper_target_cb(self, msg: Float32MultiArray) -> None:
        self.gripper_target = [float(value) for value in msg.data]
        self._mark("gripper_target")

    def _gripper_force_cb(self, msg: Float32MultiArray) -> None:
        self.gripper_force = [float(value) for value in msg.data]
        self._mark("gripper_force")

    def _running_cb(self, msg: Bool) -> None:
        self.robot_running = bool(msg.data)
        self._mark("robot_running")

    def _trajectory_cb(self, msg: JointTrajectory) -> None:
        self.trajectory = msg
        self.trajectory_start = time.monotonic()
        self.trajectory_sequence += 1
        self._mark("arm_command")

    def _all_fruits_cb(self, msg: Float32MultiArray) -> None:
        values = [float(value) for value in msg.data]
        self.all_fruits = [values[i : i + 3] for i in range(0, len(values) - 2, 3)]
        self._mark("all_fruits")

    def _fruit_radius_cb(self, msg: Float32) -> None:
        self.fruit_radius = float(msg.data)
        self._mark("fruit_radius")

    def _bbox_cb(self, msg: Float32MultiArray) -> None:
        self.fruit_bbox = [float(value) for value in msg.data]
        self._mark("fruit_bbox")

    def _direction_cb(self, msg: Vector3Stamped) -> None:
        self.fruit_direction = {
            "frame_id": msg.header.frame_id,
            "xyz": [msg.vector.x, msg.vector.y, msg.vector.z],
        }
        self._mark("fruit_direction")

    def _gap_cb(self, msg: Float32MultiArray) -> None:
        self.gap_info = [float(value) for value in msg.data]
        self._mark("gap_info")

    def _candidate_cb(self, msg: Float32MultiArray) -> None:
        self.grasp_candidate = [float(value) for value in msg.data]
        self._mark("grasp_candidate")

    def _vision_score_cb(self, msg: String) -> None:
        self.vision_score = msg.data
        self._mark("vision_score")

    def _camera_status_cb(self, msg: String) -> None:
        self.camera_status = msg.data
        self._mark("camera_status")

    def _fingertip_cb(self, msg: String) -> None:
        self.fingertip_verification = msg.data
        self._mark("fingertip_verification")

    def _grasp_attempt_cb(self, msg: PointStamped) -> None:
        self.grasp_attempt_sequence += 1
        self.grasp_attempt = {
            "sequence": self.grasp_attempt_sequence,
            **_point_dict(msg),
        }
        self._mark("grasp_attempt")

    def _grasp_result_cb(self, msg: String) -> None:
        try:
            parsed = json.loads(msg.data)
        except json.JSONDecodeError:
            parsed = None
        self.grasp_result = {"raw": msg.data, "parsed": parsed}
        self._mark("grasp_result")

    def _goal_info_cb(self, msg: String) -> None:
        try:
            self.controller_info = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._mark("controller_info")

    def _image_cb(self, msg: Image) -> None:
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = f"frame_{self.image_count:06d}.jpg"
            path = self.image_dir / filename
            if not self.cv2.imwrite(str(path), image, [self.cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError("cv2.imwrite returned false")
            self.latest_image = {
                "path": str(path.relative_to(self.output_path.parent)),
                "ros_stamp_ns": int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec),
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": msg.encoding,
            }
            self.image_count += 1
            self._mark("camera_image")
        except Exception as exc:
            if not self.image_error_reported:
                self.get_logger().error(f"Camera frame recording failed: {exc}")
                self.image_error_reported = True

    def _lookup_ee_pose(self) -> Optional[dict]:
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "tool0", Time())
        except TransformException:
            return self.last_ee_pose
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self.last_ee_pose = {
            "frame_id": "base_link",
            "child_frame_id": "tool0",
            "position": [translation.x, translation.y, translation.z],
            "orientation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
        }
        self._mark("ee_pose")
        return self.last_ee_pose

    def _active_arm_command(self) -> Optional[dict]:
        msg = self.trajectory
        if msg is None or not msg.points:
            return None
        elapsed = max(time.monotonic() - self.trajectory_start, 0.0)
        fallback = self.physical_joints[0] if self.physical_joints else [0.0] * len(JOINT_ORDER)
        previous_time = 0.0
        previous_position = list(fallback)
        for point in msg.points:
            point_time = _duration_s(point.time_from_start)
            position = self._ordered_trajectory_values(msg, point.positions, previous_position)
            if elapsed <= point_time:
                span = max(point_time - previous_time, 1e-6)
                alpha = min(max((elapsed - previous_time) / span, 0.0), 1.0)
                target = [
                    (1.0 - alpha) * start + alpha * end
                    for start, end in zip(previous_position, position)
                ]
                velocity = self._ordered_trajectory_values(msg, point.velocities, [])
                if len(velocity) != len(JOINT_ORDER):
                    velocity = [
                        (end - start) / span
                        for start, end in zip(previous_position, position)
                    ]
                return {
                    "trajectory_sequence": self.trajectory_sequence,
                    "elapsed_s": elapsed,
                    "active": True,
                    "target_joint_pos": target,
                    "target_joint_vel": velocity,
                }
            previous_time = point_time
            previous_position = position
        return {
            "trajectory_sequence": self.trajectory_sequence,
            "elapsed_s": elapsed,
            "active": False,
            "target_joint_pos": previous_position,
            "target_joint_vel": [0.0] * len(JOINT_ORDER),
        }

    @staticmethod
    def _ordered_trajectory_values(msg, values, fallback) -> list[float]:
        if len(values) != len(msg.joint_names):
            return list(fallback)
        lookup = {name: float(value) for name, value in zip(msg.joint_names, values)}
        if not all(name in lookup for name in JOINT_ORDER):
            return list(fallback)
        return [lookup[name] for name in JOINT_ORDER]

    def _freshness(self) -> dict[str, float]:
        now = time.monotonic()
        return {
            name: now - received_at for name, received_at in self.topic_times.items()
        }

    def _record_frame(self) -> None:
        elapsed = time.monotonic() - self.start_monotonic
        if self.duration_s > 0.0 and elapsed >= self.duration_s:
            self.get_logger().info("Requested recording duration reached.")
            self.finished = True
            return
        if self.physical_joints is None or self.isaac_joints is None:
            return

        physical_pos, physical_vel = self.physical_joints
        isaac_pos, isaac_vel = self.isaac_joints
        frame = {
            "record_type": "frame",
            "schema_version": 2,
            "episode_id": self.episode_id,
            "frame_index": self.frames,
            "ros_time_ns": self.get_clock().now().nanoseconds,
            "elapsed_s": elapsed,
            "joint_names": JOINT_ORDER,
            "physical_joint_pos": physical_pos,
            "physical_joint_vel": physical_vel,
            "isaac_joint_pos": isaac_pos,
            "isaac_joint_vel": isaac_vel,
            "arm_command": self._active_arm_command(),
            "ee_pose": self._lookup_ee_pose(),
            "sync": self._sync_dict(),
            "gripper": {
                "joint_state": self.gripper_joints,
                "target": self.gripper_target,
                "force": self.gripper_force,
            },
            "perception": {
                "target_pose": self.target_pose,
                "all_fruits": self.all_fruits,
                "fruit_radius": self.fruit_radius,
                "bbox_norm": self.fruit_bbox,
                "direction": self.fruit_direction,
                "gap_info": self.gap_info,
                "grasp_candidate": self.grasp_candidate,
                "vision_score": self.vision_score,
                "camera_status": self.camera_status,
                "fingertip_verification": self.fingertip_verification,
                "image": self.latest_image,
            },
            "events": {
                "grasp_attempt": self.grasp_attempt,
                "grasp_result": self.grasp_result,
            },
            "controller": self.controller_info,
            "robot_program_running": self.robot_running,
            "freshness_s": self._freshness(),
        }
        self.output_file.write(json.dumps(frame, separators=(",", ":")) + "\n")
        self.frames += 1
        if self.sync_metrics is not None and len(self.sync_metrics) >= 9:
            self.sync_samples += 1
            self.sync_mean_abs_sum += self.sync_metrics[6]
            self.sync_peak_abs = max(self.sync_peak_abs, self.sync_metrics[7])
            self.sync_age_sum += self.sync_metrics[8]
        if self.frames % 20 == 0:
            self.output_file.flush()

    def _sync_dict(self):
        if self.sync_metrics is None or len(self.sync_metrics) < 9:
            return None
        return {
            "joint_error_rad": self.sync_metrics[:6],
            "mean_abs_error_rad": self.sync_metrics[6],
            "max_abs_error_rad": self.sync_metrics[7],
            "physical_state_age_s": self.sync_metrics[8],
        }

    def close(self, outcome: str) -> None:
        if not self.output_file.closed:
            self.output_file.flush()
            self.output_file.close()
        sync_summary = None
        if self.sync_samples:
            sync_summary = {
                "mean_abs_error_rad": self.sync_mean_abs_sum / self.sync_samples,
                "peak_abs_error_rad": self.sync_peak_abs,
                "mean_state_age_s": self.sync_age_sum / self.sync_samples,
            }
        metadata = {
            "schema_version": 2,
            "episode_id": self.episode_id,
            "started_utc": self.started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": time.monotonic() - self.start_monotonic,
            "frames": self.frames,
            "images": self.image_count,
            "joint_names": JOINT_ORDER,
            "outcome": outcome,
            "success": True if outcome == "success" else False if outcome == "failure" else None,
            "notes": self.notes,
            "topic_message_counts": dict(self.topic_counts),
            "sync": sync_summary,
            "data_file": self.output_path.name,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[TWIN RECORDER] Saved {self.frames} synchronized frames to {self.output_path}")
        print(f"[TWIN RECORDER] Episode metadata: {self.metadata_path}")
        if sync_summary:
            print(
                "[TWIN RECORDER] Twin synchronization: "
                f"mean_abs={sync_summary['mean_abs_error_rad']:.5f} rad, "
                f"peak_abs={sync_summary['peak_abs_error_rad']:.5f} rad, "
                f"mean_state_age={sync_summary['mean_state_age_s']:.4f} s"
            )


def _parse_args(default_demo: bool = False):
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_dir = Path("demos/hardware_demos" if default_demo else "demos/hardware_twin")
    prefix = "demo" if default_demo else "twin"
    parser.add_argument("--out", type=Path, default=default_dir / f"{prefix}_{timestamp}.jsonl")
    parser.add_argument("--episode-id", default=f"{prefix}_{timestamp}")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0 if default_demo else 60.0,
        help="Recording duration in seconds; use 0 to record until Ctrl+C.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--outcome",
        choices=("success", "failure", "unknown", "prompt"),
        default="prompt" if default_demo else "unknown",
    )
    parser.add_argument(
        "--images",
        action=argparse.BooleanOptionalAction,
        default=default_demo,
        help="Save /vision/raw frames as JPEG files.",
    )
    return parser.parse_args()


def _resolve_outcome(requested: str) -> str:
    if requested != "prompt":
        return requested
    if not sys.stdin.isatty():
        return "unknown"
    while True:
        answer = input("Episode outcome [s=success, f=failure, u=unknown]: ").strip().lower()
        if answer in ("s", "success"):
            return "success"
        if answer in ("f", "failure"):
            return "failure"
        if answer in ("u", "unknown", ""):
            return "unknown"


def _run(default_demo: bool) -> None:
    args = _parse_args(default_demo)
    rclpy.init()
    node = TwinRecorder(
        args.out.expanduser().resolve(),
        args.rate,
        args.duration,
        args.episode_id,
        args.notes,
        args.images,
    )
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        outcome = _resolve_outcome(args.outcome)
        node.close(outcome)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    _run(default_demo=False)


def hardware_demo_main() -> None:
    _run(default_demo=True)


if __name__ == "__main__":
    main()
