"""ROS hardware-interface mirror for Isaac simulation.

This node deliberately mirrors the *hardware* side of the real date-palm stack:

- consumes the same JointTrajectory command topic as the UR driver;
- publishes /joint_states and /robot_program_running like ros2_control/UR;
- publishes the same vision outputs as the ZED date detector;
- consumes Delto gripper target commands and publishes force feedback.

The first backend is an in-process kinematic mirror. IsaacLab can later replace
only the backend methods while the ROS contract remains identical.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
import os
from pathlib import Path
import socket
import time
from typing import List, Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectory

try:
    from ur_msgs.srv import SetIO
except Exception:  # pragma: no cover - lets syntax checks run outside ROS envs
    SetIO = None


def _load_hardware_profile() -> tuple[List[str], List[float], bool]:
    """Read hardware constants without importing the CUDA-dependent planner."""
    config_path = Path(__file__).resolve().parents[1] / "config.py"
    tree = ast.parse(config_path.read_text(), filename=str(config_path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue

    profile = values["ROBOT_PROFILE"]
    environment = os.environ.get("UR10E_ENVIRONMENT", "outdoor").strip().lower()
    if environment not in ("lab", "outdoor"):
        raise ValueError(
            f"UR10E_ENVIRONMENT must be 'lab' or 'outdoor', got {environment!r}"
        )
    joint_set_name = f"_{profile.upper()}_{environment.upper()}_JOINTS"
    joint_order = [str(name) for name in values["JOINT_ORDER"]]
    home_joints = [float(value) for value in values[joint_set_name]["home"]]
    return joint_order, home_joints, profile == "new"


JOINT_ORDER, DEFAULT_HOME_JOINTS, X_FORWARD_Y_LATERAL = _load_hardware_profile()


GRIPPER_TARGET_TOPIC = "/gripper/target_joint"
GRIPPER_FORCE_TOPIC = "/gripper/force"
TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
GRIPPER_JOINT_STATE_TOPIC = "/gripper/joint_states"
DELTO_JOINT_NAMES = [f"F{finger}M{motor}" for finger in (1, 2, 3) for motor in (1, 2, 3, 4)]
# As the real vision node: the best-target choice skips dates within 8 cm of an excluded position.
EXCLUDE_RADIUS_M = 0.08

LATEST_ONLY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# TRANSIENT_LOCAL so a panel or node that starts after the mirror still
# receives the warning. RELIABLE because missing this one is the difference
# between believing you are on hardware and knowing you are not.
LATCHED_STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass
class SimFruit:
    """One simulated fruit in base_link frame."""

    x: float = 0.62
    y: float = -0.04
    z: float = 0.82
    radius: float = 0.024
    score: float = 1.0

    def xyz(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class HardwareMirrorState:
    """Small kinematic state used until IsaacLab owns the backend."""

    joints: List[float] = field(default_factory=lambda: [0.0] * len(JOINT_ORDER))
    velocities: List[float] = field(default_factory=lambda: [0.0] * len(JOINT_ORDER))
    gripper_target: List[float] = field(default_factory=list)
    gripper_open_reference: List[float] = field(default_factory=list)
    last_traj_time: float = 0.0
    active_traj: Optional[JointTrajectory] = None
    active_start_joints: List[float] = field(default_factory=lambda: [0.0] * len(JOINT_ORDER))


class HardwareMirrorBackend:
    """Backend boundary for IsaacLab integration.

    The ROS node calls these methods. A future Isaac backend should implement
    them by reading/writing Articulation state and camera/perception data.
    """

    def __init__(self, initial_joints: Sequence[float]):
        self.state = HardwareMirrorState(joints=list(initial_joints))

    def set_trajectory(self, traj: JointTrajectory) -> None:
        self.state.active_traj = traj
        self.state.active_start_joints = list(self.state.joints)
        self.state.last_traj_time = time.monotonic()

    def set_gripper_target(self, target: Sequence[float]) -> None:
        values = [float(v) for v in target]
        if not values:
            return
        if len(self.state.gripper_open_reference) != len(values):
            self.state.gripper_open_reference = list(values)
        else:
            # Opening commands reduce the primary curl joints. Preserve the
            # lowest observed command as the zero-contact reference.
            for index in _finger_curl_indices(values):
                self.state.gripper_open_reference[index] = min(
                    self.state.gripper_open_reference[index], values[index]
                )
        self.state.gripper_target = values

    def step(self) -> None:
        traj = self.state.active_traj
        if traj is None or not traj.points:
            self.state.velocities = [0.0] * len(JOINT_ORDER)
            return

        elapsed = time.monotonic() - self.state.last_traj_time
        points = traj.points
        prev_t = 0.0
        prev_q = self.state.active_start_joints
        for point in points:
            point_t = _duration_to_seconds(point.time_from_start)
            q = _positions_for_joint_order(traj.joint_names, point.positions, prev_q)
            if elapsed <= point_t:
                denom = max(point_t - prev_t, 1e-6)
                alpha = min(max((elapsed - prev_t) / denom, 0.0), 1.0)
                new_q = [
                    (1.0 - alpha) * float(a) + alpha * float(b)
                    for a, b in zip(prev_q, q)
                ]
                dt = 0.02
                self.state.velocities = [
                    (n - old) / dt for n, old in zip(new_q, self.state.joints)
                ]
                self.state.joints = new_q
                return
            prev_t = point_t
            prev_q = q

        self.state.joints = prev_q
        self.state.velocities = [0.0] * len(JOINT_ORDER)
        self.state.active_traj = None

    def joint_state(self) -> tuple[List[float], List[float]]:
        return list(self.state.joints), list(self.state.velocities)

    def gripper_force(self) -> List[float]:
        # The real Delto controller reports three finger forces. This simple
        # baseline makes the force classifier and GUI alive in sim; Isaac contact
        # force can replace this once the physics backend is attached.
        target = self.state.gripper_target
        reference = self.state.gripper_open_reference
        if not target or len(target) != len(reference):
            return [0.0, 0.0, 0.0]
        forces = []
        for index in _finger_curl_indices(target):
            closure = max(target[index] - reference[index], 0.0)
            forces.append(min(closure * 10.0, 6.0))
        while len(forces) < 3:
            forces.append(0.0)
        return forces[:3]

    def connected(self) -> bool:
        return True

    def vision_state(self) -> Optional[dict]:
        return None

    def close(self) -> None:
        pass


class IsaacUdpBackend:
    """Local adapter between ROS Python and Isaac Lab's Python runtime."""

    def __init__(
        self,
        initial_joints: Sequence[float],
        command_host: str,
        command_port: int,
        state_host: str,
        state_port: int,
    ):
        self.command_model = HardwareMirrorBackend(initial_joints)
        self.actual_joints = list(initial_joints)
        self.actual_velocities = [0.0] * len(JOINT_ORDER)
        self.actual_gripper = []
        self.actual_gripper_effort = []
        self.actual_forces = [0.0, 0.0, 0.0]
        self.perception: Optional[dict] = None
        self.ros_arm_target: Optional[List[float]] = None
        self.ros_arm_velocity: Optional[List[float]] = None
        self.last_ros_joint_time = 0.0
        self.last_state_time = 0.0
        self.command_address = (command_host, command_port)
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.state_socket.bind((state_host, state_port))
        self.state_socket.setblocking(False)

    def set_trajectory(self, traj: JointTrajectory) -> None:
        self.command_model.state.joints = list(self.actual_joints)
        self.command_model.set_trajectory(traj)

    def set_gripper_target(self, target: Sequence[float]) -> None:
        self.command_model.set_gripper_target(target)

    def set_ros_joint_state(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float] = (),
    ) -> None:
        if not names or len(names) != len(positions):
            return
        lookup = {name: float(position) for name, position in zip(names, positions)}
        if all(name in lookup for name in JOINT_ORDER):
            now = time.monotonic()
            previous_target = self.ros_arm_target
            previous_time = self.last_ros_joint_time
            self.ros_arm_target = [lookup[name] for name in JOINT_ORDER]
            if len(velocities) == len(names):
                velocity_lookup = {
                    name: float(velocity)
                    for name, velocity in zip(names, velocities)
                }
                self.ros_arm_velocity = [
                    velocity_lookup[name] for name in JOINT_ORDER
                ]
            elif previous_target is not None and previous_time > 0.0:
                dt = max(now - previous_time, 1e-6)
                self.ros_arm_velocity = [
                    (position - previous) / dt
                    for position, previous in zip(
                        self.ros_arm_target, previous_target
                    )
                ]
            else:
                self.ros_arm_velocity = [0.0] * len(JOINT_ORDER)
            self.last_ros_joint_time = now

    def step(self) -> None:
        self._receive_latest_state()
        self.command_model.step()
        arm_target, _ = self.command_model.joint_state()
        arm_velocity = [0.0] * len(JOINT_ORDER)
        arm_mode = "target"
        if self.ros_arm_target is not None:
            arm_target = self.ros_arm_target
            arm_velocity = self.ros_arm_velocity or arm_velocity
            arm_mode = "state"
        payload = {
            "arm_target": arm_target,
            "arm_velocity": arm_velocity,
            "arm_mode": arm_mode,
            "gripper_target": list(self.command_model.state.gripper_target),
        }
        try:
            self.command_socket.sendto(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                self.command_address,
            )
        except OSError:
            pass

    def _receive_latest_state(self) -> None:
        latest = None
        while True:
            try:
                packet, _ = self.state_socket.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                return
            try:
                latest = json.loads(packet.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if latest is None:
            return

        arm_pos = latest.get("arm_pos", [])
        arm_vel = latest.get("arm_vel", [])
        if len(arm_pos) == len(JOINT_ORDER):
            self.actual_joints = [float(v) for v in arm_pos]
        if len(arm_vel) == len(JOINT_ORDER):
            self.actual_velocities = [float(v) for v in arm_vel]
        self.actual_gripper = [float(v) for v in latest.get("gripper_pos", [])]
        self.actual_gripper_effort = [float(v) for v in latest.get("gripper_effort", [])]
        forces = latest.get("forces", [])
        if len(forces) >= 3:
            self.actual_forces = [float(v) for v in forces[:3]]
        self.perception = latest
        self.last_state_time = time.monotonic()

    def joint_state(self) -> tuple[List[float], List[float]]:
        return list(self.actual_joints), list(self.actual_velocities)

    def gripper_force(self) -> List[float]:
        return list(self.actual_forces)

    def connected(self) -> bool:
        # Isaac pauses for a second or two when it compiles shaders or loads a scene; a 1 s
        # limit flipped robot_program_running off mid-run.
        return time.monotonic() - self.last_state_time < 2.5

    def vision_state(self) -> Optional[dict]:
        return self.perception if self.connected() else None

    def joint_sync_metrics(self) -> Optional[List[float]]:
        if self.ros_arm_target is None or self.last_ros_joint_time <= 0.0:
            return None
        errors = [
            actual - target
            for actual, target in zip(self.actual_joints, self.ros_arm_target)
        ]
        absolute = [abs(error) for error in errors]
        mean_abs = sum(absolute) / max(len(absolute), 1)
        max_abs = max(absolute, default=0.0)
        source_age = time.monotonic() - self.last_ros_joint_time
        return errors + [mean_abs, max_abs, source_age]

    def close(self) -> None:
        self.state_socket.close()
        self.command_socket.close()


class IsaacHardwareMirrorNode(Node):
    """ROS node that makes simulation look like the real hardware stack."""

    def __init__(self):
        super().__init__("isaac_hardware_mirror")

        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("fruit_x", 0.62)
        self.declare_parameter("fruit_y", -0.04)
        self.declare_parameter("fruit_z", 0.82)
        self.declare_parameter("fruit_radius", 0.024)
        self.declare_parameter("trunk_xyz", _default_trunk_xyz())
        self.declare_parameter("initial_joints", DEFAULT_HOME_JOINTS)
        self.declare_parameter("goal_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("publish_fake_vision", True)
        self.declare_parameter("backend", "kinematic")
        self.declare_parameter("udp_command_host", "127.0.0.1")
        self.declare_parameter("udp_command_port", 55000)
        self.declare_parameter("udp_state_host", "127.0.0.1")
        self.declare_parameter("udp_state_port", 55001)
        self.declare_parameter("joint_state_source", "isaac")

        initial_joints = [
            float(v) for v in self.get_parameter("initial_joints").value
        ]
        backend_name = str(self.get_parameter("backend").value).strip().lower()
        if backend_name == "udp":
            self.backend = IsaacUdpBackend(
                initial_joints,
                str(self.get_parameter("udp_command_host").value),
                int(self.get_parameter("udp_command_port").value),
                str(self.get_parameter("udp_state_host").value),
                int(self.get_parameter("udp_state_port").value),
            )
        elif backend_name == "kinematic":
            self.backend = HardwareMirrorBackend(initial_joints)
        else:
            raise ValueError(f"Unsupported Isaac mirror backend: {backend_name!r}")

        self.joint_state_source = str(
            self.get_parameter("joint_state_source").value
        ).strip().lower()
        if self.joint_state_source not in ("isaac", "ros"):
            raise ValueError(
                "joint_state_source must be 'isaac' or 'ros', got "
                f"{self.joint_state_source!r}"
            )
        if self.joint_state_source == "ros" and not isinstance(
            self.backend, IsaacUdpBackend
        ):
            raise ValueError("joint_state_source='ros' requires backend='udp'")

        self.publish_fake_vision = bool(
            self.get_parameter("publish_fake_vision").value
        )
        self.fruit = SimFruit(
            x=float(self.get_parameter("fruit_x").value),
            y=float(self.get_parameter("fruit_y").value),
            z=float(self.get_parameter("fruit_z").value),
            radius=float(self.get_parameter("fruit_radius").value),
        )
        self.trunk_xyz = [float(v) for v in self.get_parameter("trunk_xyz").value]
        self.goal_quat = [float(v) for v in self.get_parameter("goal_quat_wxyz").value]

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.isaac_joint_pub = self.create_publisher(
            JointState, "/isaac/joint_states", 10
        )
        self.sync_error_pub = self.create_publisher(
            Float32MultiArray, "/isaac/joint_sync_error", 10
        )
        self.running_pub = self.create_publisher(
            Bool, "/io_and_status_controller/robot_program_running", 10
        )
        self.force_pub = self.create_publisher(Float32MultiArray, GRIPPER_FORCE_TOPIC, 10)
        self.gripper_joint_pub = self.create_publisher(JointState, GRIPPER_JOINT_STATE_TOPIC, 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/external_goal_pose", 10)
        self.all_fruits_pub = self.create_publisher(
            Float32MultiArray, "/vision/all_fruit_poses", 10
        )
        self.radius_pub = self.create_publisher(Float32, "/fruit_radius", 10)
        self.trunk_pub = self.create_publisher(PointStamped, "/trunk_position", 10)
        self.direction_pub = self.create_publisher(Vector3Stamped, "/datefruit_direction", 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, "/fruit_image_bbox_norm", 10)
        self.vision_score_pub = self.create_publisher(String, "/vision/fruit_score", 10)

        # Say out loud that a mirror is running, because nothing else does.
        # Every other topic this node publishes is deliberately
        # indistinguishable from the real UR/Delto/ZED stack -- that is the
        # point of a mirror -- so an operator looking at the RViz panel sees
        # moving joints, a green robot_program_running and live gripper force
        # whether or not a physical arm exists. Latched so a panel that starts
        # later still learns about it.
        #
        # ISAAC means /joint_states is simulated; MIRROR means a real arm is
        # the authority and Isaac is only watching. Those are very different
        # things to be told, so they are not collapsed into one flag.
        self.twin_status_pub = self.create_publisher(
            String, "/isaac/twin_status", LATCHED_STATUS_QOS)

        self.create_subscription(JointTrajectory, TRAJECTORY_TOPIC, self._traj_cb, 10)
        self.create_subscription(Float32MultiArray, GRIPPER_TARGET_TOPIC, self._gripper_cb, 10)
        self.excluded_positions: List[List[float]] = []
        self.create_subscription(
            Float32MultiArray, "/exclude_fruit_positions", self._exclude_cb, 10
        )
        # GPU handoff with the control node, as the ZED vision node does: it publishes
        # "paused:<id>" before planning and waits for the same "paused:<id>" back (else it stops
        # the harvest). Sim vision runs on this PC's GPU, so pausing frees nothing; it only stops
        # the detections, like the real node.
        self.vision_mode = "full"
        self.mode_state_pub = self.create_publisher(String, "/vision/mode_state", 10)
        self.create_subscription(String, "/vision/mode", self._vision_mode_cb, 10)
        # Set once Isaac reports wrist-camera detections (NuRec palm scene); from then on there
        # is no fixed fallback fruit, so an empty view never sends the arm to a phantom target.
        self.wrist_vision = False
        # Per-frame observation packet (observations.FruitObservation): what reacquire searches.
        self.observation_pub = self.create_publisher(String, "/vision/fruit_observation", 10)
        self.observation_seq = 0
        self.last_vision_frame = None
        if self.joint_state_source == "ros":
            self.create_subscription(
                JointState,
                "/isaac/physical_joint_states",
                self._joint_state_cb,
                LATEST_ONLY_QOS,
            )

        if SetIO is not None:
            self.create_service(SetIO, "/io_and_status_controller/set_io", self._set_io_cb)

        self.twin_status = ("ISAAC" if self.joint_state_source == "isaac"
                            else "MIRROR")
        self.twin_status_pub.publish(String(data=self.twin_status))
        # Republished on a slow timer as well: a subscriber that misses the
        # latched sample (restarted node, late join on a lossy transport) must
        # not be left believing it is on real hardware.
        self.create_timer(
            2.0, lambda: self.twin_status_pub.publish(String(data=self.twin_status)))

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self.get_logger().info(
            f"Isaac hardware mirror started with {backend_name} backend and "
            f"{self.joint_state_source} joint-state authority. "
            "ROS contract matches UR/ZED/Delto stack."
        )

    def _traj_cb(self, msg: JointTrajectory) -> None:
        self.backend.set_trajectory(msg)
        self.get_logger().debug(f"Received trajectory with {len(msg.points)} points")

    def _gripper_cb(self, msg: Float32MultiArray) -> None:
        self.backend.set_gripper_target(msg.data)

    def _joint_state_cb(self, msg: JointState) -> None:
        if isinstance(self.backend, IsaacUdpBackend):
            self.backend.set_ros_joint_state(msg.name, msg.position, msg.velocity)

    def _vision_mode_cb(self, msg: String) -> None:
        mode, _, request_id = msg.data.partition(":")
        # The same modes as the ZED node: reacquire and the FINAL in-flight monitor use
        # reacquire_fast; ignoring it left the mirror paused and both saw no observations.
        if mode not in ("full", "reacquire", "reacquire_fast", "paused"):
            return
        self.vision_mode = mode
        self.mode_state_pub.publish(String(data=msg.data if mode == "paused" else mode))

    def _exclude_cb(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        self.excluded_positions = [
            [float(v) for v in data[i:i + 3]] for i in range(0, len(data) - 2, 3)
        ]

    def _set_io_cb(self, request, response):
        # Sim accepts suction/IO commands as successful no-ops.
        response.success = True
        return response

    def _tick(self) -> None:
        self.backend.step()
        # In physical digital-twin mode, the UR and Delto drivers remain the
        # authoritative ROS publishers. The mirror only forwards their state to
        # Isaac and must not create competing feedback/status topics.
        if self.joint_state_source == "isaac":
            self._publish_joint_state()
            self.running_pub.publish(Bool(data=self.backend.connected()))
            self.force_pub.publish(Float32MultiArray(data=self.backend.gripper_force()))
            self._publish_gripper_joint_state()
        elif isinstance(self.backend, IsaacUdpBackend):
            self._publish_joint_state(self.isaac_joint_pub)
            metrics = self.backend.joint_sync_metrics()
            if metrics is not None:
                self.sync_error_pub.publish(Float32MultiArray(data=metrics))
        if self.publish_fake_vision and self.vision_mode != "paused":
            self._publish_vision_contract()

    def _publish_gripper_joint_state(self) -> None:
        """Delto feedback like the real driver: positions plus per-motor effort (the sim joint
        torque, standing in for motor current) for the controller's non-tactile contact check."""
        positions = getattr(self.backend, "actual_gripper", [])
        efforts = getattr(self.backend, "actual_gripper_effort", [])
        if len(positions) != len(DELTO_JOINT_NAMES) or len(efforts) != len(DELTO_JOINT_NAMES):
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(DELTO_JOINT_NAMES)
        msg.position = list(positions)
        msg.velocity = [0.0] * len(DELTO_JOINT_NAMES)
        msg.effort = list(efforts)
        self.gripper_joint_pub.publish(msg)

    def _publish_joint_state(self, publisher=None) -> None:
        positions, velocities = self.backend.joint_state()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_ORDER)
        msg.position = positions
        msg.velocity = velocities
        msg.effort = [0.0] * len(JOINT_ORDER)
        (publisher or self.joint_pub).publish(msg)

    def _publish_vision_contract(self) -> None:
        now = self.get_clock().now().to_msg()
        vision = self.backend.vision_state()
        if vision is not None and "fruits_b" in vision:
            self.wrist_vision = True
        if self.wrist_vision:
            self._publish_wrist_vision(vision, now)
            return
        fruit_xyz = vision.get("fruit_xyz_b") if vision else None
        trunk_xyz = vision.get("trunk_xyz_b") if vision else None
        bbox = vision.get("bbox") if vision else None
        if not fruit_xyz or len(fruit_xyz) != 3:
            fruit_xyz = self.fruit.xyz()
        if not trunk_xyz or len(trunk_xyz) != 3:
            trunk_xyz = self.trunk_xyz
        if not bbox or len(bbox) != 4:
            bbox = [0.5, 0.5, 0.5, 0.5]
        x, y, z = [float(v) for v in fruit_xyz]

        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = "base_link"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        goal.pose.orientation.w = self.goal_quat[0]
        goal.pose.orientation.x = self.goal_quat[1]
        goal.pose.orientation.y = self.goal_quat[2]
        goal.pose.orientation.z = self.goal_quat[3]
        self.goal_pub.publish(goal)

        self.all_fruits_pub.publish(Float32MultiArray(data=[x, y, z]))
        self.radius_pub.publish(Float32(data=self.fruit.radius))

        trunk = PointStamped()
        trunk.header.stamp = now
        trunk.header.frame_id = "base_link"
        trunk.point.x = float(trunk_xyz[0])
        trunk.point.y = float(trunk_xyz[1])
        trunk.point.z = float(trunk_xyz[2])
        self.trunk_pub.publish(trunk)

        direction = Vector3Stamped()
        direction.header.stamp = now
        direction.header.frame_id = "base_link"
        direction.vector.x = -1.0 if X_FORWARD_Y_LATERAL else 0.0
        direction.vector.y = 0.0 if X_FORWARD_Y_LATERAL else 1.0
        direction.vector.z = 0.0
        self.direction_pub.publish(direction)

        # cx, cy, bunch-relative-x, bunch-relative-y. Center fruit by default.
        self.bbox_pub.publish(Float32MultiArray(data=[float(v) for v in bbox]))
        source = vision.get("perception_source", "sim_static") if vision else "sim_static"
        reason = vision.get("perception_reason", "kinematic fallback") if vision else "kinematic fallback"
        self.vision_score_pub.publish(String(data=f"{source} score=1.000 reason={reason}"))

    def _publish_observation(self, observed) -> None:
        """One packet per processed wrist frame, like the ZED node, including empty ones.

        Stamped at publication less 50 ms: Isaac runs at ~0.13x real time, so a frame that is 1-2 s
        old in wall time shows the scene ~0.2 s ago; the controller only accepts packets 0-0.5 s
        old by its clock, and this PC's clock runs ~20 ms ahead of the Jetson's.
        """
        self.observation_seq += 1
        stamp_ns = self.get_clock().now().nanoseconds - 50_000_000
        self.observation_pub.publish(String(data=json.dumps({
            "frame_id": "base_link",
            "stamp_ns": int(stamp_ns),
            "sequence": self.observation_seq,
            "poses": [[float(v) for v in o[:3]] for o in observed],
            "qualities": [[float(v) for v in o[3:8]] for o in observed],
        })))

    def _publish_wrist_vision(self, vision: Optional[dict], now) -> None:
        """Every date the sim wrist camera sees, in score order, like the ZED date detector."""
        fruits = [list(map(float, f)) for f in (vision or {}).get("fruits_b", [])]
        flat = [v for xyz in fruits for v in xyz]
        self.all_fruits_pub.publish(Float32MultiArray(data=flat))
        frame = (vision or {}).get("vision_frame", -1)
        if vision is not None and frame != -1 and frame != self.last_vision_frame:
            self.last_vision_frame = frame
            self._publish_observation(vision.get("observed", []))

        trunk_xyz = (vision or {}).get("trunk_xyz_b")
        if trunk_xyz and len(trunk_xyz) == 3:
            trunk = PointStamped()
            trunk.header.stamp = now
            trunk.header.frame_id = "base_link"
            trunk.point.x, trunk.point.y, trunk.point.z = (float(v) for v in trunk_xyz)
            self.trunk_pub.publish(trunk)

        best = next((
            xyz for xyz in fruits
            if not any(math.dist(xyz, ex) < EXCLUDE_RADIUS_M for ex in self.excluded_positions)
        ), None)
        reason = (vision or {}).get("perception_reason", "Isaac not connected")
        if best is None:
            self.vision_score_pub.publish(String(data=f"isaac-wrist-yolo score=0.000 reason={reason}"))
            return
        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = "base_link"
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = best
        goal.pose.orientation.w = self.goal_quat[0]
        goal.pose.orientation.x = self.goal_quat[1]
        goal.pose.orientation.y = self.goal_quat[2]
        goal.pose.orientation.z = self.goal_quat[3]
        self.goal_pub.publish(goal)
        radius = vision.get("fruit_radius") or self.fruit.radius
        self.radius_pub.publish(Float32(data=float(radius)))

        direction = Vector3Stamped()
        direction.header.stamp = now
        direction.header.frame_id = "base_link"
        direction.vector.x = -1.0 if X_FORWARD_Y_LATERAL else 0.0
        direction.vector.y = 0.0 if X_FORWARD_Y_LATERAL else 1.0
        self.direction_pub.publish(direction)
        self.bbox_pub.publish(Float32MultiArray(data=[float(v) for v in vision.get("bbox", [0.5] * 4)]))
        self.vision_score_pub.publish(String(data=f"isaac-wrist-yolo score=1.000 reason={reason}"))

    def destroy_node(self):
        self.backend.close()
        return super().destroy_node()


def _duration_to_seconds(duration: Duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _finger_curl_indices(target: Sequence[float]) -> List[int]:
    if len(target) >= 11:
        return [2, 6, 10]
    return list(range(min(len(target), 3)))


def _positions_for_joint_order(
    names: Sequence[str], positions: Sequence[float], fallback: Sequence[float]
) -> List[float]:
    if not names or len(names) != len(positions):
        return [float(v) for v in positions[: len(JOINT_ORDER)]]
    lookup = {name: float(pos) for name, pos in zip(names, positions)}
    return [lookup.get(name, float(fallback[i])) for i, name in enumerate(JOINT_ORDER)]


def _default_trunk_xyz() -> List[float]:
    return [1.00, 0.16, 0.60] if X_FORWARD_Y_LATERAL else [0.16, -1.00, 0.60]


def main() -> None:
    rclpy.init()
    node = IsaacHardwareMirrorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
