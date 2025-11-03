#!/usr/bin/env python3
# ruff: noqa
import sys
import math
import time
import json
import tty
import termios
import select
import threading
from typing import List, Optional, Tuple, Iterable

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from geometry_msgs.msg import Point, Pose as ROSPose, PoseStamped, PointStamped, Quaternion
from sensor_msgs.msg import JointState as ROSJointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import InteractiveMarkerFeedback, Marker
from std_msgs.msg import Bool, Float32MultiArray, String

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from delto_gripper_controller import DeltoGripperController
from grasp_outcome_classifier import GraspOutcomeClassifier


# =========================
# Configuration & Constants
# =========================
JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# QoS for low-latency goal intake & feedback
DEFAULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# World obstacles (example)
WORLD_CONFIG = {
    "cuboid": {
        "table": {
            "dims": [5.0, 5.0, 0.2],
            "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],
        },
        "pole": {
            "dims": [0.02, 0.02, 1.0],
            "pose": [0.0, -0.95, 0.5, 1, 0, 0, 0],
        },
    }
}

PLAN_CFG_DEFAULT = MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)

# Helper typing
Vec7 = List[float]
Vec3 = List[float]


class UR10eCuroboMoveIt(Node):
    """UR10e controller that plans with cuRobo and executes via ROS trajectory controller."""

    # ---------------------
    # Lifecycle & Init
    # ---------------------
    def __init__(self) -> None:
        super().__init__("ur10e_curobo_moveit_node")

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)
        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)
        self.wrist_publisher_ = self.trajectory_pub  # single controller topic used consistently

        # Subscriptions
        self.create_subscription(ROSJointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(
            InteractiveMarkerFeedback,
            "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback",
            self._marker_cb,
            10,
        )
        self.create_subscription(Float32MultiArray, "/gripper/force", self._force_cb, 10)
        self.create_subscription(Bool, "/emergency_stop", self._stop_cb, 10)
        self.create_subscription(
            Bool, "/io_and_status_controller/robot_program_running", self._robot_running_cb, 10
        )

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timers
        self.create_timer(0.1, self._track_robot_path)  # path trace
        self.create_timer(0.02, self._classifier_tick)  # 50 Hz for classifier
        self.timer_wait_js = self.create_timer(0.5, self._check_joint_states)

        # cuRobo
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            "ur10e.yml", WORLD_CONFIG, interpolation_dt=0.004
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()

        # State
        self.qos = DEFAULT_QOS
        self.joint_order = JOINT_ORDER
        self.current_joint_positions: Optional[List[float]] = None
        self.current_joint_velocities: Optional[List[float]] = None
        self.latest_marker_pose: Optional[ROSPose] = None
        self.goal_poses: List[Vec7] = []
        self.path_points: List[Point] = []
        self.running = True

        self.speed_scale = 1.8
        self.stop_requested = False
        self.robot_running = False

        # Gripper & grasp classification
        self.gripper_controller = DeltoGripperController(self)
        self.gripper_closed = False
        self.slip_detection = False
        self.grab_miss = False
        self.weak_grab = False
        self.last_grasp_end_template: Optional[str] = None
        self.classifier = GraspOutcomeClassifier(
            on_outcome=self._on_grasp_outcome, dead_time_thresh_s=1.40, hold_time_s=0.5
        )

        # Home & staging joints
        self.home_joints = [
            -1.5916569868670862,
            -1.649402920399801,
            2.113215446472168,
            -4.4819199482547205,
            4.604945659637451,
            -0.05743295351137334,
        ]
        self.dropoff_joints = [
            -2.16858417192568,
            -1.3347657362567347,
            2.0885677337646484,
            -2.6394265333758753,
            4.78283166885376,
            0.013545919209718704,
        ]
        self.predropoff_joints = [
            -1.6832264105426233,
            -2.020153347645895,
            2.238132953643799,
            -3.9681833426104944,
            4.682962894439697,
            -0.010893646870748341,
        ]

        # Goal capture helpers
        self.goal_capture_active = False
        self.goal_capture_timer = None
        self.goal_capture_count = 0
        self.goal_sort_ref: Optional[Vec3] = None
        self.goal_sort_ascending = True

        # Keyboard thread
        self.keyboard_thread = threading.Thread(target=self._wait_for_key_press, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info("UR10e cuRobo node initialized. Waiting for joint states…")

    # ---------------------
    # Core ROS Callbacks
    # ---------------------
    def _check_joint_states(self) -> None:
        if self.current_joint_positions is not None:
            self.get_logger().info(f"Received initial joints: {np.round(self.current_joint_positions, 3)}")
            self.destroy_timer(self.timer_wait_js)

    def _joint_state_cb(self, msg: ROSJointState) -> None:
        jm = dict(zip(msg.name, msg.position))
        vm = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        self.current_joint_positions = [jm[j] for j in self.joint_order if j in jm]
        self.current_joint_velocities = [vm.get(j, 0.0) for j in self.joint_order]

    def _marker_cb(self, msg: InteractiveMarkerFeedback) -> None:
        self.latest_marker_pose = msg.pose

    def _robot_running_cb(self, msg: Bool) -> None:
        self.robot_running = msg.data
        self.get_logger().info("✅ Robot program is running." if msg.data else "⚠️ Robot program is NOT running.")

    def _stop_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.get_logger().warn("🛑 Emergency stop requested!")
        self.stop_requested = True
        self._publish_stop_trajectory()

    def _force_cb(self, msg: Float32MultiArray) -> None:
        raw = list(msg.data)
        self.classifier.on_force(raw[:3])

    def _classifier_tick(self) -> None:
        self.classifier.tick()

    # ---------------------
    # Keyboard UI
    # ---------------------
    def _get_key(self, timeout: float = 0.1) -> Optional[str]:
        """Read a single key press (non-blocking)."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            return sys.stdin.read(1) if rlist else None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _wait_for_key_press(self) -> None:
        print("Keys: y=save marker, m=manual goal, s=subscribe goal, p=capture, n=execute,")
        print("      o=open, c=close, h=home, d=dropoff, k=stop, q=quit")

        while self.running:
            key = self._get_key()
            if key is None:
                continue

            if key == "y" and self.latest_marker_pose:
                g = self._pose_to_vec7(self.latest_marker_pose)
                self.goal_poses.append(g)
                print(f"Goal {len(self.goal_poses)} saved from marker: {np.round(g, 3)}")
                self._publish_goal_marker(g[:3])

            elif key == "m":
                self._manual_goal_input()

            elif key == "s":
                self.get_logger().info("Subscribing to external goal pose…")
                self._subscribe_to_goal_pose()

            elif key == "p":
                self.get_logger().info("Timed capture of external goals (10s)…")
                self.start_goal_capture(duration=10.0)

            elif key == "n" and self.goal_poses:
                if self.goal_capture_active:
                    self.stop_goal_capture()
                if hasattr(self, "goal_pose_sub"):
                    self.destroy_subscription(self.goal_pose_sub)
                    del self.goal_pose_sub
                    print("Unsubscribed from /external_goal_pose.")
                self.get_logger().info("Executing stored goals…")
                self.plan_and_execute()
                self.get_logger().info("All goals executed. Waiting for gripper command…")

            elif key == "o":
                self.control_gripper("OPEN")

            elif key == "c":
                self.control_gripper("CLOSE")
                self.move_backward(-0.01)
                self.rotate_wrist(120)

            elif key == "h":
                self.move_to_home_position()

            elif key == "d":
                self.get_logger().info("Moving to drop-off zone…")
                self.move_to_dropoff_position()
                self.control_gripper("OPEN")

            elif key == "k":
                self.get_logger().warn("Stop requested by user.")
                self.stop_requested = True
                self._publish_stop_trajectory()

            elif key == "q":
                print("Exiting…")
                self.running = False
                break

    def _manual_goal_input(self) -> None:
        """Prompt user for XYZ and keep current orientation."""
        cur = self.get_end_effector_pose()
        if cur:
            print(f"Current EE (m): X={cur[0]:.3f} Y={cur[1]:.3f} Z={cur[2]:.3f}")
        else:
            print("⚠️ Could not read current EE pose.")

        try:
            x = float(input("Enter X: "))
            y = float(input("Enter Y: "))
            z = float(input("Enter Z: "))
        except ValueError:
            print("⚠️ Invalid input, please enter numbers.")
            return

        goal: Vec7 = [x, y, z] + (cur[3:] if cur else [1.0, 0.0, 0.0, 0.0])
        self.goal_poses.append(goal)
        print(f"Manual goal saved: {np.round(goal, 3)}")
        self._publish_goal_marker(goal[:3])

    @staticmethod
    def _pose_to_vec7(p: ROSPose) -> Vec7:
        """Convert ROS Pose to [x,y,z,qw,qx,qy,qz]."""
        return [
            p.position.x,
            p.position.y,
            p.position.z,
            p.orientation.w,
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
        ]

    # ---------------------
    # FK / Pose Utilities
    # ---------------------
    def get_end_effector_pose(self) -> Optional[Vec7]:
        """Compute current end-effector pose using cuRobo FK."""
        if self.current_joint_positions is None:
            self.get_logger().warn("Joint states not yet received.")
            return None
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            js = JointState.from_position(
                torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )
            ee = self.motion_gen.rollout_fn.compute_kinematics(js)
            pos = ee.ee_pos_seq[0].cpu().tolist()
            quat = ee.ee_quat_seq[0].cpu().tolist()
            self.get_logger().info(f"[FK] pos={np.round(pos,3)} quat={np.round(quat,3)}")
            return pos + quat
        except Exception as e:
            self.get_logger().warn(f"FK failed: {e}")
            return None

    def forward_kinematics(self, joint_positions: List[float]) -> Optional[Point]:
        """FK for an arbitrary joint vector."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            js = JointState.from_position(
                torch.tensor([joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )
            ee = self.motion_gen.rollout_fn.compute_kinematics(js)
            pos = ee.ee_pos_seq.squeeze().tolist()
            return Point(x=pos[0], y=pos[1], z=pos[2])
        except Exception as e:
            self.get_logger().warn(f"CuRobo FK failed: {e}")
            return None

    # ---------------------
    # Markers / Path Trace
    # ---------------------
    def _publish_goal_marker(self, position: Vec3, rank: Optional[int] = None) -> None:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "goal_positions"
        m.id = len(self.goal_poses)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = position
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0, 0.0
        self.goal_marker_pub.publish(m)

        if rank is not None:
            t = Marker()
            t.header.frame_id = "base_link"
            t.header.stamp = m.header.stamp
            t.ns = "goal_labels"
            t.id = 1000 + m.id
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = position[0], position[1], position[2] + 0.06
            t.scale.z = 0.05
            t.color.a = t.color.r = t.color.g = t.color.b = 1.0
            t.text = str(rank)
            self.goal_marker_pub.publish(t)

    def _publish_path_marker(self) -> None:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "robot_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.01
        m.color.a = 1.0
        m.color.g = 1.0
        m.points = self.path_points
        self.path_marker_pub.publish(m)

    def _track_robot_path(self) -> None:
        """Keep a breadcrumb trail of tool0 in base_link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            p = Point(
                x=tf.transform.translation.x,
                y=tf.transform.translation.y,
                z=tf.transform.translation.z,
            )
            if not self.path_points or (
                p.x != self.path_points[-1].x or p.y != self.path_points[-1].y or p.z != self.path_points[-1].z
            ):
                self.path_points.append(p)
                self._publish_path_marker()
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Path trace TF failed: {e}")

    # ---------------------
    # Motion Helpers
    # ---------------------
    def _build_traj_from_states(
        self, states: Iterable[List[float]], vel: float = 0.1, dt: float = 0.03
    ) -> JointTrajectory:
        msg = JointTrajectory()
        msg.joint_names = self.joint_order
        t = 0.0
        for q in states:
            if self.stop_requested:
                break
            pt = JointTrajectoryPoint()
            pt.positions = list(q)
            pt.velocities = [vel] * len(self.joint_order)
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
            msg.points.append(pt)
            t += dt
        return msg

    def _publish_stop_trajectory(self) -> None:
        if self.current_joint_positions is None:
            return
        stop = JointTrajectory()
        stop.joint_names = self.joint_order
        pt = JointTrajectoryPoint()
        pt.positions = list(self.current_joint_positions)
        pt.velocities = [0.0] * len(self.joint_order)
        pt.accelerations = [0.0] * len(self.joint_order)
        pt.time_from_start.nanosec = 1_000_000
        stop.points = [pt]
        self.trajectory_pub.publish(stop)

    def _interpolated_positions(self, result) -> List[List[float]]:
        """Return list of joint vectors from a cuRobo plan result."""
        interp = result.get_interpolated_plan()
        if isinstance(interp, JointState):
            interp = interp.position
        if not isinstance(interp, torch.Tensor):
            interp = torch.tensor(interp, dtype=torch.float32)
        return interp.to("cpu").tolist()

    def _wait_until_cartesian(self, goal_xyz: Vec3, tol: float = 0.005) -> None:
        self.get_logger().info(f"Waiting for EE to reach {np.round(goal_xyz, 3)}…")
        while self.running:
            if self.stop_requested:
                self.get_logger().warn("Stop during wait. Holding at current pose.")
                self._publish_stop_trajectory()
                self.stop_requested = False
                break
            cur = self.get_end_effector_pose()
            if not cur:
                time.sleep(0.05)
                continue
            dist = math.dist(cur[:3], goal_xyz)
            if dist < tol:
                self.get_logger().info(f"Goal reached (dist={dist:.4f}).")
                break
            time.sleep(0.05)

    def is_robot_moving(self, velocity_threshold: float = 0.001) -> bool:
        if not self.current_joint_velocities:
            return False
        return any(abs(v) > velocity_threshold for v in self.current_joint_velocities)

    # ---------------------
    # High-level Moves
    # ---------------------
    def execute_single_pose(self, pose: Vec7) -> None:
        """Plan & execute to a Cartesian pose [x,y,z,qw,qx,qy,qz]."""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state; cannot execute pose.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        goal = Pose.from_list(pose)
        res = self.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT)
        if not res.success:
            self.get_logger().warn("Plan failed for single pose.")
            return
        traj = self._build_traj_from_states(self._interpolated_positions(res), vel=0.1, dt=0.03)
        self.trajectory_pub.publish(traj)

    def move_to_home_position(self, cfg: Optional[MotionGenPlanConfig] = None) -> None:
        self._plan_execute_js(self.home_joints, cfg, label="HOME")

    def move_to_dropoff_position(self, cfg: Optional[MotionGenPlanConfig] = None) -> None:
        self._plan_execute_js(self.dropoff_joints, cfg, label="DROP-OFF")

    def move_to_predropoff_position(self, cfg: Optional[MotionGenPlanConfig] = None) -> None:
        self._plan_execute_js(self.predropoff_joints, cfg, label="preDROP-OFF", dt=0.04)

    def _plan_execute_js(
        self, target_joints: List[float], cfg: Optional[MotionGenPlanConfig], label: str, dt: float = 0.008
    ) -> None:
        """Plan & execute a joint-space move and wait at the final Cartesian position."""
        if self.current_joint_positions is None:
            self.get_logger().warn(f"No joint state; skipping {label} move.")
            return
        if not self.robot_running:
            self.get_logger().error("Robot program OFF; cannot execute joint motion.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = JointState.from_position(
            torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        goal_js = JointState.from_position(
            torch.tensor([target_joints], dtype=torch.float32, device=device),
            joint_names=self.joint_order,
        )
        res = self.motion_gen.plan_single_js(start, goal_js, cfg or PLAN_CFG_DEFAULT)
        if not res.success:
            self.get_logger().warn(f"Joint-space plan to {label} failed.")
            return
        states = self._interpolated_positions(res)
        traj = self._build_traj_from_states(states, vel=0.1, dt=dt / max(self.speed_scale, 1e-6))
        self.trajectory_pub.publish(traj)
        self.get_logger().info(f"Moving to {label} joints…")

        fk_last = self.forward_kinematics(states[-1])
        if fk_last:
            self._wait_until_cartesian([fk_last.x, fk_last.y, fk_last.z])

    def rotate_wrist(self, degrees: float, duration: float = 0.30) -> None:
        """Rotate wrist_3 by +/- degrees and return."""
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state; cannot rotate wrist.")
            return
        try:
            while self.is_robot_moving(velocity_threshold=0.01):
                time.sleep(0.01)
        except Exception:
            pass
        rad = math.radians(degrees)
        start = self.current_joint_positions.copy()
        plus = start.copy()
        plus[5] = start[5] + rad
        back = start.copy()

        traj = JointTrajectory()
        traj.joint_names = self.joint_order

        def _pt(q: List[float], tsec: float) -> JointTrajectoryPoint:
            p = JointTrajectoryPoint()
            p.positions = q
            p.velocities = [0.0] * len(q)
            p.time_from_start.sec = int(tsec)
            p.time_from_start.nanosec = int((tsec - int(tsec)) * 1e9)
            return p

        traj.points = [_pt(start, 0.0), _pt(plus, duration), _pt(back, 2 * duration)]
        # tiny hold
        hold_ns = traj.points[-1].time_from_start.nanosec + 50_000_000
        p3 = _pt(back, float(traj.points[-1].time_from_start.sec))
        p3.time_from_start.sec += 1 if hold_ns >= 1_000_000_000 else 0
        p3.time_from_start.nanosec = hold_ns % 1_000_000_000
        traj.points.append(p3)
        self.trajectory_pub.publish(traj)
        self.get_logger().info(f"Wrist +{degrees}° then back in {2*duration:.2f}s")

    def move_backward(self, delta: float) -> None:
        """Nudge by modifying shoulder_lift (index 1). Positive delta raises; negative lowers."""
        if self.current_joint_positions is None:
            self.get_logger().error("No joint state; cannot move backward.")
            return
        q = self.current_joint_positions.copy()
        q[1] -= delta
        traj = JointTrajectory()
        traj.joint_names = self.joint_order
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start.sec = 1
        traj.points.append(p)
        self.wrist_publisher_.publish(traj)
        self.get_logger().info(f"Nudged shoulder_lift by {-delta} rad")
        time.sleep(1.0)

    # ---------------------
    # Gripper
    # ---------------------
    def control_gripper(self, action: str) -> None:
        act = action.upper()
        if act == "OPEN":
            self.gripper_controller.open_gripper()
            self.gripper_closed = False
            self.slip_detection = False
            self.grab_miss = False
            self.classifier.start_opening()
            self.get_logger().info("Gripper OPEN → slip detection paused")
        elif act == "CLOSE":
            self.classifier.start_closing()
            self.gripper_controller.run_closure_loop()
            self.classifier.mark_close_done()
            self.gripper_closed = True
            self.get_logger().info("Gripper CLOSED → slip detection active")

    def _on_grasp_outcome(self, outcome: str, end: str) -> None:
        self.slip_detection = outcome == "SLIPPED"
        self.grab_miss = outcome == "NO_GRAB"
        self.weak_grab = (outcome == "GRABBED") and (end == "WEAK")
        self.last_grasp_end_template = end
        self.get_logger().info(
            f"[grasp] outcome={outcome} end={end} -> slip={self.slip_detection} "
            f"miss={self.grab_miss} weak={self.weak_grab}"
        )

    # ---------------------
    # External Goals: subscribe / capture / sort / reacquire
    # ---------------------
    def _subscribe_to_goal_pose(self) -> None:
        if self.is_robot_moving():
            self.get_logger().warn("Robot moving. Delaying subscription…")
            self.create_timer(0.5, lambda: (not self.is_robot_moving()) and self._subscribe_to_goal_pose())
            return

        if hasattr(self, "goal_pose_sub"):
            self.get_logger().info("Already subscribed. Rebinding.")
            self.destroy_subscription(self.goal_pose_sub)
            del self.goal_pose_sub

        cur = self.get_end_effector_pose()
        current_orientation = cur[3:] if cur else [1.0, 0.0, 0.0, 0.0]

        self.goal_received = False
        self.goal_poses.clear()

        def _goal_cb(msg: PoseStamped) -> None:
            self.goal_received = True
            if hasattr(self, "idle_timer"):
                self.idle_timer.cancel()
                self.get_logger().info("Idle motion stopped.")
                self._publish_stop_trajectory()
            if self.is_robot_moving():
                self.get_logger().warn("Goal arrived while moving; ignoring.")
                return
            time.sleep(0.1)
            g = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, *current_orientation]
            if not any(math.dist(g[:3], e[:3]) < 0.01 for e in self.goal_poses):
                self.goal_poses.append(g)
                self._publish_goal_marker(g[:3])
                self.get_logger().info(f"Goal saved: {np.round(g,3)}")
            if hasattr(self, "goal_pose_sub"):
                self.destroy_subscription(self.goal_pose_sub)
                del self.goal_pose_sub
                self.get_logger().info("Unsubscribed /external_goal_pose.")

        self.goal_pose_sub = self.create_subscription(PoseStamped, "/external_goal_pose", _goal_cb, self.qos)
        self.get_logger().info("Subscribed to /external_goal_pose")

        # simple idle patrol (UP, LEFT, RIGHT, DOWN)
        sequence = [("UP", 0.0, 0.0, 0.2), ("LEFT", 0.0, 0.1, 0.0), ("RIGHT", 0.0, -0.1, 0.0), ("DOWN", 0.0, 0.0, -0.1)]
        idx = {"i": 0}

        def _idle_cb():
            if self.goal_received or idx["i"] >= len(sequence):
                if hasattr(self, "idle_timer"):
                    self.idle_timer.cancel()
                return
            curp = self.get_end_effector_pose()
            if curp and not self.is_robot_moving():
                _, dx, dy, dz = sequence[idx["i"]]
                tgt = [curp[0] + dx, curp[1] + dy, curp[2] + dz, *current_orientation]
                self.get_logger().info(f"Idle step {idx['i']+1}/{len(sequence)}")
                self.execute_single_pose(tgt)
                idx["i"] += 1

        self.idle_timer = self.create_timer(5.0, _idle_cb)

    def start_goal_capture(self, duration: float = 30.0) -> None:
        if self.goal_capture_active:
            self.stop_goal_capture()

        self.goal_poses.clear()
        self.goal_capture_count = 0

        ee = self.get_end_effector_pose()
        self.goal_sort_ref = ee[:3] if ee else [0.0, 0.0, 0.0]
        if ee:
            self.get_logger().info(f"Sorting by distance to EE: {np.round(self.goal_sort_ref,3)}")
        else:
            self.get_logger().warn("EE pose unavailable; sorting to [0,0,0].")

        self.goal_pose_sub = self.create_subscription(
            PoseStamped, "/external_goal_pose", self._capture_goal_cb, self.qos
        )
        self.goal_capture_active = True
        self.get_logger().info(f"Started goal capture for {duration:.0f}s")

        def _stop_once():
            if self.goal_capture_timer:
                self.goal_capture_timer.cancel()
            self.stop_goal_capture()

        self.goal_capture_timer = self.create_timer(duration, _stop_once)

    def stop_goal_capture(self) -> None:
        if not self.goal_capture_active:
            return
        self.goal_capture_active = False
        if hasattr(self, "goal_pose_sub"):
            self.destroy_subscription(self.goal_pose_sub)
            del self.goal_pose_sub
        self._sort_goals_by_distance()
        self.get_logger().info(f"Goal capture stopped. Collected {self.goal_capture_count} goals.")

    def _capture_goal_cb(self, msg: PoseStamped) -> None:
        if self.is_robot_moving():
            return
        gx, gy, gz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        cur = self.get_end_effector_pose()
        qw, qx, qy, qz = (cur[3:] if cur else [1.0, 0.0, 0.0, 0.0])
        if any(math.dist([gx, gy, gz], g[:3]) < 0.01 for g in self.goal_poses):
            return
        goal = [gx, gy, gz, qw, qx, qy, qz]
        self.goal_poses.append(goal)
        self._sort_goals_by_distance()
        self.goal_capture_count += 1
        self._publish_goal_marker(goal[:3])
        self.get_logger().info(f"Captured goal #{self.goal_capture_count}")

    def _goal_distance(self, gxyz: Vec3, ref: Optional[Vec3] = None) -> float:
        r = ref or self.goal_sort_ref or [0.0, 0.0, 0.0]
        return math.dist(gxyz, r)

    def _sort_goals_by_distance(self, ref: Optional[Vec3] = None) -> None:
        rr = ref or self.goal_sort_ref or [0.0, 0.0, 0.0]
        self.goal_poses.sort(
            key=lambda g: self._goal_distance(g[:3], rr),
            reverse=not self.goal_sort_ascending,
        )

    def reacquire_goal_pose(
        self, seed_xyz: Vec3, timeout: float = 3.5, radius: float = 0.08, stable_eps: float = 0.004, stable_need: int = 2
    ) -> Optional[Vec3]:
        """Reacquire a detection near seed. If timeout, retreat, then retry with relaxed criteria."""
        def _try_once(seed: Vec3, timeout_s: float, rad: float, need: int) -> Optional[Tuple[float, float, float]]:
            latest = None
            stable = 0
            last_hit = time.time()

            def _cb(msg: PoseStamped) -> None:
                nonlocal latest, stable, last_hit
                if self.is_robot_moving():
                    return
                x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
                if math.hypot(x - seed[0], y - seed[1]) > rad:
                    return
                latest = (x, y, z)
                stable += 1
                last_hit = time.time()

            sub = self.create_subscription(PoseStamped, "/external_goal_pose", _cb, self.qos)
            try:
                while stable < max(1, need):
                    if timeout_s is not None and (time.time() - last_hit) > timeout_s:
                        break
                    time.sleep(0.01)
            finally:
                self.destroy_subscription(sub)
            return latest

        # wait for stop
        t0 = time.time()
        while self.is_robot_moving() and (time.time() - t0) < 0.25:
            time.sleep(0.01)

        first = _try_once(seed_xyz, timeout, radius, stable_need)
        if first:
            self.get_logger().info(f"Reacquired near seed {np.round(seed_xyz,3)} → {np.round(first,3)}")
            return list(first)

        self.get_logger().warn("Reacquire timed out; backing off then retrying…")
        cur = self.get_end_effector_pose()
        if cur:
            self.execute_single_pose([cur[0], cur[1] + 0.10, cur[2], *cur[3:]])
            self._wait_until_cartesian([cur[0], cur[1] + 0.10, cur[2]])

        second = _try_once(seed_xyz, timeout_s=2.0, rad=max(radius, 0.15), need=max(1, stable_need - 1))
        if second:
            self.get_logger().info(f"Reacquired after retreat → {np.round(second,3)}")
            return list(second)

        self.get_logger().warn("Reacquire failed after retreat.")
        return None

    # ---------------------
    # Full Pipeline
    # ---------------------
    def plan_and_execute(self) -> None:
        """For each stored goal: approach → reacquire → final insert → grip checks → drop → home."""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state yet.")
            return
        if not self.goal_poses:
            self.get_logger().warn("No stored goals.")
            return
        if not self.robot_running:
            self.get_logger().error("Robot program OFF; proceeding may fail on the controller.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        while self.goal_poses and self.running:
            start_state = JointState.from_position(
                torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )

            goal = self.goal_poses.pop(0)
            x, y, z = goal[:3]

            # Orientation heuristic by height
            if z > 1.30:
                orientation = self.quaternion_from_approach(pitch_deg=-35.0)
                approach = [x, y + 0.12, z, *orientation]
                y = y - 0.004
                z = z - 0.4
                self.get_logger().info(f"High goal (z={z:.3f}) → side approach.")
            else:
                orientation = goal[3:]
                approach = [x, y + 0.10, z - 0.12, *orientation]
                z = z + 0.055
                y = y - 0.003
                self.get_logger().info(f"Normal goal (z={z:.3f}) → top approach.")

            # 1) Approach
            if not self._plan_and_send(start_state, Pose.from_list(approach), dt=0.01, label="APPROACH"):
                continue
            self._wait_until_cartesian(approach[:3])

            # Update start state
            start_state = JointState.from_position(
                torch.tensor([self.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=self.joint_order,
            )

            # 2) Reacquire near target
            seed = [x, y, z]
            reacq = self.reacquire_goal_pose(seed_xyz=seed, timeout=3.5, stable_eps=0.004, stable_need=3)
            if reacq:
                x, y, z = reacq
                self._publish_goal_marker([x, y, z])
            else:
                self.get_logger().warn("No reacquire; skipping goal.")
                continue

            # 3) Final insert
            final_target = [x, y - 0.001, z + 0.001, *orientation]
            if not self._plan_and_send(start_state, Pose.from_list(final_target), dt=0.03, label="FINAL"):
                continue
            self._wait_until_cartesian(final_target[:3])

            # 4) Grip + checks / corrective micro-moves
            self.control_gripper("CLOSE")
            time.sleep(0.7)
            if self.slip_detection or self.grab_miss or self.weak_grab:
                self.get_logger().info("Slip/miss/weak → reopen, nudge up, retry close.")
                self.control_gripper("OPEN")
                cur = self.get_end_effector_pose()
                if cur:
                    up = [cur[0], cur[1], cur[2] + 0.015, *cur[3:]]
                    self.execute_single_pose(up)
                self.slip_detection = self.grab_miss = False
                self.control_gripper("CLOSE")

            # 5) Wrist rotate + retract → pre-dropoff → dropoff
            self.rotate_wrist(90)
            time.sleep(0.7)
            self.move_to_predropoff_position()
            self.move_to_dropoff_position()
            self.control_gripper("OPEN")
            self.move_to_home_position()

        self.get_logger().info("✅ Finished all goals.")

    def _plan_and_send(
        self, start_state: JointState, goal_pose: Pose, dt: float, label: str
    ) -> bool:
        """Utility: plan single Cartesian move and publish."""
        res = self.motion_gen.plan_single(start_state, goal_pose, PLAN_CFG_DEFAULT)
        if not res.success:
            self.get_logger().warn(f"Plan failed for {label}.")
            return False
        states = self._interpolated_positions(res)
        traj = self._build_traj_from_states(states, vel=0.1, dt=dt)
        if self.stop_requested:
            self.get_logger().warn(f"Stop requested before sending {label} trajectory.")
            self.stop_requested = False
            return False
        self.trajectory_pub.publish(traj)
        return True

    # ---------------------
    # Orientation helper
    # ---------------------
    def quaternion_from_approach(
        self, direction_xyz: Optional[Tuple[float, float, float]] = None, pitch_deg: Optional[float] = None, world: bool = False
    ) -> List[float]:
        """
        If pitch_deg is provided: apply pitch-only rotation (about Y).
        - world=False: rotate about tool's local Y (preferred)
        Returns [qw,qx,qy,qz].
        """
        if pitch_deg is not None:
            cur = self.get_end_effector_pose()
            if cur:
                qw, qx, qy, qz = cur[3], cur[4], cur[5], cur[6]
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            half = math.radians(pitch_deg) / 2.0
            cw, cx, cy, cz = math.cos(half), math.sin(half), 0.0, 0.0
            if world:
                nw = cw * qw - cx * qx - cy * qy - cz * qz
                nx = cw * qx + cx * qw + cy * qz - cz * qy
                ny = cw * qy - cx * qz + cy * qw + cz * qx
                nz = cw * qz + cx * qy - cy * qx + cz * qw
            else:
                nw = qw * cw - qx * cx - qy * cy - qz * cz
                nx = qw * cx + qx * cw + qy * cz - qz * cy
                ny = qw * cy - qx * cz + qy * cw + qz * cx
                nz = qw * cz + qx * cy - qy * cx + qz * cw
            return [nw, nx, ny, nz]

        d = np.array(direction_xyz or (0.0, 0.0, -1.0), dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            return [1.0, 0.0, 0.0, 0.0]
        d /= n
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        x_axis = np.cross(up, d)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(d, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = d
        Rm = np.array(
            [[x_axis[0], y_axis[0], z_axis[0]], [x_axis[1], y_axis[1], z_axis[1]], [x_axis[2], y_axis[2], z_axis[2]]],
            dtype=float,
        )
        t = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            qw = 0.25 * s
            qx = (Rm[2, 1] - Rm[1, 2]) / s
            qy = (Rm[0, 2] - Rm[2, 0]) / s
            qz = (Rm[1, 0] - Rm[0, 1]) / s
        else:
            if Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
                s = math.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2.0
                qw = (Rm[2, 1] - Rm[1, 2]) / s
                qx = 0.25 * s
                qy = (Rm[0, 1] + Rm[1, 0]) / s
                qz = (Rm[0, 2] + Rm[2, 0]) / s
            elif Rm[1, 1] > Rm[2, 2]:
                s = math.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2.0
                qw = (Rm[0, 2] - Rm[2, 0]) / s
                qx = (Rm[0, 1] + Rm[1, 0]) / s
                qy = 0.25 * s
                qz = (Rm[1, 2] + Rm[2, 1]) / s
            else:
                s = math.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2.0
                qw = (Rm[1, 0] - Rm[0, 1]) / s
                qx = (Rm[0, 2] + Rm[2, 0]) / s
                qy = (Rm[1, 2] + Rm[2, 1]) / s
                qz = 0.25 * s
        return [qw, qx, qy, qz]


# =========================
# Main
# =========================
def main() -> None:
    print("Starting UR10e MoveIt Node…")
    rclpy.init()
    node = UR10eCuroboMoveIt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down node.")
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


