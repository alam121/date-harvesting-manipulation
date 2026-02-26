# ur10e_curobo/managers/motion_executor.py
"""Motion planning and execution for UR10e cuRobo node."""

import copy
import threading
import time
from typing import Optional, List, TYPE_CHECKING
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from ..config import WORLD_CONFIG, STATIC_OBSTACLES
from ..dynamic_obstacle import DynamicObstacleManager
from ..voxel_obstacle import VoxelObstacleManager
from .. import static_obstacles
from .. import fk as fk_mod

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .state_manager import StateManager


class MotionExecutor:
    """Handles cuRobo motion planning, trajectory execution, and teleop."""

    def __init__(self, node: Node, config: "ConfigManager", state: "StateManager"):
        self._node = node
        self._config = config
        self._state = state

        # cuRobo planner
        self.motion_gen_config: Optional[MotionGenConfig] = None
        self.motion_gen: Optional[MotionGen] = None

        # Obstacles
        self.obstacles: Optional[DynamicObstacleManager] = None
        self.voxel_obstacles: Optional[VoxelObstacleManager] = None
        self.static_obstacles: List = []

        # Publishers
        self.trajectory_pub = None
        self.env_marker_pub = None

        # Motion lock (prevents concurrent motions)
        self._motion_lock = threading.Lock()

        # Teleop state
        self.teleop_enabled: bool = True
        self.latest_teleop_twist: Optional[Twist] = None
        self.teleop_timeout: float = 0.2  # seconds
        self.last_teleop_time: float = 0.0

        # Smoothing state for teleop (x, y, z, yaw)
        self.teleop_smoothed_delta: List[float] = [0.0, 0.0, 0.0, 0.0]
        self.teleop_smooth_alpha: float = 0.4  # Lower = smoother but slower
        self.teleop_traj_duration: float = 0.08  # seconds
        self._teleop_was_inactive: bool = True  # Skip first frame after re-activation

    def initialize(self) -> None:
        """Initialize cuRobo planner, obstacles, and motion infrastructure."""
        # Publishers
        self.trajectory_pub = self._node.create_publisher(
            JointTrajectory,
            self._config.traj_cmd_topic,
            10
        )
        self.env_marker_pub = self._node.create_publisher(
            Marker,
            "/env_marker",
            10
        )

        # Wait for trunk detection to set pole position before cuRobo init
        world_config = self._wait_for_trunk_and_build_world()

        # Rebuild static obstacle specs from (now-updated) STATIC_OBSTACLES for RViz
        from ..static_obstacles import StaticObstacleSpec
        self.static_obstacles = [
            StaticObstacleSpec(
                marker_id=i,
                position=(float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2])),
                scale=(float(obs["dims"][0]), float(obs["dims"][1]), float(obs["dims"][2])),
                color=obs["color"],
                orientation=(float(obs["pose"][4]), float(obs["pose"][5]), float(obs["pose"][6]), float(obs["pose"][3])),
            )
            for i, obs in enumerate(STATIC_OBSTACLES)
        ]

        # cuRobo setup
        self._node.get_logger().info("Loading cuRobo configuration...")
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self._config.cfg.planner.urdf_config,
            world_config,
            interpolation_dt=self._config.cfg.planner.interpolation_dt
        )

        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup()
        self._node.get_logger().info("cuRobo warmup done")

        # Dynamic obstacle manager
        self.obstacles = DynamicObstacleManager(
            node=self._node,
            motion_gen=self.motion_gen,
            world_model=self.motion_gen.world_model
        )
        self.obstacles.add_sphere("dyn_sphere", radius=0.1)
        self.obstacles.add_sphere("fruit_obstacle", radius=0.06)

        # Voxel obstacle manager for depth-based collision avoidance
        self.voxel_obstacles = VoxelObstacleManager(
            node=self._node,
            motion_gen=self.motion_gen
        )

        # Teleop subscription
        self._node.create_subscription(
            Twist,
            "/teleop_delta",
            self._teleop_cb,
            10
        )

        # Timers
        self._node.create_timer(0.025, self._teleop_servo_tick)  # 40 Hz
        self._node.create_timer(1.0, self._publish_static_obstacles)

        # One-shot timer to take initial voxel snapshot once depth data is available
        self._initial_voxel_timer = self._node.create_timer(2.0, self._initial_voxel_snapshot)

        self._node.get_logger().info("MotionExecutor initialized")

    def _wait_for_trunk_and_build_world(self, timeout: float = 15.0, collect_secs: float = 2.0) -> dict:
        """Wait for trunk position from vision, collect samples, then build WORLD_CONFIG."""
        import numpy as np
        self._node.get_logger().info(
            f"Waiting up to {timeout:.0f}s for trunk detection on /trunk_position..."
        )
        samples = []

        def _cb(msg):
            samples.append((msg.point.x, msg.point.y))

        sub = self._node.create_subscription(PointStamped, "/trunk_position", _cb, 10)

        # Wait for first message
        t0 = time.time()
        while len(samples) == 0 and (time.time() - t0) < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.5)

        # Collect more samples for averaging
        if len(samples) > 0:
            self._node.get_logger().info(
                f"First trunk detection received, collecting {collect_secs:.0f}s of samples..."
            )
            t1 = time.time()
            while (time.time() - t1) < collect_secs:
                rclpy.spin_once(self._node, timeout_sec=0.1)

        self._node.destroy_subscription(sub)

        world_config = copy.deepcopy(WORLD_CONFIG)

        if len(samples) > 0:
            arr = np.array(samples)
            tx = float(np.median(arr[:, 0]))
            ty = float(np.median(arr[:, 1]))
            self._node.get_logger().info(
                f"Trunk detected! {len(samples)} samples → pole at x={tx:.3f}, y={ty:.3f}"
            )
            if "pole" in world_config["cuboid"]:
                pole = world_config["cuboid"]["pole"]
                pole["pose"][0] = tx
                pole["pose"][1] = ty
            # Also update static obstacles for RViz
            for obs in STATIC_OBSTACLES:
                if obs["name"] == "pole":
                    obs["pose"][0] = tx
                    obs["pose"][1] = ty
        else:
            self._node.get_logger().warn(
                "No trunk detected — using default pole position from config"
            )

        return world_config

    # ============ Public Methods ============

    def get_end_effector_pose(self) -> Optional[List[float]]:
        """Compute FK for current joint state."""
        return fk_mod.get_end_effector_pose(self._create_fk_facade())

    def acquire_motion_lock(self, blocking: bool = False) -> bool:
        """Acquire motion lock for exclusive motion control."""
        return self._motion_lock.acquire(blocking=blocking)

    def release_motion_lock(self) -> None:
        """Release motion lock."""
        self._motion_lock.release()

    def publish_stop_trajectory(self) -> None:
        """Publish stop trajectory to halt robot immediately."""
        current_pos = self._state.current_joint_positions
        if current_pos is None:
            return

        stop = JointTrajectory()
        stop.joint_names = self._config.joint_order

        pt = JointTrajectoryPoint()
        pt.positions = list(current_pos)
        pt.velocities = [0.0] * len(self._config.joint_order)
        pt.accelerations = [0.0] * len(self._config.joint_order)
        pt.time_from_start.nanosec = 1_000_000
        stop.points = [pt]

        self.trajectory_pub.publish(stop)

    def debug_print_world(self) -> None:
        """Print current cuRobo world model state."""
        wm = self.motion_gen.world_model

        print("\n========== CURRENT CUROBO WORLD ==========")

        # SPHERES
        print("SPHERES:")
        if wm.sphere:
            for s in wm.sphere:
                print(f"  - name={s.name}, pose={s.pose}, radius={s.radius}")
        else:
            print("  (none)")

        # CUBOIDS
        print("\nCUBOIDS:")
        if wm.cuboid:
            for c in wm.cuboid:
                print(f"  - name={c.name}, dims={c.dims}, pose={c.pose}")
        else:
            print("  (none)")

        # CAPSULES
        print("\nCAPSULES:")
        if wm.capsule:
            for cap in wm.capsule:
                print(f"  - name={cap.name}, dims={cap.dims}, pose={cap.pose}")
        else:
            print("  (none)")

        # CYLINDERS
        print("\nCYLINDERS:")
        if wm.cylinder:
            for cyl in wm.cylinder:
                print(f"  - name={cyl.name}, dims={cyl.dims}, pose={cyl.pose}")
        else:
            print("  (none)")

        # MESH
        print("\nMESHES:")
        if wm.mesh:
            for m in wm.mesh:
                print(f"  - name={m.name}, pose={m.pose}")
        else:
            print("  (none)")

        # VOXEL
        print("\nVOXEL GRIDS:")
        if wm.voxel:
            for v in wm.voxel:
                print(f"  - name={v.name}, pose={v.pose}")
        else:
            print("  (none)")

        print("============================================\n")

    # ============ Internal Methods ============

    def _create_fk_facade(self):
        """Create facade object for FK module compatibility."""
        class FKFacade:
            pass
        facade = FKFacade()
        facade.current_joint_positions = self._state.current_joint_positions
        facade.joint_order = self._config.joint_order
        facade.motion_gen = self.motion_gen
        facade.get_logger = self._node.get_logger
        return facade

    def _teleop_cb(self, msg: Twist) -> None:
        """Cache latest teleop twist and timestamp."""
        self.latest_teleop_twist = msg
        self.last_teleop_time = time.time()

    def _teleop_servo_tick(self) -> None:
        """40Hz teleop servo loop - convert cached twist into smoothed trajectory."""
        # Teleop enabled?
        if not self.teleop_enabled:
            return

        # Need current joint feedback to seed IK
        current_pos = self._state.current_joint_positions
        if current_pos is None:
            return

        # Get raw deltas from latest twist (or zero if timed out)
        raw_dx = raw_dy = raw_dz = raw_dyaw = 0.0
        now = time.time()
        timed_out = now - self.last_teleop_time > self.teleop_timeout

        if self.latest_teleop_twist is not None and not timed_out:
            # Skip first frame after re-activation to prevent jump
            if self._teleop_was_inactive:
                self._teleop_was_inactive = False
                self.teleop_smoothed_delta = [0.0, 0.0, 0.0, 0.0]
                return
            raw_dx = float(self.latest_teleop_twist.linear.x)
            raw_dy = float(self.latest_teleop_twist.linear.y)
            raw_dz = float(self.latest_teleop_twist.linear.z)
            raw_dyaw = float(self.latest_teleop_twist.angular.z)
        else:
            # Reset smoothed delta when timed out (deadman released) to prevent jump on re-enable
            self._teleop_was_inactive = True
            self.teleop_smoothed_delta = [0.0, 0.0, 0.0, 0.0]
            return

        # If all inputs are effectively zero, reset and skip (prevents drift and jumps)
        # Use 0.0001 (0.1 mm/s) threshold to filter out joystick noise/drift
        if (abs(raw_dx) < 0.0001 and abs(raw_dy) < 0.0001 and
            abs(raw_dz) < 0.0001 and abs(raw_dyaw) < 0.0001):
            self.teleop_smoothed_delta = [0.0, 0.0, 0.0, 0.0]
            return

        # Exponential smoothing of deltas
        alpha = self.teleop_smooth_alpha
        smoothed = self.teleop_smoothed_delta

        smoothed[0] = alpha * raw_dx + (1 - alpha) * smoothed[0]
        smoothed[1] = alpha * raw_dy + (1 - alpha) * smoothed[1]
        smoothed[2] = alpha * raw_dz + (1 - alpha) * smoothed[2]
        smoothed[3] = alpha * raw_dyaw + (1 - alpha) * smoothed[3]

        self.teleop_smoothed_delta = smoothed

        # Skip if movement is negligible (use larger threshold to prevent micro-movements)
        if (abs(smoothed[0]) < 0.0005 and abs(smoothed[1]) < 0.0005 and
            abs(smoothed[2]) < 0.0005 and abs(smoothed[3]) < 0.0005):
            return

        # Current TCP pose
        ee = self.get_end_effector_pose()
        if ee is None:
            return

        x, y, z, qw, qx, qy, qz = ee

        # Rotate delta by end effector orientation (end-effector-relative control)
        # This makes joystick forward = gripper forward, not world X
        def _quat_rotate_vec(q, v):
            """Rotate vector v by quaternion q (w, x, y, z format)."""
            qw_, qx_, qy_, qz_ = q
            vx, vy, vz = v
            tx = 2.0 * (qy_ * vz - qz_ * vy)
            ty = 2.0 * (qz_ * vx - qx_ * vz)
            tz = 2.0 * (qx_ * vy - qy_ * vx)
            return (
                vx + qw_ * tx + qy_ * tz - qz_ * ty,
                vy + qw_ * ty + qz_ * tx - qx_ * tz,
                vz + qw_ * tz + qx_ * ty - qy_ * tx,
            )

        delta_local = (smoothed[0], smoothed[1], smoothed[2])
        delta_world = _quat_rotate_vec((qw, qx, qy, qz), delta_local)
        x_new = x + delta_world[0]
        y_new = y + delta_world[1]
        z_new = z + delta_world[2]

        # Apply yaw rotation if non-zero
        dyaw = smoothed[3]
        if abs(dyaw) > 0.0001:
            # Quaternion for yaw rotation around Z axis
            import math
            half_angle = dyaw / 2.0
            dqw = math.cos(half_angle)
            dqz = math.sin(half_angle)
            # Multiply quaternions: q_new = q_delta * q_current
            # q_delta = (dqw, 0, 0, dqz)
            qw_new = dqw * qw - dqz * qz
            qx_new = dqw * qx - dqz * qy
            qy_new = dqw * qy + dqz * qx
            qz_new = dqw * qz + dqz * qw
            # Normalize
            norm = math.sqrt(qw_new**2 + qx_new**2 + qy_new**2 + qz_new**2)
            qw, qx, qy, qz = qw_new/norm, qx_new/norm, qy_new/norm, qz_new/norm
            self._node.get_logger().info(f"YAW: dyaw={dyaw:.4f} rad")

        # Build target pose (vec7)
        target_pose = [x_new, y_new, z_new, qw, qx, qy, qz]

        # Debug: log target pose when yaw is applied
        if abs(smoothed[3]) > 0.0001:
            self._node.get_logger().info(
                f"TELEOP TARGET: pos=({x_new:.4f}, {y_new:.4f}, {z_new:.4f}) "
                f"quat=({qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f})"
            )

        # Solve IK (fast path)
        try:
            q_cmd = fk_mod.solve_ik_fast(
                self._create_fk_facade(),
                target_pose,
                seed=current_pos
            )
        except Exception as e:
            self._node.get_logger().warn(f"IK exception: {e}")
            return

        if q_cmd is None:
            if abs(smoothed[3]) > 0.0001:
                self._node.get_logger().warn("IK returned None for yaw rotation")
            return

        # Build smooth trajectory with velocity
        traj = JointTrajectory()
        traj.joint_names = self._config.joint_order

        traj_duration = self.teleop_traj_duration

        # Compute velocities based on position difference
        velocities = [(q_cmd[i] - current_pos[i]) / traj_duration for i in range(len(q_cmd))]

        # Single point trajectory with velocity for smooth tracking
        point = JointTrajectoryPoint()
        point.positions = list(q_cmd)
        point.velocities = velocities
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(traj_duration * 1e9)
        traj.points.append(point)

        try:
            self.trajectory_pub.publish(traj)
            if abs(smoothed[3]) > 0.0001:
                self._node.get_logger().info(f"TELEOP: Published trajectory with yaw rotation")
        except Exception as e:
            self._node.get_logger().warn(f"Trajectory publish exception: {e}")
            return

    def _initial_voxel_snapshot(self) -> None:
        """Take initial voxel snapshot once depth data arrives, then cancel timer."""
        vo = self.voxel_obstacles
        if vo._latest_points is not None:
            try:
                vo.snapshot()
                self._node.get_logger().info("Initial voxel snapshot taken from depth data.")
            except Exception as e:
                self._node.get_logger().warn(f"Initial voxel snapshot failed: {e}")
            self._initial_voxel_timer.cancel()

    def _publish_static_obstacles(self) -> None:
        """Publish static obstacles for visualization."""
        static_obstacles.publish_static_obstacles(
            self.env_marker_pub,
            self._node.get_clock(),
            frame_id="base_link",
            specs=self.static_obstacles,
        )

    def shutdown(self) -> None:
        """Clean shutdown of motion executor."""
        self.teleop_enabled = False
        self.publish_stop_trajectory()
