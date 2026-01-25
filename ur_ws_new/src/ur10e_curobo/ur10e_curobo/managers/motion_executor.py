# ur10e_curobo/managers/motion_executor.py
"""Motion planning and execution for UR10e cuRobo node."""

import threading
import time
from typing import Optional, List, TYPE_CHECKING
from rclpy.node import Node
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from ..config import WORLD_CONFIG
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

        # Smoothing state for teleop
        self.teleop_smoothed_delta: List[float] = [0.0, 0.0, 0.0]
        self.teleop_smooth_alpha: float = 0.4  # Lower = smoother but slower
        self.teleop_traj_duration: float = 0.08  # seconds

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

        # Static obstacles
        self.static_obstacles = list(static_obstacles.DEFAULT_STATIC_OBSTACLES)

        # cuRobo setup
        self._node.get_logger().info("Loading cuRobo configuration...")
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self._config.cfg.planner.urdf_config,
            WORLD_CONFIG,
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

        self._node.get_logger().info("MotionExecutor initialized")

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
        raw_dx = raw_dy = raw_dz = 0.0
        now = time.time()
        timed_out = now - self.last_teleop_time > self.teleop_timeout

        if self.latest_teleop_twist is not None and not timed_out:
            raw_dx = float(self.latest_teleop_twist.linear.x)
            raw_dy = float(self.latest_teleop_twist.linear.y)
            raw_dz = float(self.latest_teleop_twist.linear.z)

        # Exponential smoothing of deltas
        alpha = self.teleop_smooth_alpha
        smoothed = self.teleop_smoothed_delta

        smoothed[0] = alpha * raw_dx + (1 - alpha) * smoothed[0]
        smoothed[1] = alpha * raw_dy + (1 - alpha) * smoothed[1]
        smoothed[2] = alpha * raw_dz + (1 - alpha) * smoothed[2]

        self.teleop_smoothed_delta = smoothed

        # Skip if movement is negligible
        if abs(smoothed[0]) < 0.0001 and abs(smoothed[1]) < 0.0001 and abs(smoothed[2]) < 0.0001:
            return

        # Current TCP pose
        ee = self.get_end_effector_pose()
        if ee is None:
            return

        x, y, z, qw, qx, qy, qz = ee

        # Apply smoothed deltas
        x_new = x + smoothed[0]
        y_new = y + smoothed[1]
        z_new = z + smoothed[2]

        # Build target pose (vec7)
        target_pose = [x_new, y_new, z_new, qw, qx, qy, qz]

        # Solve IK (fast path)
        try:
            q_cmd = fk_mod.solve_ik_fast(
                self._create_fk_facade(),
                target_pose,
                seed=current_pos
            )
        except Exception:
            return

        if q_cmd is None:
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
        except Exception:
            # Be defensive - don't let teleop servo crash the node
            return

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
