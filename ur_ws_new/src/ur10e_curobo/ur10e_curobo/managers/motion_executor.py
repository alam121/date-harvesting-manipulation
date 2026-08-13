# ur10e_curobo/managers/motion_executor.py
"""Motion planning and execution for UR10e cuRobo node."""

import copy
import re
import threading
import time
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml

from ..config import ROBOT_PROFILE, WORLD_CONFIG, STATIC_OBSTACLES, ENVIRONMENT
from ..dynamic_obstacle import DynamicObstacleManager
from ..voxel_obstacle import VoxelObstacleManager
from .. import static_obstacles
from .. import fk as fk_mod
from ..vision.calibration_profiles import load_camera_profile

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
        self.golfcart_pallet_marker_pub = None

        # Motion lock (prevents concurrent motions)
        self._motion_lock = threading.Lock()
        self._last_ee_pose: Optional[List[float]] = None

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
        self.golfcart_pallet_marker_pub = self._node.create_publisher(
            Marker,
            "/golfcart_pallet_marker",
            10
        )

        vision_enabled = self._config.cfg.perception.enabled
        if vision_enabled:
            # Use the detected trunk position before constructing the planner world.
            world_config = self._wait_for_trunk_and_build_world(timeout=40.0)
        else:
            world_config = copy.deepcopy(WORLD_CONFIG)
            self._node.get_logger().info(
                "Vision disabled: skipping trunk detection wait and using "
                "configured static obstacles"
            )

        # Rebuild static obstacle specs from (now-updated) STATIC_OBSTACLES for RViz
        from ..static_obstacles import _obs_to_spec
        self.static_obstacles = [_obs_to_spec(i, obs) for i, obs in enumerate(STATIC_OBSTACLES)]

        # cuRobo setup
        robot_config = self._profile_curobo_config()
        self._node.get_logger().info(
            f"Loading cuRobo configuration for robot profile {ROBOT_PROFILE!r}...")
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_config,
            world_config,
            interpolation_dt=self._config.cfg.planner.interpolation_dt,
            # Pre-allocate room for static obstacles + the 6 safe-zone wall panels (+ head-
            # room). Without this the OBB cache is sized to the initial world and adding the
            # walls fails with "number of OBB is larger than collision cache".
            collision_cache={"obb": 64, "mesh": 10},
        )

        self.motion_gen = MotionGen(self.motion_gen_config)

        # Initialize isolated FK model BEFORE warmup so its GPU allocation
        # is included in CUDA graph capture (avoids 20s+ delay on first plan)
        fk_mod.init_fk_model(self._create_fk_facade())

        self.motion_gen.warmup()
        self._node.get_logger().info("cuRobo warmup done")

        # Wire the RViz safe-zone box to keep-out walls in cuRobo's world (whole-arm bound).
        self._node._safe_zone_on_change = self._update_safe_zone_walls

        # Dynamic obstacle manager
        self.obstacles = DynamicObstacleManager(
            node=self._node,
            motion_gen=self.motion_gen,
            world_model=self.motion_gen.world_model
        )
        self.obstacles.add_sphere("dyn_sphere", radius=0.1)
        self.obstacles.add_sphere("fruit_obstacle", radius=0.06)

        if vision_enabled:
            # Voxel collision data is published by the vision node.
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
        self._node.create_timer(1.0, self._publish_golfcart_pallet_marker)

        if vision_enabled:
            # Take the initial snapshot once the depth stream is available.
            self._initial_voxel_timer = self._node.create_timer(
                2.0, self._initial_voxel_snapshot
            )

        self._node.get_logger().info("MotionExecutor initialized")

    def _profile_curobo_config(self):
        """Return a cuRobo config matching the robot and camera profile."""
        config_name = self._config.cfg.planner.urdf_config
        robot_config = load_yaml(
            join_path(get_robot_configs_path(), config_name))
        kinematics = robot_config["robot_cfg"]["kinematics"]
        source_urdf = Path(join_path(get_assets_path(), kinematics["urdf_path"]))
        text = source_urdf.read_text(encoding="utf-8")

        camera_profile = load_camera_profile()
        profile_robot = str(camera_profile.get("robot_profile", "")).lower()
        profile_environment = str(camera_profile.get("environment", "")).lower()
        if profile_robot and profile_robot != ROBOT_PROFILE:
            raise RuntimeError(
                "Camera calibration robot-profile mismatch: selected "
                f"{camera_profile.get('camera_profile', '-')!r} is for "
                f"{profile_robot!r}, but the active robot is {ROBOT_PROFILE!r}")
        if profile_environment and profile_environment != ENVIRONMENT:
            raise RuntimeError(
                "Camera calibration environment mismatch: selected "
                f"{camera_profile.get('camera_profile', '-')!r} is for "
                f"{profile_environment!r}, but the active environment is "
                f"{ENVIRONMENT!r}")

        hand_eye = camera_profile.get("hand_eye", {})
        translation = hand_eye.get("translation", {})
        rpy = hand_eye.get("rpy", {})
        child_frame = str(hand_eye.get(
            "child_frame",
            camera_profile.get("rgb_frame", "zed_mini_left_camera_frame"),
        ))
        required_translation = all(axis in translation for axis in "xyz")
        required_rpy = all(axis in rpy for axis in ("roll", "pitch", "yaw"))
        if required_translation and required_rpy:
            xyz_str = " ".join(
                f"{float(translation[axis]):.6f}" for axis in "xyz")
            rpy_str = " ".join(
                f"{float(rpy[axis]):.6f}"
                for axis in ("roll", "pitch", "yaw"))
            joint_pattern = re.compile(
                rf'(<joint\s+name="tool0_to_{re.escape(child_frame)}"'
                rf'.*?</joint>)',
                re.DOTALL,
            )
            joint_match = joint_pattern.search(text)
            if not joint_match:
                raise RuntimeError(
                    "Camera joint for active calibration profile was not "
                    f"found: tool0_to_{child_frame}")
            joint_text = joint_match.group(1)
            updated_joint, count = re.subn(
                r'<origin\b[^>]*/>',
                f'<origin rpy="{rpy_str}" xyz="{xyz_str}"/>',
                joint_text,
                count=1,
            )
            if count != 1:
                raise RuntimeError(
                    f"Camera origin was not found in tool0_to_{child_frame}")
            text = (
                text[:joint_match.start(1)] + updated_joint
                + text[joint_match.end(1):]
            )
        else:
            self._node.get_logger().warning(
                "Active camera profile has no hand-eye transform; cuRobo "
                "will use the transform embedded in its source URDF: "
                f"{camera_profile.get('profile_path', '-')}")

        profile_urdf = Path(
            f"/tmp/ur10e_curobo_{ROBOT_PROFILE}_{ENVIRONMENT}.urdf")
        profile_urdf.write_text(text, encoding="utf-8")
        kinematics["urdf_path"] = str(profile_urdf)
        self._node.get_logger().info(
            "Generated contextual cuRobo URDF: "
            f"{profile_urdf} (calibration="
            f"{camera_profile.get('camera_profile', '-')})")
        return robot_config

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
                f"Trunk detected! {len(samples)} samples → trunk at x={tx:.3f}, y={ty:.3f}"
            )
            if "cuboid" in world_config and "trunk" in world_config["cuboid"]:
                world_config["cuboid"]["trunk"]["pose"][0] = tx
                world_config["cuboid"]["trunk"]["pose"][1] = ty
            # Also update static obstacles for RViz
            for obs in STATIC_OBSTACLES:
                if obs["name"] in ("trunk", "trunk_visual"):
                    obs["pose"][0] = tx
                    obs["pose"][1] = ty
        else:
            self._node.get_logger().warn(
                "No trunk detected — using default trunk position from config"
            )

        return world_config

    # ============ Public Methods ============

    def get_end_effector_pose(self) -> Optional[List[float]]:
        """Compute FK for current joint state."""
        facade = self._create_fk_facade()
        facade._last_ee_pose = self._last_ee_pose
        pose = fk_mod.get_end_effector_pose(facade)
        if pose is not None:
            self._last_ee_pose = list(pose)
        return pose

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
        facade._planning_lock = getattr(self._node, '_planning_lock', None)
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

    def _safe_zone_wall_cuboids(self, lo, hi):
        """Four VERTICAL (lateral) panels fencing the box's back/front/sides — the blind
        directions the camera can't see. Floor and ceiling are intentionally omitted: a
        floor panel intersects the robot's own base/shoulder links (the arm reaches in from
        its base, which sits below the work box) and would put the start state in collision,
        bricking all planning. So this bounds the arm horizontally while leaving the base
        region unobstructed. Panels are centered on each face (thickness T), oversized in z
        by V so they fence tall reaches, and by M laterally to seal corners. Any panel that
        still contains the base origin is skipped."""
        from curobo.geom.types import Cuboid
        T, M, V = 0.10, 0.10, 0.40
        sx, sy, sz = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
        cx, cy, cz = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0
        q = [1.0, 0.0, 0.0, 0.0]
        specs = [
            ("xmin", (lo[0] - T / 2, cy, cz), (T, sy + M, sz + V)),
            ("xmax", (hi[0] + T / 2, cy, cz), (T, sy + M, sz + V)),
            ("ymin", (cx, lo[1] - T / 2, cz), (sx + M, T, sz + V)),
            ("ymax", (cx, hi[1] + T / 2, cz), (sx + M, T, sz + V)),
            # Floor just below the box bottom. Only safe when the box bottom is at/below the
            # ground (below the base) — otherwise it slices the base and is auto-skipped.
            ("zmin", (cx, cy, lo[2] - T / 2), (sx + M, sy + M, T)),
            # Roof just above the box top. Drag the top high enough to clear the arm or it
            # will clip the upper links (start-collision).
            ("zmax", (cx, cy, hi[2] + T / 2), (sx + M, sy + M, T)),
        ]
        walls = []
        for nm, c, d in specs:
            if abs(c[0]) <= d[0] / 2 and abs(c[1]) <= d[1] / 2 and abs(c[2]) <= d[2] / 2:
                self._node.get_logger().warn(
                    f"safezone_{nm} panel overlaps the base origin — skipped "
                    f"(move that side of the box away from the robot base).")
                continue
            walls.append(Cuboid(name=f"safezone_{nm}", pose=[*c, *q], dims=list(d)))
        return walls

    def _update_safe_zone_walls(self) -> None:
        """(Re)apply keep-out walls to cuRobo's world from node.safe_zone_*, or clear them
        when the zone is disabled. Persists across voxel updates (those only touch the voxel
        list, not cuboids)."""
        if self.motion_gen is None:
            return
        node = self._node
        try:
            wm = self.motion_gen.world_model
            wm.cuboid = [c for c in (wm.cuboid or [])
                         if not c.name.startswith("safezone_")]
            enabled = bool(getattr(node, "safe_zone_enabled", False))
            if enabled:
                lo = [min(a, b) for a, b in zip(node.safe_zone_min, node.safe_zone_max)]
                hi = [max(a, b) for a, b in zip(node.safe_zone_min, node.safe_zone_max)]
                wm.cuboid.extend(self._safe_zone_wall_cuboids(lo, hi))
            self.motion_gen.update_world(wm)
            n = len([c for c in wm.cuboid if c.name.startswith("safezone_")])
            node.get_logger().info(
                f"Safe-zone walls {'applied' if enabled else 'cleared'} ({n} panels).")
        except Exception as e:
            node.get_logger().warn(f"Safe-zone wall update failed: {e}")

    def _publish_static_obstacles(self) -> None:
        """Publish static obstacles for visualization."""
        static_obstacles.publish_static_obstacles(
            self.env_marker_pub,
            self._node.get_clock(),
            frame_id="base_link",
            specs=self.static_obstacles,
        )
        stamp = self._node.get_clock().now().to_msg()
        for marker_id in range(len(self.static_obstacles), 10):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = "environment"
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.env_marker_pub.publish(marker)

    def _publish_golfcart_pallet_marker(self) -> None:
        """Publish the pallet STL as a standalone RViz marker (lab + outdoor)."""
        if self.golfcart_pallet_marker_pub is None:
            return

        stamp = self._node.get_clock().now().to_msg()

        # Clear previous experimental marker namespaces if RViz still has them cached.
        stale_markers = (
            (0, "golfcart_pallet_stl"),
            (10, "golfcart_pallet_base"),
            (11, "golfcart_robot_mount"),
            (20, "golfcart_robot_screws"),
            (21, "golfcart_robot_screws"),
            (22, "golfcart_robot_screws"),
            (23, "golfcart_robot_screws"),
        )
        for marker_id, ns in stale_markers:
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.golfcart_pallet_marker_pub.publish(marker)

        pallet_z = -0.140
        meshes = (
            (
                0,
                "golfcart_pallet_front",
                "package://ur10e_curobo/meshes/Golfcart_pallet_front.stl",
                (0.34, 0.55, 0.72, 0.45),
                (-0.500, 0.670, pallet_z),  # rotated -90° about base_link Z: (x,y)->(y,-x)
            ),
            (
                1,
                "golfcart_pallet_mount",
                "package://ur10e_curobo/meshes/Golfcart_pallet_mount.stl",
                (0.45, 0.72, 0.95, 0.78),
                (-1.920, 0.670, pallet_z),  # rotated -90° about base_link Z: (x,y)->(y,-x)
            ),
        )
        for marker_id, ns, resource, color, xyz in meshes:
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.mesh_resource = resource
            marker.mesh_use_embedded_materials = False
            marker.pose.position.x = xyz[0]
            marker.pose.position.y = xyz[1]
            marker.pose.position.z = xyz[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            # Rotated -90 degrees about base_link Z (yaw). To flip direction to
            # +90, set z = +0.70710678 and use (x,y)->(-y,x) for the positions.
            marker.pose.orientation.z = -0.70710678
            marker.pose.orientation.w = 0.70710678
            marker.scale.x = 0.001
            marker.scale.y = 0.001
            marker.scale.z = 0.001
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            self.golfcart_pallet_marker_pub.publish(marker)

        screw_z = pallet_z + 0.350 + 0.004
        for i, (x, y) in enumerate(
            (
                (0.0601040764, -0.0601040764),
                (-0.0601040764, 0.0601040764),
                (-0.0601040764, -0.0601040764),
                (0.0601040764, 0.0601040764),
            ),
            start=1,
        ):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = stamp
            marker.ns = "golfcart_robot_screws"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = screw_z
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.030
            marker.scale.y = 0.030
            marker.scale.z = 0.008
            marker.color.r = 1.0
            marker.color.g = 0.82
            marker.color.b = 0.12
            marker.color.a = 1.0
            self.golfcart_pallet_marker_pub.publish(marker)

    def shutdown(self) -> None:
        """Clean shutdown of motion executor."""
        self.teleop_enabled = False
        self.publish_stop_trajectory()
