# ur10e_curobo/voxel_obstacle.py
"""Voxel-based obstacle manager using ZED depth for collision avoidance."""

import numpy as np
import torch
from typing import Optional, List, TYPE_CHECKING
from threading import Lock
import time

from curobo.geom.types import VoxelGrid, WorldConfig
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

if TYPE_CHECKING:
    from rclpy.node import Node
    from curobo.wrap.reacher.motion_gen import MotionGen

from typing import Tuple
from .config import VOXEL_CONFIG
from curobo.types.robot import JointState


class VoxelObstacleManager:
    """
    Converts ZED depth point cloud to ESDF voxel grid for cuRobo collision avoidance.

    This allows the robot to avoid ALL objects seen by the depth camera,
    not just detected dates.
    """

    def __init__(self, node: "Node", motion_gen: "MotionGen"):
        self.node = node
        self.motion_gen = motion_gen
        self._lock = Lock()

        # Voxel grid parameters from config
        self.dims = VOXEL_CONFIG["dims"]
        self.pose = VOXEL_CONFIG["pose"]
        self.voxel_size = VOXEL_CONFIG["voxel_size"]
        self.max_esdf_distance = VOXEL_CONFIG["max_esdf_distance"]

        # Compute grid shape
        self.grid_shape = [
            int(np.ceil(self.dims[0] / self.voxel_size)),
            int(np.ceil(self.dims[1] / self.voxel_size)),
            int(np.ceil(self.dims[2] / self.voxel_size)),
        ]

        # Device for torch tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache for ESDF computation
        self._last_update_time = 0.0
        self._min_update_interval = 0.1  # Max 10Hz updates

        # Publisher for visualization
        self.marker_pub = node.create_publisher(MarkerArray, "/voxel_obstacles", 10)

        # Initialize empty voxel grid
        self._voxel_grid: Optional[VoxelGrid] = None
        self._initialized = False

        # Subscribe to depth point cloud from date_v1.9.py
        # Store latest point cloud for on-demand updates (not continuous)
        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._latest_points: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._points_lock = Lock()

        self._depth_sub = node.create_subscription(
            PointCloud2,
            "/zed_depth_pointcloud",
            self._depth_callback,
            fast_qos
        )

        node.get_logger().info(
            f"VoxelObstacleManager initialized: dims={self.dims}, "
            f"voxel_size={self.voxel_size}m, grid_shape={self.grid_shape}, "
            f"(on-demand updates, call snapshot() before planning)"
        )

    def _depth_callback(self, msg: PointCloud2) -> None:
        """Cache incoming PointCloud2 messages for on-demand snapshot."""
        try:
            # Parse PointCloud2 message
            points = self._pointcloud2_to_numpy(msg)
            if points is None or points.shape[0] < 100:
                return

            # Cache for on-demand snapshot (don't update voxels continuously)
            with self._points_lock:
                self._latest_points = points
                self._latest_stamp = msg.header.stamp

        except Exception as e:
            self.node.get_logger().debug(f"Depth callback failed: {e}")

    def snapshot(self, exclude_xyz=None, exclude_radius=0.10) -> bool:
        """
        Take a snapshot of current depth and update voxel obstacles.
        Call this BEFORE planning a trajectory.

        Args:
            exclude_xyz: [x,y,z] in base_link to exclude (e.g. the target fruit).
                         Points within exclude_radius of this position are removed
                         so cuRobo doesn't treat the fruit as an obstacle.
            exclude_radius: radius around exclude_xyz to clear (meters).

        Returns:
            True if snapshot succeeded, False otherwise
        """
        with self._points_lock:
            if self._latest_points is None:
                self.node.get_logger().warn("No depth data available for snapshot")
                return False
            points = self._latest_points.copy()
            stamp = self._latest_stamp

        self.node.get_logger().info("Taking voxel obstacle snapshot...")
        success = self.update_from_points(
            points, msg_stamp=stamp,
            exclude_xyz=exclude_xyz, exclude_radius=exclude_radius,
        )
        if success:
            self.node.get_logger().info(f"Voxel snapshot complete: {points.shape[0]} points")
        return success

    def _pointcloud2_to_numpy(self, msg: PointCloud2) -> Optional[np.ndarray]:
        """Convert PointCloud2 message to numpy array of XYZ points."""
        try:
            # Simple parsing for XYZ float32 point cloud
            point_step = msg.point_step
            num_points = msg.width * msg.height

            if num_points == 0:
                return None

            # Convert bytes to numpy array
            data = np.frombuffer(msg.data, dtype=np.float32)
            points = data.reshape(-1, point_step // 4)[:, :3]  # Take XYZ only

            return points

        except Exception as e:
            self.node.get_logger().debug(f"PointCloud2 parsing failed: {e}")
            return None

    def update_from_points(self, points_cam: np.ndarray, msg_stamp=None,
                           exclude_xyz=None, exclude_radius=0.10) -> bool:
        """
        Update voxel grid from XYZ points in camera frame.

        Args:
            points_cam: (N, 3) array of XYZ points in camera frame (meters)
            msg_stamp: Optional ROS timestamp for TF lookup (use point cloud capture time)
            exclude_xyz: [x,y,z] in base_link to exclude from voxel grid
            exclude_radius: radius around exclude_xyz to clear

        Returns:
            True if update succeeded, False otherwise
        """
        # Rate limit updates
        now = time.time()
        if now - self._last_update_time < self._min_update_interval:
            return False

        with self._lock:
            try:
                if points_cam.shape[0] < 100:
                    return False

                # Transform to base_link frame using the message timestamp
                points_base = self._transform_points_to_base(points_cam, stamp=msg_stamp)
                if points_base is None:
                    return False

                # Filter points within workspace bounds
                center = np.array(self.pose[:3])
                half_dims = np.array(self.dims) / 2.0

                in_bounds = (
                    (points_base[:, 0] >= center[0] - half_dims[0]) &
                    (points_base[:, 0] <= center[0] + half_dims[0]) &
                    (points_base[:, 1] >= center[1] - half_dims[1]) &
                    (points_base[:, 1] <= center[1] + half_dims[1]) &
                    (points_base[:, 2] >= center[2] - half_dims[2]) &
                    (points_base[:, 2] <= center[2] + half_dims[2])
                )
                points_base = points_base[in_bounds]

                # Exclude points near the target (fruit) so cuRobo doesn't avoid it
                if exclude_xyz is not None:
                    exc = np.array(exclude_xyz, dtype=np.float32)
                    dists = np.linalg.norm(points_base - exc, axis=1)
                    keep = dists > exclude_radius
                    removed = np.count_nonzero(~keep)
                    points_base = points_base[keep]
                    self.node.get_logger().info(
                        f"Excluded {removed} voxel points within {exclude_radius}m of goal"
                    )

                if points_base.shape[0] < 50:
                    return False

                # Build occupancy grid and compute ESDF
                esdf_tensor = self._compute_esdf_from_points(points_base)

                # Create/update VoxelGrid
                self._voxel_grid = VoxelGrid(
                    name="depth_obstacles",
                    pose=self.pose.copy(),
                    dims=self.dims.copy(),
                    voxel_size=self.voxel_size,
                    feature_tensor=esdf_tensor,
                    feature_dtype=torch.float32,
                )

                # self._update_motion_gen_world()  # voxels visualization-only

                self._last_update_time = now
                self._initialized = True

                # Publish visualization
                self._publish_voxel_markers(points_base)

                return True

            except Exception as e:
                self.node.get_logger().warn(f"Voxel update from points failed: {e}")
                return False

    def update_from_depth(
        self,
        xyz_np: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cam_to_base_transform: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Update voxel grid from ZED XYZ point cloud.

        Args:
            xyz_np: (H, W, 4) point cloud from ZED in camera frame (meters)
            mask: Optional (H, W) boolean mask to EXCLUDE regions (e.g., target date)
            cam_to_base_transform: Optional 4x4 transform matrix from camera to base_link

        Returns:
            True if update succeeded, False otherwise
        """
        # Rate limit updates
        now = time.time()
        if now - self._last_update_time < self._min_update_interval:
            return False

        with self._lock:
            try:
                # 1. Filter valid points
                valid = np.isfinite(xyz_np[..., :3]).all(axis=-1)

                # Filter by depth range (0.1m to 2.0m)
                z_vals = xyz_np[..., 2]
                valid &= (z_vals > 0.1) & (z_vals < 2.0)

                # Apply exclusion mask if provided
                if mask is not None:
                    valid &= ~mask

                # Extract valid points
                points_cam = xyz_np[valid, :3]

                if points_cam.shape[0] < 100:
                    self.node.get_logger().debug("Too few valid points for voxel update")
                    return False

                # 2. Transform to base_link frame
                if cam_to_base_transform is not None:
                    # Apply 4x4 transform
                    ones = np.ones((points_cam.shape[0], 1))
                    points_homog = np.hstack([points_cam, ones])
                    points_base = (cam_to_base_transform @ points_homog.T).T[:, :3]
                else:
                    # Try to get transform from TF
                    points_base = self._transform_points_to_base(points_cam)
                    if points_base is None:
                        return False

                # 3. Filter points within workspace bounds
                center = np.array(self.pose[:3])
                half_dims = np.array(self.dims) / 2.0

                in_bounds = (
                    (points_base[:, 0] >= center[0] - half_dims[0]) &
                    (points_base[:, 0] <= center[0] + half_dims[0]) &
                    (points_base[:, 1] >= center[1] - half_dims[1]) &
                    (points_base[:, 1] <= center[1] + half_dims[1]) &
                    (points_base[:, 2] >= center[2] - half_dims[2]) &
                    (points_base[:, 2] <= center[2] + half_dims[2])
                )
                points_base = points_base[in_bounds]

                if points_base.shape[0] < 50:
                    self.node.get_logger().debug("Too few points in workspace bounds")
                    return False

                # 4. Build occupancy grid and compute ESDF
                esdf_tensor = self._compute_esdf_from_points(points_base)

                # 5. Create/update VoxelGrid
                self._voxel_grid = VoxelGrid(
                    name="depth_obstacles",
                    pose=self.pose.copy(),
                    dims=self.dims.copy(),
                    voxel_size=self.voxel_size,
                    feature_tensor=esdf_tensor,
                    feature_dtype=torch.float32,
                )

                # 6. self._update_motion_gen_world()  # voxels visualization-only

                self._last_update_time = now
                self._initialized = True

                # 7. Publish visualization
                self._publish_voxel_markers(points_base)

                return True

            except Exception as e:
                self.node.get_logger().warn(f"Voxel update failed: {e}")
                return False

    def _transform_points_to_base(self, points_cam: np.ndarray, stamp=None) -> Optional[np.ndarray]:
        """Transform points from camera frame to base_link using TF.

        Args:
            points_cam: (N, 3) points in camera frame
            stamp: Optional ROS timestamp for TF lookup. If None, uses latest available.
        """
        try:
            import rclpy.time
            from rclpy.duration import Duration as rclpyDuration

            # Use provided timestamp or latest available
            if stamp is not None:
                lookup_time = rclpy.time.Time.from_msg(stamp)
            else:
                lookup_time = rclpy.time.Time()  # Latest available

            # Get transform at the time the point cloud was captured
            from tf2_ros import ExtrapolationException
            try:
                tf_stamped = self.node.tf_buffer.lookup_transform(
                    "base_link",
                    "zed2_left_camera_frame",
                    lookup_time,
                    timeout=rclpyDuration(seconds=0.1),
                )
            except ExtrapolationException:
                # Timestamp not yet in TF buffer — fall back to latest available
                tf_stamped = self.node.tf_buffer.lookup_transform(
                    "base_link",
                    "zed2_left_camera_frame",
                    rclpy.time.Time(),
                    timeout=rclpyDuration(seconds=0.1),
                )

            # Extract rotation and translation
            t = tf_stamped.transform.translation
            q = tf_stamped.transform.rotation

            # Convert quaternion to rotation matrix
            R = self._quat_to_rotation_matrix(q.w, q.x, q.y, q.z)
            translation = np.array([t.x, t.y, t.z])

            # Apply transform: p_base = R @ p_cam + t
            points_base = (R @ points_cam.T).T + translation

            return points_base

        except Exception as e:
            self.node.get_logger().warn(f"TF lookup failed: {e}")
            return None

    def _quat_to_rotation_matrix(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R

    def _compute_esdf_from_points(self, points_base: np.ndarray) -> torch.Tensor:
        """
        Compute ESDF (Euclidean Signed Distance Field) from point cloud.

        Positive values = free space (distance to nearest obstacle)
        Negative values = inside obstacle
        Zero = on surface
        """
        # Voxel grid origin (bottom-left-back corner)
        center = np.array(self.pose[:3])
        half_dims = np.array(self.dims) / 2.0
        origin = center - half_dims

        # Create occupancy grid
        occupancy = np.zeros(self.grid_shape, dtype=np.float32)

        # Convert points to voxel indices
        voxel_indices = ((points_base - origin) / self.voxel_size).astype(np.int32)

        # Clip to valid range
        voxel_indices[:, 0] = np.clip(voxel_indices[:, 0], 0, self.grid_shape[0] - 1)
        voxel_indices[:, 1] = np.clip(voxel_indices[:, 1], 0, self.grid_shape[1] - 1)
        voxel_indices[:, 2] = np.clip(voxel_indices[:, 2], 0, self.grid_shape[2] - 1)

        # Mark occupied voxels
        occupancy[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

        # Dilate occupancy slightly for safety margin
        from scipy import ndimage
        occupancy = ndimage.binary_dilation(occupancy, iterations=1).astype(np.float32)

        # Compute distance transform (ESDF)
        # Distance to nearest occupied voxel for free space
        dist_to_obstacle = ndimage.distance_transform_edt(1 - occupancy) * self.voxel_size

        # Distance inside obstacles (negative)
        dist_inside = ndimage.distance_transform_edt(occupancy) * self.voxel_size

        # Combine: positive outside, negative inside
        esdf = dist_to_obstacle - dist_inside

        # Clamp to max distance
        esdf = np.clip(esdf, -self.max_esdf_distance, self.max_esdf_distance)

        # Convert to tensor and flatten
        esdf_tensor = torch.tensor(esdf, dtype=torch.float32, device=self.device)
        esdf_tensor = esdf_tensor.flatten()

        return esdf_tensor

    def _update_motion_gen_world(self) -> None:
        """Update the motion generator's world model with new voxel grid."""
        if self._voxel_grid is None:
            return

        try:
            # Get current world model
            current_world = self.motion_gen.world_model

            # Update or add voxel grid
            if current_world.voxel:
                # Replace existing voxel grid
                found = False
                for i, v in enumerate(current_world.voxel):
                    if v.name == "depth_obstacles":
                        current_world.voxel[i] = self._voxel_grid
                        found = True
                        break
                if not found:
                    current_world.voxel.append(self._voxel_grid)
            else:
                current_world.voxel = [self._voxel_grid]

            # Update motion gen
            self.motion_gen.update_world(current_world)

        except Exception as e:
            self.node.get_logger().warn(f"Failed to update motion gen world: {e}")

    def _publish_voxel_markers(self, points_base: np.ndarray) -> None:
        """Publish occupied voxels as markers for RViz visualization."""
        try:
            marker_array = MarkerArray()

            # Clear previous markers
            clear_marker = Marker()
            clear_marker.header.frame_id = "base_link"
            clear_marker.header.stamp = self.node.get_clock().now().to_msg()
            clear_marker.ns = "voxel_obstacles"
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)

            # Subsample points for visualization (max 1000)
            if points_base.shape[0] > 1000:
                indices = np.random.choice(points_base.shape[0], 1000, replace=False)
                vis_points = points_base[indices]
            else:
                vis_points = points_base

            # Create point marker
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = "voxel_obstacles"
            marker.id = 1
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.scale.x = self.voxel_size
            marker.scale.y = self.voxel_size
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.5

            from geometry_msgs.msg import Point
            for p in vis_points:
                pt = Point()
                pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
                marker.points.append(pt)

            marker_array.markers.append(marker)
            self.marker_pub.publish(marker_array)

        except Exception as e:
            self.node.get_logger().debug(f"Voxel marker publish failed: {e}")

    def verify_trajectory_collision(
        self,
        waypoints: List[List[float]],
        safety_margin: Optional[float] = None,
        check_interval: Optional[int] = None,
        exclude_position: Optional[List[float]] = None,
        exclude_radius: Optional[float] = None,
    ) -> Tuple[bool, Optional[int]]:
        """
        Verify trajectory doesn't collide with latest depth obstacles.

        Takes fresh depth snapshot and checks each waypoint's end-effector position
        against the ESDF voxel grid.

        Args:
            waypoints: List of joint positions per waypoint
            safety_margin: Distance buffer around robot (default from config)
            check_interval: Check every Nth waypoint for speed (default from config)
            exclude_position: Target position to exclude from collision checking (we WANT to reach it)
            exclude_radius: Radius around exclude_position to skip collision checks

        Returns:
            (is_safe, first_collision_idx) - True if safe, collision index if not
        """
        if not self._initialized:
            self.node.get_logger().debug("Voxel grid not initialized, skipping verification")
            return True, None

        # Use config defaults if not specified
        if safety_margin is None:
            safety_margin = VOXEL_CONFIG.get("collision_safety_margin", 0.03)
        if check_interval is None:
            check_interval = VOXEL_CONFIG.get("collision_check_interval", 5)
        if exclude_radius is None:
            exclude_radius = VOXEL_CONFIG.get("target_exclusion_radius", 0.08)

        # Convert exclude_position to numpy array if provided
        exclude_pos_np = np.array(exclude_position) if exclude_position is not None else None

        if exclude_pos_np is not None:
            self.node.get_logger().info(
                f"Collision verification with exclusion zone: {exclude_position} (radius={exclude_radius}m)"
            )

        # Take fresh snapshot for verification
        self.snapshot()

        with self._lock:
            try:
                if self._voxel_grid is None:
                    return True, None

                # Get ESDF tensor and grid parameters
                esdf = self._voxel_grid.feature_tensor.cpu().numpy()
                esdf = esdf.reshape(self.grid_shape)

                center = np.array(self.pose[:3])
                half_dims = np.array(self.dims) / 2.0
                origin = center - half_dims

                # Check waypoints at specified interval
                for i in range(0, len(waypoints), check_interval):
                    joint_positions = waypoints[i]

                    # Compute FK to get end-effector position
                    ee_pos = self._compute_ee_position(joint_positions)
                    if ee_pos is None:
                        continue

                    # Skip collision check if near target (we WANT to reach it)
                    if exclude_pos_np is not None:
                        dist_to_target = np.linalg.norm(ee_pos - exclude_pos_np)
                        if dist_to_target < exclude_radius:
                            self.node.get_logger().debug(
                                f"Skipping collision check at waypoint {i}: "
                                f"EE {ee_pos} is {dist_to_target:.3f}m from target (within {exclude_radius}m exclusion)"
                            )
                            continue  # Don't flag collision near target

                    # Check if EE position is in collision
                    is_collision = self._check_point_collision(
                        ee_pos, esdf, origin, safety_margin
                    )

                    if is_collision:
                        self.node.get_logger().warn(
                            f"Collision detected at waypoint {i}: EE pos {ee_pos}"
                        )
                        return False, i

                return True, None

            except Exception as e:
                self.node.get_logger().warn(f"Collision verification failed: {e}")
                return True, None  # Fail open - allow execution on error

    def _compute_ee_position(self, joint_positions: List[float]) -> Optional[np.ndarray]:
        """Compute end-effector position for given joint configuration using cuRobo FK."""
        try:
            js = JointState.from_position(
                torch.tensor([joint_positions], dtype=torch.float32, device=self.device),
                joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            )
            ee = self.motion_gen.rollout_fn.compute_kinematics(js)
            pos = ee.ee_pos_seq[0].cpu().numpy()
            return pos
        except Exception as e:
            self.node.get_logger().debug(f"FK computation failed: {e}")
            return None

    def _check_point_collision(
        self,
        point: np.ndarray,
        esdf: np.ndarray,
        origin: np.ndarray,
        safety_margin: float
    ) -> bool:
        """Check if a point is in collision with the ESDF grid."""
        # Convert point to voxel indices
        voxel_idx = ((point - origin) / self.voxel_size).astype(np.int32)

        # Check bounds
        for d in range(3):
            if voxel_idx[d] < 0 or voxel_idx[d] >= self.grid_shape[d]:
                return False  # Outside grid = not in collision with depth obstacles

        # Get ESDF value (negative = inside obstacle)
        esdf_value = esdf[voxel_idx[0], voxel_idx[1], voxel_idx[2]]

        # Collision if ESDF value is less than safety margin
        # (meaning we're too close to or inside an obstacle)
        return esdf_value < safety_margin

    def clear(self) -> None:
        """Clear the voxel obstacle from the world."""
        with self._lock:
            try:
                current_world = self.motion_gen.world_model
                if current_world.voxel:
                    current_world.voxel = [v for v in current_world.voxel if v.name != "depth_obstacles"]
                    self.motion_gen.update_world(current_world)
                self._voxel_grid = None
                self._initialized = False
            except Exception as e:
                self.node.get_logger().warn(f"Failed to clear voxel obstacles: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if voxel grid has been initialized with depth data."""
        return self._initialized
