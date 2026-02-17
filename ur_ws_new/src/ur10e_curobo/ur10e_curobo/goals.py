# ruff: noqa
import time, math, torch
import threading
import numpy as np
from geometry_msgs.msg import Pose as ROSPose, PoseStamped, PointStamped
from curobo.types.math import Pose
from curobo.types.robot import JointState


class ThreadSafeGoalList:
    """Thread-safe wrapper for goal_poses list to prevent race conditions between ROS callbacks and main thread."""
    def __init__(self):
        self._lock = threading.Lock()
        self._goals = []

    def append(self, goal):
        with self._lock:
            self._goals.append(goal)

    def pop(self, index=0):
        with self._lock:
            if self._goals:
                return self._goals.pop(index)
            return None

    def clear(self):
        with self._lock:
            self._goals.clear()

    def __len__(self):
        with self._lock:
            return len(self._goals)

    def __bool__(self):
        with self._lock:
            return bool(self._goals)

    def any_within_distance(self, pos, threshold):
        """Check if any goal is within threshold distance of pos[:3]."""
        with self._lock:
            return any(math.dist(pos[:3], e[:3]) < threshold for e in self._goals)

    def sort(self, key=None, reverse=False):
        """Sort goals in place with optional key function."""
        with self._lock:
            self._goals.sort(key=key, reverse=reverse)
from .motions import execute_single_pose as _exec
from .motions import publish_stop_trajectory
from .config import PLAN_CFG_DEFAULT, VOXEL_CONFIG
from .utils import build_trajectory, wait_until_xyz
from .markers import publish_goal_marker, publish_planned_path
from .motions import interpolated_positions, execute_single_pose
from .motions import execute_single_pose as exec_pose
from .motions import rotate_wrist, move_to_predropoff_position, move_to_dropoff_position, move_to_home_position
from .motions import blend_motion
from .dynamic_obstacle import DynamicObstacleManager
from .utils import compute_visibility_approach
from .fk import forward_kinematics, forward_kinematics_batch

def pose_to_vec7(p: ROSPose):
    return [p.position.x, p.position.y, p.position.z, p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]


def log_path_deviation(node, label: str):
    """Compare actual EE position to the planned path stored by plan_and_send.
    Logs: distance to planned endpoint + max deviation from nearest planned waypoint."""
    planned = getattr(node, '_planned_cartesian_path', None)
    if not planned:
        return
    actual_pose = node.get_end_effector_pose()
    if not actual_pose:
        return
    ax, ay, az = actual_pose[0], actual_pose[1], actual_pose[2]

    # Distance to planned final point
    fp = planned[-1]
    end_dist = math.sqrt((ax - fp.x)**2 + (ay - fp.y)**2 + (az - fp.z)**2)

    # Distance to nearest planned waypoint
    min_dist = float('inf')
    min_idx = 0
    for i, p in enumerate(planned):
        d = math.sqrt((ax - p.x)**2 + (ay - p.y)**2 + (az - p.z)**2)
        if d < min_dist:
            min_dist = d
            min_idx = i

    node.get_logger().info(
        f"[PATH_DEV] {label}: end_err={end_dist*100:.1f}cm | "
        f"nearest_wp={min_idx}/{len(planned)} dist={min_dist*100:.1f}cm"
    )


def notify_grasp_attempt(node, position_xyz):
    """
    Publish grasp attempt notification to vision system for fruit tracking.
    This increments the attempt count for the fruit at the given position.
    """
    if not hasattr(node, "_grasp_attempt_pub"):
        node._grasp_attempt_pub = node.create_publisher(PointStamped, "/fruit_grasp_attempt", 10)

    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.point.x = float(position_xyz[0])
    msg.point.y = float(position_xyz[1])
    msg.point.z = float(position_xyz[2])
    node._grasp_attempt_pub.publish(msg)


def lock_target(node, position_xyz):
    """
    Lock vision system onto a specific target position.
    Vision will track this target instead of switching to a "better" one.
    Call this when starting approach to a date.
    """
    if not hasattr(node, "_target_lock_pub"):
        node._target_lock_pub = node.create_publisher(PointStamped, "/target_lock", 10)

    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.point.x = float(position_xyz[0])
    msg.point.y = float(position_xyz[1])
    msg.point.z = float(position_xyz[2])
    node._target_lock_pub.publish(msg)
    print(f"🔒 Target lock sent: [{position_xyz[0]:.3f}, {position_xyz[1]:.3f}, {position_xyz[2]:.3f}]")


def unlock_target(node):
    """
    Release target lock, allowing vision to select the best target again.
    Call this after grasp attempt completes (success or final failure).
    """
    if not hasattr(node, "_target_lock_pub"):
        node._target_lock_pub = node.create_publisher(PointStamped, "/target_lock", 10)

    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.point.x = msg.point.y = msg.point.z = 0.0  # Zero = unlock signal
    node._target_lock_pub.publish(msg)
    print("🔓 Target lock released")


def quat_dot(q1, q2):
    """Dot product of two quaternions (measures similarity)."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]


def quat_flip_z(q):
    """Rotate quaternion by 180° around Z axis (flip gripper orientation)."""
    # q_z180 = (0, 0, 0, 1) represents 180° rotation around Z
    # q_new = q * q_z180
    w, x, y, z = q
    # Multiply by (0, 0, 0, 1):
    return [-z, y, -x, w]


def quat_slerp(q0, q1, t):
    """
    Spherical linear interpolation between quaternions.
    t=0 returns q0, t=1 returns q1.
    """
    dot = quat_dot(q0, q1)

    # If dot < 0, negate one quat to take shorter path
    if dot < 0:
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]]
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = [q0[i] + t * (q1[i] - q0[i]) for i in range(4)]
        # Normalize
        n = math.sqrt(sum(x*x for x in result))
        return [x/n for x in result]

    # Standard slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return [s0 * q0[i] + s1 * q1[i] for i in range(4)]


def minimize_rotation_orientation(current_quat, target_quat, blend_weight=0.25):
    """
    Blend current and target orientation, prioritizing current.

    Since gripper can grasp from either direction (180° apart),
    first pick the closer one, then blend towards current.

    Args:
        current_quat: [qw, qx, qy, qz] current end-effector orientation
        target_quat: [qw, qx, qy, qz] target orientation from vision
        blend_weight: 0.0 = keep current, 1.0 = use target fully (default 0.3)

    Returns:
        [qw, qx, qy, qz] blended orientation
    """
    if current_quat is None or target_quat is None:
        return target_quat

    # Compute similarity (dot product) with target
    dot_orig = abs(quat_dot(current_quat, target_quat))

    # Compute flipped version (180° around Z)
    flipped = quat_flip_z(target_quat)
    dot_flip = abs(quat_dot(current_quat, flipped))

    # Higher dot = more similar = less rotation needed
    if dot_flip > dot_orig:
        best_target = flipped
    else:
        best_target = list(target_quat)

    # Blend: slerp from current towards best_target
    # blend_weight=0.3 means 70% current, 30% target
    return quat_slerp(list(current_quat), best_target, blend_weight)


def _jacobian_ik(node, target_xyz, start_js, max_iters=10, tol=0.003):
    """Solve position-only IK via damped least-squares Jacobian.

    For small displacements (~15cm), converges in 2-3 iterations.
    Guaranteed to stay near current joint configuration.

    Returns: goal joint positions (list) or None
    """
    from .fk import _get_kin_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = _get_kin_model(node)
    q = np.array(start_js, dtype=np.float64)
    target = np.array(target_xyz, dtype=np.float64)
    eps = 1e-4
    damping = 1e-3

    for iteration in range(max_iters):
        q_t = torch.tensor([q.tolist()], dtype=torch.float32, device=device)
        with torch.no_grad():
            ee_pos, _, _, _, _, _, _ = kin.forward(q_t)
        current_xyz = ee_pos[0].cpu().numpy()

        error = target - current_xyz
        if np.linalg.norm(error) < tol:
            return q.tolist()

        # Numerical Jacobian (3x6)
        J = np.zeros((3, len(q)))
        for j in range(len(q)):
            q_pert = q.copy()
            q_pert[j] += eps
            q_pt = torch.tensor([q_pert.tolist()], dtype=torch.float32, device=device)
            with torch.no_grad():
                ee_pert, _, _, _, _, _, _ = kin.forward(q_pt)
            J[:, j] = (ee_pert[0].cpu().numpy() - current_xyz) / eps

        # Damped least-squares: dq = J^T (J J^T + λ²I)^-1 * error
        JJT = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)
        dq = np.clip(dq, -0.1, 0.1)
        q = q + dq

    # Check final error after max iterations
    q_t = torch.tensor([q.tolist()], dtype=torch.float32, device=device)
    with torch.no_grad():
        ee_pos, _, _, _, _, _, _ = kin.forward(q_t)
    final_err = np.linalg.norm(target - ee_pos[0].cpu().numpy())
    if final_err < 0.01:
        return q.tolist()
    return None


def _direct_ik_move(node, target_pose_list, label="FINAL", motion_type="final",
                    store_trajectory=False, num_steps=30):
    """Use Jacobian IK to find goal joints, then interpolate directly.
    Bypasses trajectory optimization — goes straight to the IK solution."""
    if node.current_joint_positions is None:
        node.get_logger().warn(f"[DIRECT] {label}: no joint state"); return False

    start_js = list(node.current_joint_positions)

    # 1) Jacobian IK — position-only, stays near current joints
    best_js = _jacobian_ik(node, target_pose_list[:3], start_js)
    if best_js is None:
        node.get_logger().warn(f"[DIRECT] {label}: Jacobian IK failed"); return False

    best_delta = max(abs(g - c) for g, c in zip(best_js, start_js))
    node.get_logger().info(
        f"[DIRECT] {label}: max_joint_delta={best_delta*57.3:.1f}deg, {num_steps}-step interpolation"
    )

    # Safety: if IK solution is too far, fall back
    if best_delta > 1.05:  # ~60 degrees
        node.get_logger().warn(f"[DIRECT] {label}: IK too far ({best_delta*57.3:.1f}deg)"); return False

    # 3) Linearly interpolate in joint space
    states = []
    for i in range(num_steps + 1):
        t = i / num_steps
        wp = [s + t * (g - s) for s, g in zip(start_js, best_js)]
        states.append(wp)

    # 4) Build slow, smooth trajectory
    planner = node.cfg.planner
    base_dt = getattr(planner, "base_dt", 0.02)
    global_scale = max(getattr(node, "speed_scale", 1.0), 1e-6)
    type_scale = getattr(planner, f"speed_{motion_type}", getattr(planner, "speed_final", 1.0))
    scale = global_scale * type_scale
    dt = min(max(base_dt / max(scale, 1e-6), 0.012), 0.05)
    vel = min(0.05 * scale, 0.15)

    traj = build_trajectory(node.joint_order, states, vel=vel, dt=dt,
                            stop_flag=lambda: node.stop_requested,
                            max_vel=planner.max_joint_velocity * 0.5,
                            max_acc=planner.max_joint_acceleration * 0.3,
                            ramp_points=planner.ramp_points * 2)
    if node.stop_requested:
        node.stop_requested = False; return False

    # 5) Visualize & send
    cart_path = forward_kinematics_batch(node, states)
    publish_planned_path(node, states, label, cartesian_points=cart_path)
    node.trajectory_pub.publish(traj)

    # Wait for motion to finish, then blend to avoid abrupt stop
    wait_until_xyz(node, target_pose_list[:3], tol=0.015, timeout=8.0)
    blend_motion(node)

    if store_trajectory:
        if not hasattr(node, 'stored_trajectory_states'):
            node.stored_trajectory_states = []
        node.stored_trajectory_states.extend(states)

    return True


def plan_and_send(node, start_state, goal_pose: Pose, label: str, motion_type: str = "default", goal_xyz: list = None, store_trajectory: bool = False) -> bool:
    # Take voxel obstacle snapshot before planning (uses latest depth from date_v1.9.py)
    # Exclude points near goal so the fruit doesn't become an obstacle
    # if hasattr(node, 'voxel_obstacles') and node.voxel_obstacles is not None:
    #     node.voxel_obstacles.snapshot(exclude_xyz=goal_xyz, exclude_radius=0.10)

    # 1) Plan with cuRobo
    plan_cfg = PLAN_CFG_DEFAULT
    res = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    if not res.success:
        node.get_logger().warn(f"Plan failed for {label}.")
        return False

    states = interpolated_positions(res)

    # 1b) Verify trajectory against latest depth data before execution
    if (VOXEL_CONFIG.get("verify_before_execute", True) and
        hasattr(node, 'voxel_obstacles') and node.voxel_obstacles is not None):

        # Use provided goal_xyz for exclusion zone (more reliable than extracting from Pose)
        goal_position = goal_xyz

        max_attempts = VOXEL_CONFIG.get("max_replan_attempts", 2)
        for attempt in range(max_attempts):
            is_safe, collision_idx = node.voxel_obstacles.verify_trajectory_collision(
                states,
                exclude_position=goal_position,  # Skip collision check near target
            )

            if is_safe:
                break

            node.get_logger().warn(
                f"Collision detected at waypoint {collision_idx}/{len(states)} for {label} "
                f"(attempt {attempt + 1}/{max_attempts})"
            )

            # Replan with updated obstacles (snapshot already taken in verify)
            res = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
            if not res.success:
                node.get_logger().error(f"Replan failed for {label}")
                return False
            states = interpolated_positions(res)
        else:
            # All replan attempts failed
            node.get_logger().error(f"Collision persists after {max_attempts} replans for {label}")
            return False

    # 1c) Path-length sanity check using subsampled single-waypoint FK (batch=1, planner-safe)
    def _sampled_path_len(waypoints, step=20):
        """Compute approximate Cartesian path length by FK on every `step`-th waypoint."""
        indices = list(range(0, len(waypoints), step))
        if indices[-1] != len(waypoints) - 1:
            indices.append(len(waypoints) - 1)
        pts = []
        for idx in indices:
            p = forward_kinematics(node, waypoints[idx])
            if p is None:
                return None, None  # FK failed
            pts.append(p)
        total = 0.0
        for i in range(1, len(pts)):
            total += math.sqrt((pts[i].x - pts[i-1].x)**2 + (pts[i].y - pts[i-1].y)**2 + (pts[i].z - pts[i-1].z)**2)
        straight = math.sqrt((pts[-1].x - pts[0].x)**2 + (pts[-1].y - pts[0].y)**2 + (pts[-1].z - pts[0].z)**2)
        return total, straight

    if goal_xyz is not None and len(states) >= 2:
        path_len, straight = _sampled_path_len(states)
        if path_len is not None and straight is not None:
            ratio = path_len / max(straight, 0.001)
            node.get_logger().info(
                f"[PATH] {label}: path_len={path_len*100:.1f}cm, straight={straight*100:.1f}cm, ratio={ratio:.1f}x"
            )
            # If path is >3x the straight-line distance, try replanning up to 2 more times
            if ratio > 3.0 and straight > 0.02:
                best_states = states
                best_path_len = path_len
                for retry in range(2):
                    res2 = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
                    if not res2.success:
                        continue
                    s2 = interpolated_positions(res2)
                    pl2, _ = _sampled_path_len(s2)
                    if pl2 is None:
                        continue
                    node.get_logger().info(
                        f"[PATH] {label} retry {retry+1}: path_len={pl2*100:.1f}cm ({pl2/max(straight,0.001):.1f}x)"
                    )
                    if pl2 < best_path_len:
                        best_path_len = pl2
                        best_states = s2
                if best_path_len < path_len:
                    node.get_logger().info(
                        f"[PATH] {label}: picked shorter path {best_path_len*100:.1f}cm (was {path_len*100:.1f}cm)"
                    )
                states = best_states

    # 1d) Batched FK + trim overshoot (single GPU call — done AFTER all replanning)
    cart_path = forward_kinematics_batch(node, states)

    if goal_xyz is not None and cart_path:
        # Find closest-to-goal waypoint in the last 40% of trajectory
        n = len(cart_path)
        search_start = max(0, int(n * 0.6))
        min_dist = float('inf')
        trim_idx = n - 1
        for i in range(search_start, n):
            p = cart_path[i]
            d = math.sqrt((p.x - goal_xyz[0])**2 + (p.y - goal_xyz[1])**2 + (p.z - goal_xyz[2])**2)
            if d < min_dist:
                min_dist = d
                trim_idx = i
        # Only trim if the closest point is actually near the goal (<5cm)
        # and there are overshoot waypoints after it
        original_len = len(states)
        if min_dist < 0.05 and trim_idx < original_len - 2:
            states = states[:trim_idx + 1]
            cart_path = cart_path[:trim_idx + 1]
            node.get_logger().info(
                f"[TRIM] {label}: trimmed {original_len} → {len(states)} waypoints "
                f"(removed {original_len - len(states)} overshoot, closest dist={min_dist*100:.1f}cm)"
            )

    node._planned_cartesian_path = cart_path
    node._planned_label = label

    # 1d) Visualize planned path in RViz (blue line) — reuse pre-computed points
    publish_planned_path(node, states, label, cartesian_points=cart_path)

    # 2) Speed scaling
    #    - global scalar from node.speed_scale
    #    - per-motion scalar from cfg.planner
    base_dt = getattr(node.cfg.planner, "base_dt", 0.02)   # e.g. 0.02 s
    planner = node.cfg.planner

    global_scale = max(getattr(node, "speed_scale", 1.0), 1e-6)

    if motion_type == "approach":
        type_scale = getattr(planner, "speed_approach", 1.0)
    elif motion_type == "final":
        type_scale = getattr(planner, "speed_final", 1.0)
    elif motion_type in ["home", "dropoff", "predropoff"]:
        type_scale = getattr(planner, f"speed_{motion_type}", 1.0)
    else:
        type_scale = getattr(planner, "speed_home", 1.0)  # default

    scale = global_scale * type_scale

    # 3) UR10e-friendly dt and velocity
    #    - dt too small => jerk
    #    - keep dt in [12 ms, 30 ms]
    raw_dt = base_dt / max(scale, 1e-6)
    dt = min(max(raw_dt, 0.012), 0.03)

    # Velocity: linear scaling with cap
    base_vel = 0.08        # slightly gentler than 0.1
    vel = base_vel * scale
    vel = min(vel, 0.25)   # hard cap for safety

    # 4) Build trajectory
    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
    )

    if node.stop_requested:
        node.get_logger().warn(f"Stop before sending {label} trajectory.")
        node.stop_requested = False
        return False

    node.trajectory_pub.publish(traj)

    # Store trajectory for later reversal if requested
    if store_trajectory:
        if not hasattr(node, 'stored_trajectory_states'):
            node.stored_trajectory_states = []
        node.stored_trajectory_states.extend(states)

    return True


def execute_reversed_trajectory(node, motion_type: str = "predropoff"):
    """
    Execute the stored trajectory in reverse order to return along the same collision-free path.
    """
    if not hasattr(node, 'stored_trajectory_states') or not node.stored_trajectory_states:
        node.get_logger().warn("No stored trajectory to reverse; falling back to normal motion.")
        return False

    # Reverse the stored states
    reversed_states = list(reversed(node.stored_trajectory_states))

    # Clear stored trajectory after use
    node.stored_trajectory_states = []

    planner = node.cfg.planner

    # First: very slow motion from current position to first reversed waypoint
    if node.current_joint_positions is not None:
        current_pos = list(node.current_joint_positions)
        first_waypoint = reversed_states[0]
        max_diff = max(abs(c - f) for c, f in zip(current_pos, first_waypoint))

        if max_diff > 0.005:  # Any significant difference - do slow initial motion
            node.get_logger().info(f"Slow initial motion to first waypoint (max_diff={max_diff:.4f} rad)")

            # Create slow trajectory from current to first waypoint
            initial_traj = build_trajectory(
                node.joint_order,
                [current_pos, first_waypoint],
                vel=0.02,   # Very slow
                dt=0.04,    # Long time steps
                stop_flag=lambda: node.stop_requested,
                max_vel=0.3,   # Very low max velocity
                max_acc=0.2,   # Very low acceleration
                ramp_points=20,  # Many ramp points
            )
            node.trajectory_pub.publish(initial_traj)

            # Wait for this slow motion to complete
            time.sleep(0.3)
            timeout_start = time.time()
            while time.time() - timeout_start < 10.0:
                if not is_robot_moving(node, velocity_threshold=0.005):
                    break
                time.sleep(0.05)
            time.sleep(0.1)  # Small settle time

    # Now execute the main reversed trajectory
    base_dt = getattr(planner, "base_dt", 0.02)
    scale = getattr(planner, "speed_predropoff", 1.0) * getattr(planner, "global_speed_multiplier", 1.0)

    dt = base_dt / max(scale, 1e-6)
    dt = min(max(dt, getattr(planner, "min_dt", 0.012)), getattr(planner, "max_dt", 0.03))

    base_vel = 0.08
    vel = min(base_vel * scale, getattr(planner, "max_traj_velocity", 0.25))

    max_acc = getattr(planner, "max_joint_acceleration", 1.0)
    ramp_pts = getattr(planner, "ramp_points", 10)

    traj = build_trajectory(
        node.joint_order,
        reversed_states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=getattr(planner, "max_joint_velocity", 2.0) * getattr(planner, "global_speed_multiplier", 1.0),
        max_acc=max_acc,
        ramp_points=ramp_pts,
    )

    if node.stop_requested:
        node.get_logger().warn("Stop before sending reversed trajectory.")
        node.stop_requested = False
        return False

    node.get_logger().info(f"Executing reversed trajectory with {len(reversed_states)} waypoints")
    node.trajectory_pub.publish(traj)
    return True


def reacquire_goal_pose(node, seed_xyz, timeout=3.0, stable_needed=3, radius=0.08, z_tolerance=0.05):
    """
    Reacquire goal pose with improved Z-axis accuracy.

    Uses median filtering and Z-specific stability checks to reduce
    depth noise from stereo camera.
    """
    stable_count = 0
    last_pose = None
    start = time.time()

    # Track Z history for median filtering (reduces depth noise)
    z_history = []
    stable_poses = []
    prev_pose_tuple = None  # Track previous to skip duplicates

    while time.time() - start < timeout:
        # Read fresh vision data, not the corrupted best_goal_xyz
        if node.latest_goal_pose:
            pose = node.latest_goal_pose[:3]  # [x, y, z]
        else:
            pose = None
        if pose is None:
            time.sleep(0.001)
            continue

        # Skip if same reading as last iteration (wait for fresh data)
        pose_tuple = tuple(pose)
        if pose_tuple == prev_pose_tuple:
            time.sleep(0.001)
            continue
        prev_pose_tuple = pose_tuple

        x, y, z = pose

        # XY radius check
        if math.hypot(x - seed_xyz[0], y - seed_xyz[1]) > radius:
            time.sleep(0.001)
            continue

        # Z tolerance check against seed (reject if too far from original)
        if abs(z - seed_xyz[2]) > z_tolerance:
            #print(f"⚠️ Reacquire Z too far from seed: {z:.3f} vs {seed_xyz[2]:.3f}")
            time.sleep(0.001)
            continue

        # Track Z history for filtering
        z_history.append(z)
        if len(z_history) > 20:
            z_history.pop(0)

        # Z outlier rejection: skip if Z deviates >8mm from median
        if len(z_history) >= 5:
            z_median = sorted(z_history)[len(z_history) // 2]
            if abs(z - z_median) > 0.008:
                time.sleep(0.001)
                continue

        # Stability check with tighter thresholds
        if last_pose is not None:
            delta = math.dist(last_pose, pose)
            z_delta = abs(z - last_pose[2])

            # Require both total delta < 2mm AND Z delta < 2mm
            if delta < 0.002 and z_delta < 0.002:
                stable_count += 1
                print(f"Stable reacquire pose: {[round(v,3) for v in pose]} (stable {stable_count}/{stable_needed})")
                stable_poses.append(pose)
            else:
                stable_count = 0
                stable_poses.clear()

        last_pose = pose

        if stable_count >= stable_needed:
            # Return median-filtered Z for accuracy
            stable_z_values = [p[2] for p in stable_poses]
            median_z = sorted(stable_z_values)[len(stable_z_values) // 2]
            return (x, y, median_z)

        time.sleep(0.001)

    # Timeout fallback: only use z_history if we got valid readings
    if z_history:
        z_median = sorted(z_history)[len(z_history) // 2]
        return (seed_xyz[0], seed_xyz[1], z_median)

    # No valid readings - return original seed (don't use corrupted best_goal_xyz)
    print("⚠️ Reacquire timeout - using original seed position")
    return seed_xyz



def subscribe_to_goal_pose(node):
    """Subscribe to /external_goal_pose and stop idle motion immediately when goal is received."""

    # Wait until robot stops before subscribing
    if is_robot_moving(node):
        node.create_timer(0.5, lambda: (not is_robot_moving(node)) and subscribe_to_goal_pose(node))
        return

    # Cancel any existing idle timer first (prevents multiple timers from stacking)
    if hasattr(node, 'idle_timer'):
        try:
            node.idle_timer.cancel()
            del node.idle_timer
        except Exception:
            pass

    # Reset all tracking state for fresh cycle
    node.reset_goal_tracking()

    # Invalidate cached goal - always wait for fresh pose from vision
    node.latest_goal_pose = None
    node.latest_goal_time = 0

    # Destroy previous subscription if it exists
    if hasattr(node, 'goal_pose_sub'):
        node.destroy_subscription(node.goal_pose_sub)
        del node.goal_pose_sub

    cur = node.get_end_effector_pose()
    current_orientation = cur[3:] if cur else [1.0, 0.0, 0.0, 0.0]
    node.goal_poses.clear()

    # No cache - always wait for fresh goal from vision
    # Track position stability before accepting
    goal_history = {"poses": [], "stable_count": 0, "accepted": False}

    def _goal_cb(msg: PoseStamped):
        nonlocal goal_history

        new_xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # ---- ALWAYS STORE LATEST GOAL POSE ----
        node.latest_goal_pose = [*new_xyz,
            msg.pose.orientation.w, msg.pose.orientation.x,
            msg.pose.orientation.y, msg.pose.orientation.z]
        node.latest_goal_time = time.time()

        # Already accepted a goal this cycle
        if goal_history["accepted"]:
            return

        # Check for position jump (vision switched to new fruit)
        if goal_history["poses"]:
            last_xyz = goal_history["poses"][-1]
            jump = math.dist(new_xyz, last_xyz)
            if jump > 0.10:  # >10cm = new fruit, reset
                print(f"📍 Position jump {jump:.2f}m - waiting for stable...")
                goal_history["poses"].clear()
                goal_history["stable_count"] = 0

        goal_history["poses"].append(new_xyz)

        # Need at least 2 consistent readings before accepting
        if len(goal_history["poses"]) >= 2:
            recent = goal_history["poses"][-2:]
            if math.dist(recent[0], recent[1]) < 0.02:  # <2cm = stable
                goal_history["stable_count"] += 1
            else:
                goal_history["stable_count"] = 0

        # Accept after 2 stable readings (or after 5 total messages as fallback)
        if goal_history["stable_count"] >= 2 or len(goal_history["poses"]) >= 5:
            goal_history["accepted"] = True
            node.goal_received = True
            print("✅ Goal position stable, accepting...")

            # Stop idle timer
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass

            publish_stop_trajectory(node)

            # Use latest stable position
            node.goal_seed_xy = [new_xyz[0], new_xyz[1]]
            node.best_goal_xyz = new_xyz
            node.best_goal_score = float("inf")

            #g = [*new_xyz, *current_orientation]
            g = [*new_xyz, 
                msg.pose.orientation.w, msg.pose.orientation.x,
                msg.pose.orientation.y, msg.pose.orientation.z]
            
            
            if not node.goal_poses.any_within_distance(g, 0.01):
                node.goal_poses.append(g)
                publish_goal_marker(node, g[:3])
                goal_type = "HIGH (back-then-forward)" if g[2] > 0.90 else "LOW (side approach)"
                print(f"🟢 Accepted goal pose: {g}")
                print(f"   → Goal type: {goal_type} (z={g[2]:.2f}m)")
                node.obstacles.update_pose("fruit_obstacle", g[:3])

            # Destroy subscription
            try:
                if hasattr(node, 'goal_pose_sub'):
                    node.destroy_subscription(node.goal_pose_sub)
                    del node.goal_pose_sub
            except Exception:
                pass

    # Subscribe to /external_goal_pose with VOLATILE QoS
    # VOLATILE = don't receive old buffered messages, only fresh ones
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    fresh_qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )
    node.goal_pose_sub = node.create_subscription(
        PoseStamped,
        '/external_goal_pose',
        _goal_cb,
        fresh_qos,
    )

    # Define idle micro-motions while waiting for goals
    sequence = [
        (0.0, 0.0, 0.2),
        (0.0, 0.1, 0.0),
        (0.0, -0.1, 0.0),
        (0.0, 0.0, -0.1)
    ]
    idx = {"i": 0}

    def _idle_cb():
        """Perform gentle idle motions until a goal is received."""
        # Guard against shutdown/stop - cancel timer and exit early
        if not getattr(node, 'running', True) or getattr(node, 'stop_requested', False):
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass
            return

        if node.goal_received or idx["i"] >= len(sequence):
            if hasattr(node, 'idle_timer'):
                node.idle_timer.cancel()
                del node.idle_timer
            return

        curp = node.get_end_effector_pose()
        if curp and not is_robot_moving(node):
            print("Performing idle micro-motion...")
            dx, dy, dz = sequence[idx['i']]
            tgt = [curp[0] + dx, curp[1] + dy, curp[2] + dz, *current_orientation]
            print(f"Idle move to: {tgt}")
            _exec(node, tgt)
            idx['i'] += 1

    # Start idle motion timer
    node.idle_timer = node.create_timer(5.0, _idle_cb)



def is_robot_moving(node, velocity_threshold: float = 0.001) -> bool:
    """Check if robot is moving based on joint velocities.

    Returns True (assume moving) if velocity data is unavailable - safer default.
    """
    v = getattr(node, 'current_joint_velocities', None)
    if v is None or len(v) == 0:
        # No velocity data available - assume robot IS moving (safer default)
        return True
    return any(abs(x) > velocity_threshold for x in v)

def blend_approach_direction(node, x, y, z, vis_ratio=1.0, z_std=0.01):
    d_vis_pt = compute_visibility_approach(node, x, y, z, dist=0.05)
    d_vis = -np.array([x - d_vis_pt[0], y - d_vis_pt[1], z - d_vis_pt[2]])
    d_vis /= np.linalg.norm(d_vis)

    prev_dir = getattr(node, "_prev_blend_dir", None)
    prev_dot = None
    d_prev = None
    fruit_dir = getattr(node, "fruit_direction", None)
    if fruit_dir is not None:
        d_dir = np.array(fruit_dir, dtype=float)
        n_dir = np.linalg.norm(d_dir)
        if n_dir > 1e-9:
            d_dir /= n_dir
        else:
            d_dir = d_vis.copy()

        # Align with previous blended direction to avoid 180 flips.
        if prev_dir is not None:
            d_prev = np.array(prev_dir, dtype=float)
            n_prev = np.linalg.norm(d_prev)
            if n_prev > 1e-9:
                d_prev /= n_prev
                prev_dot = float(np.dot(d_dir, d_prev))
                if prev_dot < 0.0:
                    d_dir = -d_dir
                    prev_dot = -prev_dot
            else:
                d_prev = None

        dir_conf = 1.0
    else:
        d_dir = d_vis
        dir_conf = 0.0

    # vis_conf based on visibility quality (higher vis_ratio + lower z_std = more confident)
    # Note: Currently disabled (vis_conf=0) to rely purely on fruit_direction from vision
    vis_conf = 0.0  # np.clip(vis_ratio * np.exp(-z_std / 0.02), 0.0, 1.0)

    d = dir_conf * d_dir + vis_conf * d_vis
    n_blend = np.linalg.norm(d)
    d_norm = d / n_blend if n_blend > 1e-9 else d_vis.copy()
    node._prev_blend_dir = d_norm.tolist()

    # Quick log to verify sign convention (throttled).
    now = time.time()
    last = getattr(node, "_last_dir_log", 0.0)
    if now - last > 1.0:
        node._last_dir_log = now
        if fruit_dir is None:
            fruit_repr = "None"
            prev_repr = "None"
            dot_repr = "n/a"
        else:
            fruit_repr = np.round(np.array(fruit_dir, dtype=float), 3)
            prev_repr = "None" if d_prev is None else np.round(d_prev, 3)
            dot_repr = "n/a" if prev_dot is None else f"{prev_dot:.3f}"
        node.get_logger().info(
            f"dir dbg: fruit_direction={fruit_repr} d_vis={np.round(d_vis, 3)} "
            f"d_dir_aligned={np.round(d_dir, 3)} prev_dir={prev_repr} prev_dot={dot_repr} "
            f"d_blend={np.round(d_norm, 3)}"
        )

    return d_norm



# Main goal-execution pipeline — runs through all saved goals and performs motion + gripper actions in sequence.
def plan_and_execute(node):
    
    if node.current_joint_positions is None:
        node.get_logger().warn("No joint state yet."); return
    if not node.goal_poses:
        node.get_logger().warn("No stored goals."); return
    if not node.robot_running:
        node.get_logger().error("Robot program OFF; may fail.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    while node.goal_poses and getattr(node, 'running', True):
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("Stop requested; aborting goal execution.")
            publish_stop_trajectory(node)
            node.goal_poses.clear()
            node.stop_requested = False
            break
        
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        
        
        # 0. Get next goal (thread-safe pop returns None if empty)
        goal = node.goal_poses.pop(0)
        if goal is None:
            node.get_logger().warn("Goal queue empty during pop; skipping.")
            continue
        x,y,z = goal[:3]
        grasp_orientation = goal[3:]  # Store grasp orientation for pre-dropoff
        yoffset = node.yoffset

        # Lock vision onto this target (prevents switching to different "best" during approach)
        lock_target(node, goal[:3])

        # Clear stored trajectory for this new goal (will be filled during approach and final)
        node.stored_trajectory_states = []

        # Direction-biased pre-grasp: use fruit direction if available
        standoff = 0.0  # 12cm standoff distance

        d_blend = blend_approach_direction(node, x, y, z)
        ax = x + d_blend[0] * standoff
        ay = y + d_blend[1] * standoff
        az = z + d_blend[2] * standoff

        # 1. Plan approach - different strategy based on height
        # Get current orientation and minimize rotation
        cur_pose = node.get_end_effector_pose()
        cur_quat = cur_pose[3:] if cur_pose else None
        target_quat = goal[3:]
        print("Current quat:", cur_quat)
        print("Target quat:", target_quat)

        if z > 1.90:  # High dates: retreat back first, then approach towards
            print(f"High date detected (z={z:.2f}m) - using back-then-forward approach")
            # Align gripper with approach direction (pointing towards fruit)
            approach_dir = [-d_blend[0], -d_blend[1], -d_blend[2]]  # Invert: gripper faces fruit
            dir_quat = quaternion_from_approach(node, direction_xyz=approach_dir)
            # Pick closer of dir_quat vs 180° flipped to minimize rotation from current
            orientation = minimize_rotation_orientation(cur_quat, dir_quat, blend_weight=1.0)
            print(f"High date orientation (dir={approach_dir}): {orientation}")

            # Phase 1: Retreat position (15cm back from fruit along d_blend direction)
            retreat_dist = 0.15
            retreat_pos = [
                x + d_blend[0] * retreat_dist,
                y + d_blend[1] * retreat_dist,
                z - 0.10,
                *orientation
            ]
            print(f"Retreat position: {retreat_pos[:3]} (d_blend={d_blend})")

            if not plan_and_send(node, start, Pose.from_list(retreat_pos),
                                 label="RETREAT", motion_type="approach",
                                 goal_xyz=retreat_pos[:3], store_trajectory=True):
                unlock_target(node)
                continue
            wait_until_xyz(node, retreat_pos[:3])
            log_path_deviation(node, "RETREAT")
            blend_motion(node)

            # Update start state for approach phase
            start = JointState.from_position(
                torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=node.joint_order,
            )

            # Phase 2: Approach towards fruit (3cm back along d_blend, same height)
            approach_dist = 0.03
            approach = [
                x + d_blend[0] * approach_dist,
                y + d_blend[1] * approach_dist,
                z,
                *orientation
            ]
            print(f"Approach position (high): {approach[:3]}")

        else:  # Lower dates: approach from below (existing logic)
            # Use blended orientation for downward approach
            orientation = minimize_rotation_orientation(cur_quat, target_quat)
            approach = [ax, ay+0.01, az-0.12, *orientation]
            print(f"Going for side approach (low): {approach[:3]}")

        if not plan_and_send(node, start, Pose.from_list(approach), label="APPROACH", motion_type="approach", goal_xyz=approach[:3], store_trajectory=True):
            unlock_target(node)
            continue
        wait_until_xyz(node, approach[:3])
        log_path_deviation(node, "APPROACH")
        blend_motion(node)
        
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        
        # 2. Reacquire
        seed = [x,y,z]
        print("Reacquiring goal pose near:", seed)
        reacq = reacquire_goal_pose(node, seed_xyz=seed)
        
        if reacq:
            x,y,z = reacq; publish_goal_marker(node, [x,y,z])
        else:
            node.get_logger().warn("No reacquire; skipping goal.")
            unlock_target(node)
            continue
            
        # 3. Final slow precise grasp — IK + direct joint interpolation (no cuRobo trajectory)
        #    _direct_ik_move handles wait + blend internally
        final_target = [x, y, z+0.02, *orientation]
        final_ok = _direct_ik_move(node, final_target, label="FINAL",
                                   motion_type="final", store_trajectory=True)
        if not final_ok:
            # Fallback: cuRobo planner
            start = JointState.from_position(
                torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=node.joint_order,
            )
            if not plan_and_send(node, start, Pose.from_list(final_target), label="FINAL",
                                 motion_type="final", goal_xyz=final_target[:3], store_trajectory=True):
                unlock_target(node)
                continue
            wait_until_xyz(node, final_target[:3])
            blend_motion(node)
        log_path_deviation(node, "FINAL")

        node.control_gripper("CLOSE"); time.sleep(0.7)
        # Notify vision system about grasp attempt for fruit tracking
        notify_grasp_attempt(node, final_target[:3])

        # Verify 3-finger contact before moving
        def check_3finger_contact():
            forces = node.gripper_controller.force_data
            threshold = node.gripper_controller.force_threshold
            finger_names = ["Finger 0 (left)", "Finger 1 (center)", "Finger 2 (right)"]
            bad_fingers = []
            for i, f in enumerate(forces):
                has_contact = abs(f) >= threshold or f <= -threshold
                if not has_contact:
                    bad_fingers.append(f"{finger_names[i]}: {f:.2f}")
            return len(bad_fingers) == 0, forces, bad_fingers

        is_proper, forces, bad_fingers = check_3finger_contact()


        if not is_proper:
            print(f"⚠️ Weak grip - no contact on: {bad_fingers}")
            node.control_gripper("OPEN"); time.sleep(0.3)

            # Move slightly closer for re-grip (~2cm, use exec_pose directly)
            cur = node.get_end_effector_pose()
            if cur:
                closer_target = [cur[0], cur[1] - 0.001, cur[2] + 0.02, *cur[3:]]
                exec_pose(node, closer_target)
                wait_until_xyz(node, closer_target[:3], tol=0.01, timeout=3.0)

            node.control_gripper("CLOSE"); time.sleep(0.7)
            is_proper, forces, bad_fingers = check_3finger_contact()
            if is_proper:
                print(f"Re-grip result: {forces} → PROPER ✅")
            else:
                print(f"Re-grip result: {forces} → STILL WEAK on: {bad_fingers}")
            
        # 4. Drop-off and return
        time.sleep(0.5)

        # Pre-dropoff: reverse the approach trajectory (reuses the collision-free path)
        # Flow: grasp → reverse(final) → reverse(approach) → home → dropoff → home
        if execute_reversed_trajectory(node, motion_type="predropoff"):
            # Wait for reversed trajectory to complete
            time.sleep(0.5)  # Initial delay for trajectory to start
            timeout_start = time.time()
            while time.time() - timeout_start < 15.0:  # 15s max timeout
                if not is_robot_moving(node, velocity_threshold=0.005):
                    break
                time.sleep(0.1)
            print("Reversed trajectory completed - returned along collision-free path.")
            blend_motion(node)
        else:
            # Fallback: go directly to predropoff if no stored trajectory
            node.get_logger().warn("No stored trajectory; using predropoff position.")
            move_to_predropoff_position(node)
            blend_motion(node)

        # Go to home first (ensures clean position before dropoff)
        time.sleep(0.1)
        move_to_home_position(node)
        time.sleep(0.1)

        # Now go to dropoff
        move_to_dropoff_position(node)
        time.sleep(0.2)  # small delay to allow state update
        node.control_gripper("OPEN")
        move_to_home_position(node)

        # Release target lock and reset tracking state for next goal
        unlock_target(node)
        node.reset_goal_tracking()

        if len(node.goal_poses) == 0:
            # Cancel idle timer when cycle completes
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass
            node.get_logger().info("All goals completed; returned HOME.")
        else:
            node.get_logger().info("Preparing for next goal...")


def quaternion_from_approach(node, direction_xyz=None, pitch_deg=None, world=False):
    import numpy as np, math
    if pitch_deg is not None:
        cur = node.get_end_effector_pose()
        if cur: qw, qx, qy, qz = cur[3], cur[4], cur[5], cur[6]
        else:   qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        half = math.radians(pitch_deg)/2.0
        cw, cx, cy, cz = math.cos(half), math.sin(half), 0.0, 0.0
        if world:
            nw = cw*qw - cx*qx - cy*qy - cz*qz
            nx = cw*qx + cx*qw + cy*qz - cz*qy
            ny = cw*qy - cx*qz + cy*qw + cz*qx
            nz = cw*qz + cx*qy - cy*qx + cz*qw
        else:
            nw = qw*cw - qx*cx - qy*cy - qz*cz
            nx = qw*cx + qx*cw + qy*cz - qz*cy
            ny = qw*cy - qx*cz + qy*cw + qz*cx
            nz = qw*cz + qx*cy - qy*cx + qz*cw
        return [nw, nx, ny, nz]
    d = np.array(direction_xyz or (0.0, 0.0, -1.0), dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9: return [1.0, 0.0, 0.0, 0.0]
    d /= n
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(up, d); x_axis = x_axis/np.linalg.norm(x_axis) if np.linalg.norm(x_axis) >= 1e-6 else np.array([1.0,0.0,0.0])
    y_axis = np.cross(d, x_axis); y_axis /= np.linalg.norm(y_axis)
    z_axis = d
    Rm = np.array([[x_axis[0], y_axis[0], z_axis[0]],[x_axis[1], y_axis[1], z_axis[1]],[x_axis[2], y_axis[2], z_axis[2]]], dtype=float)
    t = Rm[0,0] + Rm[1,1] + Rm[2,2]
    if t>0.0:
        s = math.sqrt(t+1.0)*2.0; qw = 0.25*s
        qx = (Rm[2,1]-Rm[1,2])/s; qy = (Rm[0,2]-Rm[2,0])/s; qz = (Rm[1,0]-Rm[0,1])/s
    else:
        if Rm[0,0]>Rm[1,1] and Rm[0,0]>Rm[2,2]:
            s = math.sqrt(1.0+Rm[0,0]-Rm[1,1]-Rm[2,2])*2.0; qw = (Rm[2,1]-Rm[1,2])/s; qx = 0.25*s
            qy = (Rm[0,1]+Rm[1,0])/s; qz = (Rm[0,2]+Rm[2,0])/s
        elif Rm[1,1]>Rm[2,2]:
            s = math.sqrt(1.0+Rm[1,1]-Rm[0,0]-Rm[2,2])*2.0; qw = (Rm[0,2]-Rm[2,0])/s
            qx = (Rm[0,1]+Rm[1,0])/s; qy = 0.25*s; qz = (Rm[1,2]+Rm[2,1])/s
        else:
            s = math.sqrt(1.0+Rm[2,2]-Rm[0,0]-Rm[1,1])*2.0; qw = (Rm[1,0]-Rm[0,1])/s
            qx = (Rm[0,2]+Rm[2,0])/s; qy = (Rm[1,2]+Rm[2,1])/s; qz = 0.25*s
    return [qw, qx, qy, qz]
