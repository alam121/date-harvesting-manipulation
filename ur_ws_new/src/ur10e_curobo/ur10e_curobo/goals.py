# ruff: noqa
import time, math, torch
import threading
import numpy as np
from geometry_msgs.msg import Pose as ROSPose, PoseStamped, PointStamped
from curobo.types.math import Pose
from curobo.types.robot import JointState


def try_cuda_recovery(node) -> bool:
    """Attempt to recover from a CUDA fault by resetting the device state.
    Returns True if CUDA is usable again, False if still broken."""
    if not getattr(node, "_cuda_faulted", False):
        return True
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Test with a small tensor operation
        t = torch.tensor([1.0], device="cuda")
        _ = t + t
        del t
        torch.cuda.empty_cache()
        node._cuda_faulted = False
        node.get_logger().info("CUDA recovery successful — GPU planning re-enabled")
        return True
    except Exception as e:
        node.get_logger().warn(f"CUDA recovery failed: {e}")
        return False


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

    def __iter__(self):
        with self._lock:
            return iter(list(self._goals))

    def any_within_distance(self, pos, threshold):
        """Check if any goal is within threshold distance of pos[:3]."""
        with self._lock:
            return any(math.dist(pos[:3], e[:3]) < threshold for e in self._goals)

    def peek(self, index=0):
        """Return goal at index without removing it, or None if out of range."""
        with self._lock:
            if index < len(self._goals):
                return list(self._goals[index])
            return None

    def sort(self, key=None, reverse=False):
        """Sort goals in place with optional key function."""
        with self._lock:
            self._goals.sort(key=key, reverse=reverse)
from .motions import execute_single_pose as _exec
from .motions import publish_stop_trajectory
from .config import LOW_Z_THRESH, LATERAL_THRESH, PLAN_CFG_DEFAULT, VOXEL_CONFIG
from .utils import build_trajectory, wait_until_xyz
from .markers import publish_goal_marker, publish_planned_path, clear_path_markers
from .motions import interpolated_positions, get_curobo_dt, execute_single_pose
from .motions import execute_single_pose as exec_pose
from .motions import rotate_wrist, move_to_predropoff_position, move_to_dropoff_position, move_to_home_position
from .motions import blend_motion, preplan_js, plan_execute_js
from .dynamic_obstacle import DynamicObstacleManager
from .utils import compute_visibility_approach
from .fk import forward_kinematics, forward_kinematics_batch, solve_ik_fast
from .grasp_learner import GraspRecord
from . import gripper as gripper_mod
from . import markers as markers_mod

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
    node.get_logger().info(f"Target lock sent: [{position_xyz[0]:.3f}, {position_xyz[1]:.3f}, {position_xyz[2]:.3f}]")


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
    node.get_logger().info("Target lock released")


def quat_multiply(a, b):
    """Multiply two quaternions [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ]


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


def minimize_rotation_orientation(current_quat, target_quat, blend_weight=0.50):
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

    try:
        use_cuda = torch.cuda.is_available() and not getattr(node, "_cuda_faulted", False)
        device = torch.device("cuda" if use_cuda else "cpu")
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

        # Check final error after max iterations (allow 2x convergence tol as fallback)
        q_t = torch.tensor([q.tolist()], dtype=torch.float32, device=device)
        with torch.no_grad():
            ee_pos, _, _, _, _, _, _ = kin.forward(q_t)
        final_err = np.linalg.norm(target - ee_pos[0].cpu().numpy())
        if final_err < tol * 2:
            return q.tolist()
        return None
    except Exception as e:
        msg = str(e)
        if "CUDA error" in msg or "illegal memory access" in msg:
            node._cuda_faulted = True
            try_cuda_recovery(node)
        node.get_logger().warn(f"[DIRECT] Jacobian IK failed: {e}")
        return None


def _direct_ik_move(node, target_pose_list, label="FINAL", motion_type="final",
                    store_trajectory=True, num_steps=None):
    """Use Jacobian IK to find goal joints, then interpolate directly.
    Bypasses trajectory optimization — goes straight to the IK solution."""
    if node.current_joint_positions is None:
        node.get_logger().warn(f"[DIRECT] {label}: no joint state"); return False

    start_js = list(node.current_joint_positions)

    # 1) cuRobo native IK (position + orientation), fallback to Jacobian IK (position-only)
    best_js = None
    if getattr(node, "_cuda_faulted", False):
        node.get_logger().warn(f"[DIRECT] {label}: skipping cuRobo IK (CUDA previously faulted)")
    else:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # goal_pose: position [1,3], quaternion [1,4]
            pos = torch.tensor([target_pose_list[:3]], dtype=torch.float32, device=device)
            quat = torch.tensor([target_pose_list[3:]], dtype=torch.float32, device=device)
            goal_pose = Pose(position=pos, quaternion=quat)
            # seed_config: (n_seeds, 1, dof)
            seed = torch.tensor([start_js], dtype=torch.float32, device=device).unsqueeze(0)
            # retract_config: (1, dof) — pull solution towards current joints
            retract = torch.tensor([start_js], dtype=torch.float32, device=device)
            ik_result = node.motion_gen.ik_solver.solve_single(
                goal_pose, seed_config=seed, retract_config=retract
            )
            if ik_result.success.item():
                best_js = ik_result.js_solution.position.squeeze().cpu().tolist()
                node.get_logger().info(f"[DIRECT] {label}: cuRobo IK solved (pos_err={ik_result.position_error.item():.4f}, rot_err={ik_result.rotation_error.item():.4f})")
        except Exception as e:
            msg = str(e)
            if "CUDA error" in msg or "illegal memory access" in msg:
                node._cuda_faulted = True
                try_cuda_recovery(node)
            node.get_logger().info(f"[DIRECT] {label}: cuRobo IK exception: {e}")
    if best_js is None:
        node.get_logger().info(f"[DIRECT] {label}: cuRobo IK failed, trying Jacobian fallback")
        try:
            best_js = _jacobian_ik(node, target_pose_list[:3], start_js)
        except Exception as e:
            node.get_logger().warn(f"[DIRECT] {label}: Jacobian fallback exception: {e}")
            best_js = None
    if best_js is None:
        node.get_logger().warn(f"[DIRECT] {label}: all IK solvers failed"); return False

    best_delta = max(abs(g - c) for g, c in zip(best_js, start_js))

    # Scale num_steps with Cartesian distance (1 step per 5mm, clamped 10-60)
    if num_steps is None:
        cur_pose = node.get_end_effector_pose()
        if cur_pose:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(cur_pose[:3], target_pose_list[:3])))
            num_steps = max(10, min(60, int(dist / 0.005)))
        else:
            num_steps = 30

    node.get_logger().info(
        f"[DIRECT] {label}: max_joint_delta={best_delta*57.3:.1f}deg, {num_steps}-step interpolation"
    )

    # Safety: if IK solution is too far, fall back
    if best_delta > 1.05:  # ~60 degrees
        node.get_logger().warn(f"[DIRECT] {label}: IK too far ({best_delta*57.3:.1f}deg)"); return False

    # 3) S-curve (cosine) interpolation in joint space — avoids velocity discontinuities
    states = []
    for i in range(num_steps + 1):
        alpha = i / num_steps
        t_smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
        wp = [s + t_smooth * (g - s) for s, g in zip(start_js, best_js)]
        states.append(wp)

    # 4) Build slow, smooth trajectory
    planner = node.cfg.planner
    base_dt = getattr(planner, "base_dt", 0.02)
    global_scale = max(getattr(node, "speed_scale", 1.0), 1e-6)
    type_scale = getattr(planner, f"speed_{motion_type}", getattr(planner, "speed_final", 1.0))
    scale = global_scale * type_scale
    dt = min(max(base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    vel = min(0.05 * scale, 0.15)

    traj = build_trajectory(node.joint_order, states, vel=vel, dt=dt,
                            stop_flag=lambda: node.stop_requested,
                            max_vel=planner.max_joint_velocity * 0.5,
                            max_acc=planner.max_joint_acceleration * 0.3,
                            ramp_points=0)
    if node.stop_requested:
        node.stop_requested = False; return False

    # 5) Visualize & send
    cart_path = forward_kinematics_batch(node, states)
    publish_planned_path(node, states, label, cartesian_points=cart_path)
    node.trajectory_pub.publish(traj)

    # Wait for motion to finish (with orientation + velocity checks), then blend
    target_quat = target_pose_list[3:] if len(target_pose_list) > 3 else None
    wait_until_xyz(node, target_pose_list[:3], tol=0.008, timeout=8.0, target_quat=target_quat)
    blend_motion(node)

    # Reject stall-acceptance far from target — prevents gripper close from wrong position
    cur_pose = node.get_end_effector_pose()
    if cur_pose:
        final_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(cur_pose[:3], target_pose_list[:3])))
        if final_dist > 0.015:
            node.get_logger().warn(
                f"[DIRECT] {label}: stalled {final_dist*100:.1f}cm from target — treating as failure")
            return False

    if store_trajectory:
        if not hasattr(node, 'stored_trajectory_states'):
            node.stored_trajectory_states = []
        node.stored_trajectory_states.extend(states)

    return True


def plan_and_send(node, start_state, goal_pose: Pose, label: str, motion_type: str = "default", goal_xyz: list = None, store_trajectory: bool = False) -> bool:

    # 1) Plan with cuRobo
    plan_cfg = PLAN_CFG_DEFAULT
    lock = getattr(node, '_planning_lock', None)
    if lock: lock.acquire()
    try:
        res = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    finally:
        if lock: lock.release()
    if not res.success:
        status = getattr(res, 'status', 'unknown')
        node.get_logger().warn(f"Plan failed for {label}. status={status}")
        # Flush any deferred async CUDA errors before the next planning call
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        return False

    states = interpolated_positions(res)
    curobo_dt = get_curobo_dt(res)

    # Safety: reject trajectories with joint wraparound (>300 deg on any joint), retry up to 3x
    for _plan_attempt in range(3):
        wraparound = False
        if len(states) > 1:
            for j in range(len(states[0])):
                total_rot = sum(abs(states[i+1][j] - states[i][j]) for i in range(len(states)-1))
                if total_rot > math.radians(300):
                    node.get_logger().warn(
                        f"Joint {j} wraparound: {math.degrees(total_rot):.0f}deg for {label} — replanning ({_plan_attempt+1}/3)"
                    )
                    wraparound = True
                    break
        if not wraparound:
            break
        if lock: lock.acquire()
        try:
            res = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
        finally:
            if lock: lock.release()
        if not res.success:
            node.get_logger().warn(f"Replan failed for {label}")
            return False
        states = interpolated_positions(res)
        curobo_dt = get_curobo_dt(res)
    else:
        node.get_logger().warn(f"All replans had joint wraparound for {label} — rejecting")
        return False

    # Early return if trajectory is trivial (already at goal)
    if len(states) <= 2:
        node.get_logger().info(f"{label}: already at goal ({len(states)} waypoints), skipping motion")
        if store_trajectory:
            if not hasattr(node, 'stored_trajectory_states'):
                node.stored_trajectory_states = []
            node.stored_trajectory_states.extend(states)
        return True

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
            if lock: lock.acquire()
            try:
                res = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
            finally:
                if lock: lock.release()
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
            # If path is roundabout (>1.8x straight line), try 1 more plan and pick shorter
            best_states = states
            best_path_len = path_len
            if ratio > 1.8 and straight > 0.02:
                if lock: lock.acquire()
                try:
                    res2 = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
                finally:
                    if lock: lock.release()
                if res2.success:
                    s2 = interpolated_positions(res2)
                    skip = False
                    for j in range(len(s2[0])):
                        tot = sum(abs(s2[i+1][j] - s2[i][j]) for i in range(len(s2)-1))
                        if tot > math.radians(300):
                            skip = True
                            break
                    if not skip:
                        pl2, _ = _sampled_path_len(s2)
                        if pl2 is not None:
                            node.get_logger().info(
                                f"[PATH] {label} replan: path_len={pl2*100:.1f}cm ({pl2/max(straight,0.001):.1f}x)"
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
        # Only trim if the closest point is actually near the goal (<5cm),
        # there are overshoot waypoints after it, and we retain enough waypoints
        original_len = len(states)
        min_keep = max(10, int(original_len * 0.2))  # keep at least 20% or 10 waypoints
        if min_dist < 0.05 and trim_idx < original_len - 2 and (trim_idx + 1) >= min_keep:
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
    raw_dt = base_dt / max(scale, 1e-6)
    # Never go faster than cuRobo's own interpolation timing: the positions were
    # planned for curobo_dt intervals; compressing them produces velocities and
    # accelerations that exceed what the robot can physically follow → jerks.
    # Scale > 1 ("go faster") has no effect — cuRobo already runs at max speed.
    dt = min(max(raw_dt, curobo_dt), planner.max_dt)

    # Velocity: linear scaling with cap
    base_vel = 0.08        # slightly gentler than 0.1
    vel = base_vel * scale
    vel = min(vel, 0.25)   # hard cap for safety

    # 4) Build trajectory — constant speed, no ramping
    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=0,
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

    # Save for pre-planning dropoff from the end position
    node._last_reversed_states = reversed_states

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
                ramp_points=0,
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

    traj = build_trajectory(
        node.joint_order,
        reversed_states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=getattr(planner, "max_joint_velocity", 2.0) * getattr(planner, "global_speed_multiplier", 1.0),
        max_acc=getattr(planner, "max_joint_acceleration", 1.0),
        ramp_points=0,
    )

    if node.stop_requested:
        node.get_logger().warn("Stop before sending reversed trajectory.")
        node.stop_requested = False
        return False

    node.get_logger().info(f"Executing reversed trajectory with {len(reversed_states)} waypoints")
    node.trajectory_pub.publish(traj)
    return True


def execute_partial_reverse(node, clearance_m: float = 0.28):
    """
    Reverse only enough of the stored trajectory to pull back `clearance_m` from
    the grasp position, then stop. This clears the date bunch so cuRobo can plan
    directly to dropoff without hitting fruit.
    """
    if not hasattr(node, 'stored_trajectory_states') or not node.stored_trajectory_states:
        node.get_logger().warn("No stored trajectory for partial reverse.")
        return False

    stored = node.stored_trajectory_states
    reversed_states = list(reversed(stored))
    node.stored_trajectory_states = []

    # Use FK to find how many waypoints = clearance_m of Cartesian distance
    grasp_fk = forward_kinematics(node, reversed_states[0])
    if not grasp_fk:
        node.get_logger().warn("FK failed for partial reverse; using full reverse.")
        partial = reversed_states
    else:
        gx, gy, gz = grasp_fk.x, grasp_fk.y, grasp_fk.z
        partial = [reversed_states[0]]
        for wp in reversed_states[1:]:
            partial.append(wp)
            fk = forward_kinematics(node, wp)
            if fk:
                dist = math.sqrt((fk.x - gx)**2 + (fk.y - gy)**2 + (fk.z - gz)**2)
                if dist >= clearance_m:
                    break

    node.get_logger().info(
        f"Partial reverse: {len(partial)}/{len(reversed_states)} waypoints "
        f"({clearance_m*100:.0f}cm clearance)"
    )

    planner = node.cfg.planner
    base_dt = getattr(planner, "base_dt", 0.02)
    scale = getattr(planner, "speed_predropoff", 1.0) * getattr(planner, "global_speed_multiplier", 1.0)
    dt = base_dt / max(scale, 1e-6)
    dt = min(max(dt, getattr(planner, "min_dt", 0.012)), getattr(planner, "max_dt", 0.03))
    base_vel = 0.08
    vel = min(base_vel * scale, getattr(planner, "max_traj_velocity", 0.25))

    traj = build_trajectory(
        node.joint_order,
        partial,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=getattr(planner, "max_joint_velocity", 2.0) * getattr(planner, "global_speed_multiplier", 1.0),
        max_acc=getattr(planner, "max_joint_acceleration", 1.0),
        ramp_points=0,
    )

    node.trajectory_pub.publish(traj)

    # Wait for partial reverse to complete
    time.sleep(0.3)
    timeout_start = time.time()
    while time.time() - timeout_start < 10.0:
        if not is_robot_moving(node, velocity_threshold=0.005):
            break
        time.sleep(0.1)

    blend_motion(node)
    return True


def reacquire_goal_pose(node, seed_xyz, candidate_seeds=None, timeout=5.5, radius=0.08, z_tolerance=0.05):
    """Fast reacquire across multiple candidate seeds.

    Checks vision against all candidates. Returns first stable match.
    With multiple candidates, accepts after just 1 stable reading.

    Args:
        seed_xyz: primary seed [x,y,z]
        candidate_seeds: list of [x,y,z,...] alternate candidates (optional)
        timeout: max wait time (default 1.5s, reduced from 3s)
    """
    # Build seed list: primary first, then candidates
    seeds = [seed_xyz[:3]]
    if candidate_seeds:
        for c in candidate_seeds:
            xyz = c[:3] if len(c) > 3 else c
            if all(math.dist(xyz, s) > 0.03 for s in seeds):
                seeds.append(list(xyz))

    multi = len(seeds) > 1
    stable_needed = 1 if multi else 2

    last_pose = None
    stable_count = 0
    matched_seed = None
    start = time.time()
    prev_pose_tuple = None

    while time.time() - start < timeout:
        # Check best fruit first, then fall back to all visible fruits.
        # This allows reacquire to find the target even when it isn't best-ranked
        # (e.g. another fruit is closer during the approach phase).
        pose = None
        if node.latest_goal_pose:
            pose = node.latest_goal_pose[:3]

        # If best fruit doesn't match any seed, scan all visible fruits
        candidate_poses = [pose] if pose is not None else []
        all_visible = getattr(node, 'all_fruit_poses', [])
        for fp in all_visible:
            if not any(fp == p for p in candidate_poses):
                candidate_poses.append(fp)

        best_seed = None
        best_dist = float('inf')
        pose = None
        for candidate in candidate_poses:
            if candidate is None:
                continue
            cx, cy, cz = candidate
            for s in seeds:
                d_xy = math.hypot(cx - s[0], cy - s[1])
                d_z = abs(cz - s[2])
                d = math.dist(candidate, s)
                if d_xy <= radius and d_z <= z_tolerance and d < best_dist:
                    best_dist = d
                    best_seed = s
                    pose = candidate

        if best_seed is None or pose is None:
            time.sleep(0.001)
            continue

        pose_tuple = tuple(pose)
        if pose_tuple == prev_pose_tuple:
            time.sleep(0.001)
            continue
        prev_pose_tuple = pose_tuple

        x, y, z = pose

        # Stability check
        if matched_seed != best_seed:
            # Switched seeds — reset stability
            stable_count = 0
            matched_seed = best_seed

        if last_pose is not None:
            delta = math.dist(last_pose, pose)
            if delta < 0.003:
                stable_count += 1
            else:
                stable_count = 0

        last_pose = pose

        if stable_count >= stable_needed:
            node.get_logger().info(
                f"Reacquire stable ({stable_count}): "
                f"[{x:.3f},{y:.3f},{z:.3f}] near seed "
                f"[{matched_seed[0]:.3f},{matched_seed[1]:.3f},{matched_seed[2]:.3f}]")
            return (x, y, z)

        time.sleep(0.001)

    # Timeout — no stable detection found
    node.get_logger().warn("Reacquire timeout — using primary seed")
    return None



def subscribe_to_goal_pose(node):
    """Subscribe to /external_goal_pose and collect up to 3 candidate goals.

    Accepts first stable goal immediately, then keeps listening briefly
    to collect additional distinct candidates (>5cm apart). All candidates
    are stored for fast reacquire during approach.
    """

    MAX_CANDIDATES = 3
    COLLECT_WINDOW = 0.8  # seconds to keep listening after first accept

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
    node.candidate_goals = []  # list of [x,y,z,qw,qx,qy,qz]

    # Track position stability before accepting
    goal_history = {
        "poses": [], "stable_count": 0,
        "accepted": False, "accept_time": 0.0,
        "collecting": False,
    }

    def _destroy_sub():
        try:
            if hasattr(node, 'goal_pose_sub'):
                node.destroy_subscription(node.goal_pose_sub)
                del node.goal_pose_sub
        except Exception:
            pass

    def _goal_cb(msg: PoseStamped):
        nonlocal goal_history

        new_xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        new_quat = [msg.pose.orientation.w, msg.pose.orientation.x,
                     msg.pose.orientation.y, msg.pose.orientation.z]

        # ---- ALWAYS STORE LATEST GOAL POSE ----
        node.latest_goal_pose = [*new_xyz, *new_quat]
        node.latest_goal_time = time.time()

        # Phase 2: collecting additional candidates after first accept
        if goal_history["collecting"]:
            elapsed = time.time() - goal_history["accept_time"]
            if elapsed > COLLECT_WINDOW or len(node.candidate_goals) >= MAX_CANDIDATES:
                goal_history["collecting"] = False
                n = len(node.candidate_goals)
                node.get_logger().info(f"Collected {n} candidate goal(s)")
                _destroy_sub()
                return

            # Add if distinct from all existing candidates (>5cm apart)
            g = [*new_xyz, *new_quat]
            is_distinct = all(
                math.dist(new_xyz, c[:3]) > 0.05
                for c in node.candidate_goals
            )
            if is_distinct:
                node.candidate_goals.append(g)
                publish_goal_marker(node, new_xyz)
                h_type = "LOW" if new_xyz[2] < LOW_Z_THRESH else "MID/HIGH"
                _trunk = getattr(node, 'trunk_x', None) or 0.16
                _dx = abs(new_xyz[0] - _trunk)
                l_type = ("LEFT" if new_xyz[0] > _trunk else "RIGHT") if _dx > LATERAL_THRESH else "CENTER"
                node.get_logger().info(
                    f"Candidate #{len(node.candidate_goals)}: {h_type} | {l_type} "
                    f"(z={new_xyz[2]:.2f}m) [{new_xyz[0]:.3f},{new_xyz[1]:.3f},{new_xyz[2]:.3f}]")
            return

        # Phase 1: waiting for first stable goal
        if goal_history["accepted"]:
            return

        # Check for position jump (vision switched to new fruit)
        if goal_history["poses"]:
            last_xyz = goal_history["poses"][-1]
            jump = math.dist(new_xyz, last_xyz)
            if jump > 0.10:  # >10cm = new fruit, reset
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
            goal_history["accept_time"] = time.time()
            goal_history["collecting"] = True  # start collecting more candidates
            node.goal_received = True

            # Stop idle timer
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass

            publish_stop_trajectory(node)

            node.goal_seed_xy = [new_xyz[0], new_xyz[1]]
            node.best_goal_xyz = new_xyz
            node.best_goal_score = float("inf")

            g = [*new_xyz, *new_quat]

            # First candidate = primary goal
            node.candidate_goals = [g]
            node.goal_poses.append(g)
            publish_goal_marker(node, new_xyz)
            is_low = new_xyz[2] < LOW_Z_THRESH
            height_type = "LOW" if is_low else "MID/HIGH"

            # Lateral classification: left / center / right relative to trunk
            trunk_x = getattr(node, 'trunk_x', None)
            if trunk_x is None:
                trunk_x = 0.16  # fallback
            dx_trunk = abs(new_xyz[0] - trunk_x)
            if dx_trunk > LATERAL_THRESH:
                lateral_type = "LEFT" if new_xyz[0] > trunk_x else "RIGHT"
            else:
                lateral_type = "CENTER"

            node.latest_goal_classification = f"{height_type} | {lateral_type}"
            node.get_logger().info(
                f"Primary goal accepted: {height_type} | {lateral_type} "
                f"(z={new_xyz[2]:.2f}m, fruit_x={new_xyz[0]:.3f}, trunk_x={trunk_x:.3f})")
            node.obstacles.update_pose("fruit_obstacle", new_xyz)

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
            dx, dy, dz = sequence[idx['i']]
            tgt = [curp[0] + dx, curp[1] + dy, curp[2] + dz, *current_orientation]
            node.get_logger().info(f"Idle micro-motion to: [{tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}]")
            _exec(node, tgt)
            idx['i'] += 1

    # Start idle motion timer
    node.idle_timer = node.create_timer(5.0, _idle_cb)


def subscribe_multi_goals(node, max_goals=3, timeout=10.0):
    """Subscribe to /external_goal_pose and collect up to max_goals distinct goals within timeout seconds.

    Each goal must be stable (2 readings <2cm) and distinct (>8cm from all previously accepted goals).
    Goals are queued in node.goal_poses for sequential execution.
    """
    DISTINCT_DIST = 0.02  # 8cm apart to count as a separate goal

    # Wait until robot stops before subscribing
    if is_robot_moving(node):
        node.create_timer(0.5, lambda: (not is_robot_moving(node)) and subscribe_multi_goals(node, max_goals, timeout))
        return

    # Cancel any existing idle timer
    if hasattr(node, 'idle_timer'):
        try:
            node.idle_timer.cancel()
            del node.idle_timer
        except Exception:
            pass

    # Reset all tracking state
    node.reset_goal_tracking()
    node.latest_goal_pose = None
    node.latest_goal_time = 0

    # Destroy previous subscription
    if hasattr(node, 'goal_pose_sub'):
        node.destroy_subscription(node.goal_pose_sub)
        del node.goal_pose_sub

    node.goal_poses.clear()
    node.candidate_goals = []

    # Clear vision exclusions at start
    from std_msgs.msg import Float32MultiArray
    def _publish_exclusions(positions):
        """Publish excluded positions to vision node so it picks different targets."""
        msg = Float32MultiArray()
        for pos in positions:
            msg.data.extend([float(pos[0]), float(pos[1]), float(pos[2])])
        node.exclude_pub.publish(msg)

    _publish_exclusions([])  # clear any previous exclusions

    state = {
        "poses": [],
        "stable_count": 0,
        "accepted_goals": [],  # list of [x,y,z] for distinctness check
        "start_time": time.time(),
        "pending_xyz": None,
        "pending_quat": None,
    }

    def _destroy_sub():
        try:
            if hasattr(node, 'goal_pose_sub'):
                node.destroy_subscription(node.goal_pose_sub)
                del node.goal_pose_sub
        except Exception:
            pass

    def _cancel_timeout():
        if hasattr(node, '_multi_timeout_timer'):
            try:
                node._multi_timeout_timer.cancel()
                del node._multi_timeout_timer
            except Exception:
                pass

    def _finish():
        if state.get("finished"):
            return
        state["finished"] = True
        _destroy_sub()
        _cancel_timeout()
        _publish_exclusions([])  # clear exclusions so vision returns to normal
        n = len(state["accepted_goals"])
        node.get_logger().info(f"Subscribe multi: collected {n}/{max_goals} goals in {time.time() - state['start_time']:.1f}s")
        if n > 0:
            node.goal_received = True

    def _goal_cb(msg: PoseStamped):
        nonlocal state

        # Check timeout
        if time.time() - state["start_time"] > timeout:
            _finish()
            return

        # Already have enough goals
        if len(state["accepted_goals"]) >= max_goals:
            _finish()
            return

        new_xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        new_quat = [msg.pose.orientation.w, msg.pose.orientation.x,
                     msg.pose.orientation.y, msg.pose.orientation.z]

        # Always store latest
        node.latest_goal_pose = [*new_xyz, *new_quat]
        node.latest_goal_time = time.time()

        # Check for position jump (vision switched to new fruit)
        if state["poses"]:
            last_xyz = state["poses"][-1]
            jump = math.dist(new_xyz, last_xyz)
            if jump > 0.10:  # >10cm = new fruit, reset stability
                state["poses"].clear()
                state["stable_count"] = 0

        state["poses"].append(new_xyz)
        state["pending_xyz"] = new_xyz
        state["pending_quat"] = new_quat

        # Need at least 2 consistent readings
        if len(state["poses"]) >= 2:
            recent = state["poses"][-2:]
            if math.dist(recent[0], recent[1]) < 0.02:
                state["stable_count"] += 1
            else:
                state["stable_count"] = 0

        # Accept after 2 stable readings
        if state["stable_count"] >= 2:
            xyz = state["pending_xyz"]
            quat = state["pending_quat"]

            # Check distinctness from all previously accepted goals
            is_distinct = all(
                math.dist(xyz, prev) > DISTINCT_DIST
                for prev in state["accepted_goals"]
            )

            if not is_distinct:
                # Don't reset stability — just keep waiting for vision to switch targets
                return

            # Accept this goal
            g = [*xyz, *quat]
            state["accepted_goals"].append(xyz)
            node.goal_poses.append(g)
            node.candidate_goals.append(g)
            publish_goal_marker(node, xyz)

            # Tell vision to exclude this position so it picks next-best
            _publish_exclusions(state["accepted_goals"])

            goal_type = "LOW" if xyz[2] < LOW_Z_THRESH else "MID/HIGH"
            n = len(state["accepted_goals"])
            node.get_logger().info(
                f"Multi goal #{n}/{max_goals}: {goal_type} "
                f"(z={xyz[2]:.2f}m) [{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]")

            if n == 1:
                node.best_goal_xyz = xyz
                node.goal_seed_xy = [xyz[0], xyz[1]]

            # Check if we have enough
            if n >= max_goals:
                _finish()
                return

            # Reset stability for next goal
            state["poses"].clear()
            state["stable_count"] = 0

    # Subscribe with VOLATILE QoS
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

    # Auto-finish after timeout
    def _timeout_cb():
        node.get_logger().info("Subscribe multi: timeout reached")
        _finish()

    node._multi_timeout_timer = node.create_timer(timeout, _timeout_cb)


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

    def _safe_return_home():
        """Return to center HOME directly. cuRobo + voxel obstacles handles trunk avoidance."""
        move_to_home_position(node)

    def _check_stop():
        """Check if stop was requested; if so, halt robot and clear goals."""
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("STOP requested — aborting immediately.")
            publish_stop_trajectory(node)
            node.goal_poses.clear()
            node.stop_requested = False
            unlock_target(node)
            return True
        return False

    while node.goal_poses and getattr(node, 'running', True):
        if _check_stop():
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

        # Clear previous trajectory markers from RViz
        clear_path_markers(node)

        # Lock vision onto this target (prevents switching to different "best" during approach)
        lock_target(node, goal[:3])

        # Clear stored trajectory for partial reverse after grasp
        node.stored_trajectory_states = []

        # Adaptive gripper open: deferred until approach motion starts (opens during arm travel)
        fruit_radius = getattr(node, 'latest_fruit_radius', None)
        gripper_opened = False

        # Height-based approach strategy
        is_low = z < LOW_Z_THRESH

        standoff = 0.25  # standoff distance from fruit for approach pose
        d_blend = blend_approach_direction(node, x, y, z)

        if is_low:
            # Low-hanging: original master_new approach (no d_blend for position)
            ax = x
            ay = y
            az = z
            node.get_logger().info(
                f"LOW approach (z={z:.2f} < {LOW_Z_THRESH}): "
                f"fruit=[{x:.3f},{y:.3f},{z:.3f}]")
        else:
            # Mid/high: d_blend direction-driven approach from front
            ax = x - standoff
            ay = y + standoff  # ALWAYS toward robot
            az = z - standoff
            node.get_logger().info(
                f"MID/HIGH approach (z={z:.2f} >= {LOW_Z_THRESH}): "
                f"fruit=[{x:.3f},{y:.3f},{z:.3f}] standoff=[{ax:.3f},{ay:.3f},{az:.3f}] "
                f"d_blend=[{d_blend[0]:.3f},{d_blend[1]:.3f},{d_blend[2]:.3f}]")

        # 1. Plan approach - different strategy based on height and lateral position
        # Get current orientation and minimize rotation
        cur_pose = node.get_end_effector_pose()
        cur_quat = cur_pose[3:] if cur_pose else None
        target_quat = goal[3:]

        # Side HOME: use predefined home_left / home_right based on fruit vs trunk position
        is_side_approach = False
        trunk_x = node.trunk_x  # live trunk x from /trunk_position topic
        if trunk_x is None:
            trunk_x = 0.16  # fallback if vision hasn't published yet
            node.get_logger().warn("No trunk_position received yet, using default trunk_x=0.16")
        if cur_pose is not None:
            dx_ee_to_fruit = abs(x - trunk_x)
            node.get_logger().info(
                f"EE-to-fruit x distance: {dx_ee_to_fruit:.2f}m, trunk_x={trunk_x:.3f}")
            if dx_ee_to_fruit > LATERAL_THRESH and not is_low:   # skip side HOME for low fruits
                if x > trunk_x:
                    side_joints = node.home_left_joints
                    side_label = "HOME_LEFT"
                else:
                    side_joints = node.home_right_joints
                    side_label = "HOME_RIGHT"
                node.get_logger().info(
                    f"{side_label}: fruit x={x:.2f}, trunk_x={trunk_x:.3f}")
                plan_execute_js(node, side_joints, label=side_label, motion_type="home", speed_factor=0.5)
                is_side_approach = True
                # Update start state and cur_pose after side HOME
                cur_pose = node.get_end_effector_pose()
                cur_quat = cur_pose[3:] if cur_pose else cur_quat
                start = JointState.from_position(
                    torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
                    joint_names=node.joint_order,
                )
                # Reacquire goal after side HOME — target may have shifted during the move
                candidates = getattr(node, 'candidate_goals', [])
                reacq = reacquire_goal_pose(node, seed_xyz=[x, y, z], candidate_seeds=candidates, timeout=10.0)
                if reacq:
                    x, y, z = reacq
                    node.get_logger().info(
                        f"Reacquired after {side_label}: [{x:.3f}, {y:.3f}, {z:.3f}]")
                    publish_goal_marker(node, [x, y, z])
                    # Recompute approach standoff with updated position
                    is_low = z < LOW_Z_THRESH
                    d_blend = blend_approach_direction(node, x, y, z)
                    if is_low:
                        ax, ay, az = x, y, z
                    else:
                        ax = x - standoff
                        ay = y + standoff
                        az = z - standoff
                else:
                    node.get_logger().warn(
                        f"Reacquire after {side_label} failed — using original goal position")
            else:
                node.get_logger().info("Fruit near center; using default HOME without side move.")

        # Skip approach if EE is already close to the goal
        skip_approach = False
        if cur_pose is not None:
            ee_dist = math.sqrt((cur_pose[0] - x)**2 + (cur_pose[1] - y)**2 + (cur_pose[2] - z)**2)
            node.get_logger().info(f"EE-to-goal distance: {ee_dist*100:.1f}cm")
            if ee_dist < 0.1:  # within 15cm — skip approach, go straight to final
                node.get_logger().info("EE already close to goal — skipping approach, going direct to final.")
                orientation = minimize_rotation_orientation(cur_quat, target_quat)
                skip_approach = True

        # 2-finger mode: detect between-branches scenario from vision depth analysis
        between_branches = getattr(node, 'fruit_between_branches', False)
        gap_angle = getattr(node, 'fruit_gap_angle', 0.0)

        if between_branches:
            node.get_logger().info(
                f"Between-branches detected! gap_angle={math.degrees(gap_angle):.1f} deg — using 2-finger mode")
            # Apply roll correction so left+right fingers align with gap
            half = gap_angle / 2.0
            q_roll = [math.cos(half), 0.0, 0.0, math.sin(half)]
            target_quat = quat_multiply(list(target_quat), q_roll)
            node.gripper_controller.frozen_fingers = {1}  # freeze center finger
        else:
            node.gripper_controller.frozen_fingers = set()  # all 3 fingers active

        if skip_approach:
            pass  # jump straight to reacquire + final below


        elif is_low:
            side_blend = 0.25
            orientation = minimize_rotation_orientation(cur_quat, target_quat, blend_weight=side_blend)
            approach = [ax, ay + 0.07, az - 0.09, *orientation]
            node.get_logger().info(
                f"LOW approach pose: {approach[:3]}, is_side={is_side_approach}, blend={side_blend:.2f}")
        else:
            side_blend =0.0
            orientation = minimize_rotation_orientation(cur_quat, target_quat, blend_weight=side_blend)
            approach = [ax, ay, az, *orientation]
            node.get_logger().info(
                f"MID/HIGH approach pose: {approach[:3]}, is_side={is_side_approach}, blend={side_blend:.2f}, "
                f"d_blend=[{d_blend[0]:.3f},{d_blend[1]:.3f},{d_blend[2]:.3f}]")

        # === DEBUG PLAN PREVIEW (RViz visualization) ===
        if node.cfg.planner.debug_plan_preview:
            preview_steps = []
            # 1. Current position (HOME or side HOME)
            if is_side_approach:
                side_label = "HOME_LEFT" if x > (node.trunk_x or 0.16) else "HOME_RIGHT"
                side_js = node.home_left_joints if "LEFT" in side_label else node.home_right_joints
                preview_steps.append({"label": side_label, "joints": side_js})
            else:
                preview_steps.append({"label": "HOME", "joints": node.home_joints})
            # 2. Approach
            if not skip_approach:
                preview_steps.append({"label": "APPROACH", "position": approach[:3]})
            # 3. Final
            preview_steps.append({"label": "FINAL", "position": [x, y + 0.03, z + 0.03]})
            # 4. Dropoff
            preview_steps.append({"label": "DROPOFF", "joints": node.dropoff_joints})
            # 5. Return HOME
            preview_steps.append({"label": "HOME", "joints": node.home_joints})

            markers_mod.publish_plan_preview(node, preview_steps)

            trunk_x_val = node.trunk_x or 0.16
            dx_trunk = abs(x - trunk_x_val)
            height_label = "LOW" if is_low else "MID/HIGH"
            lateral_label = ("LEFT" if x > trunk_x_val else "RIGHT") if dx_trunk > LATERAL_THRESH else "CENTER"
            node.get_logger().info(
                f"PLAN PREVIEW: {height_label} | {lateral_label} | "
                f"side={is_side_approach} | goal=[{x:.3f},{y:.3f},{z:.3f}] | "
                f"Waiting for confirm (GUI 'confirm'/'cancel' or keyboard ENTER/k)")

            # Wait for confirmation via event (set by keyboard thread or GUI command)
            node.plan_confirmed = None
            node.plan_confirm_event.clear()
            node.plan_waiting = True
            node.plan_confirm_event.wait()  # blocks until set
            node.plan_waiting = False
            markers_mod.clear_plan_preview(node)
            if not node.plan_confirmed:
                node.get_logger().info("Plan cancelled by user.")
                unlock_target(node)
                continue

        if not skip_approach:
            if not plan_and_send(node, start, Pose.from_list(approach), label="APPROACH", motion_type="approach", goal_xyz=approach[:3], store_trajectory=True):
                unlock_target(node)
                continue
            # Open gripper during approach motion (arm is already moving)
            if not gripper_opened:
                gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius)
                gripper_opened = True
            wait_until_xyz(node, approach[:3])
            if _check_stop(): break
            log_path_deviation(node, "APPROACH")
            blend_motion(node)

            # Wait for joint state to be available after approach
            for _ in range(20):
                if node.current_joint_positions is not None:
                    break
                time.sleep(0.05)
            if node.current_joint_positions is None:
                node.get_logger().warn("No joint state after approach; skipping goal.")
                unlock_target(node)
                continue

            start = JointState.from_position(
                torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
                joint_names=node.joint_order,
            )

        # Ensure gripper is open before final approach (fallback for skip_approach case)
        if not gripper_opened:
            gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius)
            gripper_opened = True

        if _check_stop(): break

        # 2. Reacquire — check primary + all candidate seeds for fastest lock-on
        seed = [x,y,z]
        candidates = getattr(node, 'candidate_goals', [])
        reacq = reacquire_goal_pose(node, seed_xyz=seed, candidate_seeds=candidates)

        if reacq is None:
            # Nudge EE down a few cm and retry — fruit may be just below FOV
            node.get_logger().warn("Reacquire timed out — nudging down to search for detection.")
            cur = node.get_end_effector_pose()
            if cur and not _check_stop():
                nudge = [cur[0], cur[1], cur[2] - 0.02, *cur[3:]]
                _direct_ik_move(node, nudge, label="REACQ_NUDGE",
                                motion_type="final", store_trajectory=False)
                reacq = reacquire_goal_pose(node, seed_xyz=seed, candidate_seeds=candidates, timeout=4.0)

        if reacq:
            x,y,z = reacq
            publish_goal_marker(node, [x,y,z])
        else:
            node.get_logger().warn("No reacquire after nudge — falling back to original seed.")
            x, y, z = seed
            
        if _check_stop(): break

        # 3. Final slow precise grasp — IK + direct joint interpolation (no cuRobo trajectory)
        #    _direct_ik_move handles wait + blend internally
        # Reuse orientation from APPROACH step — all orientation changes happen during approach only
        node.get_logger().info(f"FINAL orientation: reusing APPROACH orientation (is_low={is_low})")
        z_offset = 0.015  # small downward adjustment during grasp
        y_offset = 0.005  # no lateral adjustment
        # Pull back along the approach direction by 1× estimated fruit radius.
        # This adapts to fruit size and approach angle rather than a fixed Y offset,
        # preventing overshoot when depth is measured at the centroid vs near-surface.
        fruit_radius = getattr(node, 'latest_fruit_radius', None) or 0.035
        _sm_dir = getattr(getattr(node, 'state_manager', None), 'fruit_direction', None)
        approach_dir = list(_sm_dir) if _sm_dir is not None else None  # [dx,dy,dz] unit vec in base_link
        if approach_dir is not None and len(approach_dir) == 3:
            # Move the grasp point back by 1 radius along the approach direction
            pullback = float(fruit_radius)
            gx = x - approach_dir[0] * pullback
            gy = y - approach_dir[1] * pullback + y_offset  # add fixed Y offset on top of directional pullback
            gz = z - approach_dir[2] * pullback + z_offset
        else:
            # Fallback: fixed Y pullback
            gx, gy, gz = x, y + fruit_radius, z + z_offset
        final_target = [gx, gy, gz, *orientation]
        final_ok = _direct_ik_move(node, final_target, label="FINAL",
                                   motion_type="final", store_trajectory=True)
        if not final_ok:
            # IK too far — pull back a few cm and retry from a different config
            node.get_logger().warn("FINAL IK failed — pulling back and retrying...")
            cur = node.get_end_effector_pose()
            if cur:
                # Move 5cm back (away from fruit, along -Y in base frame)
                retreat = [cur[0], cur[1] + 0.05, cur[2], *cur[3:]]
                _direct_ik_move(node, retreat, label="FINAL_RETREAT",
                                motion_type="final", store_trajectory=False)

            # Retry FINAL from new position
            final_ok = _direct_ik_move(node, final_target, label="FINAL_RETRY",
                                       motion_type="final", store_trajectory=False)
        if not final_ok:
            # Both attempts failed — skip this goal
            node.get_logger().warn("FINAL IK retry also failed — skipping goal.")
            unlock_target(node)
            continue
        log_path_deviation(node, "FINAL")

        if _check_stop(): break
        node.control_gripper("CLOSE"); time.sleep(0.5)
        if _check_stop(): break
        # Notify vision system about grasp attempt for fruit tracking
        notify_grasp_attempt(node, final_target[:3])

        # Evaluate grasp using force profile (when did contact start during closure)
        gc = node.gripper_controller
        first_contact = getattr(gc, 'closure_first_contact_step', gc.steps)
        stopped_early = getattr(gc, 'closure_stopped_early', False)
        closure_step = getattr(gc, 'closure_step_stopped', gc.steps)
        deltas = getattr(gc, 'closure_deltas', [0.0, 0.0, 0.0])

        learner = getattr(node, 'grasp_learner', None)
        if learner and learner.enabled and learner.has_enough_data():
            prediction = learner.predict_grasp_success(first_contact, gc.steps, stopped_early)
            action = learner.suggest_action(first_contact, gc.steps, stopped_early)
            node.get_logger().info(f"[LEARNER] prediction={prediction}, action={action}, first_contact={first_contact}/{gc.steps}")
        else:
            # Fallback: early contact or stopped_early = PROCEED
            if stopped_early or first_contact < gc.steps - 2:
                prediction = "PROPER"
                action = "PROCEED"
            else:
                prediction = "NO_CONTACT"
                action = "REGRIP"

        if action == "REGRIP":
            node.get_logger().warn(f"Weak grip ({prediction}) — re-gripping...")
            gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius); time.sleep(0.1)
            cur = node.get_end_effector_pose()
            if cur:
                f0, f1, f2 = deltas[0], deltas[1], deltas[2]  # left, center, right

                # Lateral correction: imbalance between left (F0) and right (F2)
                # F0 > F2 → fruit is left of center → shift gripper left (−X)
                # F2 > F0 → fruit is right of center → shift gripper right (+X)
                lateral_imbalance = f0 - f2
                lateral_correction = -float(lateral_imbalance) * 0.008  # ~8mm per 1N imbalance
                lateral_correction = max(-0.02, min(0.02, lateral_correction))  # clamp ±20mm

                # Forward correction: late contact = gripper too far from fruit → move forward
                # first_contact close to gc.steps = near fully closed before touching
                contact_ratio = first_contact / max(gc.steps, 1)
                forward_correction = -(contact_ratio - 0.5) * 0.04  # up to 20mm forward if late
                forward_correction = max(-0.025, min(0.0, forward_correction))  # clamp, never pull back

                # Vertical correction: if center (F1) much weaker than sides → fruit is below center
                center_vs_sides = f1 - (f0 + f2) / 2.0
                vertical_correction = -float(center_vs_sides) * 0.005  # small, ±5mm max
                vertical_correction = max(-0.01, min(0.01, vertical_correction))

                node.get_logger().info(
                    f"[REGRIP] F=[{f0:.2f},{f1:.2f},{f2:.2f}]N "
                    f"contact={first_contact}/{gc.steps} ({contact_ratio:.0%}) | "
                    f"corrections: lateral={lateral_correction*1000:+.1f}mm "
                    f"forward={forward_correction*1000:+.1f}mm "
                    f"vertical={vertical_correction*1000:+.1f}mm"
                )

                closer_target = [
                    cur[0] + lateral_correction,
                    cur[1] + forward_correction,
                    cur[2] + vertical_correction,
                    *cur[3:]
                ]
                exec_pose(node, closer_target)
                wait_until_xyz(node, closer_target[:3], tol=0.01, timeout=3.0)
            node.control_gripper("CLOSE"); time.sleep(0.1)
            # Re-read after re-grip
            first_contact = getattr(gc, 'closure_first_contact_step', gc.steps)
            stopped_early = getattr(gc, 'closure_stopped_early', False)
            closure_step = getattr(gc, 'closure_step_stopped', gc.steps)
            deltas = getattr(gc, 'closure_deltas', [0.0, 0.0, 0.0])
            if learner and learner.enabled and learner.has_enough_data():
                prediction = learner.predict_grasp_success(first_contact, gc.steps, stopped_early)
            else:
                prediction = "PROPER" if (stopped_early or first_contact < gc.steps - 2) else "NO_CONTACT"
            node.get_logger().info(f"Re-grip result: {prediction}, first_contact={first_contact}/{gc.steps}")

        # Log grasp attempt for learning (success filled in by user feedback later)
        if learner:
            node.pending_grasp_record = GraspRecord(
                fruit_x=x, fruit_y=y, fruit_z=z,
                first_contact_step=first_contact,
                stopped_early=stopped_early,
                closure_step=closure_step,
                total_steps=gc.steps,
                delta_f0=deltas[0], delta_f1=deltas[1], delta_f2=deltas[2],
            )

        # 4. Drop-off and return
        if _check_stop(): 
            break
        time.sleep(0.2)

        # Attempt CUDA recovery before dropoff/home planning
        if getattr(node, "_cuda_faulted", False):
            try_cuda_recovery(node)

        # Reverse along the stored approach path (28cm clearance from fruit).
        execute_partial_reverse(node, clearance_m=0.28)

        # After partial reverse, return to the correct home position:
        # - Side approach: go back to home_left or home_right (arm is near trunk, needs to clear)
        # - Center approach: try direct dropoff, fall back to center HOME only if planning fails
        if is_side_approach:
            node.get_logger().info("Side approach — returning to center HOME before dropoff")
            _safe_return_home()

        if not move_to_dropoff_position(node):
            node.get_logger().info("Direct dropoff failed, going to center HOME first")
            _safe_return_home()
            move_to_dropoff_position(node)
        time.sleep(0.2)
        # Reset 2-finger mode before dropoff open (all fingers active for release)
        node.gripper_controller.frozen_fingers = set()
        node.control_gripper("OPEN")

        if _check_stop(): 
            break

        # Smart return: if there are more goals, try direct approach instead of going home first
        if node.goal_poses:
            # Peek at next goal (don't pop it yet)
            next_goal = node.goal_poses.peek(0)
            if next_goal is not None:
                nx, ny, nz = next_goal[:3]
                next_quat = next_goal[3:]
                next_is_low = nz < LOW_Z_THRESH

                # Compute approach pose for next goal
                next_d_blend = blend_approach_direction(node, nx, ny, nz)
                next_standoff = 0.12
                if next_is_low:
                    nax, nay, naz = nx, ny - 0.01, nz - 0.12
                else:
                    nax = nx - next_d_blend[0] * next_standoff
                    nay = ny + abs(next_d_blend[1]) * next_standoff
                    naz = nz - next_d_blend[2] * next_standoff

                # Try planning from current (dropoff) position to next approach
                cur_joints = node.current_joint_positions
                if cur_joints is not None:
                    next_start = JointState.from_position(
                        torch.tensor([cur_joints], dtype=torch.float32, device=device),
                        joint_names=node.joint_order,
                    )
                    cur_pose = node.get_end_effector_pose()
                    cur_quat = cur_pose[3:] if cur_pose else [1.0, 0.0, 0.0, 0.0]
                    next_orient = minimize_rotation_orientation(cur_quat, next_quat)
                    next_approach = [nax, nay, naz, *next_orient]

                    # Test plan (don't execute yet, just check feasibility)
                    plan_cfg = PLAN_CFG_DEFAULT
                    lock = getattr(node, '_planning_lock', None)
                    if lock: lock.acquire()
                    try:
                        test_res = node.motion_gen.plan_single(
                            next_start, Pose.from_list(next_approach), plan_cfg)
                    finally:
                        if lock: lock.release()

                    if test_res.success:
                        node.get_logger().info(
                            f"Direct path to next goal approach is COLLISION-FREE — skipping HOME")
                        # Don't go home, the main loop will handle approach planning
                    else:
                        node.get_logger().info(
                            f"Direct path to next goal BLOCKED — going HOME first")
                        _safe_return_home()
                else:
                    _safe_return_home()
            else:
                _safe_return_home()
        else:
            _safe_return_home()

        # Wait for grasp feedback from RViz GUI (Y/N keys) or terminal
        if node.cfg.grasp.learning_enabled and hasattr(node, 'pending_grasp_record') and node.pending_grasp_record is not None:
            node._grasp_feedback = None  # reset
            node.get_logger().info("Waiting for grasp feedback (Y=success / N=fail)...")
            t0 = time.time()
            while node._grasp_feedback is None and (time.time() - t0) < 10.0:
                time.sleep(0.1)
            if node._grasp_feedback is not None:
                record = node.pending_grasp_record
                record.success = node._grasp_feedback
                node.grasp_learner.log_attempt(record)
                node.get_logger().info(f"Grasp logged: {'SUCCESS' if record.success else 'FAIL'}")
            else:
                node.get_logger().info("Grasp feedback timeout — skipping.")
            node._grasp_feedback = None
            node.pending_grasp_record = None

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
