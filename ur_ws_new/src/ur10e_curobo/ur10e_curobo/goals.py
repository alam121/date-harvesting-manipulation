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
from .motions import blend_motion, preplan_js, plan_execute_js, nearest_joint_config
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

        for _ in range(max_iters):
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

    # Normalise IK solution to the same 2π branch as current joints.
    # cuRobo IK can return an equivalent config that is ±2π away, which would
    # make the interpolation travel a full revolution instead of staying put.
    best_js = nearest_joint_config(start_js, best_js)

    # If the IK solution is on a different kinematic branch (large joint delta even
    # after 2π normalisation), retry with perturbed seeds to find the nearest branch.
    # This prevents the arm taking a long arc when a shorter path exists.
    _total_delta = sum(abs(g - c) for g, c in zip(best_js, start_js))
    _RETRY_THRESH_RAD = 0.70  # ~40° total — above this, search for a closer branch
    if _total_delta > _RETRY_THRESH_RAD and not getattr(node, "_cuda_faulted", False):
        try:
            import random as _random
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pos  = torch.tensor([target_pose_list[:3]], dtype=torch.float32, device=device)
            quat = torch.tensor([target_pose_list[3:]], dtype=torch.float32, device=device)
            goal_pose = Pose(position=pos, quaternion=quat)
            retract   = torch.tensor([start_js], dtype=torch.float32, device=device)

            # Build a batch of seeds: current config + 8 small random perturbations
            _seeds = [start_js]
            for _ in range(8):
                _perturb = [j + _random.uniform(-0.3, 0.3) for j in start_js]
                _seeds.append(_perturb)

            _best_alt = best_js
            _best_total = _total_delta
            for _s in _seeds:
                _seed_t = torch.tensor([_s], dtype=torch.float32, device=device).unsqueeze(0)
                _r = node.motion_gen.ik_solver.solve_single(goal_pose, seed_config=_seed_t, retract_config=retract)
                if _r.success.item():
                    _candidate = nearest_joint_config(start_js, _r.js_solution.position.squeeze().cpu().tolist())
                    _d = sum(abs(g - c) for g, c in zip(_candidate, start_js))
                    if _d < _best_total:
                        _best_total = _d
                        _best_alt = _candidate
            if _best_alt is not best_js:
                node.get_logger().info(
                    f"[DIRECT] {label}: branch retry found shorter path "
                    f"({_total_delta*57.3:.1f}° → {_best_total*57.3:.1f}° total)")
                best_js = _best_alt
        except Exception as _e:
            node.get_logger().warn(f"[DIRECT] {label}: branch retry exception: {_e}")

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

    # 3) Cartesian IK waypoints → per-segment S-curve interpolation
    #
    # Direct joint-space interpolation from approach to final can arc through
    # dangerous wrist configurations (lower-arm / tool-flange clamping) on
    # left+low fruits.  Instead, solve IK at N evenly-spaced Cartesian positions
    # along the straight EE line — each seeded from the previous solution so the
    # arm stays on the same kinematic branch throughout.
    _N_CART = 5   # intermediate IK waypoints (6 segments total)
    _states_built = False
    states = []
    _cur_ee = node.get_end_effector_pose()
    if _cur_ee is not None and not getattr(node, "_cuda_faulted", False):
        try:
            _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _start_quat = list(_cur_ee[3:]) if len(_cur_ee) > 3 else [1.0, 0.0, 0.0, 0.0]
            _target_quat = list(target_pose_list[3:])
            _wp_js = [start_js]
            _prev_js = start_js
            _cart_ok = True
            for _k in range(1, _N_CART + 1):
                _a = _k / (_N_CART + 1)
                _wp_xyz = [_cur_ee[i] + _a * (target_pose_list[i] - _cur_ee[i]) for i in range(3)]
                # Slerp orientation from current to target so wrist rotates gradually —
                # using fixed target orientation for all waypoints forces an immediate wrist
                # snap at the first IK point, which can create a tool-flange/lower-arm
                # proximity violation in the interpolated states between waypoints.
                _wp_quat = quat_slerp(_start_quat, _target_quat, _a)
                _pos_t   = torch.tensor([_wp_xyz],  dtype=torch.float32, device=_dev)
                _quat_t  = torch.tensor([_wp_quat], dtype=torch.float32, device=_dev)
                _seed_t  = torch.tensor([_prev_js], dtype=torch.float32, device=_dev).unsqueeze(0)
                _ret_t   = torch.tensor([_prev_js], dtype=torch.float32, device=_dev)
                _r = node.motion_gen.ik_solver.solve_single(
                    Pose(position=_pos_t, quaternion=_quat_t),
                    seed_config=_seed_t, retract_config=_ret_t)
                if _r.success.item():
                    _wj = nearest_joint_config(
                        _prev_js, _r.js_solution.position.squeeze().cpu().tolist())
                    # Reject if this waypoint jumps to a different kinematic branch.
                    # nearest_joint_config only fixes ±2π wrapping — a branch jump
                    # still shows up as a large per-joint delta, which causes a jerk
                    # in the trajectory and a zig-zag in the RViz path visualization.
                    _wp_delta = max(abs(a - b) for a, b in zip(_wj, _prev_js))
                    if _wp_delta > 0.52:  # ~30° — branch jumped, abort Cartesian path
                        _cart_ok = False
                        node.get_logger().warn(
                            f"[DIRECT] {label}: Cartesian waypoint {_k} jumped "
                            f"{_wp_delta*57.3:.1f}deg — aborting (branch switch)")
                        break
                    _wp_js.append(_wj)
                    _prev_js = _wj
                else:
                    _cart_ok = False
                    node.get_logger().warn(
                        f"[DIRECT] {label}: Cartesian IK failed at step {_k}/{_N_CART}")
                    break
            if _cart_ok:
                # Re-solve final endpoint seeded from last intermediate waypoint
                _pos_final = torch.tensor([target_pose_list[:3]], dtype=torch.float32, device=_dev)
                _seed_final = torch.tensor([_prev_js], dtype=torch.float32, device=_dev).unsqueeze(0)
                _ret_final  = torch.tensor([_prev_js], dtype=torch.float32, device=_dev)
                _r_final = node.motion_gen.ik_solver.solve_single(
                    Pose(position=_pos_final, quaternion=_quat_t),
                    seed_config=_seed_final, retract_config=_ret_final)
                if _r_final.success.item():
                    _final_js = nearest_joint_config(
                        _prev_js, _r_final.js_solution.position.squeeze().cpu().tolist())
                    _final_delta = max(abs(g - c) for g, c in zip(_final_js, _prev_js))
                    node.get_logger().info(
                        f"[DIRECT] {label}: final-seg re-solve delta={_final_delta*57.3:.1f}deg")
                    _wp_js.append(_final_js)
                else:
                    _final_delta = max(abs(g - c) for g, c in zip(best_js, _prev_js))
                    node.get_logger().warn(
                        f"[DIRECT] {label}: final-seg re-solve failed, using best_js "
                        f"(delta={_final_delta*57.3:.1f}deg)")
                    if _final_delta > 1.05:
                        node.get_logger().warn(
                            f"[DIRECT] {label}: final segment too large ({_final_delta*57.3:.1f}deg) — aborting")
                        _cart_ok = False
                    else:
                        _wp_js.append(best_js)
            if _cart_ok:
                # Linear interpolation — build_trajectory central-difference gives
                # one smooth continuous velocity profile across all waypoints.
                # Minimum 10 steps per segment for smooth central-difference velocities.
                _steps_per = max(10, num_steps // max(len(_wp_js) - 1, 1))
                for _si in range(len(_wp_js) - 1):
                    _s0, _s1 = _wp_js[_si], _wp_js[_si + 1]
                    for _i in range(_steps_per + 1):
                        if _i == 0 and _si > 0:
                            continue  # avoid duplicate at segment boundary
                        _t = _i / _steps_per
                        states.append([_s0[j] + _t * (_s1[j] - _s0[j]) for j in range(len(_s0))])
                _states_built = True
                node.get_logger().info(
                    f"[DIRECT] {label}: Cartesian IK path — "
                    f"{len(_wp_js)} waypoints, {len(states)} states")
        except Exception as _ce:
            node.get_logger().warn(f"[DIRECT] {label}: Cartesian IK path failed: {_ce}")

    if not _states_built:
        # Fallback: original single-step joint-space S-curve
        node.get_logger().info(f"[DIRECT] {label}: fallback to joint-space interpolation")
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
                            ramp_points=0, include_acc=False)
    if node.stop_requested:
        node.stop_requested = False; return False

    # 5) Visualize & send
    cart_path = forward_kinematics_batch(node, states)
    node._planned_cartesian_path = cart_path  # keep PATH_DEV in sync with this motion
    publish_planned_path(node, states, label, cartesian_points=cart_path)
    node.trajectory_pub.publish(traj)

    # Wait for motion to finish (with orientation + velocity checks), then blend
    target_quat = target_pose_list[3:] if len(target_pose_list) > 3 else None
    reached = wait_until_xyz(node, target_pose_list[:3], tol=0.008, timeout=8.0, target_quat=target_quat)
    blend_motion(node)

    # Abort goal if robot has stalled twice — something is obstructing or IK is wrong
    if not reached and getattr(node, '_goal_stall_count', 0) >= 2:
        node.get_logger().warn(
            f"[DIRECT] {label}: stall count={node._goal_stall_count} — aborting goal.")
        return False

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
            node._last_plan_ratio = ratio  # expose to caller for retry decisions
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
        _clearance_reached = False
        for wp in reversed_states[1:]:
            partial.append(wp)
            fk = forward_kinematics(node, wp)
            if fk:
                dist = math.sqrt((fk.x - gx)**2 + (fk.y - gy)**2 + (fk.z - gz)**2)
                if dist >= clearance_m:
                    _clearance_reached = True
                    break
        # If all waypoints were consumed without reaching clearance, the reverse
        # goes all the way back to the approach start (home area). cuRobo's plan
        # can front-load wrist rotation at the beginning of the approach; reversed,
        # this becomes a concentrated wrist snap at the END of the reverse — the jerk.
        # In this case truncate the last 15% of waypoints (the jerky wrist portion)
        # and let cuRobo re-plan the remainder smoothly after the reverse.
        if not _clearance_reached:
            keep = max(10, int(len(partial) * 0.85))
            partial = partial[:keep]
            node.get_logger().info(
                f"Partial reverse: all waypoints consumed — truncating to {keep} "
                f"(dropping last 15% to avoid wrist-snap near home)"
            )

    # Wait for the arm to fully stop before publishing the reverse trajectory.
    # A fixed sleep isn't reliable — poll velocities until settled or timeout.
    _vel_tol = 0.05   # rad/s — joint considered stopped below this
    _settle_timeout = 1.0
    _t0 = time.time()
    while (time.time() - _t0) < _settle_timeout:
        vels = node.current_joint_velocities
        if vels is not None and all(abs(v) < _vel_tol for v in vels):
            break
        time.sleep(0.02)
    else:
        node.get_logger().warn("Partial reverse: arm did not settle within 1s — publishing anyway")

    # Prepend current position to bridge any gap (e.g. after depth correction nudge)
    # as part of the single reverse trajectory — no separate bridge + stop needed,
    # which avoids the stop→restart jerk at the transition.
    if node.current_joint_positions is not None:
        current_pos = list(node.current_joint_positions)
        max_diff = max(abs(c - f) for c, f in zip(current_pos, partial[0]))
        if max_diff > 0.002:
            node.get_logger().info(
                f"Partial reverse: prepending current pos ({max_diff*57.3:.2f}° gap)"
            )
            partial = [current_pos] + partial

    node.get_logger().info(
        f"Partial reverse: {len(partial)}/{len(reversed_states)} waypoints "
        f"({clearance_m*100:.0f}cm clearance)"
    )

    planner = node.cfg.planner
    base_dt = getattr(planner, "base_dt", 0.02)
    # Reverse uses a slow dedicated scale — do NOT use speed_predropoff (full speed).
    # global_speed_multiplier is intentionally NOT applied to max_vel here to avoid
    # 10 rad/s peaks that cause jerk at the start of the reverse motion.
    dt = getattr(planner, "min_dt", 0.012) * 1.5

    # Append deceleration tail: duplicate the last waypoint several times so the
    # controller has multiple dt steps to decelerate the wrist to zero velocity.
    # Without this, build_trajectory sets velocity=0 only at the final point while
    # the second-to-last still carries full central-difference velocity — the
    # controller must stop in one dt (~18ms), causing a wrist jerk.
    DECEL_TAIL = 8
    partial = partial + [partial[-1]] * DECEL_TAIL

    traj = build_trajectory(
        node.joint_order,
        partial,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=getattr(planner, "max_joint_velocity", 2.0) * 0.7,
        max_acc=getattr(planner, "max_joint_acceleration", 1.0) * 0.7,
        ramp_points=0,
    )

    node.trajectory_pub.publish(traj)

    # Wait for partial reverse to complete naturally — do NOT send a new trajectory
    # (blend_motion) while the controller is still decelerating; that preemption
    # causes a jerk at the last waypoint.  Instead: wait until fully stopped, then
    # hold an additional 0.3s so the controller settles before the next trajectory.
    time.sleep(0.3)
    timeout_start = time.time()
    while time.time() - timeout_start < 10.0:
        if not is_robot_moving(node, velocity_threshold=0.005):
            break
        time.sleep(0.1)
    time.sleep(0.3)  # extra settle before next trajectory
    return True


def reacquire_goal_pose(node, seed_xyz, candidate_seeds=None, timeout=5.5, radius=0.04, z_tolerance=0.05, depth_settle_s=2.0, stable_needed=None):
    """Fast reacquire across multiple candidate seeds.

    Checks vision against all candidates. Returns first stable match.
    With multiple candidates, accepts after just 1 stable reading.

    Args:
        seed_xyz: primary seed [x,y,z]
        candidate_seeds: list of [x,y,z,...] alternate candidates (optional)
        timeout: max wait time including settle period
        depth_settle_s: how long to wait for depth to stabilise (reduce for small nudges)
    """
    # Switch vision to lightweight mode: no heatmap, no trunk, no viz
    if hasattr(node, 'set_vision_mode'):
        node.set_vision_mode("reacquire")

    try:
        return _reacquire_goal_pose_impl(
            node, seed_xyz, candidate_seeds, timeout, radius, z_tolerance, depth_settle_s,
            stable_needed=stable_needed)
    finally:
        # Always restore full mode so the next approach cycle works normally
        if hasattr(node, 'set_vision_mode'):
            node.set_vision_mode("full")


def _reacquire_goal_pose_impl(node, seed_xyz, candidate_seeds=None, timeout=5.5, radius=0.04, z_tolerance=0.05, depth_settle_s=2.0, stable_needed=None):
    import numpy as _np
    # Build seed list: primary first, then candidates
    seeds = [seed_xyz[:3]]
    if candidate_seeds:
        for c in candidate_seeds:
            xyz = c[:3] if len(c) > 3 else c
            if all(math.dist(xyz, s) > 0.03 for s in seeds):
                seeds.append(list(xyz))

    if stable_needed is None:
        stable_needed = 5      # readings that must match before accepting
    Z_STABLE_THRESH = 0.012    # max std of Z across stable readings (12 mm)

    stable_count = 0
    matched_seed = None
    start = time.time()
    prev_pose_tuple = None
    recent_zs = []             # Z values of the last stable_needed matched readings

    # Let depth settle before starting to match — ZED stereo depth needs a few
    # frames to stabilize after the arm stops moving at the approach position.
    DEPTH_SETTLE_S = depth_settle_s
    search_s = max(timeout - DEPTH_SETTLE_S, 1.0)
    node.get_logger().info(
        f"[REACQ] Settling {DEPTH_SETTLE_S:.1f}s | "
        f"seeds={len(seeds)} primary=[{seeds[0][0]:.3f},{seeds[0][1]:.3f},{seeds[0][2]:.3f}] | "
        f"search budget={search_s:.1f}s")
    while time.time() - start < DEPTH_SETTLE_S:
        time.sleep(0.05)
    node.get_logger().info(f"[REACQ] Settle done — searching ({search_s:.1f}s remaining)")

    _last_progress_log = 0.0   # throttle per-frame progress to once/sec

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
        _used_xy_only = False
        _closest_miss = None   # for diagnostics
        _closest_miss_d = float('inf')
        for candidate in candidate_poses:
            if candidate is None:
                continue
            cx, cy, cz = candidate
            for s in seeds:
                d_xy = math.hypot(cx - s[0], cy - s[1])
                d_z = abs(cz - s[2])
                d = math.dist(candidate, s)
                # XY-only match: only when we have enough readings AND they are
                # genuinely noisy (high Z std). Do NOT open this gate early just
                # because stable_count is low — that allows neighbouring fruits
                # at 4-5cm to be accepted before we have evidence of depth noise.
                depth_unstable = (
                    len(recent_zs) >= stable_needed and
                    float(_np.std(recent_zs)) > Z_STABLE_THRESH
                )
                xy_only_match = depth_unstable and d_xy <= 0.05
                full_match = d_xy <= radius and d_z <= z_tolerance
                if (full_match or xy_only_match) and d_xy < best_dist:
                    best_dist = d_xy
                    best_seed = s
                    _used_xy_only = xy_only_match and not full_match
                    node.get_logger().debug(
                        f"[REACQ] match: [{cx:.3f},{cy:.3f},{cz:.3f}] → seed [{s[0]:.3f},{s[1]:.3f},{s[2]:.3f}] "
                        f"dxy={d_xy*100:.1f}cm dz={d_z*100:.1f}cm "
                        f"{'XY-only' if _used_xy_only else 'full'}")
                    if _used_xy_only:
                        # Use detected XY but keep seed Z — depth unreliable
                        pose = [cx, cy, s[2]]
                    else:
                        pose = candidate
                elif d < _closest_miss_d:
                    _closest_miss_d = d
                    _closest_miss = (candidate, s, d_xy, d_z)

        if best_seed is None or pose is None:
            # Log once per second so we can see why reacquire keeps missing
            _now = time.time()
            if not hasattr(_reacquire_goal_pose_impl, '_last_miss_log') or \
                    _now - _reacquire_goal_pose_impl._last_miss_log > 1.0:
                _reacquire_goal_pose_impl._last_miss_log = _now
                if _closest_miss is not None:
                    _cm, _cs, _dxy, _dz = _closest_miss
                    node.get_logger().warn(
                        f"[REACQ] No match — closest candidate "
                        f"[{_cm[0]:.3f},{_cm[1]:.3f},{_cm[2]:.3f}] "
                        f"seed [{_cs[0]:.3f},{_cs[1]:.3f},{_cs[2]:.3f}] "
                        f"d_xy={_dxy*100:.1f}cm d_z={_dz*100:.1f}cm "
                        f"(limit: xy={radius*100:.0f}cm z={z_tolerance*100:.0f}cm)"
                    )
                else:
                    node.get_logger().warn("[REACQ] No candidates visible at all")
            time.sleep(0.001)
            continue

        pose_tuple = tuple(pose)
        if pose_tuple == prev_pose_tuple:
            time.sleep(0.001)
            continue
        prev_pose_tuple = pose_tuple

        x, y, z = pose

        # Stability check: count how many distinct readings land near the seed.
        # Depth noise is typically 5-20mm so a tight consecutive-delta gate (old 3mm)
        # would reset on every noisy frame. Instead just count distinct detections.
        if matched_seed != best_seed:
            stable_count = 0
            matched_seed = best_seed
            recent_zs.clear()

        stable_count += 1
        recent_zs.append(z)
        if len(recent_zs) > stable_needed:
            recent_zs.pop(0)

        z_std = float(_np.std(recent_zs)) if len(recent_zs) >= 2 else float('nan')

        # Throttled progress log — once per second
        _now = time.time()
        if _now - _last_progress_log >= 1.0:
            _last_progress_log = _now
            elapsed = _now - start - DEPTH_SETTLE_S
            node.get_logger().info(
                f"[REACQ] t={elapsed:.1f}s | stable={stable_count}/{stable_needed} "
                f"Z_std={z_std*1000:.1f}mm | pos=[{x:.3f},{y:.3f},{z:.3f}] | "
                f"visible={len(candidate_poses)} fruits")

        if stable_count >= stable_needed:
            if z_std > Z_STABLE_THRESH:
                # Depth still fluctuating — keep collecting, don't reset count
                node.get_logger().info(
                    f"[REACQ] Waiting for depth: Z_std={z_std*1000:.1f}mm > {Z_STABLE_THRESH*1000:.0f}mm "
                    f"(stable_count={stable_count})")
                time.sleep(0.001)
                continue
            match_type = "XY-only (depth unstable)" if _used_xy_only else "full 3D"
            dx = (x - matched_seed[0]) * 1000
            dy = (y - matched_seed[1]) * 1000
            dz = (z - matched_seed[2]) * 1000
            drift_xy = math.hypot(dx, dy)
            _accept_msg = (
                f"[REACQ] ACCEPTED ({match_type}) after {time.time()-start-DEPTH_SETTLE_S:.1f}s | "
                f"count={stable_count} Z_std={z_std*1000:.1f}mm | "
                f"[{x:.3f},{y:.3f},{z:.3f}] | seed=[{matched_seed[0]:.3f},{matched_seed[1]:.3f},{matched_seed[2]:.3f}] | "
                f"drift=({dx:.0f},{dy:.0f},{dz:.0f})mm XY={drift_xy:.0f}mm"
            )
            if drift_xy > 25:
                node.get_logger().warn(_accept_msg + " ← LARGE DRIFT, verify correct fruit")
            else:
                node.get_logger().info(_accept_msg)
            return (x, y, z)

        time.sleep(0.001)

    # Timeout — soft-accept if we have ≥3 stable readings with good Z
    elapsed = time.time() - start - DEPTH_SETTLE_S
    if stable_count >= 3 and matched_seed is not None and len(recent_zs) >= 2:
        z_std = float(_np.std(recent_zs))
        if z_std <= Z_STABLE_THRESH and prev_pose_tuple is not None:
            x, y, z = prev_pose_tuple
            dx = (x - matched_seed[0]) * 1000
            dy = (y - matched_seed[1]) * 1000
            dz = (z - matched_seed[2]) * 1000
            drift_xy = math.hypot(dx, dy)
            node.get_logger().warn(
                f"[REACQ] SOFT-ACCEPT at timeout | stable={stable_count}/{stable_needed} "
                f"Z_std={z_std*1000:.1f}mm | [{x:.3f},{y:.3f},{z:.3f}] | "
                f"drift=({dx:.0f},{dy:.0f},{dz:.0f})mm XY={drift_xy:.0f}mm")
            return (x, y, z)

    # Hard timeout — nothing usable
    if _closest_miss is not None:
        _cm, _cs, _dxy, _dz = _closest_miss
        node.get_logger().warn(
            f"[REACQ] TIMEOUT after {elapsed:.1f}s | stable={stable_count}/{stable_needed} | "
            f"closest was [{_cm[0]:.3f},{_cm[1]:.3f},{_cm[2]:.3f}] "
            f"dxy={_dxy*100:.1f}cm dz={_dz*100:.1f}cm (need xy<{radius*100:.0f}cm z<{z_tolerance*100:.0f}cm)")
    else:
        node.get_logger().warn(
            f"[REACQ] TIMEOUT after {elapsed:.1f}s | stable={stable_count}/{stable_needed} | "
            f"no fruits visible in {len(seeds)} seed windows")
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
            try:
                node.obstacles.update_pose("fruit_obstacle", new_xyz)
            except Exception as _e:
                node.get_logger().warn(f"obstacle update_pose failed (CUDA faulted?): {_e}")

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

        standoff = 0.10  # standoff distance from fruit for approach pose
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
            # Mid/high: standoff directly behind fruit in Y only — same X and Z as fruit
            # Right-side fruits get a larger standoff to improve approach angle
            _is_right = x < (node.trunk_x or 0.16)
            _midhi_standoff = 0.12 if _is_right else standoff
            ax = x
            ay = y + _midhi_standoff
            az = z
            node.get_logger().info(
                f"MID/HIGH approach (z={z:.2f} >= {LOW_Z_THRESH}): "
                f"fruit=[{x:.3f},{y:.3f},{z:.3f}] standoff=[{ax:.3f},{ay:.3f},{az:.3f}] "
                f"standoff_dist={_midhi_standoff:.2f} right={_is_right}")

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
            LOW_SIDE_HOME_THRESH = 0.07  # only trigger side HOME for LOW when >7cm lateral
            if dx_ee_to_fruit > (LOW_SIDE_HOME_THRESH if is_low else LATERAL_THRESH):
                if x > trunk_x:
                    side_joints = node.home_left_low_joints if is_low else node.home_left_joints
                    side_label = "HOME_LEFT_LOW" if is_low else "HOME_LEFT"
                else:
                    side_joints = node.home_right_low_joints if is_low else node.home_right_joints
                    side_label = "HOME_RIGHT_LOW" if is_low else "HOME_RIGHT"
                node.get_logger().info(
                    f"{side_label}: fruit x={x:.2f}, trunk_x={trunk_x:.3f}")
                if node.current_joint_positions is not None:
                    side_joints = nearest_joint_config(node.current_joint_positions, side_joints)
                plan_execute_js(node, side_joints, label=side_label, motion_type="home", speed_factor=0.5)
                is_side_approach = True
                # Update start state and cur_pose after side HOME
                cur_pose = node.get_end_effector_pose()
                cur_quat = cur_pose[3:] if cur_pose else cur_quat
                start = JointState.from_position(
                    torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
                    joint_names=node.joint_order,
                )
            else:
                node.get_logger().info("Fruit near center; using default HOME without side move.")

        # Skip approach if EE is already close to the goal
        skip_approach = False
        if cur_pose is not None:
            ee_dist = math.sqrt((cur_pose[0] - x)**2 + (cur_pose[1] - y)**2 + (cur_pose[2] - z)**2)
            node.get_logger().info(f"EE-to-goal distance: {ee_dist*100:.1f}cm")

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
            node.reacquire_result = ""  # reset at start of each attempt
            pass  # jump straight to reacquire + final below


        elif is_low:
            is_low_lateral = abs(x - trunk_x) > LATERAL_THRESH
            side_blend = 0.25 if (is_low_lateral and is_side_approach) else (0.10 if is_low_lateral else 0.25)
            orientation = minimize_rotation_orientation(cur_quat, target_quat, blend_weight=side_blend)
            approach = [ax, ay + 0.07, az - 0.09, *orientation]
            node.get_logger().info(
                f"LOW approach pose: {approach[:3]}, is_side={is_side_approach}, "
                f"is_low_lateral={is_low_lateral}, blend={side_blend:.2f}")
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

        node._goal_stall_count = 0  # reset stall counter for each new goal

        if not skip_approach:
            node.reacquire_result = ""  # reset at start of each attempt
            node.motion_phase = "APPROACH"

            _ap_dist = math.sqrt(sum((cur_pose[i] - approach[i])**2 for i in range(3))) if cur_pose else 999.0
            _skip_approach_wait = False
            node.get_logger().info(f"EE-to-approach distance: {_ap_dist*100:.1f}cm")
            if _ap_dist < 0.20:
                node.get_logger().info("Approach standoff close — using direct IK (skipping cuRobo plan)")
                _approach_ok = _direct_ik_move(node, approach, label="APPROACH", motion_type="approach", store_trajectory=True)
                if not _approach_ok:
                    # IK failed (branch mismatch) — cuRobo handles branch switching via TRAJOPT.
                    # Path will be a detour but gets the arm to the correct approach position.
                    node.get_logger().info(
                        f"IK failed — cuRobo plan to approach {[round(v,3) for v in approach[:3]]}"
                    )
                    _start_after_ik_fail = JointState.from_position(
                        torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
                        joint_names=node.joint_order,
                    )
                    _approach_ok = plan_and_send(
                        node, _start_after_ik_fail,
                        Pose.from_list(approach),
                        label="APPROACH",
                        motion_type="approach",
                        goal_xyz=approach[:3],
                        store_trajectory=True,
                    )
                if not _approach_ok:
                    node.get_logger().info("All approach attempts failed — re-homing then skipping to FINAL")
                    move_to_home_position(node)
                    if _check_stop(): break
                    _approach_ok = True
                    _skip_approach_wait = True
            else:
                _approach_ok = plan_and_send(node, start, Pose.from_list(approach), label="APPROACH", motion_type="approach", goal_xyz=approach[:3], store_trajectory=True)
            if not _approach_ok:
                unlock_target(node)
                continue
            # Open gripper during approach motion (arm is already moving)
            if not gripper_opened:
                gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius)
                gripper_opened = True
            if not _skip_approach_wait:
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

        # 2. Soft reacquire — quick confirmation within tight radius of original seed.
        # Initial detection is accurate enough; this just refines the position slightly.
        # Falls back to seed immediately if no match found — no nudge, no failure.
        seed = [x, y, z]
        node.motion_phase = "REACQUIRE"
        node.reacquire_result = "SEARCH"
        node._publish_goal_info()
        reacq = reacquire_goal_pose(
            node,
            seed_xyz=seed,
            candidate_seeds=[],   # tight to seed only — no roaming
            timeout=3.0,
            radius=0.03,          # 3cm — tighter than default 4cm
            z_tolerance=0.002,    # 2mm
            depth_settle_s=1.2,
            stable_needed=2,
        )
        if reacq:
            x, y, z = reacq
            publish_goal_marker(node, [x, y, z])
            node.get_logger().info(f"[REACQ] Refined: [{x:.3f}, {y:.3f}, {z:.3f}]")
            node.reacquire_result = "OK"
        else:
            x, y, z = seed
            node.get_logger().info(f"[REACQ] No match — using original seed: [{x:.3f}, {y:.3f}, {z:.3f}]")
            node.reacquire_result = "SEED"
        node._publish_goal_info()
            
        if _check_stop(): break

        # If reacquired position drifted too far from seed, fall back to seed.
        # Large drift means a neighbouring fruit was matched or depth was unreliable.
        _drift_xy = math.hypot(x - seed[0], y - seed[1]) * 1000  # mm
        _drift_z  = abs(z - seed[2]) * 1000                       # mm
        if _drift_xy > 35.0 or _drift_z > 20.0:
            node.get_logger().warn(
                f"[REACQ] Large drift (XY={_drift_xy:.0f}mm Z={_drift_z:.0f}mm) — reverting to original seed "
                f"[{seed[0]:.3f},{seed[1]:.3f},{seed[2]:.3f}]")
            x, y, z = seed[0], seed[1], seed[2]

        # 3. Final slow precise grasp — IK + direct joint interpolation (no cuRobo trajectory)
        #    _direct_ik_move handles wait + blend internally
        # Reuse orientation from APPROACH step — all orientation changes happen during approach only
        node.get_logger().info(f"FINAL orientation: reusing APPROACH orientation (is_low={is_low})")
        z_offset = 0.025  # small downward adjustment during grasp
        y_offset = -0.032  # no lateral adjustment
        fruit_radius = getattr(node, 'latest_fruit_radius', None) or 0.035
        # Recompute approach direction from current EE → reacquired fruit (not stale state_manager).
        # This ensures the approach vector is accurate after the arm has settled at standoff.
        _cur_ee = node.get_end_effector_pose()
        approach_dir = None
        if _cur_ee is not None:
            _dx = x - _cur_ee[0]; _dy = y - _cur_ee[1]; _dz = z - _cur_ee[2]
            _dist = math.sqrt(_dx**2 + _dy**2 + _dz**2)
            if _dist > 0.01:
                approach_dir = [_dx/_dist, _dy/_dist, _dz/_dist]
                node.get_logger().info(
                    f"FINAL approach_dir recomputed from EE→fruit: "
                    f"[{approach_dir[0]:.3f},{approach_dir[1]:.3f},{approach_dir[2]:.3f}] dist={_dist:.3f}m"
                )
        if approach_dir is None:
            _sm_dir = getattr(getattr(node, 'state_manager', None), 'fruit_direction', None)
            approach_dir = list(_sm_dir) if _sm_dir is not None else None
        # Place TCP at the fruit centroid (no radius pullback).
        # Pulling back by fruit_radius leaves the fruit at the gripper entrance — easy to slip.
        # With TCP at the centroid, the fruit sits deep inside the three-finger cup.
        # approach_dir is retained for logging/debug but no longer shifts the target.
        gx, gy, gz = x, y - y_offset, z + z_offset
        final_target = [gx, gy, gz, *orientation]
        node.motion_phase = "FINAL"
        # Suppress heatmap/scoring during final grasp — target is locked, no new selection needed
        if hasattr(node, 'set_vision_mode'):
            node.set_vision_mode("reacquire")
        final_ok = _direct_ik_move(node, final_target, label="FINAL",
                                   motion_type="final", store_trajectory=True)
        if not final_ok:
            # IK branch mismatch — redo approach with blend_weight=0.0 (keep current orientation,
            # no target blend) to find a different IK branch, then retry FINAL.
            node.get_logger().warn("FINAL IK failed — redoing approach with blend_weight=0.0...")
            _cur_quat_now = node.get_end_effector_pose()
            _cur_quat_now = _cur_quat_now[3:] if _cur_quat_now else orientation
            _orient_zero = minimize_rotation_orientation(_cur_quat_now, target_quat, blend_weight=0.0)
            _approach_reorient = list(approach[:3]) + list(_orient_zero)
            node.get_logger().info(
                f"FINAL retry: approach with blend_weight=0.0 → {[round(v,3) for v in _approach_reorient[:3]]}"
            )
            _direct_ik_move(node, _approach_reorient, label="APPROACH_REORIENT",
                            motion_type="approach", store_trajectory=False)
            final_ok = _direct_ik_move(node, final_target, label="FINAL_RETRY",
                                       motion_type="final", store_trajectory=True)
        if not final_ok:
            node.get_logger().warn("FINAL IK retry also failed — skipping goal.")
            if hasattr(node, 'set_vision_mode'):
                node.set_vision_mode("full")
            unlock_target(node)
            continue
        log_path_deviation(node, "FINAL")

        if _check_stop(): break
        node.control_gripper("CLOSE")
        time.sleep(0.5)
        if _check_stop():
            break
        # Restore full vision mode now that the grasp is done
        if hasattr(node, 'set_vision_mode'):
            node.set_vision_mode("full")
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
            # Sensor-driven: target contact at ~40% closure (step 4/10).
            # Too late  (>60%) = fruit at entrance → move forward.
            # Too early (<20%) = gripper overshot fruit → pull back.
            # Stopped early counts as proper regardless of contact step.
            contact_ratio = first_contact / max(gc.steps, 1)
            # Force confirmation: if ≥2 fingers show meaningful contact force,
            # the grip is physically real — don't disturb it even if timing is off.
            _force_confirmed = sum(1 for d in deltas if d > 1.5) >= 2
            if stopped_early or (0.20 <= contact_ratio <= 0.60) or _force_confirmed:
                prediction = "PROPER"
                action = "PROCEED"
                if _force_confirmed and not (stopped_early or (0.20 <= contact_ratio <= 0.60)):
                    node.get_logger().info(
                        f"Grip timing off ({contact_ratio:.0%}) but force confirms contact "
                        f"(deltas={[f'{d:.2f}' for d in deltas]}N) — skipping regrip"
                    )
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

                # Depth correction: target contact_ratio = 0.40 (step 4/10).
                # Negative = move forward (toward fruit), positive = pull back (away from fruit).
                # Both directions allowed: late contact → forward, early contact → back.
                contact_ratio = first_contact / max(gc.steps, 1)
                forward_correction = -(contact_ratio - 0.40) * 0.04  # 16mm range each direction
                forward_correction = max(-0.025, min(0.015, forward_correction))  # clamp ±

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
                # Use _direct_ik_move — stays on the same kinematic branch.
                # exec_pose (full cuRobo planner) can plan a wild arc for a tiny correction.
                _direct_ik_move(node, closer_target, label="REGRIP",
                                motion_type="final", store_trajectory=False)
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

        # Depth correction: nudge gripper forward (gripper closed) to bring contact
        # ratio closer to ideal 0.40. Only applied when grip is accepted (no regrip
        # needed) but fruit is shallower than ideal (ratio > 0.40).
        # Gripper stays closed — this deepens the cup around the fruit without releasing.
        contact_ratio_now = first_contact / max(gc.steps, 1)
        _depth_correction = -(contact_ratio_now - 0.40) * 0.04  # same formula as REGRIP
        _depth_correction = max(-0.020, min(0.0, _depth_correction))  # forward only, ≤20mm
        if abs(_depth_correction) > 0.003:  # only move if correction > 3mm
            _cur_for_depth = node.get_end_effector_pose()
            if _cur_for_depth:
                _depth_target = [
                    _cur_for_depth[0],
                    _cur_for_depth[1] + _depth_correction,
                    _cur_for_depth[2],
                    *_cur_for_depth[3:]
                ]
                node.get_logger().info(
                    f"[DEPTH] contact_ratio={contact_ratio_now:.0%} → nudging "
                    f"{_depth_correction*1000:+.1f}mm forward to deepen grip"
                )
                _direct_ik_move(node, _depth_target, label="DEPTH_CORRECT",
                                motion_type="final", store_trajectory=False)

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
        node.motion_phase = "REVERSING"
        execute_partial_reverse(node, clearance_m=0.28)

        # Flush any async CUDA errors that accumulated during FINAL IK/planning.
        # They surface at the next CUDA op — force them here so DROP-OFF gets a clean state.
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            node._cuda_faulted = True
        if getattr(node, "_cuda_faulted", False):
            try_cuda_recovery(node)

        # After partial reverse, return to the correct home position:
        # - Side approach: go back to home_left or home_right (arm is near trunk, needs to clear)
        # - Center approach: try direct dropoff, fall back to center HOME only if planning fails
        if is_side_approach:
            node.get_logger().info("Side approach — returning to center HOME before dropoff")
            node.motion_phase = "HOME"
            move_to_home_position(node)

        node.motion_phase = "DROPOFF"
        if not move_to_dropoff_position(node):
            node.get_logger().info("Direct dropoff failed, going to center HOME first")
            node.motion_phase = "HOME"
            move_to_home_position(node)
            node.motion_phase = "DROPOFF"
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
                    nax = nx
                    nay = ny + next_standoff
                    naz = nz

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
                        node.motion_phase = "HOME"
                        move_to_home_position(node)
                else:
                    node.motion_phase = "HOME"
                    move_to_home_position(node)
            else:
                node.motion_phase = "HOME"
                move_to_home_position(node)
        else:
            node.motion_phase = "HOME"
            move_to_home_position(node)

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
