# ruff: noqa
import sys, termios, tty, select, time, math
from typing import Iterable, List, Optional, Callable
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import rclpy
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime


def read_key(timeout=0.1):
    """Non-blocking single key read that keeps Ctrl-C working."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        # start from raw
        tty.setraw(fd)
        # re-enable signals so ^C (^Z) still generate SIGINT/SIGTSTP
        new = termios.tcgetattr(fd)
        lflag = new[3]  # lflags
        lflag |= termios.ISIG      # keep signals
        new[3] = lflag
        termios.tcsetattr(fd, termios.TCSADRAIN, new)

        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def ease_out_tail(states: List[List[float]], tail_count: int) -> List[List[float]]:
    """Re-parameterize the last `tail_count` waypoints of a joint-space path
    along arc length using an ease-out curve, so the path approaches its
    endpoint with decaying rate of change instead of a hard stop.

    build_trajectory() forces every trajectory's first and last waypoint to
    zero velocity, but with uneased, evenly-arc-spaced samples the
    second-to-last waypoint can still imply a large central-difference
    velocity right up to that forced drop -- an abrupt one-dt velocity cliff
    at the very end, not a smooth deceleration. Remapping the tail's arc-length
    progress with f(u)=u+u^2-u^3 (slope 1 at u=0, slope 0 at u=1) makes equal
    fixed-dt time steps correspond to shrinking arc-length steps near the end,
    which is what actually produces a decelerating velocity profile from a
    dt-only trajectory. Interpolating on the existing polyline preserves its
    path -- only the spacing of samples along it changes.

    Returns a new list of the same length as `states`; the input is not
    mutated. If `states` has fewer than 4 points, or the requested tail is
    degenerate (zero arc length), it is returned unchanged.
    """
    n = len(states)
    if n < 4:
        return list(states)
    tail_count = min(max(0, int(tail_count)), n - 1)
    if tail_count < 2:
        return list(states)
    tail_start = n - tail_count - 1
    tail_path = np.asarray(states[tail_start:], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(tail_path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if arc[-1] <= 1e-9:
        return list(states)
    u = np.linspace(0.0, 1.0, tail_count + 1)
    eased_u = u + u * u - u * u * u
    sample_arc = eased_u * arc[-1]
    eased_tail = np.column_stack([
        np.interp(sample_arc, arc, tail_path[:, joint_idx])
        for joint_idx in range(tail_path.shape[1])
    ])
    out = list(states)
    out[tail_start:] = eased_tail.tolist()
    return out


def ease_in_head(states: List[List[float]], head_count: int) -> List[List[float]]:
    """Mirror of ease_out_tail() for the START of a path: re-parameterize the
    first `head_count` waypoints so the path leaves its start with a rising
    rate of change instead of jumping straight to full speed.

    build_trajectory() forces the first waypoint to zero velocity (and the
    controller gets one dt to reach it from wherever it actually is), but with
    uneased, evenly-arc-spaced samples the SECOND waypoint can already imply a
    large central-difference velocity -- a one-dt jump from rest straight to
    the trajectory's cruising speed, not a smooth acceleration. This is the
    mirror-image of the tail problem ease_out_tail() fixes, and reuses its
    exact arc-length + eased-u math on the reversed head segment (an ease-out
    read backwards is an ease-in) so the two stay numerically consistent.

    Returns a new list of the same length as `states`; the input is not
    mutated. If `states` has fewer than 4 points, or the requested head is
    degenerate (zero arc length), it is returned unchanged.
    """
    n = len(states)
    if n < 4:
        return list(states)
    head_count = min(max(0, int(head_count)), n - 1)
    if head_count < 2:
        return list(states)
    head_end = head_count + 1
    reversed_head = list(reversed(states[:head_end]))
    eased_reversed = ease_out_tail(reversed_head, head_count)
    out = list(states)
    out[:head_end] = list(reversed(eased_reversed))
    return out


def build_trajectory(joint_names: List[str], states: Iterable[List[float]], vel: float = 0.1, dt: float = 0.02,
                     stop_flag: Optional[Callable[[], bool]] = None,
                     max_vel: float = 1.5, max_acc: float = 2.0, ramp_points: int = 8,
                     include_acc: bool = False) -> JointTrajectory:
    """Build a uniformly retimed Hermite trajectory within derivative limits.

    Exact cubic extrema (or Bernstein bounds for quintics) cover interpolation
    between samples, including zero-velocity endpoints. The measured-state-to-first-point bridge must be
    supplied by the caller; it cannot be validated without the measured state.
    """
    msg = JointTrajectory()
    msg.joint_names = joint_names
    states_list = list(states)
    if not states_list:
        return msg
    if not all(math.isfinite(v) and v > 0 for v in (dt, max_vel, max_acc)):
        raise ValueError("dt, max_vel and max_acc must be positive and finite")
    q = np.asarray(states_list, dtype=float)
    if q.shape != (len(states_list), len(joint_names)) or not joint_names or not np.isfinite(q).all():
        raise ValueError("Trajectory must contain complete finite joint positions")
    # Derivatives with respect to one unit sample interval, retimed together.
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    if len(q) > 2:
        v[1:-1] = (q[2:] - q[:-2]) / 2.0
        if include_acc:
            a[1:-1] = q[2:] - 2.0*q[1:-1] + q[:-2]
    if len(q) > 1:
        if include_acc:
            controls = np.stack((q[:-1], q[:-1]+v[:-1]/5,
                q[:-1]+2*v[:-1]/5+a[:-1]/20,
                q[1:]-2*v[1:]/5+a[1:]/20, q[1:]-v[1:]/5, q[1:]), axis=1)
            degree = 5
            speed_bound = float(np.max(np.abs(degree*np.diff(controls, axis=1))))
            accel_bound = float(np.max(np.abs(degree*(degree-1)*np.diff(controls, n=2, axis=1))))
        else:
            # Exact cubic Hermite extrema avoid unnecessary slowing from the
            # looser Bernstein hull. q(u)=q0+v0*u+c2*u²+c3*u³, 0<=u<=1.
            delta = q[1:] - q[:-1]
            c2 = 3*delta - 2*v[:-1] - v[1:]
            c3 = -2*delta + v[:-1] + v[1:]
            extremum = np.zeros_like(c2)
            np.divide(-c2, 3*c3, out=extremum, where=np.abs(c3) > 1e-15)
            extremum = np.clip(extremum, 0.0, 1.0)
            interior_v = v[:-1] + 2*c2*extremum + 3*c3*extremum**2
            speed_bound = max(float(np.max(np.abs(v))), float(np.max(np.abs(interior_v))))
            accel_bound = max(float(np.max(np.abs(2*c2))),
                              float(np.max(np.abs(2*c2+6*c3))))
        dt = max(dt, speed_bound/max_vel, math.sqrt(accel_bound/max_acc))
    # Round upward once so serialized intervals do not undershoot the bound.
    step_ns = max(1, math.ceil(dt * 1e9))
    dt = step_ns / 1e9
    for i, positions in enumerate(q):
        if stop_flag and stop_flag():
            msg.points.clear()
            return msg
        pt = JointTrajectoryPoint()
        pt.positions = positions.tolist()
        pt.velocities = (v[i]/dt).tolist()
        if include_acc:
            pt.accelerations = (a[i]/(dt*dt)).tolist()
        stamp_ns = (i+1)*step_ns
        pt.time_from_start.sec, pt.time_from_start.nanosec = divmod(stamp_ns, 1_000_000_000)
        msg.points.append(pt)
    return msg


def wait_until_xyz(node, target_xyz, tol: float = 0.005, timeout: float = 10.0,
                   target_quat=None, quat_tol: float = 0.99):
    """
    Wait until the end-effector reaches the target XYZ (within tolerance).
    Optionally checks orientation (quaternion dot > quat_tol) and velocity (~0).
    Stops gracefully if stop_requested or timeout occurs.
    """
    from .motions import publish_stop_trajectory
    last_dist = None
    stall_start = None
    stalled = False
    reached = False
    STALL_TIMEOUT = 2.0
    _deadline = time.monotonic() + timeout
    _min_dist_seen = float('inf')
    _prog_was_off = False

    try:
        if not getattr(node.cfg.planner, "concise_console_logs", False):
            node.get_logger().info(
                f"Waiting for EE → {[round(x, 3) for x in target_xyz]} (tol={tol})")
        _start_pose = node.get_end_effector_pose()
        _start_xyz = _start_pose[:3] if _start_pose else None
        if _start_xyz is None:
            # FK unavailable at start — skip departure check, check arrival directly
            _departed = True
        else:
            _already_there = math.dist(_start_xyz, target_xyz) < tol * 2
            _departed = _already_there
        _departure_deadline = time.monotonic() + 15.0

        while getattr(node, "running", True):

            # Completion requires the current pose, never a past closest approach.
            if time.monotonic() > _deadline:
                node.get_logger().warn("Timeout waiting for a settled target pose.")
                stalled = True
                break

            # E-stop
            if getattr(node, "stop_requested", False):
                node.get_logger().warn("Stop requested during wait; holding.")
                publish_stop_trajectory(node)
                node._stop_was_requested = True
                break

            # Robot program off — wait silently
            if not getattr(node, 'robot_running', True):
                if not _prog_was_off:
                    node.get_logger().info("Robot program OFF — waiting for restart.")
                    _prog_was_off = True
                _departure_deadline = time.monotonic() + 15.0
                stall_start = None
                last_dist = None
                time.sleep(0.5)
                continue
            if _prog_was_off:
                node.get_logger().info("Robot program back ON — resuming.")
                _prog_was_off = False

            try:
                cur = node.get_end_effector_pose()
            except Exception as e:
                node.get_logger().warn(f"FK error: {e}")
                cur = None

            state_mgr = getattr(node, '_state_mgr', None)
            received_at = getattr(state_mgr, '_joint_state_received_at', None)
            stale = received_at is not None and time.monotonic() - received_at > 0.25
            if stale or not cur or any(not math.isfinite(v) for v in cur[:3]):
                time.sleep(0.05)
                continue

            dist = math.dist(cur[:3], target_xyz)
            _min_dist_seen = min(_min_dist_seen, dist)

            # A short move can reach its goal without travelling 1 cm.
            if dist < tol:
                _departed = True

            # Departure check
            if not _departed and _start_xyz is not None:
                if math.dist(cur[:3], _start_xyz) > 0.01:
                    _departed = True
                elif time.monotonic() > _departure_deadline:
                    node.get_logger().warn("Arm has not moved 1cm in 15s — giving up.")
                    return False
            if not _departed:
                time.sleep(0.05)
                continue

            # Confirm measured position, orientation and stationary velocity.
            # No extra fixed dwell after the feedback already confirms arrival.
            orientation_ok = True
            if target_quat is not None:
                orientation_ok = False
                if len(cur) >= 7:
                    q, goal_q = cur[3:7], target_quat[:4]
                    norm = math.sqrt(sum(v*v for v in q) * sum(v*v for v in goal_q))
                    if math.isfinite(norm) and norm > 1e-12:
                        orientation_ok = abs(sum(a*b for a, b in zip(q, goal_q))) / norm >= quat_tol
            vels = getattr(node, 'current_joint_velocities', None)
            stationary = bool(vels) and all(math.isfinite(v) and abs(v) <= 0.01 for v in vels)
            if dist < tol and orientation_ok and stationary:
                reached = True
                break

            # Stall detection must use actual robot motion as well as radial
            # progress. On a curved cuRobo path the TCP can move tangentially
            # for several seconds while its straight-line distance to the goal
            # changes by less than 1 mm; distance-only detection falsely stopped
            # valid approaches in that case.
            joint_velocities = getattr(node, 'current_joint_velocities', None)
            velocity_known = (
                joint_velocities is not None and len(joint_velocities) > 0)
            robot_stationary = (
                velocity_known
                and max(abs(float(v)) for v in joint_velocities) < 0.005)
            no_radial_progress = (
                last_dist is not None and abs(dist - last_dist) < 0.001)
            if no_radial_progress and (robot_stationary or not velocity_known):
                if stall_start is None:
                    stall_start = time.monotonic()
                elif time.monotonic() - stall_start > STALL_TIMEOUT:
                    node.get_logger().warn(
                        f"Robot not moving (dist={dist*100:.1f}cm from target, "
                        f"stalled {STALL_TIMEOUT}s) — declaring motion failed.")
                    stalled = True
                    break
            else:
                stall_start = None
            last_dist = dist

            time.sleep(0.05)

    except Exception as e:
        node.get_logger().error(f"Error during wait_until_xyz: {e}")

    # Track stall count on the node — reset at start of each goal, abort after 2
    if stalled:
        node._goal_stall_count = getattr(node, '_goal_stall_count', 0) + 1

    return reached

def quat_to_rot_matrix(q):
    """
    Convert quaternion (x, y, z, w) to 3x3 rotation matrix.
    ROS2 gives quaternion as (x,y,z,w).
    """
    x, y, z, w = q

    # Compute rotation matrix
    R = np.zeros((3,3))
    R[0,0] = 1 - 2*(y*y + z*z)
    R[0,1] = 2*(x*y - z*w)
    R[0,2] = 2*(x*z + y*w)

    R[1,0] = 2*(x*y + z*w)
    R[1,1] = 1 - 2*(x*x + z*z)
    R[1,2] = 2*(y*z - x*w)

    R[2,0] = 2*(x*z - y*w)
    R[2,1] = 2*(y*z + x*w)
    R[2,2] = 1 - 2*(x*x + y*y)

    return R

def compute_visibility_approach(node, x, y, z, dist=0.15):
    """Move BACK + DOWN along the camera's viewing ray to get a clean look."""

    try:
        # Wait briefly for TF to be available
        if not node.tf_buffer.can_transform(
            "base_link",
            node.cam_frame,
            rclpyTime(),
            timeout=rclpyDuration(seconds=0.0),
        ):
            start = time.time()
            while time.time() - start < 2.0 and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.tf_buffer.can_transform(
                    "base_link",
                    node.cam_frame,
                    rclpyTime(),
                    timeout=rclpyDuration(seconds=0.0),
                ):
                    break

        t = node.tf_buffer.lookup_transform(
            "base_link",
            node.cam_frame,
            rclpyTime(),
            timeout=rclpyDuration(seconds=0.5),
        )

        q = t.transform.rotation
        R = quat_to_rot_matrix([q.x, q.y, q.z, q.w])

        # Camera forward axis = 3rd column
        cam_fwd = R[:, 2]
        cam_fwd = cam_fwd / np.linalg.norm(cam_fwd)

    except Exception as e:
        print("TF lookup failed:", e)
        # default: ZED points forward in -Y and slightly up → safe fallback
        cam_fwd = np.array([-0.1, -0.8, 0.6])
        cam_fwd /= np.linalg.norm(cam_fwd)

    # Move opposite direction of camera -> back + down
    ax = x - cam_fwd[0] * dist
    ay = y - cam_fwd[1] * dist
    az = z - cam_fwd[2] * dist

    return ax, ay, az
