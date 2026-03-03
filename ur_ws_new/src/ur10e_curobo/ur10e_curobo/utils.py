# ruff: noqa
import sys, termios, tty, select, time, math
from typing import Iterable, List, Optional, Callable
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import rclpy
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time as rclpyTime
from tf2_ros import Buffer, TransformListener


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

def build_trajectory(joint_names: List[str], states: Iterable[List[float]], vel: float = 0.1, dt: float = 0.02,
                     stop_flag: Optional[Callable[[], bool]] = None,
                     max_vel: float = 1.5, max_acc: float = 2.0, ramp_points: int = 8) -> JointTrajectory:
    """
    Build a simple trajectory — positions + constant dt only.
    cuRobo already handles velocity profiling in its interpolated plan.
    """
    msg = JointTrajectory()
    msg.joint_names = joint_names

    states_list = list(states)
    n_points = len(states_list)

    if n_points == 0:
        return msg

    n_joints = len(joint_names)
    t = 0.0
    for i, q in enumerate(states_list):
        if stop_flag and stop_flag():
            break
        pt = JointTrajectoryPoint()
        pt.positions = list(q)
        if i == 0 or i == n_points - 1:
            pt.velocities = [0.0] * n_joints
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
        msg.points.append(pt)
        t += dt

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
    STALL_TIMEOUT = 2.0  # only timeout if robot not moving for 2s

    try:
        node.get_logger().info(f"Waiting for EE → {[round(x, 3) for x in target_xyz]} (tol={tol})")
        while getattr(node, "running", True):
            # Safety exit: stop signal
            if getattr(node, "stop_requested", False):
                node.get_logger().warn("Stop requested during wait; holding.")
                publish_stop_trajectory(node)
                node.stop_requested = False
                break

            # Try to get current pose (catch FK/TF errors)
            try:
                cur = node.get_end_effector_pose()
            except Exception as e:
                node.get_logger().warn(f"FK/TF error in wait loop: {e}")
                cur = None

            # Skip if no valid FK
            if not cur or any(math.isnan(v) for v in cur[:3]):
                time.sleep(0.05)
                continue

            dist = math.dist(cur[:3], target_xyz)

            # Check distance to goal
            if dist < tol:
                # Orientation check (if requested)
                if target_quat and len(cur) >= 7:
                    dot = abs(sum(a * b for a, b in zip(cur[3:7], target_quat[:4])))
                    if dot < quat_tol:
                        time.sleep(0.05)
                        continue

                # Velocity check — ensure robot is settling, not just passing through
                vels = getattr(node, 'current_joint_velocities', None)
                if vels and max(abs(v) for v in vels) > 0.01:
                    time.sleep(0.05)
                    continue

                node.get_logger().info(f"End-effector reached target. dist={dist*100:.1f}cm")
                break

            # Stall detection: only timeout if robot has stopped moving
            if last_dist is not None and abs(dist - last_dist) < 0.001:
                if stall_start is None:
                    stall_start = time.time()
                elif time.time() - stall_start > STALL_TIMEOUT:
                    node.get_logger().warn(
                        f"Robot not moving (dist={dist*100:.1f}cm from target, stalled {STALL_TIMEOUT}s) — accepting.")
                    break
            else:
                stall_start = None
            last_dist = dist

            time.sleep(0.05)

            time.sleep(0.05)

    except Exception as e:
        node.get_logger().error(f"Error during wait_until_xyz: {e}")
        publish_stop_trajectory(node)

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
