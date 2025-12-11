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

def build_trajectory(joint_names: List[str], states: Iterable[List[float]], vel: float, dt: float,
                     stop_flag: Optional[Callable[[], bool]] = None) -> JointTrajectory:
    """
    Build smooth trajectory with velocity ramping and acceleration control.
    Ramps velocity up at start, maintains constant in middle, ramps down at end.
    """
    msg = JointTrajectory()
    msg.joint_names = joint_names

    states_list = list(states)
    if not states_list:
        return msg

    n = len(states_list)
    if n == 1:
        # Single point - just hold position
        pt = JointTrajectoryPoint()
        pt.positions = list(states_list[0])
        pt.velocities = [0.0] * len(joint_names)
        pt.accelerations = [0.0] * len(joint_names)
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 0
        msg.points.append(pt)
        return msg

    # Smooth velocity profile with ramp up/down
    # First 20% and last 20% are ramps, middle 60% is constant
    ramp_fraction = 0.2
    ramp_points = max(2, int(n * ramp_fraction))

    t = 0.0
    for i, q in enumerate(states_list):
        if stop_flag and stop_flag():
            break

        pt = JointTrajectoryPoint()
        pt.positions = list(q)

        # Compute smooth velocity profile
        if i < ramp_points:
            # Ramp up (smooth acceleration)
            alpha = i / ramp_points
            # Use sine curve for smoother acceleration
            smooth_alpha = (1 - math.cos(alpha * math.pi)) / 2
            v_scale = smooth_alpha
        elif i > n - ramp_points:
            # Ramp down (smooth deceleration)
            alpha = (n - i) / ramp_points
            smooth_alpha = (1 - math.cos(alpha * math.pi)) / 2
            v_scale = smooth_alpha
        else:
            # Constant velocity in the middle
            v_scale = 1.0

        # Apply scaled velocity
        current_vel = vel * v_scale
        pt.velocities = [current_vel] * len(joint_names)

        # Compute accelerations (for smoother motion)
        if i == 0:
            # Start from rest
            pt.accelerations = [0.0] * len(joint_names)
        elif i == n - 1:
            # End at rest
            pt.accelerations = [0.0] * len(joint_names)
        else:
            # Smooth acceleration based on velocity change
            # This helps the robot controller plan smoother motion
            accel = (v_scale - v_scale) / dt if i > 0 else 0.0
            pt.accelerations = [accel * 0.5] * len(joint_names)

        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
        msg.points.append(pt)
        t += dt

    return msg


def wait_until_xyz(node, target_xyz, tol: float = 0.005, timeout: float = 15.0):
    """
    Wait until the end-effector reaches the target XYZ (within tolerance).
    Stops gracefully if stop_requested or timeout occurs.
    """
    from .motions import publish_stop_trajectory
    start_time = time.time()
    last_pos = None
    stationary_count = 0

    try:
        node.get_logger().info(f"Waiting for EE → {[round(x, 3) for x in target_xyz]} (tol={tol})")
        while getattr(node, "running", True):
            # Safety exit: timeout
            if time.time() - start_time > timeout:
                node.get_logger().warn("Timeout waiting for EE to reach target. Allowing robot to settle...")
                # Don't stop trajectory - let it finish naturally
                time.sleep(1.0)  # Give robot time to settle
                break

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

            # Check distance to goal
            dist = math.dist(cur[:3], target_xyz)
            if dist < tol:
                node.get_logger().info(f"✅ End-effector reached target (dist={dist*1000:.1f}mm)")
                break

            # Check if robot stopped moving (reached a stable position, even if not the target)
            if last_pos is not None:
                move_dist = math.dist(cur[:3], last_pos)
                if move_dist < 0.001:  # Less than 1mm movement
                    stationary_count += 1
                    if stationary_count > 20:  # 1 second of no movement (20 * 0.05s)
                        node.get_logger().info(f"Robot stationary at {dist*1000:.1f}mm from target. Continuing...")
                        break
                else:
                    stationary_count = 0
            last_pos = cur[:3]

            # Log progress every 2 seconds
            if int(time.time() - start_time) % 2 == 0 and (time.time() - start_time) % 1 < 0.05:
                node.get_logger().info(f"Moving... distance to target: {dist*1000:.1f}mm")

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
        node.get_logger().warn(f"TF lookup failed for camera frame, using default: {e}")
        # default: ZED points forward in -Y and slightly up → safe fallback
        cam_fwd = np.array([-0.1, -0.8, 0.6])
        cam_fwd /= np.linalg.norm(cam_fwd)

    # Move opposite direction of camera -> back + down
    ax = x - cam_fwd[0] * dist
    ay = y - cam_fwd[1] * dist
    az = z - cam_fwd[2] * dist

    return ax, ay, az
