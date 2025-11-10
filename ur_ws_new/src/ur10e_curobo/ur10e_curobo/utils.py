# ruff: noqa
import sys, termios, tty, select, time, math
from typing import Iterable, List, Optional, Callable
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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
    
    msg = JointTrajectory()
    msg.joint_names = joint_names
    
    t = 0.0
    for q in states:
        if stop_flag and stop_flag():
            break
        pt = JointTrajectoryPoint()
        pt.positions = list(q)
        pt.velocities = [vel] * len(joint_names)
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
        msg.points.append(pt)
        t += dt
    return msg


def wait_until_xyz(node, target_xyz, tol: float = 0.005, timeout: float = 10.0):
    """
    Wait until the end-effector reaches the target XYZ (within tolerance).
    Stops gracefully if stop_requested or timeout occurs.
    """
    from .motions import publish_stop_trajectory
    start_time = time.time()

    try:
        node.get_logger().info(f"Waiting for EE → {[round(x, 3) for x in target_xyz]} (tol={tol})")
        while getattr(node, "running", True):
            # Safety exit: timeout
            if time.time() - start_time > timeout:
                node.get_logger().warn("Timeout waiting for EE to reach target.")
                publish_stop_trajectory(node)
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
            if math.dist(cur[:3], target_xyz) < tol:
                node.get_logger().info("✅ End-effector reached target position.")
                break

            time.sleep(0.05)

    except Exception as e:
        node.get_logger().error(f"Error during wait_until_xyz: {e}")
        publish_stop_trajectory(node)
