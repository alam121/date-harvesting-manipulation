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
    msg = JointTrajectory(); msg.joint_names = joint_names
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


def wait_until_xyz(node, target_xyz, tol: float = 0.005):
    node.get_logger().info(f"Waiting for EE→{[round(x,3) for x in target_xyz]}…")
    while getattr(node, "running", True):
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("Stop during wait; holding.")
            from .motions import publish_stop_trajectory
            publish_stop_trajectory(node)
            node.stop_requested = False
            break
        cur = node.get_end_effector_pose()
        if not cur:
            time.sleep(0.05); continue
        if math.dist(cur[:3], target_xyz) < tol:
            node.get_logger().info("Goal reached.")
            break
        time.sleep(0.05)

