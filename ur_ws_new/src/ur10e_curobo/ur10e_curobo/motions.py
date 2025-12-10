# ruff: noqa
import math, time, torch
from typing import List
from curobo.types.math import Pose
from curobo.types.robot import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectoryPoint

from .config import PLAN_CFG_DEFAULT
from .utils import build_trajectory, wait_until_xyz
from .fk import forward_kinematics


def _joint_state_ready(node, label: str) -> bool:
    """Quick guard to avoid planning with missing joint feedback."""
    if node.current_joint_positions is None:
        node.get_logger().warn(f"No joint state; skipping {label}.")
        return False
    if len(node.current_joint_positions) != len(node.joint_order):
        node.get_logger().warn(
            f"Incomplete joint state ({len(node.current_joint_positions)}/{len(node.joint_order)}); skipping {label}."
        )
        return False
    return True


def interpolated_positions(result):
    interp = result.get_interpolated_plan()
    if isinstance(interp, JointState):
        interp = interp.position
    if not isinstance(interp, torch.Tensor):
        interp = torch.tensor(interp, dtype=torch.float32)
    return interp.to("cpu").tolist()


def publish_stop_trajectory(node):
    if not _joint_state_ready(node, "stop trajectory"):
        return
    stop = JointTrajectory(); stop.joint_names = node.joint_order
    pt = JointTrajectoryPoint(); pt.positions = list(node.current_joint_positions)
    pt.velocities = [0.0]*len(node.joint_order); pt.accelerations = [0.0]*len(node.joint_order)
    pt.time_from_start.nanosec = 1_000_000; stop.points = [pt]
    node.trajectory_pub.publish(stop)


# Executes a single pose in Cartesian space.
#input: pose: List of 7 elements [x,y,z,qw,qx,qy,qz]

#output: Publishes /joint_trajectory_controller/joint_trajectory.
def execute_single_pose(node, pose: list, motion_type: str = "default"):
    if not _joint_state_ready(node, "pose execution"):
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = JointState.from_position(
        torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )
    goal = Pose.from_list(pose)
    res = node.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT) #cuRobo Cartesian planner
    if not res.success:
        node.get_logger().warn("Plan failed for single pose."); return
    
    planner = node.cfg.planner
    base_dt = planner.base_dt  # usually 0.02
    # ------------------------------
    # 2. Speed scaling
    # ------------------------------
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0)

    dt = base_dt / scale
    dt = min(max(dt, 0.015), 0.03)   # clamp for UR stability

    # ------------------------------
    # 6. velocity smoothing
    # ------------------------------
    base_vel = 0.10
    vel = min(base_vel * scale, 0.25)



    traj = build_trajectory(
        node.joint_order,
        interpolated_positions(res),
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested
)
    node.trajectory_pub.publish(traj)


# Plans and executes a joint-space motion to reach a specified set of joint angles.
# Input: target_joints: List of joint angles in radians.
        # label: A string label for logging purposes.
        # dt: Time step for trajectory interpolation.
        
# Publishes /joint_trajectory_controller/joint_trajectory.
def plan_execute_js(node, target_joints: List[float], label: str, motion_type: str = "default"):
    if not _joint_state_ready(node, f"{label} move"):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 1. Build joint states
    # ------------------------------
    start = JointState.from_position(
        torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )

    goal_js = JointState.from_position(
        torch.tensor([target_joints], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )

    # ------------------------------
    # 2. Speed scaling
    # ------------------------------
    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0)

    # ------------------------------
    # 3. cuRobo plan
    # ------------------------------
    res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_DEFAULT)
    if not res.success:
        node.get_logger().warn(f"Joint-space plan to {label} failed.")
        return

    # ------------------------------
    # 4. Interpolate (older cuRobo API)
    # ------------------------------
    # No args allowed
    states = interpolated_positions(res)

    # ------------------------------
    # 5. dt smoothing (critical)
    # ------------------------------
    base_dt = planner.base_dt  # usually 0.02
    dt = base_dt / scale
    dt = min(max(dt, 0.015), 0.03)   # clamp for UR stability

    # ------------------------------
    # 6. velocity smoothing
    # ------------------------------
    base_vel = 0.10
    vel = min(base_vel * scale, 0.25)

    # ------------------------------
    # 7. Build trajectory
    # ------------------------------
    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
    )

    node.get_logger().info(f"Moving to {label} (vel={vel:.2f}, dt={dt:.3f})")
    node.trajectory_pub.publish(traj)

    # ------------------------------
    # 8. Wait for the robot
    # ------------------------------
    fk = forward_kinematics(node, states[-1])
    if fk:
        wait_until_xyz(node, [fk.x, fk.y, fk.z])



def move_to_home_position(node):
    plan_execute_js(node, node.home_joints, label="HOME", motion_type="home")


def move_to_dropoff_position(node):
    plan_execute_js(node, node.dropoff_joints, label="DROP-OFF", motion_type="dropoff")
 

def move_to_predropoff_position(node):
    plan_execute_js(node, node.predropoff_joints, label="preDROP-OFF", motion_type="predropoff")


def rotate_wrist(
    node,
    degrees: float,
    hold_s: float = 0.2,
    return_to_start: bool = True,
    duration_s: float = None,
    vel: float = 0.14,
):
    if not _joint_state_ready(node, "wrist rotation"):
        return

    rad = math.radians(degrees)
    start = list(node.current_joint_positions)

    # UR wrist_3 is continuous → no clamping needed
    target = start.copy()
    target[5] = start[5] + rad            # <-- KEY FIX (continuous joint)

    # Build interpolated path
    base_dt = getattr(node.cfg.planner, "base_dt", 0.02)
    base_dt = min(max(base_dt, 0.015), 0.03)

    if duration_s is not None:
        steps = max(4, int(duration_s / base_dt))
    else:
        max_step = 0.12
        steps = max(8, int(abs(rad) / max_step))

    states = []
    for i in range(1, steps + 1):
        frac = i / steps
        q = start.copy()
        q[5] = start[5] + rad * frac
        states.append(q)

    hold_steps = max(1, int(hold_s / base_dt))
    states.extend([target.copy()] * hold_steps)

    if return_to_start:
        for i in range(1, steps + 1):
            frac = i / steps
            q = target.copy()
            q[5] = target[5] - rad * frac
            states.append(q)

    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=base_dt,
        stop_flag=lambda: node.stop_requested,
    )

    node.get_logger().info(
        f"Twisting wrist by {degrees:.1f}deg (hold {hold_s:.2f}s, return={return_to_start})"
    )
    node.trajectory_pub.publish(traj)



def move_backward(node, delta: float):
    if not _joint_state_ready(node, "backward move"):
        return
    
    q = node.current_joint_positions.copy(); q[1] -= delta
    traj = JointTrajectory(); traj.joint_names = node.joint_order
    p = JointTrajectoryPoint(); p.positions = q; p.time_from_start.sec = 1
    traj.points.append(p); node.trajectory_pub.publish(traj)

def blend_motion(node, pause=0.2):
    if not _joint_state_ready(node, "blend motion"):
        return
    # maintains smoothness, avoids jerk
    traj = build_trajectory(
        node.joint_order,
        [node.current_joint_positions],
        vel=0.05,
        dt=0.02,
        stop_flag=lambda: node.stop_requested
    )
    node.trajectory_pub.publish(traj)
    time.sleep(pause)
