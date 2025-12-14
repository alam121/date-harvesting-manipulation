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


def interpolated_positions(result):
    interp = result.get_interpolated_plan()
    if isinstance(interp, JointState):
        interp = interp.position
    if not isinstance(interp, torch.Tensor):
        interp = torch.tensor(interp, dtype=torch.float32)
    return interp.to("cpu").tolist()


def publish_stop_trajectory(node):
    if node.current_joint_positions is None:
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
    if node.current_joint_positions is None:
        node.get_logger().warn("No joint state; cannot execute pose."); return
    
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
    # 2. Speed scaling (applies global multiplier)
    # ------------------------------
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier

    dt = base_dt / scale
    dt = min(max(dt, planner.min_dt), planner.max_dt)   # clamp using config limits

    # ------------------------------
    # 6. velocity smoothing
    # ------------------------------
    base_vel = 0.10
    vel = min(base_vel * scale, planner.max_traj_velocity)



    traj = build_trajectory(
        node.joint_order,
        interpolated_positions(res),
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=planner.ramp_points,
    )
    node.trajectory_pub.publish(traj)


# Plans and executes a joint-space motion to reach a specified set of joint angles.
# Input: target_joints: List of joint angles in radians.
        # label: A string label for logging purposes.
        # dt: Time step for trajectory interpolation.
        
# Publishes /joint_trajectory_controller/joint_trajectory.
def plan_execute_js(node, target_joints: List[float], label: str, motion_type: str = "default"):
    if node.current_joint_positions is None:
        node.get_logger().warn(f"No joint state; skipping {label} move.")
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
    # 2. Speed scaling (applies global multiplier)
    # ------------------------------
    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier

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
    dt = min(max(dt, planner.min_dt), planner.max_dt)   # clamp using config limits

    # ------------------------------
    # 6. velocity smoothing
    # ------------------------------
    base_vel = 0.10
    vel = min(base_vel * scale, planner.max_traj_velocity)

    # ------------------------------
    # 7. Build trajectory with smoothness params
    # ------------------------------
    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=planner.ramp_points,
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


def rotate_wrist(node, degrees: float, duration: float = 0.30):
    if node.current_joint_positions is None:
        node.get_logger().error("No joint state; cannot rotate wrist."); return
    rad = math.radians(degrees)
    start = node.current_joint_positions.copy(); plus = start.copy(); plus[5] = start[5] + rad
    traj = JointTrajectory(); traj.joint_names = node.joint_order
    def _pt(q, t):
        p = JointTrajectoryPoint(); p.positions = q; p.velocities = [0.0]*len(q)
        p.time_from_start.sec = int(t); p.time_from_start.nanosec = int((t-int(t))*1e9); return p
    traj.points = [_pt(start, 0.0), _pt(plus, duration), _pt(start, 2*duration)]
    last = traj.points[-1]; hold_ns = last.time_from_start.nanosec + 50_000_000
    p3 = _pt(start, float(last.time_from_start.sec)); p3.time_from_start.sec += 1 if hold_ns>=1_000_000_000 else 0
    p3.time_from_start.nanosec = hold_ns % 1_000_000_000; traj.points.append(p3)
    node.trajectory_pub.publish(traj)


def move_backward(node, delta: float):
    if node.current_joint_positions is None:
        node.get_logger().error("No joint state; cannot move.")
        return
    
    q = node.current_joint_positions.copy(); q[1] -= delta
    traj = JointTrajectory(); traj.joint_names = node.joint_order
    p = JointTrajectoryPoint(); p.positions = q; p.time_from_start.sec = 1
    traj.points.append(p); node.trajectory_pub.publish(traj)

def blend_motion(node, pause=0.2):
    # maintains smoothness, avoids jerk
    planner = node.cfg.planner
    traj = build_trajectory(
        node.joint_order,
        [node.current_joint_positions],
        vel=0.05,
        dt=0.02,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity,
        max_acc=planner.max_joint_acceleration,
        ramp_points=planner.ramp_points,
    )
    node.trajectory_pub.publish(traj)
    time.sleep(pause)