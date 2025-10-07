# ruff: noqa
import time, math, torch
from geometry_msgs.msg import Pose as ROSPose, PoseStamped
from curobo.types.math import Pose
from curobo.types.robot import JointState

from .config import PLAN_CFG_DEFAULT
from .utils import build_trajectory, wait_until_xyz
from .markers import publish_goal_marker
from .motions import interpolated_positions, execute_single_pose


def pose_to_vec7(p: ROSPose):
    return [p.position.x, p.position.y, p.position.z, p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]


def plan_and_send(node, start_state, goal_pose: Pose, dt: float, label: str) -> bool:
    res = node.motion_gen.plan_single(start_state, goal_pose, PLAN_CFG_DEFAULT)
    if not res.success:
        node.get_logger().warn(f"Plan failed for {label}."); return False
    states = interpolated_positions(res)
    traj = build_trajectory(node.joint_order, states, vel=0.1, dt=dt,
                            stop_flag=lambda: node.stop_requested)
    if node.stop_requested:
        node.get_logger().warn(f"Stop before sending {label} trajectory."); node.stop_requested = False; return False
    node.trajectory_pub.publish(traj); return True


def reacquire_goal_pose(node, seed_xyz, timeout=3.5, radius=0.08, stable_eps=0.004, stable_need=2):
    def _try_once(seed, timeout_s, rad, need):
        latest = None; stable = 0; last_hit = time.time()
        def _cb(msg: PoseStamped):
            nonlocal latest, stable, last_hit
            if is_robot_moving(node): return
            x,y,z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
            if math.hypot(x-seed[0], y-seed[1]) > rad: return
            latest = (x,y,z); stable += 1; last_hit = time.time()
        sub = node.create_subscription(PoseStamped, '/external_goal_pose', _cb, node.qos)
        try:
            while stable < max(1, need):
                if timeout_s is not None and (time.time() - last_hit) > timeout_s: break
                time.sleep(0.01)
        finally:
            node.destroy_subscription(sub)
        return latest
    t0 = time.time();
    while is_robot_moving(node) and (time.time()-t0) < 0.25: time.sleep(0.01)
    first = _try_once(seed_xyz, timeout, radius, stable_need)
    if first: return list(first)
    cur = node.get_end_effector_pose()
    if cur:
        execute_single_pose(node, [cur[0], cur[1]+0.10, cur[2], *cur[3:]]); wait_until_xyz(node, [cur[0], cur[1]+0.10, cur[2]])
    second = _try_once(seed_xyz, 2.0, max(radius,0.15), max(1, stable_need-1))
    return list(second) if second else None


def subscribe_to_goal_pose(node):
    if is_robot_moving(node):
        node.create_timer(0.5, lambda: (not is_robot_moving(node)) and subscribe_to_goal_pose(node)); return
    if hasattr(node, 'goal_pose_sub'):
        node.destroy_subscription(node.goal_pose_sub); del node.goal_pose_sub
    cur = node.get_end_effector_pose(); current_orientation = cur[3:] if cur else [1.0,0.0,0.0,0.0]
    node.goal_received = False; node.goal_poses.clear()
    def _goal_cb(msg: PoseStamped):
        node.goal_received = True
        if hasattr(node, 'idle_timer'): node.idle_timer.cancel(); from .motions import publish_stop_trajectory; publish_stop_trajectory(node)
        if is_robot_moving(node): return
        time.sleep(0.1)
        g = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, *current_orientation]
        import math
        if not any(math.dist(g[:3], e[:3]) < 0.01 for e in node.goal_poses):
            node.goal_poses.append(g); publish_goal_marker(node, g[:3])
        if hasattr(node, 'goal_pose_sub'):
            node.destroy_subscription(node.goal_pose_sub); del node.goal_pose_sub
    node.goal_pose_sub = node.create_subscription(PoseStamped, '/external_goal_pose', _goal_cb, node.qos)
    sequence = [(0.0,0.0,0.2),(0.0,0.1,0.0),(0.0,-0.1,0.0),(0.0,0.0,-0.1)]
    idx = {"i":0}
    def _idle_cb():
        if node.goal_received or idx["i"]>=len(sequence):
            if hasattr(node,'idle_timer'): node.idle_timer.cancel(); return
        curp = node.get_end_effector_pose()
        if curp and not is_robot_moving(node):
            dx,dy,dz = sequence[idx['i']]
            tgt = [curp[0]+dx, curp[1]+dy, curp[2]+dz, *current_orientation]
            from .motions import execute_single_pose as _exec
            _exec(node, tgt); idx['i'] += 1
    node.idle_timer = node.create_timer(5.0, _idle_cb)


def is_robot_moving(node, velocity_threshold: float = 0.001) -> bool:
    v = getattr(node, 'current_joint_velocities', None) or []
    return any(abs(x) > velocity_threshold for x in v)


def plan_and_execute(node):
    if node.current_joint_positions is None:
        node.get_logger().warn("No joint state yet."); return
    if not node.goal_poses:
        node.get_logger().warn("No stored goals."); return
    if not node.robot_running:
        node.get_logger().error("Robot program OFF; may fail.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .motions import execute_single_pose as exec_pose
    while node.goal_poses and getattr(node, 'running', True):
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        goal = node.goal_poses.pop(0); x,y,z = goal[:3]
        if z > 1.30:
            orientation = quaternion_from_approach(node, pitch_deg=-35.0)
            approach = [x, y+0.12, z, *orientation]; y -= 0.004; z -= 0.4
        else:
            orientation = goal[3:]; approach = [x, y+0.10, z-0.12, *orientation]; z += 0.055; y -= 0.003
        if not plan_and_send(node, start, Pose.from_list(approach), dt=0.01, label="APPROACH"): continue
        wait_until_xyz(node, approach[:3])
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        seed = [x,y,z]; reacq = reacquire_goal_pose(node, seed_xyz=seed, timeout=3.5, stable_eps=0.004, stable_need=3)
        if reacq:
            x,y,z = reacq; publish_goal_marker(node, [x,y,z])
        else:
            node.get_logger().warn("No reacquire; skipping goal."); continue
        final_target = [x, y-0.001, z+0.001, *orientation]
        if not plan_and_send(node, start, Pose.from_list(final_target), dt=0.03, label="FINAL"): continue
        wait_until_xyz(node, final_target[:3])
        node.control_gripper("CLOSE"); time.sleep(0.7)
        if node.slip_detection or node.grab_miss or node.weak_grab:
            node.control_gripper("OPEN"); cur = node.get_end_effector_pose()
            if cur: exec_pose(node, [cur[0], cur[1], cur[2]+0.015, *cur[3:]])
            node.slip_detection = node.grab_miss = False
            node.control_gripper("CLOSE")
        from .motions import rotate_wrist, move_to_predropoff_position, move_to_dropoff_position, move_to_home_position
        rotate_wrist(node, 90); time.sleep(0.7)
        move_to_predropoff_position(node); move_to_dropoff_position(node); node.control_gripper("OPEN"); move_to_home_position(node)


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

