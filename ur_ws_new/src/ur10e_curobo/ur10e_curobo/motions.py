# ruff: noqa
import math, time, torch
from typing import List
from curobo.types.math import Pose
from curobo.types.robot import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectoryPoint

from .config import PLAN_CFG_DEFAULT, PLAN_CFG_JS, PLAN_CFG_JS_NO_FINETUNE, VOXEL_CONFIG
from .utils import build_trajectory, wait_until_xyz
from .fk import forward_kinematics


def interpolated_positions(result):
    interp = result.get_interpolated_plan()
    if isinstance(interp, JointState):
        interp = interp.position
    if not isinstance(interp, torch.Tensor):
        interp = torch.tensor(interp, dtype=torch.float32)
    return interp.to("cpu").tolist()


def get_curobo_dt(result) -> float:
    """Get cuRobo's interpolation dt from a MotionGenResult."""
    return getattr(result, 'interpolation_dt', 0.02)


def publish_stop_trajectory(node):
    if node.current_joint_positions is None:
        return
    try:
        stop = JointTrajectory(); stop.joint_names = node.joint_order
        pt = JointTrajectoryPoint(); pt.positions = list(node.current_joint_positions)
        pt.velocities = [0.0]*len(node.joint_order); pt.accelerations = [0.0]*len(node.joint_order)
        pt.time_from_start.nanosec = 1_000_000; stop.points = [pt]
        node.trajectory_pub.publish(stop)
    except Exception:
        pass  # handle already destroyed during shutdown


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
    lock = getattr(node, '_planning_lock', None)
    if lock: lock.acquire()
    try:
        res = node.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT)
    finally:
        if lock: lock.release()
    if not res.success:
        node.get_logger().warn("Plan failed for single pose."); return

    states = interpolated_positions(res)
    curobo_dt = get_curobo_dt(res)

    # Verify trajectory against latest depth data before execution
    if (VOXEL_CONFIG.get("verify_before_execute", True) and
        hasattr(node, 'voxel_obstacles') and node.voxel_obstacles is not None):

        # Extract goal position for exclusion zone (we WANT to reach the target)
        goal_position = pose[:3]  # [x, y, z] from input pose

        max_attempts = VOXEL_CONFIG.get("max_replan_attempts", 2)
        for attempt in range(max_attempts):
            is_safe, collision_idx = node.voxel_obstacles.verify_trajectory_collision(
                states,
                exclude_position=goal_position,  # Skip collision check near target
            )

            if is_safe:
                break

            node.get_logger().warn(
                f"Collision detected at waypoint {collision_idx}/{len(states)} "
                f"(attempt {attempt + 1}/{max_attempts})"
            )

            # Replan with updated obstacles
            if lock: lock.acquire()
            try:
                res = node.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT)
            finally:
                if lock: lock.release()
            if not res.success:
                node.get_logger().error("Replan failed after collision detection")
                return
            states = interpolated_positions(res)
        else:
            node.get_logger().error(f"Collision persists after {max_attempts} replans")
            return

    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier

    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, curobo_dt), planner.max_dt)

    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=0,
    )
    if traj.points:
        node.trajectory_pub.publish(traj)


# Plans and executes a joint-space motion to reach a specified set of joint angles.
# Input: target_joints: List of joint angles in radians.
        # label: A string label for logging purposes.
        # dt: Time step for trajectory interpolation.
        
# Publishes /joint_trajectory_controller/joint_trajectory.
def plan_execute_js(
    node,
    target_joints: List[float],
    label: str,
    motion_type: str = "default",
    speed_factor: float = 1.0,
):
    if node.current_joint_positions is None:
        # Wait briefly for joint state callback to fire (can be delayed after blocking ops)
        for _ in range(10):
            time.sleep(0.1)
            if node.current_joint_positions is not None:
                break
    if node.current_joint_positions is None:
        node.get_logger().warn(f"No joint state; skipping {label} move.")
        return False

    # Attempt CUDA recovery if previously faulted
    if getattr(node, "_cuda_faulted", False):
        from .goals import try_cuda_recovery
        try_cuda_recovery(node)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 1. Build joint states
    # ------------------------------
    try:
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )

        goal_js = JointState.from_position(
            torch.tensor([target_joints], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
    except Exception as e:
        msg = str(e)
        if "CUDA error" in msg or "illegal memory access" in msg:
            node._cuda_faulted = True
        node.get_logger().warn(f"Failed to create tensors for {label}: {e}")
        return False

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
    scale *= max(speed_factor, 1e-6)

    # ------------------------------
    # 3. cuRobo plan (hold lock to prevent concurrent CUDA ops)
    # ------------------------------
    lock = getattr(node, '_planning_lock', None)
    res = None
    _yolo = getattr(node, 'yolo_thread', None)
    _yolo_lock = getattr(_yolo, 'inference_lock', None)
    if _yolo_lock: _yolo_lock.acquire()
    try:
        if lock: lock.acquire()
        try:
            res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS)
        except Exception as e:
            msg = str(e)
            if "CUDA error" in msg or "illegal memory access" in msg:
                node._cuda_faulted = True
            node.get_logger().warn(f"Joint-space plan to {label} exception: {e}")
            return False
        finally:
            if lock: lock.release()

        if res is None or not res.success:
            status = getattr(res, 'status', 'unknown') if res is not None else 'None'
            # FINETUNE_TRAJOPT_FAIL means the base trajectory was found but fine-tune
            # smoothing failed. Retry without finetune to still get a valid trajectory.
            if res is not None and "FINETUNE" in str(status):
                node.get_logger().info(
                    f"Joint-space plan to {label}: finetune failed, retrying without finetune")
                if lock: lock.acquire()
                try:
                    res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS_NO_FINETUNE)
                except Exception as e:
                    node.get_logger().warn(f"Joint-space retry to {label} exception: {e}")
                    return False
                finally:
                    if lock: lock.release()
        if res is None or not res.success:
            status = getattr(res, 'status', 'unknown') if res is not None else 'None'
            node.get_logger().warn(f"Joint-space plan to {label} failed. status={status}")
            return False
    finally:
        if _yolo_lock: _yolo_lock.release()

    # ------------------------------
    # 4. Interpolate (older cuRobo API)
    # ------------------------------
    # No args allowed
    states = interpolated_positions(res)
    curobo_dt = get_curobo_dt(res)

    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, planner.min_dt), planner.max_dt)

    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=math.pi,  # UR10e physical joint limit; prevents clipping cuRobo's natural velocities
    )

    if getattr(node.cfg.planner, "log_phase_timings", False):
        node.get_logger().info(f"Moving to {label} (dt={dt:.3f})")
    if traj.points:
        node.trajectory_pub.publish(traj)
    else:
        node.get_logger().warn(f"Skipping publish for {label} — empty trajectory (stop requested?)")
        return False

    # ------------------------------
    # 8. Wait for the robot, then blend to avoid abrupt stop
    # ------------------------------
    target_joints = states[-1]
    fk = forward_kinematics(node, target_joints)
    if fk:
        wait_until_xyz(node, [fk.x, fk.y, fk.z])
    # Secondary joint-space check: if EE arrived instantly (different IK branch),
    # wait until joints actually converge to the target configuration.
    _joint_tol = 0.05  # rad (~3°)
    _joint_deadline = time.time() + 15.0
    while time.time() < _joint_deadline and getattr(node, 'running', True):
        if getattr(node, 'stop_requested', False):
            break
        cur_joints = node.current_joint_positions
        if cur_joints is not None:
            max_err = max(abs(c - t) for c, t in zip(cur_joints, target_joints))
            if max_err < _joint_tol:
                break
        time.sleep(0.05)
    blend_motion(node)
    return True



def nearest_joint_config(current: List[float], target: List[float]) -> List[float]:
    """Return the joint angles equivalent to `target` that are closest to `current`.

    For each joint, picks the value in {target[i] + k*2π | k ∈ ℤ} that minimises
    |current[i] - target[i]|. This prevents cuRobo from routing through a 360°
    detour when the stored config and the current config have drifted by ≈2π.
    """
    out = []
    for c, t in zip(current, target):
        diff = c - t
        k = round(diff / (2 * math.pi))
        out.append(t + k * 2 * math.pi)
    return out


def _clear_voxels(node):
    """Clear voxel obstacle world so depth-camera noise doesn't block return paths."""
    vo = getattr(node, 'voxel_obstacles', None)
    if vo is not None:
        try:
            vo.clear()
        except Exception:
            pass


def move_to_home_position(node):
    _clear_voxels(node)
    target = node.home_joints
    if node.current_joint_positions is not None:
        target = nearest_joint_config(node.current_joint_positions, target)
    return plan_execute_js(node, target, label="HOME", motion_type="home")


def move_to_dropoff_position(node):
    _clear_voxels(node)
    target = node.dropoff_joints
    if node.current_joint_positions is not None:
        target = nearest_joint_config(node.current_joint_positions, target)
    return plan_execute_js(node, target, label="DROP-OFF", motion_type="dropoff")


def preplan_js(node, target_joints: List[float], start_joints: List[float],
               label: str = "PREPLAN", motion_type: str = "default"):
    """Plan a joint-space trajectory without executing it.
    Returns the built JointTrajectory message, or None on failure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = JointState.from_position(
        torch.tensor([start_joints], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )
    goal_js = JointState.from_position(
        torch.tensor([target_joints], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )
    lock = getattr(node, '_planning_lock', None)
    if lock: lock.acquire()
    try:
        res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS)
    finally:
        if lock: lock.release()
    if not res.success:
        return None
    states = interpolated_positions(res)
    curobo_dt = get_curobo_dt(res)
    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier
    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order, states, dt=dt,
        stop_flag=lambda: node.stop_requested,
    )
    return traj, states
 

def move_to_predropoff_position(node):
    target = node.predropoff_joints
    if node.current_joint_positions is not None:
        target = nearest_joint_config(node.current_joint_positions, target)
    plan_execute_js(node, target, label="preDROP-OFF", motion_type="predropoff")


def rotate_wrist(node, degrees: float,
                 rotate_time: float = 1.2,
                 hold_time: float = 0.1,
                 return_time: float = 1.2):

    if node.current_joint_positions is None:
        node.get_logger().error("No joint state; cannot rotate wrist.")
        return

    rad = math.radians(degrees)

    start = node.current_joint_positions.copy()
    peak = start.copy()
    peak[5] += rad  # wrist_3_joint

    traj = JointTrajectory()
    traj.joint_names = node.joint_order

    def pt(q, t):
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int((t - int(t)) * 1e9)
        return p

    t1 = rotate_time
    t2 = t1 + hold_time
    t3 = t2 + return_time

    traj.points = [
        pt(start, 0.0),    # start
        pt(peak, t1),      # rotate
        pt(peak, t2),      # short hold
        pt(start, t3)      # return
    ]

    node.trajectory_pub.publish(traj)




def move_backward(node, delta: float):
    if node.current_joint_positions is None:
        node.get_logger().error("No joint state; cannot move.")
        return
    
    q = node.current_joint_positions.copy(); q[1] -= delta
    traj = JointTrajectory(); traj.joint_names = node.joint_order
    p = JointTrajectoryPoint(); p.positions = q; p.time_from_start.sec = 1
    traj.points.append(p); node.trajectory_pub.publish(traj)

def blend_motion(node, pause=0.1):
    # maintains smoothness, avoids jerk
    if node.current_joint_positions is None:
        time.sleep(pause)
        return

    planner = node.cfg.planner
    traj = build_trajectory(
        node.joint_order,
        [node.current_joint_positions],
        dt=0.02,
        stop_flag=lambda: node.stop_requested,
    )
    if traj.points:
        node.trajectory_pub.publish(traj)
    time.sleep(pause)
