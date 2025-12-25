# ruff: noqa
import time, math, torch
import numpy as np
from geometry_msgs.msg import Pose as ROSPose, PoseStamped
from curobo.types.math import Pose
from curobo.types.robot import JointState
from .motions import execute_single_pose as _exec
from .motions import publish_stop_trajectory
from .config import PLAN_CFG_DEFAULT
from .utils import build_trajectory, wait_until_xyz
from .markers import publish_goal_marker
from .motions import interpolated_positions, execute_single_pose
from .motions import execute_single_pose as exec_pose
from .motions import rotate_wrist, move_to_predropoff_position, move_to_dropoff_position, move_to_home_position
from .motions import blend_motion
from .dynamic_obstacle import DynamicObstacleManager
from .utils import compute_visibility_approach

def pose_to_vec7(p: ROSPose):
    return [p.position.x, p.position.y, p.position.z, p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]


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


def minimize_rotation_orientation(current_quat, target_quat, blend_weight=0.5):
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


def plan_and_send(node, start_state, goal_pose: Pose, label: str, motion_type: str = "default") -> bool:
    # 1) Plan with cuRobo
    res = node.motion_gen.plan_single(start_state, goal_pose, PLAN_CFG_DEFAULT)
    if not res.success:
        node.get_logger().warn(f"Plan failed for {label}.")
        return False

    states = interpolated_positions(res)

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
    #    - dt too small => jerk
    #    - keep dt in [12 ms, 30 ms]
    raw_dt = base_dt / max(scale, 1e-6)
    dt = min(max(raw_dt, 0.012), 0.03)

    # Velocity: linear scaling with cap
    base_vel = 0.08        # slightly gentler than 0.1
    vel = base_vel * scale
    vel = min(vel, 0.25)   # hard cap for safety

    # 4) Build trajectory
    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
    )

    if node.stop_requested:
        node.get_logger().warn(f"Stop before sending {label} trajectory.")
        node.stop_requested = False
        return False

    node.trajectory_pub.publish(traj)
    return True



def reacquire_goal_pose(node, seed_xyz, timeout=3.0, stable_needed=3, radius=0.08, z_tolerance=0.05):
    """
    Reacquire goal pose with improved Z-axis accuracy.

    Uses median filtering and Z-specific stability checks to reduce
    depth noise from stereo camera.
    """
    stable_count = 0
    last_pose = None
    start = time.time()

    # Track Z history for median filtering (reduces depth noise)
    z_history = []
    stable_poses = []
    prev_pose_tuple = None  # Track previous to skip duplicates

    while time.time() - start < timeout:
        # Read fresh vision data, not the corrupted best_goal_xyz
        if node.latest_goal_pose:
            pose = node.latest_goal_pose[:3]  # [x, y, z]
        else:
            pose = None
        if pose is None:
            time.sleep(0.001)
            continue

        # Skip if same reading as last iteration (wait for fresh data)
        pose_tuple = tuple(pose)
        if pose_tuple == prev_pose_tuple:
            time.sleep(0.001)
            continue
        prev_pose_tuple = pose_tuple

        x, y, z = pose

        # XY radius check
        if math.hypot(x - seed_xyz[0], y - seed_xyz[1]) > radius:
            time.sleep(0.001)
            continue

        # Z tolerance check against seed (reject if too far from original)
        if abs(z - seed_xyz[2]) > z_tolerance:
            #print(f"⚠️ Reacquire Z too far from seed: {z:.3f} vs {seed_xyz[2]:.3f}")
            time.sleep(0.001)
            continue

        # Track Z history for filtering
        z_history.append(z)
        if len(z_history) > 20:
            z_history.pop(0)

        # Z outlier rejection: skip if Z deviates >8mm from median
        if len(z_history) >= 5:
            z_median = sorted(z_history)[len(z_history) // 2]
            if abs(z - z_median) > 0.008:
                time.sleep(0.001)
                continue

        # Stability check with tighter thresholds
        if last_pose is not None:
            delta = math.dist(last_pose, pose)
            z_delta = abs(z - last_pose[2])

            # Require both total delta < 2mm AND Z delta < 2mm
            if delta < 0.002 and z_delta < 0.002:
                stable_count += 1
                print(f"Stable reacquire pose: {[round(v,3) for v in pose]} (stable {stable_count}/{stable_needed})")
                stable_poses.append(pose)
            else:
                stable_count = 0
                stable_poses.clear()

        last_pose = pose

        if stable_count >= stable_needed:
            # Return median-filtered Z for accuracy
            stable_z_values = [p[2] for p in stable_poses]
            median_z = sorted(stable_z_values)[len(stable_z_values) // 2]
            return (x, y, median_z)

        time.sleep(0.001)

    # Timeout fallback: only use z_history if we got valid readings
    if z_history:
        z_median = sorted(z_history)[len(z_history) // 2]
        return (seed_xyz[0], seed_xyz[1], z_median)

    # No valid readings - return original seed (don't use corrupted best_goal_xyz)
    print("⚠️ Reacquire timeout - using original seed position")
    return seed_xyz



def subscribe_to_goal_pose(node):
    """Subscribe to /external_goal_pose and stop idle motion immediately when goal is received."""

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

    # No cache - always wait for fresh goal from vision
    # Track position stability before accepting
    goal_history = {"poses": [], "stable_count": 0, "accepted": False}

    def _goal_cb(msg: PoseStamped):
        nonlocal goal_history

        new_xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

        # ---- ALWAYS STORE LATEST GOAL POSE ----
        node.latest_goal_pose = [*new_xyz,
            msg.pose.orientation.w, msg.pose.orientation.x,
            msg.pose.orientation.y, msg.pose.orientation.z]
        node.latest_goal_time = time.time()

        # Already accepted a goal this cycle
        if goal_history["accepted"]:
            return

        # Check for position jump (vision switched to new fruit)
        if goal_history["poses"]:
            last_xyz = goal_history["poses"][-1]
            jump = math.dist(new_xyz, last_xyz)
            if jump > 0.10:  # >10cm = new fruit, reset
                print(f"📍 Position jump {jump:.2f}m - waiting for stable...")
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
            node.goal_received = True
            print("✅ Goal position stable, accepting...")

            # Stop idle timer
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass

            publish_stop_trajectory(node)

            # Use latest stable position
            node.goal_seed_xy = [new_xyz[0], new_xyz[1]]
            node.best_goal_xyz = new_xyz
            node.best_goal_score = float("inf")

            #g = [*new_xyz, *current_orientation]
            g = [*new_xyz, 
                msg.pose.orientation.w, msg.pose.orientation.x,
                msg.pose.orientation.y, msg.pose.orientation.z]
            
            
            if not any(math.dist(g[:3], e[:3]) < 0.01 for e in node.goal_poses):
                node.goal_poses.append(g)
                publish_goal_marker(node, g[:3])
                print(f"🟢 Accepted goal pose: {g}")
                node.obstacles.update_pose("fruit_obstacle", g[:3])

            # Destroy subscription
            try:
                if hasattr(node, 'goal_pose_sub'):
                    node.destroy_subscription(node.goal_pose_sub)
                    del node.goal_pose_sub
            except Exception:
                pass

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
            print("Performing idle micro-motion...")
            dx, dy, dz = sequence[idx['i']]
            tgt = [curp[0] + dx, curp[1] + dy, curp[2] + dz, *current_orientation]
            print(f"Idle move to: {tgt}")
            _exec(node, tgt)
            idx['i'] += 1

    # Start idle motion timer
    node.idle_timer = node.create_timer(5.0, _idle_cb)



def is_robot_moving(node, velocity_threshold: float = 0.001) -> bool:
    v = getattr(node, 'current_joint_velocities', None) or []
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
    
    
    while node.goal_poses and getattr(node, 'running', True):
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("Stop requested; aborting goal execution.")
            publish_stop_trajectory(node)
            node.goal_poses.clear()
            node.stop_requested = False
            break
        
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        
        
        # 0. Get next goal
        goal = node.goal_poses.pop(0)
        x,y,z = goal[:3]
        yoffset = node.yoffset

        # Direction-biased pre-grasp: use fruit direction if available
        direction = getattr(node, 'fruit_direction', None)
        standoff = 0.08  # 12cm standoff distance

        d_blend = blend_approach_direction(node, x, y, z)
        ax = x + d_blend[0] * standoff
        ay = y + d_blend[1] * standoff
        az = z + d_blend[2] * standoff

        # 1. Plan approach
        # if z > 1.30:  #high targets: top-down approach
        #     orientation = quaternion_from_approach(node, pitch_deg=-35.0)
        #     approach = [x, y+0.12, z, *orientation] #top approach
        # else:
        # Get current orientation and minimize rotation
        cur_pose = node.get_end_effector_pose()
        cur_quat = cur_pose[3:] if cur_pose else None
        target_quat = goal[3:]
        print("Current quat:", cur_quat)
        print("Target quat:", target_quat)
        orientation = minimize_rotation_orientation(cur_quat, target_quat)
        approach = [ax, ay, az-0.12, *orientation]
        print("Going for side approach:", approach)
        #z -= 0.055; y -= 0.003
            
        if not plan_and_send(node, start, Pose.from_list(approach), label="APPROACH", motion_type="approach"): 
            continue
        wait_until_xyz(node, approach[:3])
        blend_motion(node)
        
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        
        # 2. Reacquire
        seed = [x,y,z]
        print("Reacquiring goal pose near:", seed)
        reacq = reacquire_goal_pose(node, seed_xyz=seed)
        
        if reacq:
            x,y,z = reacq; publish_goal_marker(node, [x,y,z])
        else:
            node.get_logger().warn("No reacquire; skipping goal.")
            continue
            
        # 3. Final slow precise grasp
        final_target = [x, y, z+0.02, *orientation]
        if not plan_and_send(node, start, Pose.from_list(final_target), label="FINAL", motion_type="final"): 
            continue
        wait_until_xyz(node, final_target[:3])
        blend_motion(node)

        node.control_gripper("CLOSE"); time.sleep(0.7)
        
        
        if node.slip_detection or node.grab_miss or node.weak_grab:
            print("Re-attempting grasp due to:",)
            # node.control_gripper("OPEN"); cur = node.get_end_effector_pose()
            # if cur: exec_pose(node, [cur[0], cur[1], cur[2]+0.015, *cur[3:]])
            # node.slip_detection = node.grab_miss = False
            # node.control_gripper("CLOSE")
            
        # 4. Drop-off and return
        #rotate_wrist(node, 90, rotate_time=0.5, hold_time=0.05, return_time=0.5)
        time.sleep(0.1)  # Wait for wrist rotation to complete (0.5s rotate + 0.05s hold + 0.5s return + margin)

        # #move_to_predropoff_position
        current_pose = node.get_end_effector_pose()
        target_pose = [current_pose[0], current_pose[1]+node.cfg.planner.pre_dropoff_y_offset,
                       current_pose[2]+node.cfg.planner.pre_dropoff_z_offset,
                       *current_pose[3:]]
        print("Current pose:", current_pose)
        print("Target pre-dropoff pose:", target_pose)
        execute_single_pose(node, target_pose, motion_type="predropoff")
        # Wait until robot reaches target position (with tolerance)
        wait_until_xyz(node, target_pose[:3], tol=0.02, timeout=10.0)
        print("Moved to pre-dropoff position.")
        #move_to_predropoff_position(node)
        blend_motion(node)
        time.sleep(0.1)
        move_to_dropoff_position(node)
        time.sleep(0.2)  # small delay to allow state update
        node.control_gripper("OPEN")
        move_to_home_position(node)

        # Reset tracking state for next goal
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
