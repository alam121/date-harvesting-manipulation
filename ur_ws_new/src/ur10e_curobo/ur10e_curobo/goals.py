# ruff: noqa
import time, math, torch
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



def reacquire_goal_pose(node, seed_xyz, timeout=5.5, stable_needed=3, radius=0.08):

    stable_count = 0
    last_pose = None
    start = time.time()

    # Track small movements
    small_movements = []

    while time.time() - start < timeout:

        pose = node.best_goal_xyz
        if pose is None:
            time.sleep(0.005)
            continue

        x, y, z = pose

        if math.hypot(x - seed_xyz[0], y - seed_xyz[1]) > radius:
            time.sleep(0.005)
            continue

        # Distance change between frames
        if last_pose is not None:
            delta = math.dist(last_pose, pose)

            # Collect last few deltas
            small_movements.append(delta)
            if len(small_movements) > 5:
                small_movements.pop(0)

            # Movement below 2 mm consistently
            if len(small_movements) >= 4 and max(small_movements) < 0.002:
                print("🍏 Fruit is static — early exit")
                return pose

            # Standard stability counter
            if delta < 0.004:  # 4 mm
                stable_count += 1
                print(f"🍏 Stable count: {stable_count}/{stable_needed} (delta={delta:.4f} m)")
            else:
                stable_count = 0

        last_pose = pose

        if stable_count >= stable_needed:
            print(f"🍏 Stable reacquired goal = {pose}")
            return pose

        time.sleep(0.005)

    print("⚠️ Reacquire timeout — using best estimate.")
    return node.best_goal_xyz or seed_xyz



def subscribe_to_goal_pose(node):
    """Subscribe to /external_goal_pose and stop idle motion immediately when goal is received."""

    # Wait until robot stops before subscribing
    if is_robot_moving(node):
        node.create_timer(0.5, lambda: (not is_robot_moving(node)) and subscribe_to_goal_pose(node))
        return

    # Destroy previous subscription if it exists
    if hasattr(node, 'goal_pose_sub'):
        node.destroy_subscription(node.goal_pose_sub)
        del node.goal_pose_sub

    cur = node.get_end_effector_pose()
    current_orientation = cur[3:] if cur else [1.0, 0.0, 0.0, 0.0]
    node.goal_received = False
    node.goal_poses.clear()

    def _goal_cb(msg: PoseStamped):

        # ---- ALWAYS STORE LATEST GOAL POSE ----
        node.latest_goal_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        node.latest_goal_time = time.time()

        print("Stored latest global goal pose.")
        # ---------------------------------------
        """Triggered immediately on receiving a goal pose."""
        if node.goal_received:
            return  # Ignore duplicates
        


        node.goal_received = True
        print("✅ Goal received, stopping all idle activity...")

        # Stop idle timer immediately
        if hasattr(node, 'idle_timer'):
            try:
                node.idle_timer.cancel()
                del node.idle_timer
            except Exception:
                pass

        # Stop robot motion
        publish_stop_trajectory(node)

        # Wait a bit for safety
        time.sleep(0.05)

        # Extract goal position and RELATIVE ROTATION ANGLE from vision
        # Vision publishes rotation angle as quaternion around Z-axis
        # We need to extract the angle and apply it relative to current orientation
        rotation_quat_w = msg.pose.orientation.w
        rotation_quat_z = msg.pose.orientation.z

        # Extract rotation angle from quaternion (rotation around Z-axis)
        # quat = [cos(θ/2), 0, 0, sin(θ/2)] for Z-axis rotation
        import math
        rotation_angle_rad = 2.0 * math.atan2(rotation_quat_z, rotation_quat_w)
        rotation_angle_deg = math.degrees(rotation_angle_rad)

        print(f"[GOAL] Received rotation angle: {rotation_angle_deg:+.1f}° relative to current orientation")

        # Get current EE orientation (this is our 0° baseline)
        current_ee_pose = node.get_end_effector_pose()
        if current_ee_pose:
            current_quat = current_ee_pose[3:]  # [w, x, y, z]
        else:
            current_quat = [1.0, 0.0, 0.0, 0.0]  # Fallback to identity

        # Apply relative rotation around approach axis (Z-axis in current frame)
        # Multiply quaternions: final = current * rotation
        from scipy.spatial.transform import Rotation as R
        current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])  # scipy uses [x,y,z,w]
        relative_rot = R.from_quat([0, 0, rotation_quat_z, rotation_quat_w])  # rotation around Z
        final_rot = current_rot * relative_rot
        final_quat_scipy = final_rot.as_quat()  # [x, y, z, w]
        final_quat = [final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]]  # [w, x, y, z]

        # ------------------------------------------
        # Start tracking this fruit based on the first goal message
        node.goal_seed_xy = [
            msg.pose.position.x,
            msg.pose.position.y
        ]
        node.best_goal_xyz = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        node.best_goal_quat = final_quat.copy()  # Use computed final orientation
        node.best_goal_score = float("inf")
        # ------------------------------------------

        g = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            *final_quat
        ]

        # Add goal only if it's new
        if not any(math.dist(g[:3], e[:3]) < 0.01 for e in node.goal_poses):
            node.goal_poses.append(g)
            publish_goal_marker(node, g[:3])
            print(f"🟢 Received goal pose: {g}")
            node.obstacles.update_pose("fruit_obstacle", g[:3])
        # Destroy the subscription — stop listening after first goal
        try:
            if hasattr(node, 'goal_pose_sub'):
                node.destroy_subscription(node.goal_pose_sub)
                del node.goal_pose_sub
        except Exception:
            pass

    # Subscribe to /external_goal_pose
    node.goal_pose_sub = node.create_subscription(
        PoseStamped,
        '/external_goal_pose',
        _goal_cb,
        getattr(node, "goal_qos", node.qos),
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

# Main goal-execution pipeline — runs through all saved goals and performs motion + gripper actions in sequence.
def plan_and_execute(node):
    
    if node.current_joint_positions is None or len(node.current_joint_positions) != len(node.joint_order):
        node.get_logger().warn("No joint state yet or incomplete state; aborting goal execution.")
        return
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
        optimized_orientation = goal[3:]  # Save optimized orientation from vision
        yoffset = node.yoffset
        ax, ay, az = compute_visibility_approach(node, x, y, z, dist=0.18)

        # 1. Plan approach
        # Strategy: Use current EE orientation as baseline (0°)
        # This provides smooth motion - robot doesn't rotate during approach
        # Final optimized orientation is applied in step 3 (final grasp)

        if z > 1.30:  #high targets: top-down approach
            approach_orientation = quaternion_from_approach(node, pitch_deg=-35.0)
            approach = [x, y+0.12, z, *approach_orientation] #top approach
            #y -= 0.004; z -= 0.4
        else:
            # Use current EE orientation (treat as 0° baseline for approach)
            current_ee_pose = node.get_end_effector_pose()
            if current_ee_pose:
                approach_orientation = current_ee_pose[3:]  # Current orientation = 0° reference
            else:
                approach_orientation = [1.0, 0.0, 0.0, 0.0]  # Default fallback

            #approach = [x, y+node.yoffset, z-node.zoffset, *approach_orientation] #side approch: Z negative means down, y positive means back
            approach = [ax, ay, az, *approach_orientation]
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

        # 3. Final slow precise grasp (NOW use optimized orientation from vision)
        # Use latest tracked orientation if available, otherwise use original
        if hasattr(node, 'best_goal_quat') and node.best_goal_quat is not None:
            final_orientation = node.best_goal_quat
            print(f"[GRASP] Using optimized orientation from vision tracker")
        else:
            final_orientation = optimized_orientation
            print(f"[GRASP] Using initial optimized orientation")

        final_target = [x, y, z+0.02, *final_orientation]
        if not plan_and_send(node, start, Pose.from_list(final_target), label="FINAL", motion_type="final"): 
            continue
        wait_until_xyz(node, final_target[:3])
        blend_motion(node)

        node.control_gripper("CLOSE"); time.sleep(0.7)
        
        
        if node.slip_detection or node.grab_miss or node.weak_grab:
            node.control_gripper("OPEN"); cur = node.get_end_effector_pose()
            if cur: exec_pose(node, [cur[0], cur[1], cur[2]+0.015, *cur[3:]])
            node.slip_detection = node.grab_miss = False
            node.control_gripper("CLOSE")
            
        # 4. Drop-off and return
        rotate_wrist(node, 90); time.sleep(0.9)

        # #move_to_predropoff_position(node)
        # current_pose = node.get_end_effector_pose()
        # execute_single_pose(node, [current_pose[0], current_pose[1]+node.cfg.planner.pre_droffoff_y_offset,
        #                       current_pose[2]+node.cfg.planner.pre_droffoff_z_offset,
        #                       *current_pose[3:]], motion_type="predropoff")

        move_to_predropoff_position(node)
        blend_motion(node)
        time.sleep(0.1)
        move_to_dropoff_position(node)
        time.sleep(0.2)  # small delay to allow state update
        node.control_gripper("OPEN")
        move_to_home_position(node)

        if len(node.goal_poses) == 0:
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
