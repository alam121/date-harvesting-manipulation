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
    res = node.motion_gen.plan_single(start_state, goal_pose, PLAN_CFG_DEFAULT)
    if not res.success:
        node.get_logger().warn(f"Plan failed for {label}."); return False
    states = interpolated_positions(res)
    
    
    scale = max(getattr(node, "speed_scale", 1.0), 1e-6)
    base_dt = getattr(node.cfg.planner, "base_dt", 0.02)
    

    
    if motion_type == "approach":
        scale = getattr(node.cfg.planner, "speed_approach", 1.0)
    elif motion_type == "final":
        scale = getattr(node.cfg.planner, "speed_final", 1.0)
    elif motion_type in ["home", "dropoff", "predropoff"]:
        scale = getattr(node.cfg.planner, f"speed_{motion_type}", 1.0)
    else:
        scale = getattr(node.cfg.planner, "speed_home", 1.0)  # default fast

    traj = build_trajectory(
        node.joint_order,
        states,
        vel=0.1 * scale,
        dt=base_dt / scale,
        stop_flag=lambda: node.stop_requested
    )
    
    
    #node.get_logger().info(f"Planned {label} trajectory with {len(states)} steps.")
    
    if node.stop_requested:
        node.get_logger().warn(f"Stop before sending {label} trajectory."); node.stop_requested = False; return False
    node.trajectory_pub.publish(traj)
    
    return True


def reacquire_goal_pose(node, seed_xyz, timeout=5.5, stable_needed=2, radius=0.15):
    """
    Wait in place (no robot motion) until 2 stable goal poses appear.
    If no stable pose -> return best candidate by score.
    """

    stable_count = 0
    last_pose = None
    best_candidate = None
    best_score = float("inf")

    def _cb(msg):
        nonlocal stable_count, last_pose, best_candidate, best_score

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # must match same fruit
        if math.hypot(x - seed_xyz[0], y - seed_xyz[1]) > radius:
            return
        
        print(f"🍎 Reacquire candidate: {[x, y, z]}")
        pose = [x, y, z]

        # stability measure
        if last_pose is not None:
            if math.dist(last_pose, pose) < 0.004:   # <4mm
                stable_count += 1
                print(f"   Stable count: {stable_count}")
            else:
                stable_count = 0

        last_pose = pose

        # also track best candidate by EE distance (for fallback)
        ee = node.get_end_effector_pose()
        if ee is not None:
            dx = x - ee[0]; dy = y - ee[1]; dz = z - ee[2]
            score = math.sqrt(dx*dx + dy*dy + dz*dz)
        else:
            score = z

        if score < best_score:
            best_score = score
            best_candidate = pose

    # temporary subscriber
    sub = node.create_subscription(
        PoseStamped,
        "/external_goal_pose",
        _cb,
        getattr(node, "goal_qos", node.qos),
    )

    try:
        start = time.time()
        while time.time() - start < timeout:
            if stable_count >= stable_needed:
                print(f"🍏 Stable reacquired goal: {last_pose}")
                return last_pose
            time.sleep(0.01)

        # --------------------------------------------
        # Updated fallback order using continuous tracker
        # --------------------------------------------

        # 1) Use continuously tracked BEST if available
        if hasattr(node, "best_goal_xyz") and node.best_goal_xyz is not None:
            print(f"⚠️ No stable hits. Using continuous tracker BEST = {node.best_goal_xyz}")
            return node.best_goal_xyz

        # 2) If no continuous BEST → use frame best candidate
        if best_candidate:
            print(f"⚠️ No stable hits. Using best_candidate = {best_candidate}")
            return best_candidate

        # 3) Final fallback → seed (safe old fruit)
        print("❌ No reacquire candidates. Using seed_xyz.")
        return seed_xyz

    finally:
        node.destroy_subscription(sub)




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
        node.best_goal_score = float("inf")
        # ------------------------------------------

        # Wait a bit for safety
        time.sleep(0.05)

        # Extract goal position + keep current orientation
        g = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            *current_orientation
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
    
    if node.current_joint_positions is None:
        node.get_logger().warn("No joint state yet."); return
    if not node.goal_poses:
        node.get_logger().warn("No stored goals."); return
    if not node.robot_running:
        node.get_logger().error("Robot program OFF; may fail.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    while node.goal_poses and getattr(node, 'running', True):
        
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        
        
        # 0. Get next goal
        goal = node.goal_poses.pop(0)
        x,y,z = goal[:3]
        yoffset = node.yoffset
        ax, ay, az = compute_visibility_approach(node, x, y, z, dist=0.18)
        # 1. Plan approach
        if z > 1.30:  #high targets: top-down approach
            orientation = quaternion_from_approach(node, pitch_deg=-35.0)
            approach = [x, y+0.12, z, *orientation] #top approach
            #y -= 0.004; z -= 0.4
        else:
            orientation = goal[3:]
            #approach = [x, y+node.yoffset, z-node.zoffset, *orientation] #side approch: Z negative means down, y positive means back 
            approach = [ax, ay, az, *orientation]
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
        reacq = reacquire_goal_pose(node, seed_xyz=seed)
        
        if reacq:
            x,y,z = reacq; publish_goal_marker(node, [x,y,z])
        else:
            node.get_logger().warn("No reacquire; skipping goal.")
            continue
            
        # 3. Final slow precise grasp
        final_target = [x, y, z-0.02, *orientation]
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
        move_to_predropoff_position(node)
        blend_motion(node)
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
