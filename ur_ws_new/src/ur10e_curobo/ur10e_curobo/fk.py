# ruff: noqa
import threading
from typing import List, Optional
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from geometry_msgs.msg import Point

# Module-level cache for the isolated FK kinematics model.
# This is completely separate from motion_gen.kinematics and will never be
# corrupted by graph planner buffer resizing.
_fk_kin_model: Optional[CudaRobotModel] = None
_fk_lock = threading.RLock()


def init_fk_model(node):
    """Eagerly create the isolated FK model BEFORE warmup().

    This ensures GPU memory allocation happens before CUDA graph capture,
    so the pre-captured graphs remain valid.  Must be called after
    MotionGen is constructed but before motion_gen.warmup().
    """
    global _fk_kin_model
    with _fk_lock:
        if _fk_kin_model is None:
            _fk_kin_model = CudaRobotModel(node.motion_gen.robot_cfg.kinematics)
            try:
                node.get_logger().info("FK model initialized (pre-warmup)")
            except Exception:
                pass
    return _fk_kin_model


def _is_planning_active(node) -> bool:
    """Non-blocking check: is a planning operation currently holding the lock?

    Returns True if we should skip CUDA FK to avoid corrupting
    CUDA graph capture in plan_single().
    """
    lock = getattr(node, '_planning_lock', None)
    if lock is None:
        return False
    return lock.locked()


def _get_kin_model(node):
    """Get an isolated CudaRobotModel for pure FK.

    Creates a separate instance on first call so that the graph planner's
    internal buffer resizing (7000-10000 batch) never corrupts FK results.
    """
    global _fk_kin_model
    with _fk_lock:
        if _fk_kin_model is None:
            # Create a fresh CudaRobotModel from the original robot config (clean
            # CudaRobotModelConfig, no runtime attrs like _batch_size).
            _fk_kin_model = CudaRobotModel(node.motion_gen.robot_cfg.kinematics)
    return _fk_kin_model


def get_end_effector_pose(node) -> Optional[list]:
    if node.current_joint_positions is None:
        node.get_logger().debug("Joint states not yet received.")
        return None
    # Skip CUDA FK while planning is active to avoid corrupting CUDA graph capture
    if _is_planning_active(node):
        return getattr(node, '_last_ee_pose', None)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device)
        with _fk_lock:
            kin = _get_kin_model(node)
            with torch.no_grad():
                ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
            pos = ee_pos[0].cpu().tolist()
            quat = ee_quat[0].cpu().tolist()
        result = pos + quat
        node._last_ee_pose = result  # cache for use during planning
        return result
    except Exception as e:
        node.get_logger().warn(f"FK failed: {e}")
        return None


def pose_from_joints(node, joint_positions: List[float]) -> Optional[list]:
    """Return [x, y, z, qw, qx, qy, qz] for an arbitrary joint configuration."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor([joint_positions], dtype=torch.float32, device=device)
        with _fk_lock:
            kin = _get_kin_model(node)
            with torch.no_grad():
                ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
            pos = ee_pos[0].cpu().tolist()
            quat = ee_quat[0].cpu().tolist()
        return pos + quat
    except Exception as e:
        node.get_logger().warn(f"FK pose failed: {e}")
        return None


def forward_kinematics(node, joint_positions: List[float]) -> Optional[Point]:
    """Single-waypoint FK using CudaRobotModel.forward() directly.
    Does NOT touch rollout_fn or collision state."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor([joint_positions], dtype=torch.float32, device=device)
        with _fk_lock:
            kin = _get_kin_model(node)
            with torch.no_grad():
                ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
            pos = ee_pos.reshape(-1).cpu().tolist()
        return Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    except Exception as e:
        node.get_logger().warn(f"CuRobo FK failed: {e}")
        return None


def forward_kinematics_batch(node, joint_states: List[List[float]]) -> List[Point]:
    """Compute FK for all waypoints in a single batched GPU call.
    Uses CudaRobotModel.forward() directly — no rollout_fn, no collision state.
    Does NOT reset internal buffers (that corrupts graph planner state)."""
    if not joint_states:
        return []
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor(joint_states, dtype=torch.float32, device=device)
        with _fk_lock:
            kin = _get_kin_model(node)
            with torch.no_grad():
                ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
            if ee_pos.ndim == 1:
                ee_pos = ee_pos.unsqueeze(0)
            if ee_pos.ndim != 2 or ee_pos.shape[0] != len(joint_states):
                malformed_shape = tuple(ee_pos.shape)
                position_frames = []
                with torch.no_grad():
                    for sample_q in q:
                        sample_pos = kin.forward(sample_q.unsqueeze(0))[0]
                        position_frames.append(sample_pos.reshape(-1, 3)[0].clone())
                ee_pos = torch.stack(position_frames, dim=0)
                node.get_logger().warn(
                    f"Batched FK recovered malformed output {malformed_shape} "
                    f"using {len(joint_states)} single-waypoint checks")
            positions = ee_pos.cpu().tolist()
        return [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions]
    except Exception as e:
        node.get_logger().warn(f"Batched FK failed: {e}")
        return []


def forward_kinematics_pose_batch(node, joint_states: List[List[float]]) -> List[list]:
    """Compute batched FK poses as [x, y, z, qw, qx, qy, qz]."""
    if not joint_states:
        return []
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor(joint_states, dtype=torch.float32, device=device)
        with _fk_lock:
            kin = _get_kin_model(node)
            with torch.no_grad():
                ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
            if ee_pos.ndim == 1:
                ee_pos = ee_pos.unsqueeze(0)
            if ee_quat.ndim == 1:
                ee_quat = ee_quat.unsqueeze(0)
            if (
                ee_pos.ndim != 2
                or ee_quat.ndim != 2
                or ee_pos.shape[0] != len(joint_states)
                or ee_quat.shape[0] != len(joint_states)
            ):
                poses = []
                with torch.no_grad():
                    for sample_q in q:
                        p, quat, _, _, _, _, _ = kin.forward(sample_q.unsqueeze(0))
                        poses.append(
                            p.reshape(-1, 3)[0].cpu().tolist()
                            + quat.reshape(-1, 4)[0].cpu().tolist())
                return poses
            positions = ee_pos.cpu().tolist()
            quats = ee_quat.cpu().tolist()
        return [list(p) + list(q) for p, q in zip(positions, quats)]
    except Exception as e:
        node.get_logger().warn(f"Batched FK pose failed: {e}")
        return []


def solve_ik_fast(node, target_pose: list, seed: Optional[list] = None, iters: int = 25, lr: float = 0.6):
    """Lightweight IK via gradient descent on the isolated FK model.

    Args:
        target_pose: [x,y,z,qw,qx,qy,qz]
        seed: initial joint positions (defaults to current)
        iters: gradient steps
        lr: step size

    Returns:
        list of joint positions or None
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = len(node.joint_order)

        if seed is None:
            seed = node.current_joint_positions if node.current_joint_positions is not None else [0.0] * n

        x_target = torch.tensor(target_pose[:3], dtype=torch.float32, device=device)
        q_target = torch.tensor(target_pose[3:], dtype=torch.float32, device=device)
        var = torch.tensor([seed], dtype=torch.float32, device=device, requires_grad=True)

        with _fk_lock:
            kin = _get_kin_model(node)

            for _ in range(iters):
                ee_pos, ee_quat, _, _, _, _, _ = kin.forward(var)
                pos = ee_pos.squeeze()
                quat = ee_quat.squeeze()

                pos_loss = ((pos - x_target) ** 2).sum()
                quat_norm = quat / (quat.norm() + 1e-12)
                qnorm_target = q_target / (q_target.norm() + 1e-12)
                ori_loss = 1.0 - (quat_norm * qnorm_target).sum()
                loss = pos_loss + 0.1 * ori_loss

                loss.backward()

                with torch.no_grad():
                    if var.grad is None:
                        break
                    var -= lr * var.grad
                    var.grad.zero_()

        result = var.detach().cpu().squeeze().tolist()
        return result if len(result) == n else None
    except Exception as e:
        try:
            node.get_logger().warn(f"IK solver failed: {e}")
        except Exception:
            pass
        return None
