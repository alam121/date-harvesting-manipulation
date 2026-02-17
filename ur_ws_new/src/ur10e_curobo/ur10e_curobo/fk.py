# ruff: noqa
from typing import List, Optional
import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from geometry_msgs.msg import Point

# Module-level cache for the isolated FK kinematics model.
# This is completely separate from motion_gen.kinematics and will never be
# corrupted by graph planner buffer resizing.
_fk_kin_model: Optional[CudaRobotModel] = None


def _get_kin_model(node):
    """Get an isolated CudaRobotModel for pure FK.

    Creates a separate instance on first call so that the graph planner's
    internal buffer resizing (7000-10000 batch) never corrupts FK results.
    """
    global _fk_kin_model
    if _fk_kin_model is None:
        # Create a fresh CudaRobotModel from the original robot config (clean
        # CudaRobotModelConfig, no runtime attrs like _batch_size).
        _fk_kin_model = CudaRobotModel(node.motion_gen.robot_cfg.kinematics)
    return _fk_kin_model


def get_end_effector_pose(node) -> Optional[list]:
    if node.current_joint_positions is None:
        node.get_logger().warn("Joint states not yet received.")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device)
        kin = _get_kin_model(node)
        with torch.no_grad():
            ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
        pos = ee_pos[0].cpu().tolist()
        quat = ee_quat[0].cpu().tolist()
        return pos + quat
    except Exception as e:
        node.get_logger().warn(f"FK failed: {e}")
        return None


def forward_kinematics(node, joint_positions: List[float]) -> Optional[Point]:
    """Single-waypoint FK using CudaRobotModel.forward() directly.
    Does NOT touch rollout_fn or collision state."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.tensor([joint_positions], dtype=torch.float32, device=device)
        kin = _get_kin_model(node)
        with torch.no_grad():
            ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
        pos = ee_pos.squeeze().cpu().tolist()
        return Point(x=pos[0], y=pos[1], z=pos[2])
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
        kin = _get_kin_model(node)
        with torch.no_grad():
            ee_pos, ee_quat, _, _, _, _, _ = kin.forward(q)
        positions = ee_pos.cpu().tolist()
        # Handle single-waypoint edge case
        if len(joint_states) == 1 and len(positions) == 3 and not isinstance(positions[0], list):
            positions = [positions]
        return [Point(x=p[0], y=p[1], z=p[2]) for p in positions]
    except Exception as e:
        node.get_logger().warn(f"Batched FK failed: {e}")
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
