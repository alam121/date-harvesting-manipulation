# ruff: noqa
from typing import List, Optional
import torch
from curobo.types.robot import JointState
from geometry_msgs.msg import Point


def get_end_effector_pose(node) -> Optional[list]:
    if node.current_joint_positions is None:
        node.get_logger().warn("Joint states not yet received.")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        js = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        ee = node.motion_gen.rollout_fn.compute_kinematics(js)
        pos = ee.ee_pos_seq[0].cpu().tolist()
        quat = ee.ee_quat_seq[0].cpu().tolist()
        return pos + quat
    except Exception as e:
        node.get_logger().warn(f"FK failed: {e}")
        return None


def forward_kinematics(node, joint_positions: List[float]) -> Optional[Point]:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        js = JointState.from_position(
            torch.tensor([joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        ee = node.motion_gen.rollout_fn.compute_kinematics(js)
        pos = ee.ee_pos_seq.squeeze().tolist()
        return Point(x=pos[0], y=pos[1], z=pos[2])
    except Exception as e:
        node.get_logger().warn(f"CuRobo FK failed: {e}")
        return None


def solve_ik_fast(node, target_pose: list, seed: Optional[list] = None, iters: int = 25, lr: float = 0.6):
    """A lightweight, best-effort IK solver using simple gradient descent.

    Args:
        node: the ROS node (used for motion_gen and joint_order)
        target_pose: [x,y,z,qw,qx,qy,qz]
        seed: optional initial joint positions (list)
        iters: number of gradient steps
        lr: step size (tweakable)

    Returns:
        list of joint positions (len == len(node.joint_order)) or None on failure

    Note: This is a simple numeric solver intended for small teleop deltas. It's
    not a full replacement for a robust IK solver and may fail for large jumps.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n = len(node.joint_order)

        if seed is None:
            if node.current_joint_positions is not None:
                seed = node.current_joint_positions
            else:
                seed = [0.0] * n

        # initial tensor (1 x n)
        x_target = torch.tensor(target_pose[:3], dtype=torch.float32, device=device)
        q_target = torch.tensor(target_pose[3:], dtype=torch.float32, device=device)

        var = torch.tensor([seed], dtype=torch.float32, device=device, requires_grad=True)

        for _ in range(iters):
            # build joint state and compute FK
            js = JointState.from_position(var, joint_names=node.joint_order)
            ee = node.motion_gen.rollout_fn.compute_kinematics(js)
            pos = ee.ee_pos_seq[0]
            quat = ee.ee_quat_seq[0]

            # position loss (MSE)
            pos_err = pos - x_target
            pos_loss = (pos_err * pos_err).sum()

            # orientation loss: use 1 - cos(angle) ~ 1 - dot(quat, q_target)
            # ensure normalized
            quat_norm = quat / (quat.norm() + 1e-12)
            qnorm_target = q_target / (q_target.norm() + 1e-12)
            dot = (quat_norm * qnorm_target).sum()
            ori_loss = (1.0 - dot)

            loss = pos_loss + 0.1 * ori_loss

            # backward
            loss.backward()

            # gradient step (in-place replacement)
            with torch.no_grad():
                grad = var.grad
                if grad is None:
                    break
                var -= lr * grad
                var.grad.zero_()

        result = var.detach().cpu().squeeze().tolist()
        # sanity check length
        if len(result) != n:
            return None
        return result
    except Exception as e:
        try:
            node.get_logger().warn(f"IK solver failed: {e}")
        except Exception:
            pass
        return None

