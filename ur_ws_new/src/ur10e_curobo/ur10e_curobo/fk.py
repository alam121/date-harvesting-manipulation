# ruff: noqa
from typing import List, Optional
import torch
from curobo.types.robot import JointState
from geometry_msgs.msg import Point


def get_end_effector_pose(node) -> Optional[list]:
    if node.current_joint_positions is None or len(node.current_joint_positions) != len(node.joint_order):
        node.get_logger().warn("Joint states not yet received or incomplete.")
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
