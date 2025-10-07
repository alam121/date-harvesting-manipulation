# ruff: noqa
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig


JOINT_ORDER = [
"shoulder_pan_joint",
"shoulder_lift_joint",
"elbow_joint",
"wrist_1_joint",
"wrist_2_joint",
"wrist_3_joint",
]


DEFAULT_QOS = QoSProfile(
reliability=ReliabilityPolicy.BEST_EFFORT,
history=HistoryPolicy.KEEP_LAST,
depth=1,
durability=DurabilityPolicy.VOLATILE,
)


WORLD_CONFIG = {
"cuboid": {
"table": {"dims": [5.0, 5.0, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0]},
"pole": {"dims": [0.02, 0.02, 1.0], "pose": [0.0, -0.95, 0.5, 1, 0, 0, 0]},
}
}


PLAN_CFG_DEFAULT = MotionGenPlanConfig(max_attempts=20, enable_finetune_trajopt=True)
