"""ROS topic bridge for the rqt UR10e panel."""
import json
import math

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String, Bool, Float32, Float32MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Twist


class RosBridge:
    """Wraps an existing rqt node with UR10e pub/sub topics."""

    def __init__(self, node: Node):
        self._node = node

        # Publishers
        self.cmd_pub = node.create_publisher(String, "/ui_command", 10)
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.goal_pub = node.create_publisher(PoseStamped, "/external_goal_pose", goal_qos)
        self.stop_pub = node.create_publisher(Bool, "/emergency_stop", 10)
        self.teleop_pub = node.create_publisher(Twist, "/teleop_delta", 10)

        # Subscriber data
        self.joint_state_data = None
        self.gripper_force_data = [0.0, 0.0, 0.0]
        self.robot_running = False
        self.goal_info_data = {}
        self.velocity_scale = 5.0
        self.calib_check_result = None
        self.fruit_score_data = {}

        # Subscribers
        node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        node.create_subscription(Float32MultiArray, "/gripper/force", self._gripper_force_cb, 10)
        node.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self._robot_running_cb, 10)
        node.create_subscription(String, "/goal_info", self._goal_info_cb, 10)
        node.create_subscription(Float32, "/velocity_scale", self._velocity_scale_cb, 10)
        node.create_subscription(String, "/calib_check_result", self._calib_check_cb, 10)
        node.create_subscription(String, "/vision/fruit_score", self._fruit_score_cb, 10)

    def _joint_state_cb(self, msg):
        self.joint_state_data = msg

    def _gripper_force_cb(self, msg):
        if len(msg.data) >= 3:
            self.gripper_force_data = list(msg.data[:3])

    def _robot_running_cb(self, msg):
        self.robot_running = msg.data

    def _goal_info_cb(self, msg):
        try:
            self.goal_info_data = json.loads(msg.data)
            if "velocity_scale" in self.goal_info_data:
                self.velocity_scale = self.goal_info_data["velocity_scale"]
        except json.JSONDecodeError:
            pass

    def _velocity_scale_cb(self, msg):
        self.velocity_scale = msg.data

    def _calib_check_cb(self, msg):
        self.calib_check_result = msg.data

    def _fruit_score_cb(self, msg):
        try:
            self.fruit_score_data = json.loads(msg.data)
        except Exception:
            pass

    def publish_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)

    def publish_goal(self, x, y, z, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
        qmag = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if qmag < 0.01:
            return
        qw, qx, qy, qz = qw/qmag, qx/qmag, qy/qmag, qz/qmag

        msg = PoseStamped()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        self.goal_pub.publish(msg)

    def publish_stop(self):
        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

    def publish_velocity_scale(self, scale: float):
        self.publish_cmd(f"set_velocity_scale {scale:.2f}")
