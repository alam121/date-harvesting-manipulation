from typing import Sequence, List
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.duration import Duration




class Visualizer:
def __init__(self, node, goal_topic: str, path_topic: str):
self.node = node
self.goal_pub = node.create_publisher(Marker, goal_topic, 10)
self.path_pub = node.create_publisher(Marker, path_topic, 10)
self.path_points: List[Point] = []


def publish_goal_marker(self, position: Sequence[float], idx: int, rank: int = None):
m = Marker()
m.header.frame_id = "base_link"
m.header.stamp = self.node.get_clock().now().to_msg()
m.ns = "goal_positions"
m.id = idx
m.type = Marker.SPHERE
m.action = Marker.ADD
m.pose.position.x, m.pose.position.y, m.pose.position.z = position
m.scale.x = m.scale.y = m.scale.z = 0.05
m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0, 0.0
self.goal_pub.publish(m)


if rank is not None:
t = Marker()
t.header.frame_id = "base_link"
t.header.stamp = self.node.get_clock().now().to_msg()
t.ns = "goal_labels"
t.id = 1000 + idx
t.type = Marker.TEXT_VIEW_FACING
t.action = Marker.ADD
t.pose.position.x, t.pose.position.y, t.pose.position.z = position[0], position[1], position[2] + 0.06
t.scale.z = 0.05
t.color.a = t.color.r = t.color.g = t.color.b = 1.0
t.text = str(rank)
self.goal_pub.publish(t)


def track_robot_path(self, tf_buffer):
try:
transform = tf_buffer.lookup_transform("base_link", "tool0", self.node.get_clock().now().to_msg(), timeout=Duration(seconds=1.0))
p = Point(
x=transform.transform.translation.x,
y=transform.transform.translation.y,
z=transform.transform.translation.z,
)
if not self.path_points or (self.path_points[-1].x != p.x or self.path_points[-1].y != p.y or self.path_points[-1].z != p.z):
self.path_points.append(p)
self._publish_path_marker()
except Exception as e:
self.node.get_logger().warn(f"Failed to track robot position: {e}")


def _publish_path_marker(self):
m = Marker()
m.header.frame_id = "base_link"
m.header.stamp = self.node.get_clock().now().to_msg()
m.ns = "robot_path"
m.id = 0
m.type = Marker.LINE_STRIP
m.action = Marker.ADD
m.scale.x = 0.01
m.color.a, m.color.r, m.color.g, m.color.b = 1.0, 0.0, 1.0, 0.0
m.points = self.path_points
self.path_pub.publish(m)
