# ruff: noqa
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import rclpy
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def publish_goal_marker(node, position, rank=None):
    
    m = Marker(); m.header.frame_id = "base_link"; m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "goal_positions"; m.id = len(node.goal_poses)
    m.type = Marker.SPHERE; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = position
    m.scale.x = m.scale.y = m.scale.z = 0.05
    m.color.a, m.color.r = 1.0, 1.0
    node.goal_marker_pub.publish(m)
    
    if rank is not None:
        t = Marker(); t.header.frame_id = "base_link"; t.header.stamp = m.header.stamp
        t.ns = "goal_labels"; t.id = 1000 + m.id
        t.type = Marker.TEXT_VIEW_FACING; t.action = Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = position[0], position[1], position[2] + 0.06
        t.scale.z = 0.05; t.color.a = t.color.r = t.color.g = t.color.b = 1.0
        t.text = str(rank); node.goal_marker_pub.publish(t)


def publish_path_marker(node):
    
    m = Marker(); m.header.frame_id = "base_link"; m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "robot_path"; m.id = 0
    m.type = Marker.LINE_STRIP; m.action = Marker.ADD
    m.scale.x = 0.01; m.color.a = 1.0; m.color.g = 1.0
    m.points = node.path_points
    node.path_marker_pub.publish(m)


def track_robot_path(node):
    try:
        tf = node.tf_buffer.lookup_transform(
            "base_link", "tool0", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
        )
        p = Point(x=tf.transform.translation.x, y=tf.transform.translation.y, z=tf.transform.translation.z)
        if not node.path_points or (p.x != node.path_points[-1].x or p.y != node.path_points[-1].y or p.z != node.path_points[-1].z):
            node.path_points.append(p); publish_path_marker(node)
            
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        node.get_logger().warn(f"Path trace TF failed: {e}")
