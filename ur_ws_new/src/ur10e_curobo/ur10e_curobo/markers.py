# ruff: noqa
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import rclpy
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from .fk import forward_kinematics_batch


def publish_planned_path(node, joint_states, label="planned", cartesian_points=None):
    """Publish planned trajectory as a line strip marker (blue).

    If cartesian_points is provided, uses those directly (avoids recomputing FK).
    """
    points = cartesian_points if cartesian_points else forward_kinematics_batch(node, joint_states)

    if not points:
        return

    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "planned_path"
    m.id = hash(label) % 10000  # Different ID per label
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.008  # Thinner than actual path
    m.color.a = 0.8
    m.color.b = 1.0  # Blue for planned
    m.color.r = 0.2
    m.points = points
    m.lifetime.sec = 0  # Persist until cleared
    node.path_marker_pub.publish(m)
    node.get_logger().info(f"Published planned path ({len(points)} points) for {label}")


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
            "base_link", "gripper_tip", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
        )
        p = Point(x=tf.transform.translation.x, y=tf.transform.translation.y, z=tf.transform.translation.z)
        if not node.path_points or (p.x != node.path_points[-1].x or p.y != node.path_points[-1].y or p.z != node.path_points[-1].z):
            node.path_points.append(p); publish_path_marker(node)
            if not node.tf_printed:
                node.tf_printed = True
                node.get_logger().info("Tracked robot path point. TF READY.")

            
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        if not node.tf_warning_printed:
            node.get_logger().warn(f"Waiting for TF. Path trace TF failed: {e}")
            node.tf_warning_printed = True
