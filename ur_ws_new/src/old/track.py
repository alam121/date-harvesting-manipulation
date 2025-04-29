import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class EndEffectorTracker(Node):
    def __init__(self):
        super().__init__("end_effector_tracker")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_marker_pub = self.create_publisher(Marker, "/end_effector_path", 10)
        self.path_points = []

        self.create_timer(0.1, self.track_end_effector_path)

    def track_end_effector_path(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link", "tool0", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            print(f"End-Effector Position → X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")

            point = Point(x=x, y=y, z=z)

            if not self.path_points or (
                self.path_points[-1].x != point.x or
                self.path_points[-1].y != point.y or
                self.path_points[-1].z != point.z
            ):
                self.path_points.append(point)
                self.publish_path_marker()

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

    def publish_path_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "end_effector_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.points = self.path_points
        self.path_marker_pub.publish(marker)

def main():
    rclpy.init()
    node = EndEffectorTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

