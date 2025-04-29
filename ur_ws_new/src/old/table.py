import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class VisibleBoxPublisher(Node):
    def __init__(self):
        super().__init__('visible_box_publisher')

        self.marker_pub = self.create_publisher(Marker, '/env_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)  # Every second

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'  # Make sure this matches your RViz fixed frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'test_box'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Small cube at 0.5m in front of base_link
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = -0.1  # Slightly above ground

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 2.0
        marker.scale.y = 2.0
        marker.scale.z = 0.2

        marker.color.a = 1.0  # Fully opaque
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)
        self.get_logger().info("Published visible cube marker at [0.5, 0.0, 0.1]")

def main():
    rclpy.init()
    node = VisibleBoxPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

