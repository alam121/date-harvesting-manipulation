import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class VisibleBoxPublisher(Node):
    def __init__(self):
        super().__init__('visible_box_publisher')

        self.marker_pub = self.create_publisher(Marker, '/env_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)  # Every second

    def publish_marker(self):
        now = self.get_clock().now().to_msg()

        # ─── TABLE ───
        table = Marker()
        table.header.frame_id = 'base_link'
        table.header.stamp = now
        table.ns = 'environment'
        table.id = 0
        table.type = Marker.CUBE
        table.action = Marker.ADD

        # center at [0,0,-0.1], dims [5×5×0.2]
        table.pose.position.x = 0.0
        table.pose.position.y = 0.0
        table.pose.position.z = -0.1  
        table.pose.orientation.w = 1.0

        table.scale.x = 5.0
        table.scale.y = 5.0
        table.scale.z = 0.2

        table.color.a = 1.0
        table.color.r = 1.0
        table.color.g = 0.0
        table.color.b = 0.0

        # ─── POLE ───
        pole = Marker()
        pole.header.frame_id = 'base_link'
        pole.header.stamp = now
        pole.ns = 'environment'
        pole.id = 1
        pole.type = Marker.CUBE
        pole.action = Marker.ADD

        # center at [0,0.65,0.5], dims [0.02×0.02×1.0]
        pole.pose.position.x = 0.0
        pole.pose.position.y = -0.65
        pole.pose.position.z = 0.5  
        pole.pose.orientation.w = 1.0

        pole.scale.x = 0.02
        pole.scale.y = 0.02
        pole.scale.z = 1.0

        pole.color.a = 1.0
        pole.color.r = 0.0
        pole.color.g = 1.0
        pole.color.b = 0.0

        # publish both
        self.marker_pub.publish(table)
        self.marker_pub.publish(pole)

        self.get_logger().info("Published table and pole markers.")

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

