import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

try:
    from . import static_obstacles
except ImportError:
    from ur10e_curobo import static_obstacles

class VisibleBoxPublisher(Node):
    def __init__(self):
        super().__init__('visible_box_publisher')

        self.marker_pub = self.create_publisher(Marker, '/env_marker', 10)
        self.frame_id = "base_link"
        self.static_obstacles = list(static_obstacles.DEFAULT_STATIC_OBSTACLES)
        self.timer = self.create_timer(1.0, self.publish_marker)  # Every second

    def publish_marker(self):
        static_obstacles.publish_static_obstacles(
            self.marker_pub,
            self.get_clock(),
            frame_id=self.frame_id,
            specs=self.static_obstacles,
        )

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
