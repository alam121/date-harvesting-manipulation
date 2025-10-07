# ruff: noqa
import rclpy
from .node import UR10eCuroboMoveIt


def main():
    print("Starting UR10e MoveIt Node…")
    rclpy.init()
    node = UR10eCuroboMoveIt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down node.")
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()