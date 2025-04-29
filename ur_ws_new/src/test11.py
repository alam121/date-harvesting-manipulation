import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
import time

class GoalExecutor(Node):
    def __init__(self):
        super().__init__('goal_executor')

        # Publisher to send Pose goals
        self.goal_pub = self.create_publisher(Pose, '/external_goal_pose', 10)


        # Home Pose
        self.home_pose = {
            "position": [0.023206107318401337, 0.9999909996986389, 0.735256552696228,],
            "orientation": [ 0.8613798022270203, -0.07769563049077988, 0.019696131348609924, -0.5015978813171387]
        }

        # Goals
        self.goals = [
            {"position": [0.025126459077000618, 1.0098114013671875, 0.8723804950714111],
             "orientation": [0.8579429984092712, -0.11112676560878754, 0.007773248944431543, -0.5015220642089844]},
        ]

        # Start
        self.timer = self.create_timer(2.0, self.start_mission)
        self.current_goal_idx = 0
        self.phase = 'go_home'

    def start_mission(self):
        if self.current_goal_idx >= len(self.goals):
            self.get_logger().info("✅ All goals completed.")
            self.timer.cancel()
            return

        if self.phase == 'go_home':
            self.publish_pose(self.home_pose)
            self.get_logger().info("Going to Home Position...")
            self.phase = 'go_goal'

        elif self.phase == 'go_goal':
            goal = self.goals[self.current_goal_idx]
            self.publish_pose(goal)
            self.get_logger().info(f"Moving to Goal {self.current_goal_idx + 1}...")

        elif self.phase == 'next_goal':
            self.current_goal_idx += 1
            self.phase = 'go_goal'

    def publish_pose(self, pose_dict):
        pose_msg = Pose()

        pose_msg.position.x = pose_dict["position"][0]
        pose_msg.position.y = pose_dict["position"][1]
        pose_msg.position.z = pose_dict["position"][2]

        pose_msg.orientation.w = pose_dict["orientation"][0]
        pose_msg.orientation.x = pose_dict["orientation"][1]
        pose_msg.orientation.y = pose_dict["orientation"][2]
        pose_msg.orientation.z = pose_dict["orientation"][3]

        self.goal_pub.publish(pose_msg)

def main():
    rclpy.init()
    node = GoalExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

