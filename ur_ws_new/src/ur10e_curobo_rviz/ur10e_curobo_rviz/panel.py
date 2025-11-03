import rclpy
from rclpy.node import Node
from python_qt_binding.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from rviz2.plugin import RVizPlugin

class UR10eCuroboPanel(RVizPlugin):

    def __init__(self):
        super().__init__()
        self.node = rclpy.create_node("ur10e_curobo_panel")

        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        self.status_label = QLabel("UR10e CuRobo Control Panel")
        layout.addWidget(self.status_label)

        btn_home = QPushButton("Go Home")
        btn_home.clicked.connect(self.go_home)
        layout.addWidget(btn_home)

        btn_dropoff = QPushButton("Go Dropoff")
        btn_dropoff.clicked.connect(self.go_dropoff)
        layout.addWidget(btn_dropoff)

    def go_home(self):
        self.node.get_logger().info("Sending 'home' command...")
        # publish a message on a topic ur10e_curobo.main subscribes to (e.g. /external_goal_pose)
        # implement publisher here if needed

    def go_dropoff(self):
        self.node.get_logger().info("Sending 'dropoff' command...")

    def get_widget(self):
        return self.widget

