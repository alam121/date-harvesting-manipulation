import sys
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from PyQt5 import QtWidgets, QtGui
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped


class UiBridge(Node):
    """ROS bridge that the Qt UI talks to."""

    def __init__(
        self,
        command_topic: str = "/ui_command",
        goal_topic: str = "/external_goal_pose",
        stop_topic: str = "/emergency_stop",
    ):
        super().__init__("ur10e_desktop_ui")
        self.cmd_pub = self.create_publisher(String, command_topic, 10)
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.goal_pub = self.create_publisher(PoseStamped, goal_topic, goal_qos)
        self.stop_pub = self.create_publisher(Bool, stop_topic, 10)

    def publish_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)

    def publish_goal(
        self,
        x: float,
        y: float,
        z: float,
        qw: float = 1.0,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
    ):
        msg = PoseStamped()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = qw
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        self.goal_pub.publish(msg)

    def publish_stop(self):
        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)


class MainWindow(QtWidgets.QWidget):
    def __init__(self, ros: UiBridge):
        super().__init__()
        self.ros = ros
        self.setWindowTitle("UR10e Desktop UI")
        self.setMinimumWidth(420)
        self._status: Optional[QtWidgets.QLabel] = None
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("UR10e cuRobo Desktop Control")
        title.setFont(QtGui.QFont("Helvetica", 14, QtGui.QFont.Bold))
        layout.addWidget(title)

        status = QtWidgets.QLabel("Ready — topics: /ui_command, /external_goal_pose, /emergency_stop")
        status.setStyleSheet("color: #2d6a4f;")
        self._status = status
        layout.addWidget(status)

        # Goal input section
        goal_box = QtWidgets.QGroupBox("Goal (base frame)")
        goal_layout = QtWidgets.QGridLayout(goal_box)
        self.x_in = QtWidgets.QLineEdit("0.30")
        self.y_in = QtWidgets.QLineEdit("0.00")
        self.z_in = QtWidgets.QLineEdit("0.20")
        self.qw_in = QtWidgets.QLineEdit("1.0")
        self.qx_in = QtWidgets.QLineEdit("0.0")
        self.qy_in = QtWidgets.QLineEdit("0.0")
        self.qz_in = QtWidgets.QLineEdit("0.0")

        goal_layout.addWidget(QtWidgets.QLabel("X (m)"), 0, 0)
        goal_layout.addWidget(self.x_in, 0, 1)
        goal_layout.addWidget(QtWidgets.QLabel("Y (m)"), 0, 2)
        goal_layout.addWidget(self.y_in, 0, 3)
        goal_layout.addWidget(QtWidgets.QLabel("Z (m)"), 0, 4)
        goal_layout.addWidget(self.z_in, 0, 5)

        goal_layout.addWidget(QtWidgets.QLabel("qw"), 1, 0)
        goal_layout.addWidget(self.qw_in, 1, 1)
        goal_layout.addWidget(QtWidgets.QLabel("qx"), 1, 2)
        goal_layout.addWidget(self.qx_in, 1, 3)
        goal_layout.addWidget(QtWidgets.QLabel("qy"), 1, 4)
        goal_layout.addWidget(self.qy_in, 1, 5)
        goal_layout.addWidget(QtWidgets.QLabel("qz"), 1, 6)
        goal_layout.addWidget(self.qz_in, 1, 7)

        send_goal_btn = QtWidgets.QPushButton("Publish Goal")
        send_goal_btn.clicked.connect(self._handle_send_goal)
        goal_layout.addWidget(send_goal_btn, 2, 0, 1, 8)
        layout.addWidget(goal_box)

        # Actions row 1
        actions1 = QtWidgets.QHBoxLayout()
        home_btn = QtWidgets.QPushButton("Home")
        home_btn.clicked.connect(lambda: self._send_cmd("home"))
        drop_btn = QtWidgets.QPushButton("Dropoff")
        drop_btn.clicked.connect(lambda: self._send_cmd("dropoff"))
        exec_btn = QtWidgets.QPushButton("Execute Goals")
        exec_btn.clicked.connect(lambda: self._send_cmd("execute"))
        clear_btn = QtWidgets.QPushButton("Clear Goals")
        clear_btn.clicked.connect(lambda: self._send_cmd("clear"))
        actions1.addWidget(home_btn)
        actions1.addWidget(drop_btn)
        actions1.addWidget(exec_btn)
        actions1.addWidget(clear_btn)
        layout.addLayout(actions1)

        # Actions row 2
        actions2 = QtWidgets.QHBoxLayout()
        capture_btn = QtWidgets.QPushButton("Start Capture (10s)")
        capture_btn.clicked.connect(lambda: self._send_cmd("capture 10"))
        cap_stop_btn = QtWidgets.QPushButton("Stop Capture")
        cap_stop_btn.clicked.connect(lambda: self._send_cmd("capture_stop"))
        open_btn = QtWidgets.QPushButton("Open Gripper")
        open_btn.clicked.connect(lambda: self._send_cmd("open"))
        close_btn = QtWidgets.QPushButton("Close Gripper")
        close_btn.clicked.connect(lambda: self._send_cmd("close"))
        actions2.addWidget(capture_btn)
        actions2.addWidget(cap_stop_btn)
        actions2.addWidget(open_btn)
        actions2.addWidget(close_btn)
        layout.addLayout(actions2)

        # Safety row
        safety = QtWidgets.QHBoxLayout()
        stop_btn = QtWidgets.QPushButton("Emergency Stop")
        stop_btn.setStyleSheet("background-color: #d00000; color: white; font-weight: bold;")
        stop_btn.clicked.connect(self._handle_stop)
        debug_btn = QtWidgets.QPushButton("Print World")
        debug_btn.clicked.connect(lambda: self._send_cmd("debug_world"))
        safety.addWidget(stop_btn)
        safety.addWidget(debug_btn)
        layout.addLayout(safety)

        layout.addStretch(1)

    def _handle_send_goal(self):
        try:
            x = float(self.x_in.text())
            y = float(self.y_in.text())
            z = float(self.z_in.text())
            qw = float(self.qw_in.text())
            qx = float(self.qx_in.text())
            qy = float(self.qy_in.text())
            qz = float(self.qz_in.text())
        except ValueError:
            self._set_status("Invalid goal values.", error=True)
            return
        self.ros.publish_goal(x, y, z, qw, qx, qy, qz)
        self._set_status(f"Published goal ({x:.3f}, {y:.3f}, {z:.3f}).")

    def _handle_stop(self):
        self.ros.publish_stop()
        self._send_cmd("stop")
        self._set_status("Emergency stop sent.", error=True)

    def _send_cmd(self, cmd: str):
        self.ros.publish_cmd(cmd)
        self._set_status(f"Sent command '{cmd}'.")

    def _set_status(self, text: str, error: bool = False):
        if not self._status:
            return
        color = "#d00000" if error else "#2d6a4f"
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color};")


def main():
    rclpy.init()
    ros_node = UiBridge()

    def spin_ros():
        rclpy.spin(ros_node)

    spin_thread = threading.Thread(target=spin_ros, daemon=True)
    spin_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(ros_node)
    win.show()
    ret = app.exec_()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
