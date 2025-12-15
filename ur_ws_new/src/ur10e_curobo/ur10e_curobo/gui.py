import sys
import threading
import json
from typing import Optional, List
from collections import deque
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer, pyqtSignal
from std_msgs.msg import Bool, String, Float32MultiArray, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

try:
    from pyqtgraph import PlotWidget, mkPen
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("[WARN] pyqtgraph not installed. Install with: pip install pyqtgraph")
    print("[WARN] Graphs will be disabled.")


class UiBridge(Node):
    """ROS bridge that the Qt UI talks to."""

    def __init__(
        self,
        command_topic: str = "/ui_command",
        goal_topic: str = "/external_goal_pose",
        stop_topic: str = "/emergency_stop",
    ):
        super().__init__("ur10e_desktop_ui")

        # Publishers
        self.cmd_pub = self.create_publisher(String, command_topic, 10)
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.goal_pub = self.create_publisher(PoseStamped, goal_topic, goal_qos)
        self.stop_pub = self.create_publisher(Bool, stop_topic, 10)

        # Subscribers
        self.joint_state_data = None
        self.gripper_force_data = [0.0, 0.0, 0.0]
        self.robot_running = False

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            "/gripper/force",
            self._gripper_force_cb,
            10
        )
        self.create_subscription(
            Bool,
            "/io_and_status_controller/robot_program_running",
            self._robot_running_cb,
            10
        )

        # Goal info subscriber
        self.goal_info_data = {}
        self.velocity_scale = 5.0  # Default
        self.create_subscription(
            String,
            "/goal_info",
            self._goal_info_cb,
            10
        )
        self.create_subscription(
            Float32,
            "/velocity_scale",
            self._velocity_scale_cb,
            10
        )

    def _joint_state_cb(self, msg: JointState):
        self.joint_state_data = msg

    def _gripper_force_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 3:
            self.gripper_force_data = list(msg.data[:3])

    def _robot_running_cb(self, msg: Bool):
        self.robot_running = msg.data

    def _goal_info_cb(self, msg: String):
        try:
            self.goal_info_data = json.loads(msg.data)
            if "velocity_scale" in self.goal_info_data:
                self.velocity_scale = self.goal_info_data["velocity_scale"]
        except json.JSONDecodeError:
            pass

    def _velocity_scale_cb(self, msg: Float32):
        self.velocity_scale = msg.data

    def publish_velocity_scale(self, scale: float):
        """Send velocity scale command to node."""
        self.publish_cmd(f"set_velocity_scale {scale:.2f}")

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
        # Normalize quaternion
        qmag = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if qmag < 0.01:
            self.get_logger().warn("Invalid quaternion (zero magnitude)")
            return
        qw, qx, qy, qz = qw/qmag, qx/qmag, qy/qmag, qz/qmag

        msg = PoseStamped()
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
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
    # Signals for thread-safe updates
    status_update = pyqtSignal(str, bool)

    def __init__(self, ros: UiBridge):
        super().__init__()
        self.ros = ros
        self.setWindowTitle("UR10e Enhanced Control Panel")
        self.setMinimumSize(1200, 800)

        # Data storage for graphs
        self.force_history = [deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)]
        self.joint_history = [deque(maxlen=200) for _ in range(6)]
        self.time_data = deque(maxlen=200)
        self.elapsed_time = 0.0

        self._status: Optional[QtWidgets.QLabel] = None
        self._build_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(50)  # 20Hz update

        # Connect signals
        self.status_update.connect(self._set_status_slot)

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left panel: Controls
        left_panel = self._build_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel: Monitoring & Graphs
        right_panel = self._build_monitor_panel()
        main_layout.addWidget(right_panel, stretch=2)

    def _build_control_panel(self):
        """Build left control panel with all buttons and inputs."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Title
        title = QtWidgets.QLabel("UR10e Control Panel")
        title.setFont(QtGui.QFont("Helvetica", 16, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Status indicator
        self._status = QtWidgets.QLabel("Ready — ROS2 Connected")
        self._status.setStyleSheet("color: #2d6a4f; font-weight: bold; padding: 8px; background: #e8f5e9; border-radius: 4px;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addSpacing(10)

        # Robot state display
        state_group = QtWidgets.QGroupBox("Robot State")
        state_layout = QtWidgets.QVBoxLayout(state_group)
        self.robot_state_label = QtWidgets.QLabel("Program: Unknown")
        self.robot_state_label.setStyleSheet("font-size: 12pt; padding: 5px;")
        state_layout.addWidget(self.robot_state_label)
        layout.addWidget(state_group)

        # Global Velocity Scale (with slider)
        velocity_group = QtWidgets.QGroupBox("Global Velocity Scale")
        velocity_layout = QtWidgets.QVBoxLayout(velocity_group)

        # Slider row
        slider_row = QtWidgets.QHBoxLayout()
        self.velocity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.velocity_slider.setRange(10, 100)  # 0.1 to 10.0 (x10 for integer slider)
        self.velocity_slider.setValue(50)  # Default 5.0
        self.velocity_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.velocity_slider.setTickInterval(10)
        self.velocity_slider.valueChanged.connect(self._on_velocity_slider_changed)
        slider_row.addWidget(self.velocity_slider)

        self.velocity_value_label = QtWidgets.QLabel("5.0x")
        self.velocity_value_label.setFont(QtGui.QFont("Courier", 12, QtGui.QFont.Bold))
        self.velocity_value_label.setMinimumWidth(50)
        self.velocity_value_label.setStyleSheet("color: #1976d2; font-weight: bold;")
        slider_row.addWidget(self.velocity_value_label)
        velocity_layout.addLayout(slider_row)

        # Quick preset buttons
        preset_row = QtWidgets.QHBoxLayout()
        for label, value in [("Slow", 1.0), ("Normal", 3.0), ("Fast", 5.0), ("Max", 8.0)]:
            btn = QtWidgets.QPushButton(label)
            btn.setStyleSheet("padding: 4px; font-size: 9pt;")
            btn.clicked.connect(lambda checked, v=value: self._set_velocity_scale(v))
            preset_row.addWidget(btn)
        velocity_layout.addLayout(preset_row)

        apply_velocity_btn = QtWidgets.QPushButton("Apply Velocity Scale")
        apply_velocity_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 6px;")
        apply_velocity_btn.clicked.connect(self._apply_velocity_scale)
        velocity_layout.addWidget(apply_velocity_btn)
        layout.addWidget(velocity_group)

        # Speed control (motion-specific)
        speed_group = QtWidgets.QGroupBox("Motion-Specific Speeds")
        speed_layout = QtWidgets.QGridLayout(speed_group)

        speed_layout.addWidget(QtWidgets.QLabel("Home:"), 0, 0)
        self.speed_home = QtWidgets.QDoubleSpinBox()
        self.speed_home.setRange(1.0, 6.0)
        self.speed_home.setValue(4.0)
        self.speed_home.setSingleStep(0.5)
        speed_layout.addWidget(self.speed_home, 0, 1)

        speed_layout.addWidget(QtWidgets.QLabel("Dropoff:"), 1, 0)
        self.speed_dropoff = QtWidgets.QDoubleSpinBox()
        self.speed_dropoff.setRange(1.0, 8.0)
        self.speed_dropoff.setValue(6.5)
        self.speed_dropoff.setSingleStep(0.5)
        speed_layout.addWidget(self.speed_dropoff, 1, 1)

        speed_layout.addWidget(QtWidgets.QLabel("Approach:"), 2, 0)
        self.speed_approach = QtWidgets.QDoubleSpinBox()
        self.speed_approach.setRange(0.2, 2.0)
        self.speed_approach.setValue(0.5)
        self.speed_approach.setSingleStep(0.1)
        speed_layout.addWidget(self.speed_approach, 2, 1)

        apply_speed_btn = QtWidgets.QPushButton("Apply Speeds")
        apply_speed_btn.clicked.connect(self._apply_speed_settings)
        speed_layout.addWidget(apply_speed_btn, 3, 0, 1, 2)
        layout.addWidget(speed_group)

        # Goal input section
        goal_box = QtWidgets.QGroupBox("Manual Goal (base frame)")
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
        goal_layout.addWidget(QtWidgets.QLabel("Y (m)"), 1, 0)
        goal_layout.addWidget(self.y_in, 1, 1)
        goal_layout.addWidget(QtWidgets.QLabel("Z (m)"), 2, 0)
        goal_layout.addWidget(self.z_in, 2, 1)

        goal_layout.addWidget(QtWidgets.QLabel("qw"), 3, 0)
        goal_layout.addWidget(self.qw_in, 3, 1)
        goal_layout.addWidget(QtWidgets.QLabel("qx"), 4, 0)
        goal_layout.addWidget(self.qx_in, 4, 1)
        goal_layout.addWidget(QtWidgets.QLabel("qy"), 5, 0)
        goal_layout.addWidget(self.qy_in, 5, 1)
        goal_layout.addWidget(QtWidgets.QLabel("qz"), 6, 0)
        goal_layout.addWidget(self.qz_in, 6, 1)

        send_goal_btn = QtWidgets.QPushButton("Send Goal")
        send_goal_btn.clicked.connect(self._handle_send_goal)
        send_goal_btn.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold; padding: 8px;")
        goal_layout.addWidget(send_goal_btn, 7, 0, 1, 2)
        layout.addWidget(goal_box)

        # Motion control buttons
        motion_group = QtWidgets.QGroupBox("Motion Commands")
        motion_layout = QtWidgets.QVBoxLayout(motion_group)

        btn_style = "padding: 10px; font-size: 11pt; font-weight: bold;"

        home_btn = QtWidgets.QPushButton("🏠 Home")
        home_btn.setStyleSheet(f"{btn_style} background-color: #4caf50; color: white;")
        home_btn.clicked.connect(lambda: self._send_cmd("home"))
        motion_layout.addWidget(home_btn)

        drop_btn = QtWidgets.QPushButton("📦 Dropoff")
        drop_btn.setStyleSheet(f"{btn_style} background-color: #2196f3; color: white;")
        drop_btn.clicked.connect(lambda: self._send_cmd("dropoff"))
        motion_layout.addWidget(drop_btn)

        exec_btn = QtWidgets.QPushButton("▶ Execute Goals")
        exec_btn.setStyleSheet(f"{btn_style} background-color: #ff9800; color: white;")
        exec_btn.clicked.connect(lambda: self._send_cmd("execute"))
        motion_layout.addWidget(exec_btn)

        clear_btn = QtWidgets.QPushButton("🗑 Clear Goals")
        clear_btn.setStyleSheet(f"{btn_style} background-color: #9e9e9e; color: white;")
        clear_btn.clicked.connect(lambda: self._send_cmd("clear"))
        motion_layout.addWidget(clear_btn)

        layout.addWidget(motion_group)

        # Gripper & capture controls
        aux_group = QtWidgets.QGroupBox("Gripper & Capture")
        aux_layout = QtWidgets.QVBoxLayout(aux_group)

        gripper_row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open Gripper")
        open_btn.clicked.connect(lambda: self._send_cmd("open"))
        close_btn = QtWidgets.QPushButton("Close Gripper")
        close_btn.clicked.connect(lambda: self._send_cmd("close"))
        gripper_row.addWidget(open_btn)
        gripper_row.addWidget(close_btn)
        aux_layout.addLayout(gripper_row)

        capture_row = QtWidgets.QHBoxLayout()
        capture_btn = QtWidgets.QPushButton("Start Capture (10s)")
        capture_btn.clicked.connect(lambda: self._send_cmd("capture 10"))
        cap_stop_btn = QtWidgets.QPushButton("Stop Capture")
        cap_stop_btn.clicked.connect(lambda: self._send_cmd("capture_stop"))
        capture_row.addWidget(capture_btn)
        capture_row.addWidget(cap_stop_btn)
        aux_layout.addLayout(capture_row)

        layout.addWidget(aux_group)

        # Emergency stop
        stop_btn = QtWidgets.QPushButton("⚠ EMERGENCY STOP")
        stop_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 14pt; padding: 15px;")
        stop_btn.clicked.connect(self._handle_stop)
        layout.addWidget(stop_btn)

        # Debug
        debug_btn = QtWidgets.QPushButton("Debug World")
        debug_btn.clicked.connect(lambda: self._send_cmd("debug_world"))
        layout.addWidget(debug_btn)

        layout.addStretch(1)

        return panel

    def _build_monitor_panel(self):
        """Build right monitoring panel with status displays and graphs."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # Gripper force display
        gripper_group = QtWidgets.QGroupBox("Gripper Force Sensors")
        gripper_layout = QtWidgets.QHBoxLayout(gripper_group)

        self.force_labels = []
        finger_names = ["Left", "Center", "Right"]
        for i, name in enumerate(finger_names):
            finger_widget = QtWidgets.QWidget()
            finger_layout = QtWidgets.QVBoxLayout(finger_widget)
            finger_layout.setSpacing(5)

            label = QtWidgets.QLabel(name)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setFont(QtGui.QFont("Helvetica", 10, QtGui.QFont.Bold))
            finger_layout.addWidget(label)

            value_label = QtWidgets.QLabel("0.00 N")
            value_label.setAlignment(QtCore.Qt.AlignCenter)
            value_label.setFont(QtGui.QFont("Courier", 14, QtGui.QFont.Bold))
            value_label.setStyleSheet("background: #263238; color: #00e676; padding: 8px; border-radius: 4px;")
            self.force_labels.append(value_label)
            finger_layout.addWidget(value_label)

            # Progress bar for visual feedback
            progress = QtWidgets.QProgressBar()
            progress.setRange(0, 50)  # 0 to -0.50 N scaled to 0-50
            progress.setValue(0)
            progress.setTextVisible(False)
            progress.setStyleSheet("""
                QProgressBar {
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #00e676;
                }
            """)
            finger_layout.addWidget(progress)
            setattr(self, f"force_bar_{i}", progress)

            gripper_layout.addWidget(finger_widget)

        layout.addWidget(gripper_group)

        # Joint state display
        joint_group = QtWidgets.QGroupBox("Joint Positions (rad)")
        joint_layout = QtWidgets.QGridLayout(joint_group)
        joint_layout.setSpacing(5)

        self.joint_labels = []
        joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
        for i, name in enumerate(joint_names):
            label = QtWidgets.QLabel(f"{name}:")
            label.setFont(QtGui.QFont("Helvetica", 9))
            joint_layout.addWidget(label, i // 3, (i % 3) * 2)

            value = QtWidgets.QLabel("0.000")
            value.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
            value.setStyleSheet("background: #37474f; color: #64b5f6; padding: 4px; border-radius: 3px;")
            self.joint_labels.append(value)
            joint_layout.addWidget(value, i // 3, (i % 3) * 2 + 1)

        layout.addWidget(joint_group)

        # Goals Information Panel
        goals_group = QtWidgets.QGroupBox("Goals Information")
        goals_layout = QtWidgets.QVBoxLayout(goals_group)

        # Goal count and capture status
        goals_header = QtWidgets.QHBoxLayout()
        self.goal_count_label = QtWidgets.QLabel("Goals: 0")
        self.goal_count_label.setFont(QtGui.QFont("Helvetica", 11, QtGui.QFont.Bold))
        self.goal_count_label.setStyleSheet("color: #1976d2;")
        goals_header.addWidget(self.goal_count_label)

        self.capture_status_label = QtWidgets.QLabel("Capture: Inactive")
        self.capture_status_label.setFont(QtGui.QFont("Helvetica", 10))
        self.capture_status_label.setStyleSheet("color: #757575;")
        goals_header.addWidget(self.capture_status_label)
        goals_header.addStretch()
        goals_layout.addLayout(goals_header)

        # Latest goal display
        latest_goal_row = QtWidgets.QHBoxLayout()
        latest_goal_row.addWidget(QtWidgets.QLabel("Latest:"))
        self.latest_goal_label = QtWidgets.QLabel("--")
        self.latest_goal_label.setFont(QtGui.QFont("Courier", 9))
        self.latest_goal_label.setStyleSheet("background: #e3f2fd; padding: 3px; border-radius: 3px;")
        latest_goal_row.addWidget(self.latest_goal_label)
        goals_layout.addLayout(latest_goal_row)

        # Best goal display
        best_goal_row = QtWidgets.QHBoxLayout()
        best_goal_row.addWidget(QtWidgets.QLabel("Best:"))
        self.best_goal_label = QtWidgets.QLabel("--")
        self.best_goal_label.setFont(QtGui.QFont("Courier", 9))
        self.best_goal_label.setStyleSheet("background: #e8f5e9; padding: 3px; border-radius: 3px;")
        best_goal_row.addWidget(self.best_goal_label)
        goals_layout.addLayout(best_goal_row)

        # Goal queue list
        self.goal_list_widget = QtWidgets.QListWidget()
        self.goal_list_widget.setMaximumHeight(80)
        self.goal_list_widget.setFont(QtGui.QFont("Courier", 8))
        self.goal_list_widget.setStyleSheet("background: #fafafa;")
        goals_layout.addWidget(self.goal_list_widget)

        # Current velocity scale display
        velocity_row = QtWidgets.QHBoxLayout()
        velocity_row.addWidget(QtWidgets.QLabel("Velocity Scale:"))
        self.current_velocity_label = QtWidgets.QLabel("5.0x")
        self.current_velocity_label.setFont(QtGui.QFont("Courier", 10, QtGui.QFont.Bold))
        self.current_velocity_label.setStyleSheet("color: #ff5722; font-weight: bold;")
        velocity_row.addWidget(self.current_velocity_label)
        velocity_row.addStretch()
        goals_layout.addLayout(velocity_row)

        layout.addWidget(goals_group)

        # Graphs
        if PYQTGRAPH_AVAILABLE:
            # Force graph
            force_graph_group = QtWidgets.QGroupBox("Gripper Force History")
            force_graph_layout = QtWidgets.QVBoxLayout(force_graph_group)

            self.force_plot = PlotWidget()
            self.force_plot.setBackground('w')
            self.force_plot.setLabel('left', 'Force (N)')
            self.force_plot.setLabel('bottom', 'Time (s)')
            self.force_plot.addLegend()
            self.force_plot.showGrid(x=True, y=True, alpha=0.3)

            colors = ['#f44336', '#2196f3', '#4caf50']
            self.force_curves = []
            for i, color in enumerate(colors):
                curve = self.force_plot.plot(
                    pen=mkPen(color=color, width=2),
                    name=finger_names[i]
                )
                self.force_curves.append(curve)

            force_graph_layout.addWidget(self.force_plot)
            layout.addWidget(force_graph_group)

            # Joint graph
            joint_graph_group = QtWidgets.QGroupBox("Joint Position History")
            joint_graph_layout = QtWidgets.QVBoxLayout(joint_graph_group)

            self.joint_plot = PlotWidget()
            self.joint_plot.setBackground('w')
            self.joint_plot.setLabel('left', 'Position (rad)')
            self.joint_plot.setLabel('bottom', 'Time (s)')
            self.joint_plot.addLegend()
            self.joint_plot.showGrid(x=True, y=True, alpha=0.3)

            joint_colors = ['#e91e63', '#9c27b0', '#3f51b5', '#00bcd4', '#009688', '#ff9800']
            self.joint_curves = []
            for i, color in enumerate(joint_colors):
                curve = self.joint_plot.plot(
                    pen=mkPen(color=color, width=1.5),
                    name=f"J{i+1}"
                )
                self.joint_curves.append(curve)

            joint_graph_layout.addWidget(self.joint_plot)
            layout.addWidget(joint_graph_group)
        else:
            no_graph_label = QtWidgets.QLabel("⚠ Install pyqtgraph for graphs:\npip install pyqtgraph")
            no_graph_label.setStyleSheet("color: #ff6f00; font-size: 12pt; padding: 20px;")
            no_graph_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(no_graph_label)

        return panel

    def _update_displays(self):
        """Update all displays with latest ROS data."""
        # Update robot state
        if self.ros.robot_running:
            self.robot_state_label.setText("Program: RUNNING ✓")
            self.robot_state_label.setStyleSheet("font-size: 12pt; padding: 5px; background: #c8e6c9; color: #2e7d32;")
        else:
            self.robot_state_label.setText("Program: STOPPED ⏸")
            self.robot_state_label.setStyleSheet("font-size: 12pt; padding: 5px; background: #ffccbc; color: #bf360c;")

        # Update gripper forces
        forces = self.ros.gripper_force_data
        for i, (force, label) in enumerate(zip(forces, self.force_labels)):
            label.setText(f"{force:.3f} N")

            # Update color based on magnitude
            if abs(force) > 0.20:
                color = "#ff1744"  # Red for high force
            elif abs(force) > 0.10:
                color = "#ffc107"  # Yellow for medium
            else:
                color = "#00e676"  # Green for low

            label.setStyleSheet(f"background: #263238; color: {color}; padding: 8px; border-radius: 4px; font-weight: bold;")

            # Update progress bar
            bar = getattr(self, f"force_bar_{i}")
            bar.setValue(int(abs(force) * 100))

        # Update joint positions
        if self.ros.joint_state_data and len(self.ros.joint_state_data.position) >= 6:
            positions = self.ros.joint_state_data.position[:6]
            for i, (pos, label) in enumerate(zip(positions, self.joint_labels)):
                label.setText(f"{pos:.3f}")

        # Update goals information
        goal_info = self.ros.goal_info_data
        if goal_info:
            # Goal count
            goal_count = goal_info.get("goal_count", 0)
            self.goal_count_label.setText(f"Goals: {goal_count}")
            if goal_count > 0:
                self.goal_count_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            else:
                self.goal_count_label.setStyleSheet("color: #1976d2;")

            # Capture status
            if goal_info.get("capture_active", False):
                cap_count = goal_info.get("capture_count", 0)
                self.capture_status_label.setText(f"Capture: ACTIVE ({cap_count})")
                self.capture_status_label.setStyleSheet("color: #ff5722; font-weight: bold;")
            else:
                self.capture_status_label.setText("Capture: Inactive")
                self.capture_status_label.setStyleSheet("color: #757575;")

            # Latest goal
            latest = goal_info.get("latest_goal")
            if latest:
                self.latest_goal_label.setText(f"X:{latest[0]:.3f} Y:{latest[1]:.3f} Z:{latest[2]:.3f}")
            else:
                self.latest_goal_label.setText("--")

            # Best goal
            best = goal_info.get("best_goal_xyz")
            if best:
                self.best_goal_label.setText(f"X:{best[0]:.3f} Y:{best[1]:.3f} Z:{best[2]:.3f}")
            else:
                self.best_goal_label.setText("--")

            # Goal queue
            goals = goal_info.get("goals", [])
            self.goal_list_widget.clear()
            for i, g in enumerate(goals):
                self.goal_list_widget.addItem(f"#{i+1}: X:{g[0]:.3f} Y:{g[1]:.3f} Z:{g[2]:.3f}")

            # Velocity scale
            vel_scale = goal_info.get("velocity_scale", 5.0)
            self.current_velocity_label.setText(f"{vel_scale:.1f}x")

        # Update graphs
        if PYQTGRAPH_AVAILABLE:
            self.elapsed_time += 0.05
            self.time_data.append(self.elapsed_time)

            # Force data
            for i, force in enumerate(forces):
                self.force_history[i].append(force)
                if len(self.time_data) > 0 and len(self.force_history[i]) > 0:
                    self.force_curves[i].setData(
                        list(self.time_data)[-len(self.force_history[i]):],
                        list(self.force_history[i])
                    )

            # Joint data
            if self.ros.joint_state_data and len(self.ros.joint_state_data.position) >= 6:
                positions = self.ros.joint_state_data.position[:6]
                for i, pos in enumerate(positions):
                    self.joint_history[i].append(pos)
                    if len(self.time_data) > 0 and len(self.joint_history[i]) > 0:
                        self.joint_curves[i].setData(
                            list(self.time_data)[-len(self.joint_history[i]):],
                            list(self.joint_history[i])
                        )

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
            self.status_update.emit("Invalid goal values.", True)
            return

        self.ros.publish_goal(x, y, z, qw, qx, qy, qz)
        self.status_update.emit(f"Published goal ({x:.3f}, {y:.3f}, {z:.3f}).", False)

    def _handle_stop(self):
        self.ros.publish_stop()
        self._send_cmd("stop")
        self.status_update.emit("⚠ EMERGENCY STOP ACTIVATED", True)

    def _send_cmd(self, cmd: str):
        self.ros.publish_cmd(cmd)
        self.status_update.emit(f"Sent command: {cmd}", False)

    def _apply_speed_settings(self):
        """Send speed configuration as command."""
        speeds = {
            "home": self.speed_home.value(),
            "dropoff": self.speed_dropoff.value(),
            "approach": self.speed_approach.value()
        }
        cmd = f"set_speeds {speeds['home']:.2f} {speeds['dropoff']:.2f} {speeds['approach']:.2f}"
        self._send_cmd(cmd)
        self.status_update.emit(f"Applied speeds: Home={speeds['home']:.1f}, Drop={speeds['dropoff']:.1f}, Approach={speeds['approach']:.2f}", False)

    def _on_velocity_slider_changed(self, value: int):
        """Update velocity label when slider changes."""
        scale = value / 10.0
        self.velocity_value_label.setText(f"{scale:.1f}x")

    def _set_velocity_scale(self, scale: float):
        """Set velocity scale from preset button."""
        self.velocity_slider.setValue(int(scale * 10))
        self.velocity_value_label.setText(f"{scale:.1f}x")

    def _apply_velocity_scale(self):
        """Apply the velocity scale to the robot node."""
        scale = self.velocity_slider.value() / 10.0
        self.ros.publish_velocity_scale(scale)
        self.status_update.emit(f"Velocity scale set to {scale:.1f}x", False)

    def _set_status_slot(self, text: str, error: bool):
        """Slot for thread-safe status updates."""
        if not self._status:
            return
        if error:
            color = "#d32f2f"
            bg = "#ffcdd2"
        else:
            color = "#2d6a4f"
            bg = "#e8f5e9"
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-weight: bold; padding: 8px; background: {bg}; border-radius: 4px;")


def main():
    rclpy.init()
    ros_node = UiBridge()

    def spin_ros():
        rclpy.spin(ros_node)

    spin_thread = threading.Thread(target=spin_ros, daemon=True)
    spin_thread.start()

    app = QtWidgets.QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Dark palette (optional - comment out for light theme)
    # palette = QtGui.QPalette()
    # palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    # palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    # palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    # palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    # palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    # palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    # palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    # app.setPalette(palette)

    win = MainWindow(ros_node)
    win.show()
    ret = app.exec_()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
