import sys
import threading
import json
from typing import Optional, List
from collections import deque
import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import QShortcut, QScrollArea
from PyQt5.QtGui import QKeySequence
from std_msgs.msg import Bool, String, Float32MultiArray, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Twist
class ControlMode(Enum):
    NORMAL = 0
    KEYBOARD = 1

try:
    from pyqtgraph import PlotWidget, mkPen
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("[WARN] pyqtgraph not installed. Install with: pip install pyqtgraph")
    print("[WARN] Graphs will be disabled.")


# ==================== SCORE BAR WIDGET ====================

class ScoreBarWidget(QtWidgets.QWidget):
    """Horizontal bar chart for fruit detection score components."""

    LABELS = [
        ("distance",     "Distance",    "#ef5350"),
        ("visibility",   "Visibility",  "#42a5f5"),
        ("depth_quality","Depth",       "#66bb6a"),
        ("confidence",   "Confidence",  "#ffa726"),
        ("ellipse",      "Shape",       "#ab47bc"),
        ("center_bias",  "Center",      "#26c6da"),
    ]

    def __init__(self):
        super().__init__()
        self.scores: dict = {}
        self.setMinimumHeight(115)
        self.setMaximumHeight(140)

    def update_scores(self, components: dict):
        self.scores = components
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        if not self.scores:
            p.setPen(QtGui.QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter, "No detection")
            return

        n = len(self.LABELS)
        margin = 3
        label_w = 62
        bar_area = w - label_w - margin * 3
        row_h = max(10, (h - margin * (n + 1)) // n)

        fm = QtGui.QFontMetrics(p.font())

        for i, (key, name, color) in enumerate(self.LABELS):
            y = margin + i * (row_h + margin)
            val = float(self.scores.get(key, 0.0))
            val = max(0.0, min(1.0, val))
            bar_w = int(bar_area * val)

            # Label
            p.setPen(QtGui.QColor("#333"))
            p.drawText(margin, y, label_w, row_h, Qt.AlignVCenter | Qt.AlignLeft, name)

            # Background track
            bg = QtCore.QRect(label_w + margin, y, bar_area, row_h)
            p.fillRect(bg, QtGui.QColor("#e0e0e0"))

            # Filled bar
            if bar_w > 0:
                p.fillRect(QtCore.QRect(label_w + margin, y, bar_w, row_h), QtGui.QColor(color))

            # Value text
            val_str = f"{val:.2f}"
            text_x = label_w + margin + max(bar_w - fm.horizontalAdvance(val_str) - 2, 2)
            p.setPen(QtGui.QColor("#fff") if bar_w > 30 else QtGui.QColor("#555"))
            p.drawText(text_x, y, bar_area - (text_x - label_w - margin), row_h,
                       Qt.AlignVCenter | Qt.AlignLeft, val_str)


# ==================== HARVEST RESULT WIDGET ====================

class HarvestResultWidget(QtWidgets.QWidget):
    """
    Large banner showing SUCCESS / PARTIAL / SLIPPED / FAIL for the last
    harvest cycle, plus a session tally bar (grabbed / slipped / miss).
    """

    RESULT_MAP = {
        # (outcome, end) -> (label, bg, fg)
        ("GRABBED", "PROPER"):  ("SUCCESS",  "#2e7d32", "#e8f5e9"),
        ("GRABBED", "WEAK"):    ("PARTIAL",  "#e65100", "#fff3e0"),
        ("SLIPPED", "WEAK"):    ("SLIPPED",  "#f57f17", "#fffde7"),
        ("SLIPPED", ""):        ("SLIPPED",  "#f57f17", "#fffde7"),
        ("NO_GRAB", ""):        ("FAIL",     "#b71c1c", "#ffebee"),
    }
    DEFAULT = ("—",        "#455a64", "#eceff1")

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Large result banner
        self.banner = QtWidgets.QLabel("—")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setFont(QtGui.QFont("Helvetica", 20, QtGui.QFont.Bold))
        self.banner.setFixedHeight(52)
        self.banner.setStyleSheet(
            "background: #455a64; color: #eceff1; border-radius: 8px; letter-spacing: 3px;")
        layout.addWidget(self.banner)

        # Tally row: grabbed | slipped | miss | total
        tally_row = QtWidgets.QHBoxLayout()
        tally_row.setSpacing(6)

        self._tally_labels = {}
        for key, label, color in [
            ("grabbed", "Grabbed", "#4caf50"),
            ("slipped", "Slipped", "#ff9800"),
            ("miss",    "Miss",    "#f44336"),
            ("total",   "Total",   "#90a4ae"),
        ]:
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setSpacing(1)
            cell_layout.setContentsMargins(2, 2, 2, 2)

            count = QtWidgets.QLabel("0")
            count.setAlignment(Qt.AlignCenter)
            count.setFont(QtGui.QFont("Courier", 14, QtGui.QFont.Bold))
            count.setStyleSheet(f"color: {color};")
            cell_layout.addWidget(count)

            name = QtWidgets.QLabel(label)
            name.setAlignment(Qt.AlignCenter)
            name.setStyleSheet("font-size: 8pt; color: #666;")
            cell_layout.addWidget(name)

            tally_row.addWidget(cell)
            self._tally_labels[key] = count

        layout.addLayout(tally_row)

    def update_result(self, history: list):
        """Update banner from last entry and recount session tallies."""
        if not history:
            self.banner.setText("—")
            self.banner.setStyleSheet(
                "background: #455a64; color: #eceff1; border-radius: 8px; letter-spacing: 3px;")
            for k in self._tally_labels:
                self._tally_labels[k].setText("0")
            return

        # Last grasp
        last = history[-1]
        outcome = last.get("outcome", "")
        end = last.get("end", "")
        key = (outcome, end)
        if key not in self.RESULT_MAP:
            key = (outcome, "")  # try without end
        label, bg, fg = self.RESULT_MAP.get(key, self.DEFAULT)
        self.banner.setText(label)
        self.banner.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 8px; letter-spacing: 3px;")

        # Tally
        grabbed = slipped = miss = 0
        for item in history:
            o = item.get("outcome", "")
            if o == "GRABBED":
                grabbed += 1
            elif o == "SLIPPED":
                slipped += 1
            elif o == "NO_GRAB":
                miss += 1
        total = grabbed + slipped + miss
        self._tally_labels["grabbed"].setText(str(grabbed))
        self._tally_labels["slipped"].setText(str(slipped))
        self._tally_labels["miss"].setText(str(miss))
        self._tally_labels["total"].setText(str(total))


# ==================== GRASP OUTCOME DOTS ====================

class GraspOutcomeWidget(QtWidgets.QWidget):
    """Row of coloured circles showing last N grasp outcomes."""

    COLORS = {
        ("GRABBED", "PROPER"):  "#4caf50",  # green
        ("GRABBED", "WEAK"):    "#ffeb3b",  # yellow
        ("SLIPPED", ""):        "#ff9800",  # orange
        ("NO_GRAB", ""):        "#f44336",  # red
    }
    DEFAULT_COLOR = "#9e9e9e"

    def __init__(self):
        super().__init__()
        self.outcomes: list = []
        self.setFixedHeight(28)

    def update_outcomes(self, outcomes: list):
        self.outcomes = outcomes[-15:]
        self.update()

    def _color(self, item):
        outcome = item.get("outcome", "")
        end = item.get("end", "")
        for (o, e), c in self.COLORS.items():
            if outcome == o and (e == "" or end == e):
                return c
        return self.DEFAULT_COLOR

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if not self.outcomes:
            p.setPen(QtGui.QColor("#aaa"))
            p.drawText(self.rect(), Qt.AlignCenter, "No grasps yet")
            return
        r = 10
        gap = 4
        x = gap
        for item in self.outcomes:
            color = QtGui.QColor(self._color(item))
            p.setBrush(color)
            p.setPen(QtGui.QColor("#555"))
            p.drawEllipse(x, (self.height() - r * 2) // 2, r * 2, r * 2)
            x += r * 2 + gap


class UiBridge(Node):
    """ROS bridge that the Qt UI talks to."""

    def __init__(
        self,
        command_topic: str = "/ui_command",
        goal_topic: str = "/external_goal_pose",
        stop_topic: str = "/emergency_stop",
    ):
        super().__init__("ur10e_desktop_ui")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
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

        self.teleop_pub = self.create_publisher(
            Twist,
            "/teleop_delta",
            10
        )
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

        # Calibration check result
        self.calib_check_result: Optional[str] = None
        self.create_subscription(
            String,
            "/calib_check_result",
            self._calib_check_cb,
            10
        )

        # Fruit score components from vision node
        self.fruit_score_data: dict = {}
        self.create_subscription(
            String,
            "/vision/fruit_score",
            self._fruit_score_cb,
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

    def get_tcp_pose(self) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                "base_link",
                "tool0",
                rclpy.time.Time()
            )
        except Exception:
            return None
        
    def publish_teleop_delta(self, dx, dy, dz, droll, dpitch, dyaw):
        """Publish a geometry_msgs/Twist with linear and angular deltas.

        Many teleop consumers subscribe to a plain `Twist` message. Keep the
        values relative (deltas) and publish on `/teleop_delta`.
        """
        msg = Twist()
        msg.linear.x = dx
        msg.linear.y = dy
        msg.linear.z = dz
        msg.angular.x = droll
        msg.angular.y = dpitch
        msg.angular.z = dyaw

        self.teleop_pub.publish(msg)

    def _velocity_scale_cb(self, msg: Float32):
        self.velocity_scale = msg.data

    def _calib_check_cb(self, msg: String):
        self.calib_check_result = msg.data

    def _fruit_score_cb(self, msg: String):
        try:
            self.fruit_score_data = json.loads(msg.data)
        except Exception:
            pass

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


# ==================== KEYBOARD CONTROL WINDOW ====================

class KeyboardControlWindow(QtWidgets.QWidget):
    """Separate window for keyboard/mouse control of the robot."""

    status_update = pyqtSignal(str, bool)

    def __init__(self, ros: UiBridge, parent_geometry=None):
        super().__init__(None)
        self.ros = ros
        self.setWindowTitle("Keyboard Control - UR10e")
        self.setMinimumSize(450, 600)

        # Make it a proper independent window
        self.setWindowFlags(Qt.Window)

        # Position window to the right of parent if geometry provided
        if parent_geometry:
            self.move(parent_geometry.right() + 20, parent_geometry.top())

        # Keyboard control state - START ENABLED
        self.keyboard_enabled = True
        self.step_size = 0.005  # 5mm default step (smaller for smoothness)

        # Keys currently held down (tracked via keyPress/keyRelease)
        self.keys_held = set()

        # Mouse tracking state
        self.mouse_tracking_active = False
        self.last_mouse_pos = None

        # Velocity ramping for smooth acceleration
        self.current_velocity = [0.0, 0.0, 0.0]  # dx, dy, dz
        self.max_velocity = 0.02  # m per tick at full speed
        self.accel_rate = 0.3  # how fast to ramp up (0-1, higher = faster)
        self.decel_rate = 0.5  # how fast to slow down when key released

        self._build_ui()
        self._setup_shortcuts()

        # Movement timer - run faster for smoother motion
        self.kbd_timer = QTimer()
        self.kbd_timer.timeout.connect(self._process_keyboard_movement)
        self.kbd_timer.start(25)  # 40Hz for smoother updates

        # Connect status signal
        self.status_update.connect(self._set_status)

    def _setup_shortcuts(self):
        """Setup non-movement keyboard shortcuts."""
        # Movement keys are now handled via keyPressEvent/keyReleaseEvent for smooth hold detection

        # Action shortcuts - gripper
        g_shortcut = QShortcut(QKeySequence('G'), self)
        g_shortcut.setContext(Qt.WindowShortcut)
        g_shortcut.activated.connect(lambda: self._gripper_cmd("open"))

        h_shortcut = QShortcut(QKeySequence('H'), self)
        h_shortcut.setContext(Qt.WindowShortcut)
        h_shortcut.activated.connect(lambda: self._gripper_cmd("close"))

        # Step size shortcuts
        plus_shortcut = QShortcut(QKeySequence('+'), self)
        plus_shortcut.setContext(Qt.WindowShortcut)
        plus_shortcut.activated.connect(self._increase_step)

        equal_shortcut = QShortcut(QKeySequence('='), self)
        equal_shortcut.setContext(Qt.WindowShortcut)
        equal_shortcut.activated.connect(self._increase_step)

        minus_shortcut = QShortcut(QKeySequence('-'), self)
        minus_shortcut.setContext(Qt.WindowShortcut)
        minus_shortcut.activated.connect(self._decrease_step)

        # Escape to disable
        esc_shortcut = QShortcut(QKeySequence("Esc"), self)
        esc_shortcut.setContext(Qt.WindowShortcut)
        esc_shortcut.activated.connect(self._disable_keyboard)

    def keyPressEvent(self, event):
        """Track movement keys being held down."""
        if not event.isAutoRepeat() and self.keyboard_enabled:
            key = event.key()
            if key in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
                self.keys_held.add(key)
                self._update_control_area_display()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Track movement keys being released."""
        if not event.isAutoRepeat():
            key = event.key()
            if key in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
                self.keys_held.discard(key)
                self._update_control_area_display()
                event.accept()
                return
        super().keyReleaseEvent(event)

    def _gripper_cmd(self, cmd):
        if self.keyboard_enabled:
            self.ros.publish_cmd(cmd)
            self.status_update.emit(f"Gripper: {cmd.upper()}", False)

    def _increase_step(self):
        if self.keyboard_enabled:
            self.step_spin.setValue(min(self.step_size * 2, 0.1))

    def _decrease_step(self):
        if self.keyboard_enabled:
            self.step_spin.setValue(max(self.step_size / 2, 0.001))

    def _disable_keyboard(self):
        self.enable_btn.setChecked(False)
        self._toggle_enabled(False)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QtWidgets.QLabel("Keyboard Control Mode")
        title.setFont(QtGui.QFont("Helvetica", 14, QtGui.QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Status - starts enabled
        self.status_label = QtWidgets.QLabel("ACTIVE - Press WASD/QE to move robot")
        self.status_label.setStyleSheet("padding: 10px; background: #c8e6c9; border-radius: 4px; font-weight: bold; font-size: 11pt;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # Note: keyboard window publishes teleop deltas only; no goal sending here

        # Enable button - starts checked
        self.enable_btn = QtWidgets.QPushButton("Disable Keyboard Control")
        self.enable_btn.setCheckable(True)
        self.enable_btn.setChecked(True)
        self.enable_btn.setStyleSheet("""
            QPushButton {
                background-color: #607d8b;
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 12pt;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #4caf50;
            }
        """)
        self.enable_btn.clicked.connect(self._toggle_enabled)
        layout.addWidget(self.enable_btn)

        # Visual control area (for mouse)
        control_group = QtWidgets.QGroupBox("Mouse Control Area (drag here)")
        control_layout = QtWidgets.QVBoxLayout(control_group)

        self.control_area = QtWidgets.QLabel()
        self.control_area.setMinimumSize(300, 200)
        self.control_area.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #e3f2fd, stop:1 #bbdefb);
            border: 2px dashed #1976d2;
            border-radius: 10px;
        """)
        self.control_area.setAlignment(Qt.AlignCenter)
        self._update_control_area_display()
        control_layout.addWidget(self.control_area)
        layout.addWidget(control_group)

        # Removed absolute position and orientation displays; keyboard window
        # only emits teleop deltas (no absolute pose state)

        # Step size control
        step_group = QtWidgets.QGroupBox("Step Size")
        step_layout = QtWidgets.QHBoxLayout(step_group)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.001, 0.1)
        self.step_spin.setValue(0.01)
        self.step_spin.setSingleStep(0.005)
        self.step_spin.setDecimals(3)
        self.step_spin.setSuffix(" m")
        self.step_spin.valueChanged.connect(lambda v: setattr(self, 'step_size', v))
        step_layout.addWidget(self.step_spin)

        for label, val in [("1mm", 0.001), ("5mm", 0.005), ("1cm", 0.01), ("5cm", 0.05)]:
            btn = QtWidgets.QPushButton(label)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda _, v=val: self.step_spin.setValue(v))
            step_layout.addWidget(btn)

        layout.addWidget(step_group)

        # Controls help
        help_group = QtWidgets.QGroupBox("Keyboard Controls")
        help_layout = QtWidgets.QVBoxLayout(help_group)
        help_text = QtWidgets.QLabel(
            "<table>"
            "<tr><td><b>W/S</b></td><td>Y axis (forward/back) — publishes delta</td></tr>"
            "<tr><td><b>A/D</b></td><td>X axis (left/right) — publishes delta</td></tr>"
            "<tr><td><b>Q/E</b></td><td>Z axis (up/down) — publishes delta</td></tr>"
            "<tr><td><b>Mouse drag</b></td><td>Publish X/Y deltas (no absolute pose)</td></tr>"
            "<tr><td><b>Scroll wheel</b></td><td>Publish Z delta only</td></tr>"
            "<tr><td><b>G/H</b></td><td>Open/Close gripper</td></tr>"
            "<tr><td><b>+/-</b></td><td>Adjust step size</td></tr>"
            "<tr><td><b>Esc</b></td><td>Disable keyboard mode</td></tr>"
            "</table>"
        )
        help_text.setStyleSheet("font-size: 9pt;")
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)

    # Action buttons removed: keyboard window only sends teleop deltas

        # Gripper buttons
        gripper_layout = QtWidgets.QHBoxLayout()

        open_btn = QtWidgets.QPushButton("Open Gripper (G)")
        open_btn.clicked.connect(lambda: self.ros.publish_cmd("open"))
        gripper_layout.addWidget(open_btn)

        close_btn = QtWidgets.QPushButton("Close Gripper (H)")
        close_btn.clicked.connect(lambda: self.ros.publish_cmd("close"))
        gripper_layout.addWidget(close_btn)

        layout.addLayout(gripper_layout)

        # Emergency stop
        stop_btn = QtWidgets.QPushButton("EMERGENCY STOP")
        stop_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 12pt; padding: 12px;")
        stop_btn.clicked.connect(self._emergency_stop)
        layout.addWidget(stop_btn)

    def _update_control_area_display(self):
        # Show which keys are active
        active = []
        if Qt.Key_W in self.keys_held: active.append("W")
        if Qt.Key_S in self.keys_held: active.append("S")
        if Qt.Key_A in self.keys_held: active.append("A")
        if Qt.Key_D in self.keys_held: active.append("D")
        if Qt.Key_Q in self.keys_held: active.append("Q")
        if Qt.Key_E in self.keys_held: active.append("E")
        keys_text = f"<b style='color:#4caf50'>Keys: {' '.join(active)}</b>" if active else "<i>Hold WASD/QE keys</i>"

        # Show current velocity
        vel_mag = sum(v*v for v in self.current_velocity) ** 0.5
        vel_text = f"<span style='color:#1976d2'>Vel: {vel_mag*1000:.1f} mm/tick</span>"

        # Display that the window publishes deltas rather than absolute pose
        self.control_area.setText(
            f"<center><h2>Teleop (smooth)</h2>"
            f"<p style='font-size:12pt'>Hold WASD/QE for smooth motion.<br>"
            f"Drag mouse for X/Y. Scroll for Z.</p>"
            f"<p>{keys_text}</p>"
            f"<p>{vel_text}</p></center>"
        )

    def _update_displays(self):
        # Keyboard window no longer keeps absolute pose/orientation; just
        # update the control area to reflect active keys.
        self._update_control_area_display()

    def _toggle_enabled(self, checked: bool):
        self.keyboard_enabled = checked
        if checked:
            self.enable_btn.setText("Disable Keyboard Control")
            self.status_label.setText("ACTIVE - Hold WASD/QE to move robot")
            self.status_label.setStyleSheet("padding: 10px; background: #c8e6c9; border-radius: 4px; font-weight: bold; font-size: 11pt;")
        else:
            self.enable_btn.setText("Enable Keyboard Control")
            self.status_label.setText("DISABLED - Click Enable to start")
            self.status_label.setStyleSheet("padding: 10px; background: #ffcdd2; border-radius: 4px; font-size: 11pt;")
            self.keys_held.clear()
            self.current_velocity = [0.0, 0.0, 0.0]

    def closeEvent(self, event):
        """Clean up when closing."""
        self.kbd_timer.stop()
        event.accept()

    def showEvent(self, event):
        """Focus window when showing."""
        super().showEvent(event)
        if self.keyboard_enabled:
            self.activateWindow()
            self.setFocus()


    def _set_status(self, text: str, error: bool):
        if error:
            self.status_label.setStyleSheet("padding: 8px; background: #ffcdd2; border-radius: 4px;")
        else:
            self.status_label.setStyleSheet("padding: 8px; background: #c8e6c9; border-radius: 4px;")
        self.status_label.setText(text)

    def _emergency_stop(self):
        self.ros.publish_stop()
        self.ros.publish_cmd("stop")
        self.status_update.emit("EMERGENCY STOP ACTIVATED", True)

    def _process_keyboard_movement(self):
        """Process held keys with smooth velocity ramping."""
        if not self.keyboard_enabled:
            return

        # Target velocity based on keys held
        target = [0.0, 0.0, 0.0]  # dx, dy, dz
        speed = self.step_size

        if Qt.Key_W in self.keys_held: target[1] += speed
        if Qt.Key_S in self.keys_held: target[1] -= speed
        if Qt.Key_A in self.keys_held: target[0] -= speed
        if Qt.Key_D in self.keys_held: target[0] += speed
        if Qt.Key_Q in self.keys_held: target[2] += speed
        if Qt.Key_E in self.keys_held: target[2] -= speed

        # Smooth velocity ramping
        for i in range(3):
            if abs(target[i]) > 0.0001:
                # Accelerate towards target
                diff = target[i] - self.current_velocity[i]
                self.current_velocity[i] += diff * self.accel_rate
            else:
                # Decelerate when no key held
                self.current_velocity[i] *= (1.0 - self.decel_rate)
                if abs(self.current_velocity[i]) < 0.0001:
                    self.current_velocity[i] = 0.0

        # Clamp to max velocity
        for i in range(3):
            self.current_velocity[i] = max(-self.max_velocity, min(self.max_velocity, self.current_velocity[i]))

        # Publish if there's any movement
        dx, dy, dz = self.current_velocity
        if abs(dx) > 0.0001 or abs(dy) > 0.0001 or abs(dz) > 0.0001:
            self.ros.publish_teleop_delta(dx, dy, dz, 0.0, 0.0, 0.0)
            self._update_control_area_display()
    

    def mousePressEvent(self, event):
        # Always grab focus when clicking anywhere in the window
        self.setFocus()

        if self.keyboard_enabled and event.button() == Qt.LeftButton:
            self.mouse_tracking_active = True
            self.last_mouse_pos = event.pos()
            event.accept(); return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_tracking_active = False
            self.last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.keyboard_enabled and self.mouse_tracking_active and self.last_mouse_pos:
            # Publish deltas (do not update any absolute pose)
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            scale = self.step_size * 0.001  # scale pixels->meters
            dx = delta.x() * scale
            dy = -delta.y() * scale
            # publish linear deltas only
            self.ros.publish_teleop_delta(dx, dy, 0.0, 0.0, 0.0, 0.0)
            event.accept(); return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        if self.keyboard_enabled:
            delta = event.angleDelta().y()
            if delta != 0:
                # Publish only Z delta (linear.z), scale by step_size
                dz = self.step_size * (0.5 if delta > 0 else -0.5)
                self.ros.publish_teleop_delta(0.0, 0.0, dz, 0.0, 0.0, 0.0)
                event.accept(); return
        super().wheelEvent(event)


# ==================== MAIN WINDOW ====================

class MainWindow(QtWidgets.QWidget):
    status_update = pyqtSignal(str, bool)

    def __init__(self, ros: UiBridge):
        super().__init__()
        self.ros = ros
        self.setWindowTitle("UR10e Control Panel")
        self.setMinimumSize(1000, 700)

        # Keyboard window reference
        self.kbd_window = None

        # Data storage for graphs
        self.force_history = [deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)]
        self.joint_history  = [deque(maxlen=200) for _ in range(6)]
        self.joint_vel_history = [deque(maxlen=200) for _ in range(6)]
        self.time_data = deque(maxlen=200)
        self.elapsed_time = 0.0

        self._status: Optional[QtWidgets.QLabel] = None
        self._build_ui()

        # Emergency stop shortcut
        stop_shortcut = QShortcut(QKeySequence("Ctrl+Space"), self)
        stop_shortcut.activated.connect(self._handle_stop)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(50)

        self.status_update.connect(self._set_status_slot)

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setSpacing(8)

        # Left panel: Controls (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(380)

        left_panel = self._build_control_panel()
        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        # Right panel: Monitoring
        right_panel = self._build_monitor_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def _build_control_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title & Status
        title = QtWidgets.QLabel("UR10e Control")
        title.setFont(QtGui.QFont("Helvetica", 14, QtGui.QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._status = QtWidgets.QLabel("Ready")
        self._status.setStyleSheet("color: #2d6a4f; font-weight: bold; padding: 6px; background: #e8f5e9; border-radius: 4px;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Robot state
        self.robot_state_label = QtWidgets.QLabel("Program: Unknown")
        self.robot_state_label.setStyleSheet("font-size: 11pt; padding: 4px;")
        layout.addWidget(self.robot_state_label)

        # Velocity Scale
        vel_group = QtWidgets.QGroupBox("Velocity Scale")
        vel_layout = QtWidgets.QVBoxLayout(vel_group)
        vel_layout.setSpacing(4)

        slider_row = QtWidgets.QHBoxLayout()
        self.velocity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.velocity_slider.setRange(10, 100)
        self.velocity_slider.setValue(50)
        self.velocity_slider.valueChanged.connect(self._on_velocity_slider_changed)
        slider_row.addWidget(self.velocity_slider)

        self.velocity_value_label = QtWidgets.QLabel("5.0x")
        self.velocity_value_label.setFont(QtGui.QFont("Courier", 10, QtGui.QFont.Bold))
        self.velocity_value_label.setStyleSheet("color: #1976d2;")
        slider_row.addWidget(self.velocity_value_label)
        vel_layout.addLayout(slider_row)

        preset_row = QtWidgets.QHBoxLayout()
        for label, value in [("Slow", 1.0), ("Med", 3.0), ("Fast", 5.0), ("Max", 8.0)]:
            btn = QtWidgets.QPushButton(label)
            btn.setMaximumHeight(24)
            btn.clicked.connect(lambda _, v=value: self._set_velocity_scale(v))
            preset_row.addWidget(btn)
        vel_layout.addLayout(preset_row)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self._apply_velocity_scale)
        vel_layout.addWidget(apply_btn)
        layout.addWidget(vel_group)

        # Manual Goal (compact)
        goal_group = QtWidgets.QGroupBox("Manual Goal")
        goal_layout = QtWidgets.QGridLayout(goal_group)
        goal_layout.setSpacing(4)

        self.x_in = QtWidgets.QLineEdit("0.30")
        self.y_in = QtWidgets.QLineEdit("0.00")
        self.z_in = QtWidgets.QLineEdit("0.20")
        self.qw_in = QtWidgets.QLineEdit("1.0")
        self.qx_in = QtWidgets.QLineEdit("0.0")
        self.qy_in = QtWidgets.QLineEdit("0.0")
        self.qz_in = QtWidgets.QLineEdit("0.0")

        for i, (label, widget) in enumerate([
            ("X", self.x_in), ("Y", self.y_in), ("Z", self.z_in)
        ]):
            goal_layout.addWidget(QtWidgets.QLabel(label), i, 0)
            widget.setMaximumWidth(80)
            goal_layout.addWidget(widget, i, 1)

        for i, (label, widget) in enumerate([
            ("qw", self.qw_in), ("qx", self.qx_in), ("qy", self.qy_in), ("qz", self.qz_in)
        ]):
            goal_layout.addWidget(QtWidgets.QLabel(label), i, 2)
            widget.setMaximumWidth(60)
            goal_layout.addWidget(widget, i, 3)

        send_btn = QtWidgets.QPushButton("Send Goal")
        send_btn.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        send_btn.clicked.connect(self._handle_send_goal)
        goal_layout.addWidget(send_btn, 4, 0, 1, 4)
        layout.addWidget(goal_group)

        # Motion Commands
        motion_group = QtWidgets.QGroupBox("Motion")
        motion_layout = QtWidgets.QGridLayout(motion_group)
        motion_layout.setSpacing(4)

        home_btn = QtWidgets.QPushButton("Home")
        home_btn.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;")
        home_btn.clicked.connect(lambda: self._send_cmd("home"))
        motion_layout.addWidget(home_btn, 0, 0)

        drop_btn = QtWidgets.QPushButton("Dropoff")
        drop_btn.setStyleSheet("background-color: #2196f3; color: white; font-weight: bold;")
        drop_btn.clicked.connect(lambda: self._send_cmd("dropoff"))
        motion_layout.addWidget(drop_btn, 0, 1)

        exec_btn = QtWidgets.QPushButton("Execute")
        exec_btn.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
        exec_btn.clicked.connect(lambda: self._send_cmd("execute"))
        motion_layout.addWidget(exec_btn, 1, 0)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setStyleSheet("background-color: #9e9e9e; color: white;")
        clear_btn.clicked.connect(lambda: self._send_cmd("clear"))
        motion_layout.addWidget(clear_btn, 1, 1)

        check_calib_btn = QtWidgets.QPushButton("Check Calibration (trunk)")
        check_calib_btn.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        check_calib_btn.setToolTip("Check hand-eye calibration using the detected trunk as the reference")
        check_calib_btn.clicked.connect(lambda: self._send_cmd("check_calibration"))
        motion_layout.addWidget(check_calib_btn, 2, 0, 1, 2)

        self.calib_result_label = QtWidgets.QLabel("—")
        self.calib_result_label.setWordWrap(True)
        self.calib_result_label.setStyleSheet(
            "font-size: 9pt; padding: 3px; background: #f3e5f5; border-radius: 4px;")
        motion_layout.addWidget(self.calib_result_label, 3, 0, 1, 2)

        self.debug_preview_cb = QtWidgets.QCheckBox("Debug Plan Preview")
        self.debug_preview_cb.setChecked(True)
        self.debug_preview_cb.setToolTip("Show full plan in RViz before executing")
        self.debug_preview_cb.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 4px;")
        self.debug_preview_cb.stateChanged.connect(self._on_debug_preview_changed)
        motion_layout.addWidget(self.debug_preview_cb, 4, 0, 1, 2)

        # Plan confirm/cancel buttons (shown when plan preview is waiting)
        self.plan_confirm_btn = QtWidgets.QPushButton("Confirm Plan")
        self.plan_confirm_btn.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold; font-size: 11pt; padding: 8px;")
        self.plan_confirm_btn.clicked.connect(lambda: self._send_cmd("plan_confirm"))
        self.plan_confirm_btn.setVisible(False)
        motion_layout.addWidget(self.plan_confirm_btn, 5, 0)

        self.plan_cancel_btn = QtWidgets.QPushButton("Cancel Plan")
        self.plan_cancel_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 11pt; padding: 8px;")
        self.plan_cancel_btn.clicked.connect(lambda: self._send_cmd("plan_cancel"))
        self.plan_cancel_btn.setVisible(False)
        motion_layout.addWidget(self.plan_cancel_btn, 5, 1)
        layout.addWidget(motion_group)

        # Gripper
        gripper_group = QtWidgets.QGroupBox("Gripper")
        gripper_layout = QtWidgets.QHBoxLayout(gripper_group)

        open_btn = QtWidgets.QPushButton("Open")
        open_btn.clicked.connect(lambda: self._send_cmd("open"))
        gripper_layout.addWidget(open_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(lambda: self._send_cmd("close"))
        gripper_layout.addWidget(close_btn)
        layout.addWidget(gripper_group)

        # Keyboard Mode Button (opens separate window)
        kbd_btn = QtWidgets.QPushButton("Open Keyboard Control Window")
        kbd_btn.setStyleSheet("""
            QPushButton {
                background-color: #673ab7;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 11pt;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7e57c2;
            }
        """)
        kbd_btn.clicked.connect(self._open_keyboard_window)
        layout.addWidget(kbd_btn)

        # Capture
        capture_group = QtWidgets.QGroupBox("Capture")
        capture_layout = QtWidgets.QHBoxLayout(capture_group)

        cap_btn = QtWidgets.QPushButton("Start (10s)")
        cap_btn.clicked.connect(lambda: self._send_cmd("capture 10"))
        capture_layout.addWidget(cap_btn)

        cap_stop = QtWidgets.QPushButton("Stop")
        cap_stop.clicked.connect(lambda: self._send_cmd("capture_stop"))
        capture_layout.addWidget(cap_stop)
        layout.addWidget(capture_group)

        # Emergency Stop
        stop_btn = QtWidgets.QPushButton("EMERGENCY STOP")
        stop_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 12pt; padding: 12px;")
        stop_btn.clicked.connect(self._handle_stop)
        layout.addWidget(stop_btn)

        layout.addStretch(1)
        return panel

    def _build_monitor_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(6)

        # Motion phase + reacquire result (side by side)
        phase_row = QtWidgets.QHBoxLayout()

        phase_group = QtWidgets.QGroupBox("Motion Phase")
        phase_inner = QtWidgets.QVBoxLayout(phase_group)
        self.phase_label = QtWidgets.QLabel("IDLE")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setFont(QtGui.QFont("Helvetica", 14, QtGui.QFont.Bold))
        self.phase_label.setStyleSheet(
            "background: #455a64; color: #eceff1; padding: 8px; border-radius: 6px; letter-spacing: 2px;")
        phase_inner.addWidget(self.phase_label)
        phase_row.addWidget(phase_group, stretch=3)

        reacq_group = QtWidgets.QGroupBox("Reacquire")
        reacq_inner = QtWidgets.QVBoxLayout(reacq_group)
        self.reacq_label = QtWidgets.QLabel("—")
        self.reacq_label.setAlignment(Qt.AlignCenter)
        self.reacq_label.setFont(QtGui.QFont("Helvetica", 14, QtGui.QFont.Bold))
        self.reacq_label.setStyleSheet(
            "background: #455a64; color: #eceff1; padding: 8px; border-radius: 6px;")
        reacq_inner.addWidget(self.reacq_label)
        phase_row.addWidget(reacq_group, stretch=2)

        layout.addLayout(phase_row)

        # Gripper forces (compact)
        gripper_group = QtWidgets.QGroupBox("Gripper Forces")
        gripper_layout = QtWidgets.QHBoxLayout(gripper_group)

        self.force_labels = []
        for name in ["Left", "Center", "Right"]:
            w = QtWidgets.QWidget()
            vl = QtWidgets.QVBoxLayout(w)
            vl.setSpacing(2)

            lbl = QtWidgets.QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(lbl)

            val = QtWidgets.QLabel("0.00 N")
            val.setAlignment(Qt.AlignCenter)
            val.setFont(QtGui.QFont("Courier", 12, QtGui.QFont.Bold))
            val.setStyleSheet("background: #263238; color: #00e676; padding: 6px; border-radius: 4px;")
            self.force_labels.append(val)
            vl.addWidget(val)

            gripper_layout.addWidget(w)
        layout.addWidget(gripper_group)

        # Joints
        joint_group = QtWidgets.QGroupBox("Joint Positions (rad)")
        joint_layout = QtWidgets.QGridLayout(joint_group)
        joint_layout.setSpacing(4)

        self.joint_labels = []
        names = ["Pan", "Lift", "Elbow", "W1", "W2", "W3"]
        for i, name in enumerate(names):
            joint_layout.addWidget(QtWidgets.QLabel(name), i // 3, (i % 3) * 2)
            val = QtWidgets.QLabel("0.000")
            val.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
            val.setStyleSheet("background: #37474f; color: #64b5f6; padding: 3px; border-radius: 3px;")
            self.joint_labels.append(val)
            joint_layout.addWidget(val, i // 3, (i % 3) * 2 + 1)
        layout.addWidget(joint_group)

        # Goals info
        goals_group = QtWidgets.QGroupBox("Goals")
        goals_layout = QtWidgets.QVBoxLayout(goals_group)
        goals_layout.setSpacing(4)

        row1 = QtWidgets.QHBoxLayout()
        self.goal_count_label = QtWidgets.QLabel("Goals: 0")
        self.goal_count_label.setStyleSheet("color: #1976d2; font-weight: bold;")
        row1.addWidget(self.goal_count_label)
        self.capture_status_label = QtWidgets.QLabel("Capture: --")
        row1.addWidget(self.capture_status_label)
        row1.addStretch()
        goals_layout.addLayout(row1)

        self.latest_goal_label = QtWidgets.QLabel("Latest: --")
        self.latest_goal_label.setFont(QtGui.QFont("Courier", 8))
        goals_layout.addWidget(self.latest_goal_label)

        self.goal_list_widget = QtWidgets.QListWidget()
        self.goal_list_widget.setMaximumHeight(60)
        self.goal_list_widget.setFont(QtGui.QFont("Courier", 8))
        goals_layout.addWidget(self.goal_list_widget)

        self.current_velocity_label = QtWidgets.QLabel("Velocity: 5.0x")
        self.current_velocity_label.setStyleSheet("color: #ff5722; font-weight: bold;")
        goals_layout.addWidget(self.current_velocity_label)
        layout.addWidget(goals_group)

        # Fruit score bar chart
        score_group = QtWidgets.QGroupBox("Fruit Score Components")
        score_layout = QtWidgets.QVBoxLayout(score_group)
        score_layout.setContentsMargins(4, 4, 4, 4)
        self.score_bar = ScoreBarWidget()
        score_layout.addWidget(self.score_bar)
        layout.addWidget(score_group)

        # Harvest result banner + session tally
        result_group = QtWidgets.QGroupBox("Last Harvest Result")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_layout.setContentsMargins(4, 4, 4, 4)
        self.harvest_result = HarvestResultWidget()
        result_layout.addWidget(self.harvest_result)
        layout.addWidget(result_group)

        # Grasp history dots
        grasp_group = QtWidgets.QGroupBox("Grasp History (last 15)")
        grasp_layout = QtWidgets.QVBoxLayout(grasp_group)
        grasp_layout.setContentsMargins(4, 4, 4, 4)
        legend_row = QtWidgets.QHBoxLayout()
        for ltext, lcolor in [("Proper", "#4caf50"), ("Weak", "#ffeb3b"), ("Slipped", "#ff9800"), ("No-grab", "#f44336")]:
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet(f"color: {lcolor}; font-size: 14pt;")
            legend_row.addWidget(dot)
            lbl = QtWidgets.QLabel(ltext)
            lbl.setStyleSheet("font-size: 8pt;")
            legend_row.addWidget(lbl)
        legend_row.addStretch()
        grasp_layout.addLayout(legend_row)
        self.grasp_dots = GraspOutcomeWidget()
        grasp_layout.addWidget(self.grasp_dots)
        layout.addWidget(grasp_group)

        # Graphs
        if PYQTGRAPH_AVAILABLE:
            force_graph_group = QtWidgets.QGroupBox("Force History")
            force_graph_layout = QtWidgets.QVBoxLayout(force_graph_group)

            self.force_plot = PlotWidget()
            self.force_plot.setBackground('w')
            self.force_plot.setLabel('left', 'Force (N)')
            self.force_plot.showGrid(x=True, y=True, alpha=0.3)
            self.force_plot.setMaximumHeight(150)

            colors = ['#f44336', '#2196f3', '#4caf50']
            self.force_curves = [self.force_plot.plot(pen=mkPen(color=c, width=2)) for c in colors]
            force_graph_layout.addWidget(self.force_plot)
            layout.addWidget(force_graph_group)

            joint_graph_group = QtWidgets.QGroupBox("Joint History")
            joint_graph_layout = QtWidgets.QVBoxLayout(joint_graph_group)

            self.joint_plot = PlotWidget()
            self.joint_plot.setBackground('w')
            self.joint_plot.setLabel('left', 'Pos (rad)')
            self.joint_plot.showGrid(x=True, y=True, alpha=0.3)
            self.joint_plot.setMaximumHeight(150)

            jcolors = ['#e91e63', '#9c27b0', '#3f51b5', '#00bcd4', '#009688', '#ff9800']
            self.joint_curves = [self.joint_plot.plot(pen=mkPen(color=c, width=1.5)) for c in jcolors]
            joint_graph_layout.addWidget(self.joint_plot)
            layout.addWidget(joint_graph_group)

            jvel_graph_group = QtWidgets.QGroupBox("Joint Velocity History (rad/s)")
            jvel_graph_layout = QtWidgets.QVBoxLayout(jvel_graph_group)

            self.joint_vel_plot = PlotWidget()
            self.joint_vel_plot.setBackground('w')
            self.joint_vel_plot.setLabel('left', 'Vel (rad/s)')
            self.joint_vel_plot.showGrid(x=True, y=True, alpha=0.3)
            self.joint_vel_plot.setMaximumHeight(150)
            self.joint_vel_plot.addLegend(offset=(5, 5))

            jnames = ["Pan", "Lift", "Elbow", "W1", "W2", "W3"]
            self.joint_vel_curves = [
                self.joint_vel_plot.plot(pen=mkPen(color=c, width=1.5), name=n)
                for c, n in zip(jcolors, jnames)
            ]
            jvel_graph_layout.addWidget(self.joint_vel_plot)
            layout.addWidget(jvel_graph_group)

        layout.addStretch(1)
        return panel

    def _open_keyboard_window(self):
        """Open the keyboard control window to the right of main window."""
        if self.kbd_window is None or not self.kbd_window.isVisible():
            # Pass geometry so keyboard window opens to the side
            self.kbd_window = KeyboardControlWindow(self.ros, self.geometry())
            # Keyboard window no longer accepts absolute pose from main window
        self.kbd_window.show()
        self.kbd_window.raise_()
        self.kbd_window.activateWindow()
        self.kbd_window.setFocus()

    def _update_displays(self):
        # Robot state
        if self.ros.robot_running:
            self.robot_state_label.setText("Program: RUNNING")
            self.robot_state_label.setStyleSheet("font-size: 11pt; padding: 4px; background: #c8e6c9; color: #2e7d32;")
        else:
            self.robot_state_label.setText("Program: STOPPED")
            self.robot_state_label.setStyleSheet("font-size: 11pt; padding: 4px; background: #ffccbc; color: #bf360c;")

        # Forces
        forces = self.ros.gripper_force_data
        for i, (force, label) in enumerate(zip(forces, self.force_labels)):
            label.setText(f"{force:.3f} N")
            color = "#ff1744" if abs(force) > 0.20 else "#ffc107" if abs(force) > 0.10 else "#00e676"
            label.setStyleSheet(f"background: #263238; color: {color}; padding: 6px; border-radius: 4px;")

        # Joints
        if self.ros.joint_state_data and len(self.ros.joint_state_data.position) >= 6:
            for i, pos in enumerate(self.ros.joint_state_data.position[:6]):
                self.joint_labels[i].setText(f"{pos:.3f}")

        # Motion phase badge
        goal_info = self.ros.goal_info_data
        phase = goal_info.get("motion_phase", "IDLE") if goal_info else "IDLE"
        phase_colors = {
            "IDLE":      ("#455a64", "#eceff1"),
            "APPROACH":  ("#1565c0", "#e3f2fd"),
            "REACQUIRE": ("#6a1b9a", "#f3e5f5"),
            "FINAL":     ("#e65100", "#fff3e0"),
            "REVERSING": ("#558b2f", "#f1f8e9"),
            "DROPOFF":   ("#00838f", "#e0f7fa"),
            "HOME":      ("#2e7d32", "#e8f5e9"),
        }
        bg, fg = phase_colors.get(phase, ("#455a64", "#eceff1"))
        self.phase_label.setText(phase)
        self.phase_label.setStyleSheet(
            f"background: {bg}; color: {fg}; padding: 8px; border-radius: 6px; "
            f"letter-spacing: 2px; font-size: 14pt; font-weight: bold;")

        # Reacquire result badge
        reacq = goal_info.get("reacquire_result", "") if goal_info else ""
        reacq_style = {
            "OK":    ("OK",    "#1b5e20", "#e8f5e9"),
            "NUDGE": ("NUDGE", "#e65100", "#fff3e0"),
            "FAIL":  ("FAIL",  "#b71c1c", "#ffebee"),
            "":      ("—",     "#455a64", "#eceff1"),
        }
        r_text, r_bg, r_fg = reacq_style.get(reacq, reacq_style[""])
        self.reacq_label.setText(r_text)
        self.reacq_label.setStyleSheet(
            f"background: {r_bg}; color: {r_fg}; padding: 8px; border-radius: 6px; "
            f"font-size: 14pt; font-weight: bold;")

        # Fruit score bar chart
        self.score_bar.update_scores(self.ros.fruit_score_data)

        # Harvest result + session tally
        grasp_history = goal_info.get("grasp_history", []) if goal_info else []
        self.harvest_result.update_result(grasp_history)

        # Grasp history dots
        self.grasp_dots.update_outcomes(grasp_history)

        # Goals
        if goal_info:
            self.goal_count_label.setText(f"Goals: {goal_info.get('goal_count', 0)}")
            if goal_info.get("capture_active"):
                self.capture_status_label.setText(f"Capture: ACTIVE ({goal_info.get('capture_count', 0)})")
                self.capture_status_label.setStyleSheet("color: #ff5722; font-weight: bold;")
            else:
                self.capture_status_label.setText("Capture: --")
                self.capture_status_label.setStyleSheet("")

            latest = goal_info.get("latest_goal")
            self.latest_goal_label.setText(f"Latest: {latest[0]:.3f}, {latest[1]:.3f}, {latest[2]:.3f}" if latest else "Latest: --")

            self.goal_list_widget.clear()
            for i, g in enumerate(goal_info.get("goals", [])):
                self.goal_list_widget.addItem(f"#{i+1}: {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}")

            self.current_velocity_label.setText(f"Velocity: {goal_info.get('velocity_scale', 5.0):.1f}x")

            if "debug_plan_preview" in goal_info:
                self.debug_preview_cb.blockSignals(True)
                self.debug_preview_cb.setChecked(goal_info["debug_plan_preview"])
                self.debug_preview_cb.blockSignals(False)

            waiting = goal_info.get("plan_waiting_confirm", False)
            self.plan_confirm_btn.setVisible(waiting)
            self.plan_cancel_btn.setVisible(waiting)

        # Calibration check result
        calib = self.ros.calib_check_result
        if calib is not None:
            self.calib_result_label.setText(calib)
            if calib.startswith("GOOD"):
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #e8f5e9; color: #2e7d32; border-radius: 4px;")
            elif calib.startswith("ACCEPTABLE"):
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #fff8e1; color: #e65100; border-radius: 4px;")
            elif calib.startswith("POOR") or calib.startswith("FAIL"):
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #ffebee; color: #c62828; border-radius: 4px;")
            else:
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #f3e5f5; border-radius: 4px;")

        # Graphs
        if PYQTGRAPH_AVAILABLE:
            self.elapsed_time += 0.05
            self.time_data.append(self.elapsed_time)

            for i, force in enumerate(forces):
                self.force_history[i].append(force)
                if self.time_data and self.force_history[i]:
                    self.force_curves[i].setData(list(self.time_data)[-len(self.force_history[i]):], list(self.force_history[i]))

            if self.ros.joint_state_data and len(self.ros.joint_state_data.position) >= 6:
                vels = list(self.ros.joint_state_data.velocity) if self.ros.joint_state_data.velocity else [0.0] * 6
                for i, pos in enumerate(self.ros.joint_state_data.position[:6]):
                    self.joint_history[i].append(pos)
                    if self.time_data and self.joint_history[i]:
                        self.joint_curves[i].setData(list(self.time_data)[-len(self.joint_history[i]):], list(self.joint_history[i]))
                for i in range(6):
                    v = vels[i] if i < len(vels) else 0.0
                    self.joint_vel_history[i].append(v)
                    if self.time_data and self.joint_vel_history[i]:
                        self.joint_vel_curves[i].setData(list(self.time_data)[-len(self.joint_vel_history[i]):], list(self.joint_vel_history[i]))

    def _handle_send_goal(self):
        try:
            x, y, z = float(self.x_in.text()), float(self.y_in.text()), float(self.z_in.text())
            qw, qx, qy, qz = float(self.qw_in.text()), float(self.qx_in.text()), float(self.qy_in.text()), float(self.qz_in.text())
        except ValueError:
            self.status_update.emit("Invalid goal values", True)
            return
        self.ros.publish_goal(x, y, z, qw, qx, qy, qz)
        self.status_update.emit(f"Goal sent: ({x:.3f}, {y:.3f}, {z:.3f})", False)

    def _handle_stop(self):
        self.ros.publish_stop()
        self._send_cmd("stop")
        self.status_update.emit("EMERGENCY STOP", True)

    def _send_cmd(self, cmd: str):
        self.ros.publish_cmd(cmd)
        self.status_update.emit(f"Cmd: {cmd}", False)

    def _on_velocity_slider_changed(self, value: int):
        self.velocity_value_label.setText(f"{value / 10.0:.1f}x")

    def _set_velocity_scale(self, scale: float):
        self.velocity_slider.setValue(int(scale * 10))

    def _apply_velocity_scale(self):
        scale = self.velocity_slider.value() / 10.0
        self.ros.publish_velocity_scale(scale)
        self.status_update.emit(f"Velocity: {scale:.1f}x", False)

    def _on_debug_preview_changed(self, state):
        enabled = "true" if state == Qt.Checked else "false"
        self.ros.publish_cmd(f"set_debug_preview {enabled}")
        self.status_update.emit(f"Debug preview: {enabled}", False)

    def _set_status_slot(self, text: str, error: bool):
        if not self._status:
            return
        color, bg = ("#d32f2f", "#ffcdd2") if error else ("#2d6a4f", "#e8f5e9")
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-weight: bold; padding: 6px; background: {bg}; border-radius: 4px;")


def main():
    rclpy.init()
    ros_node = UiBridge()

    def spin_ros():
        rclpy.spin(ros_node)

    spin_thread = threading.Thread(target=spin_ros, daemon=True)
    spin_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    win = MainWindow(ros_node)
    win.show()
    ret = app.exec_()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
