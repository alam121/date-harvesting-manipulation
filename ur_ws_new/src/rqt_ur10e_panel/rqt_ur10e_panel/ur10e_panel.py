"""RViz-dockable rqt panel for UR10e date harvesting robot."""
from collections import deque

from qt_gui.plugin import Plugin
from python_qt_binding.QtCore import QTimer, Qt
from python_qt_binding.QtGui import QFont
from python_qt_binding.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QSlider, QLineEdit,
    QScrollArea, QListWidget,
)

try:
    from pyqtgraph import PlotWidget, mkPen
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from .ros_bridge import RosBridge


class UR10ePanel(Plugin):

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName("UR10ePanel")

        self._widget = QWidget()
        self._widget.setObjectName("UR10ePanelWidget")

        self._node = context.node
        self._bridge = RosBridge(self._node)

        # Graph data
        self.force_history = [deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)]
        self.joint_history = [deque(maxlen=200) for _ in range(6)]
        self.time_data = deque(maxlen=200)
        self.elapsed_time = 0.0

        self._build_ui()
        context.add_widget(self._widget)

        # 20 Hz display update
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_displays)
        self._timer.start(50)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # Title & status
        title = QLabel("UR10e Control")
        title.setFont(QFont("Helvetica", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._status = QLabel("Ready")
        self._status.setStyleSheet(
            "color: #2d6a4f; font-weight: bold; padding: 4px; "
            "background: #e8f5e9; border-radius: 4px;"
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Robot program state
        self.robot_state_label = QLabel("Program: Unknown")
        self.robot_state_label.setStyleSheet("font-size: 10pt; padding: 3px;")
        layout.addWidget(self.robot_state_label)

        # Emergency stop (top, prominent)
        stop_btn = QPushButton("EMERGENCY STOP")
        stop_btn.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold; "
            "font-size: 11pt; padding: 10px;"
        )
        stop_btn.clicked.connect(self._handle_stop)
        layout.addWidget(stop_btn)

        # Motion commands
        motion_group = QGroupBox("Motion")
        motion_layout = QGridLayout(motion_group)
        motion_layout.setSpacing(4)

        for i, (label, cmd, color) in enumerate([
            ("Home", "home", "#4caf50"),
            ("Dropoff", "dropoff", "#2196f3"),
            ("Execute", "execute", "#ff9800"),
            ("Clear", "clear", "#9e9e9e"),
        ]):
            btn = QPushButton(label)
            btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold;")
            btn.clicked.connect(lambda _, c=cmd: self._send_cmd(c))
            motion_layout.addWidget(btn, i // 2, i % 2)

        check_calib_btn = QPushButton("Check Calibration (trunk)")
        check_calib_btn.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        check_calib_btn.setToolTip("Check hand-eye calibration using the detected trunk as the reference")
        check_calib_btn.clicked.connect(lambda: self._send_cmd("check_calibration"))
        motion_layout.addWidget(check_calib_btn, 2, 0, 1, 2)

        self.calib_result_label = QLabel("—")
        self.calib_result_label.setWordWrap(True)
        self.calib_result_label.setStyleSheet(
            "font-size: 9pt; padding: 3px; background: #f3e5f5; border-radius: 4px;"
        )
        motion_layout.addWidget(self.calib_result_label, 3, 0, 1, 2)
        layout.addWidget(motion_group)

        # Gripper
        gripper_group = QGroupBox("Gripper")
        gripper_layout = QHBoxLayout(gripper_group)
        for label, cmd in [("Open", "open"), ("Close", "close")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, c=cmd: self._send_cmd(c))
            gripper_layout.addWidget(btn)
        layout.addWidget(gripper_group)

        # Velocity scale
        vel_group = QGroupBox("Velocity Scale")
        vel_layout = QVBoxLayout(vel_group)
        vel_layout.setSpacing(4)

        slider_row = QHBoxLayout()
        self.velocity_slider = QSlider(Qt.Horizontal)
        self.velocity_slider.setRange(10, 100)
        self.velocity_slider.setValue(50)
        self.velocity_slider.valueChanged.connect(self._on_velocity_slider_changed)
        slider_row.addWidget(self.velocity_slider)

        self.velocity_value_label = QLabel("5.0x")
        self.velocity_value_label.setFont(QFont("Courier", 9, QFont.Bold))
        self.velocity_value_label.setStyleSheet("color: #1976d2;")
        slider_row.addWidget(self.velocity_value_label)
        vel_layout.addLayout(slider_row)

        preset_row = QHBoxLayout()
        for label, value in [("Slow", 1.0), ("Med", 3.0), ("Fast", 5.0), ("Max", 8.0)]:
            btn = QPushButton(label)
            btn.setMaximumHeight(22)
            btn.clicked.connect(lambda _, v=value: self._set_velocity_scale(v))
            preset_row.addWidget(btn)
        vel_layout.addLayout(preset_row)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self._apply_velocity_scale)
        vel_layout.addWidget(apply_btn)
        layout.addWidget(vel_group)

        # Manual goal
        goal_group = QGroupBox("Manual Goal")
        goal_layout = QGridLayout(goal_group)
        goal_layout.setSpacing(3)

        self.x_in = QLineEdit("0.30")
        self.y_in = QLineEdit("0.00")
        self.z_in = QLineEdit("0.20")
        self.qw_in = QLineEdit("1.0")
        self.qx_in = QLineEdit("0.0")
        self.qy_in = QLineEdit("0.0")
        self.qz_in = QLineEdit("0.0")

        for i, (lbl, w) in enumerate([("X", self.x_in), ("Y", self.y_in), ("Z", self.z_in)]):
            goal_layout.addWidget(QLabel(lbl), i, 0)
            w.setMaximumWidth(70)
            goal_layout.addWidget(w, i, 1)

        for i, (lbl, w) in enumerate([("qw", self.qw_in), ("qx", self.qx_in), ("qy", self.qy_in), ("qz", self.qz_in)]):
            goal_layout.addWidget(QLabel(lbl), i, 2)
            w.setMaximumWidth(50)
            goal_layout.addWidget(w, i, 3)

        send_btn = QPushButton("Send Goal")
        send_btn.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        send_btn.clicked.connect(self._handle_send_goal)
        goal_layout.addWidget(send_btn, 4, 0, 1, 4)
        layout.addWidget(goal_group)

        # Capture
        capture_group = QGroupBox("Capture")
        capture_layout = QHBoxLayout(capture_group)
        cap_btn = QPushButton("Start (10s)")
        cap_btn.clicked.connect(lambda: self._send_cmd("capture 10"))
        capture_layout.addWidget(cap_btn)
        cap_stop = QPushButton("Stop")
        cap_stop.clicked.connect(lambda: self._send_cmd("capture_stop"))
        capture_layout.addWidget(cap_stop)
        layout.addWidget(capture_group)

        # --- Monitoring section ---

        # Gripper forces
        gripper_f_group = QGroupBox("Gripper Forces")
        gripper_f_layout = QHBoxLayout(gripper_f_group)
        self.force_labels = []
        for name in ["Left", "Center", "Right"]:
            w = QWidget()
            vl = QVBoxLayout(w)
            vl.setSpacing(2)
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(lbl)
            val = QLabel("0.00 N")
            val.setAlignment(Qt.AlignCenter)
            val.setFont(QFont("Courier", 10, QFont.Bold))
            val.setStyleSheet("background: #263238; color: #00e676; padding: 4px; border-radius: 3px;")
            self.force_labels.append(val)
            vl.addWidget(val)
            gripper_f_layout.addWidget(w)
        layout.addWidget(gripper_f_group)

        # Joint positions
        joint_group = QGroupBox("Joint Positions (rad)")
        joint_layout = QGridLayout(joint_group)
        joint_layout.setSpacing(3)
        self.joint_labels = []
        names = ["Pan", "Lift", "Elbow", "W1", "W2", "W3"]
        for i, name in enumerate(names):
            joint_layout.addWidget(QLabel(name), i // 3, (i % 3) * 2)
            val = QLabel("0.000")
            val.setFont(QFont("Courier", 8, QFont.Bold))
            val.setStyleSheet("background: #37474f; color: #64b5f6; padding: 2px; border-radius: 2px;")
            self.joint_labels.append(val)
            joint_layout.addWidget(val, i // 3, (i % 3) * 2 + 1)
        layout.addWidget(joint_group)

        # Goal info
        goals_group = QGroupBox("Goals")
        goals_layout = QVBoxLayout(goals_group)
        goals_layout.setSpacing(3)

        row1 = QHBoxLayout()
        self.goal_count_label = QLabel("Goals: 0")
        self.goal_count_label.setStyleSheet("color: #1976d2; font-weight: bold;")
        row1.addWidget(self.goal_count_label)
        self.capture_status_label = QLabel("Capture: --")
        row1.addWidget(self.capture_status_label)
        row1.addStretch()
        goals_layout.addLayout(row1)

        self.latest_goal_label = QLabel("Latest: --")
        self.latest_goal_label.setFont(QFont("Courier", 7))
        goals_layout.addWidget(self.latest_goal_label)

        self.goal_list_widget = QListWidget()
        self.goal_list_widget.setMaximumHeight(50)
        self.goal_list_widget.setFont(QFont("Courier", 7))
        goals_layout.addWidget(self.goal_list_widget)

        self.current_velocity_label = QLabel("Velocity: 5.0x")
        self.current_velocity_label.setStyleSheet("color: #ff5722; font-weight: bold;")
        goals_layout.addWidget(self.current_velocity_label)
        layout.addWidget(goals_group)

        # Graphs (optional)
        if PYQTGRAPH_AVAILABLE:
            force_graph_group = QGroupBox("Force History")
            fg_layout = QVBoxLayout(force_graph_group)
            self.force_plot = PlotWidget()
            self.force_plot.setBackground('w')
            self.force_plot.setLabel('left', 'Force (N)')
            self.force_plot.showGrid(x=True, y=True, alpha=0.3)
            self.force_plot.setMaximumHeight(120)
            colors = ['#f44336', '#2196f3', '#4caf50']
            self.force_curves = [self.force_plot.plot(pen=mkPen(color=c, width=2)) for c in colors]
            fg_layout.addWidget(self.force_plot)
            layout.addWidget(force_graph_group)

            joint_graph_group = QGroupBox("Joint History")
            jg_layout = QVBoxLayout(joint_graph_group)
            self.joint_plot = PlotWidget()
            self.joint_plot.setBackground('w')
            self.joint_plot.setLabel('left', 'Pos (rad)')
            self.joint_plot.showGrid(x=True, y=True, alpha=0.3)
            self.joint_plot.setMaximumHeight(120)
            jcolors = ['#e91e63', '#9c27b0', '#3f51b5', '#00bcd4', '#009688', '#ff9800']
            self.joint_curves = [self.joint_plot.plot(pen=mkPen(color=c, width=1.5)) for c in jcolors]
            jg_layout.addWidget(self.joint_plot)
            layout.addWidget(joint_graph_group)

        layout.addStretch(1)

        scroll.setWidget(panel)
        outer = QVBoxLayout(self._widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ----------------------------------------------------------- Updates
    def _update_displays(self):
        ros = self._bridge

        # Robot state
        if ros.robot_running:
            self.robot_state_label.setText("Program: RUNNING")
            self.robot_state_label.setStyleSheet(
                "font-size: 10pt; padding: 3px; background: #c8e6c9; color: #2e7d32;"
            )
        else:
            self.robot_state_label.setText("Program: STOPPED")
            self.robot_state_label.setStyleSheet(
                "font-size: 10pt; padding: 3px; background: #ffccbc; color: #bf360c;"
            )

        # Forces
        forces = ros.gripper_force_data
        for i, (force, label) in enumerate(zip(forces, self.force_labels)):
            label.setText(f"{force:.3f} N")
            color = "#ff1744" if abs(force) > 0.20 else "#ffc107" if abs(force) > 0.10 else "#00e676"
            label.setStyleSheet(
                f"background: #263238; color: {color}; padding: 4px; border-radius: 3px;"
            )

        # Joints
        if ros.joint_state_data and len(ros.joint_state_data.position) >= 6:
            for i, pos in enumerate(ros.joint_state_data.position[:6]):
                self.joint_labels[i].setText(f"{pos:.3f}")

        # Goals
        gi = ros.goal_info_data
        if gi:
            self.goal_count_label.setText(f"Goals: {gi.get('goal_count', 0)}")
            if gi.get("capture_active"):
                self.capture_status_label.setText(f"Capture: ACTIVE ({gi.get('capture_count', 0)})")
                self.capture_status_label.setStyleSheet("color: #ff5722; font-weight: bold;")
            else:
                self.capture_status_label.setText("Capture: --")
                self.capture_status_label.setStyleSheet("")

            latest = gi.get("latest_goal")
            self.latest_goal_label.setText(
                f"Latest: {latest[0]:.3f}, {latest[1]:.3f}, {latest[2]:.3f}" if latest else "Latest: --"
            )

            self.goal_list_widget.clear()
            for i, g in enumerate(gi.get("goals", [])):
                self.goal_list_widget.addItem(f"#{i+1}: {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}")

            self.current_velocity_label.setText(f"Velocity: {gi.get('velocity_scale', 5.0):.1f}x")

        calib = ros.calib_check_result
        if calib is not None:
            self.calib_result_label.setText(calib)
            if calib.startswith("GOOD"):
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #e8f5e9; color: #2e7d32; border-radius: 4px;"
                )
            elif calib.startswith("ACCEPTABLE"):
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #fff8e1; color: #e65100; border-radius: 4px;"
                )
            elif calib.startswith("POOR") or calib.startswith("FAIL"):
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #ffebee; color: #c62828; border-radius: 4px;"
                )
            else:
                self.calib_result_label.setStyleSheet(
                    "font-size: 10pt; padding: 4px; background: #f3e5f5; border-radius: 4px;"
                )

        # Graphs
        if PYQTGRAPH_AVAILABLE:
            self.elapsed_time += 0.05
            self.time_data.append(self.elapsed_time)

            for i, force in enumerate(forces):
                self.force_history[i].append(force)
                if self.time_data and self.force_history[i]:
                    self.force_curves[i].setData(
                        list(self.time_data)[-len(self.force_history[i]):],
                        list(self.force_history[i]),
                    )

            if ros.joint_state_data and len(ros.joint_state_data.position) >= 6:
                for i, pos in enumerate(ros.joint_state_data.position[:6]):
                    self.joint_history[i].append(pos)
                    if self.time_data and self.joint_history[i]:
                        self.joint_curves[i].setData(
                            list(self.time_data)[-len(self.joint_history[i]):],
                            list(self.joint_history[i]),
                        )

    # ----------------------------------------------------------- Actions
    def _send_cmd(self, cmd):
        self._bridge.publish_cmd(cmd)
        self._set_status(f"Cmd: {cmd}")

    def _handle_stop(self):
        self._bridge.publish_stop()
        self._bridge.publish_cmd("stop")
        self._set_status("EMERGENCY STOP", error=True)

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
            self._set_status("Invalid goal values", error=True)
            return
        self._bridge.publish_goal(x, y, z, qw, qx, qy, qz)
        self._set_status(f"Goal sent: ({x:.3f}, {y:.3f}, {z:.3f})")

    def _on_velocity_slider_changed(self, value):
        self.velocity_value_label.setText(f"{value / 10.0:.1f}x")

    def _set_velocity_scale(self, scale):
        self.velocity_slider.setValue(int(scale * 10))

    def _apply_velocity_scale(self):
        scale = self.velocity_slider.value() / 10.0
        self._bridge.publish_velocity_scale(scale)
        self._set_status(f"Velocity: {scale:.1f}x")

    def _set_status(self, text, error=False):
        color, bg = ("#d32f2f", "#ffcdd2") if error else ("#2d6a4f", "#e8f5e9")
        self._status.setText(text)
        self._status.setStyleSheet(
            f"color: {color}; font-weight: bold; padding: 4px; "
            f"background: {bg}; border-radius: 4px;"
        )

    # --------------------------------------------------------- Lifecycle
    def shutdown_plugin(self):
        self._timer.stop()

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value("velocity_scale", self.velocity_slider.value())

    def restore_settings(self, plugin_settings, instance_settings):
        val = instance_settings.value("velocity_scale", 50)
        self.velocity_slider.setValue(int(val))
