"""RViz-dockable rqt panel for UR10e date harvesting robot."""
from collections import deque

from qt_gui.plugin import Plugin
from python_qt_binding.QtCore import QTimer, Qt, QRect
from python_qt_binding.QtGui import QFont, QPainter, QColor, QFontMetrics
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


# ──────────────────────────── custom painter widgets ─────────────────────────

class ScoreBarWidget(QWidget):
    """Horizontal bar chart for the 6 fruit detection score components."""

    LABELS = [
        ("distance",      "Distance",   "#ef5350"),
        ("visibility",    "Visibility", "#42a5f5"),
        ("depth_quality", "Depth",      "#66bb6a"),
        ("confidence",    "Confidence", "#ffa726"),
        ("ellipse",       "Shape",      "#ab47bc"),
        ("center_bias",   "Center",     "#26c6da"),
    ]

    def __init__(self):
        super().__init__()
        self.scores: dict = {}
        self.setMinimumHeight(110)
        self.setMaximumHeight(135)

    def update_scores(self, components: dict):
        self.scores = components
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        if not self.scores:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter, "No detection")
            return

        n = len(self.LABELS)
        margin = 3
        label_w = 58
        bar_area = w - label_w - margin * 3
        row_h = max(10, (h - margin * (n + 1)) // n)
        fm = QFontMetrics(p.font())

        for i, (key, name, color) in enumerate(self.LABELS):
            y = margin + i * (row_h + margin)
            val = max(0.0, min(1.0, float(self.scores.get(key, 0.0))))
            bar_w = int(bar_area * val)

            p.setPen(QColor("#333"))
            p.drawText(margin, y, label_w, row_h, Qt.AlignVCenter | Qt.AlignLeft, name)

            bg = QRect(label_w + margin, y, bar_area, row_h)
            p.fillRect(bg, QColor("#e0e0e0"))

            if bar_w > 0:
                p.fillRect(QRect(label_w + margin, y, bar_w, row_h), QColor(color))

            val_str = f"{val:.2f}"
            text_x = label_w + margin + max(bar_w - fm.horizontalAdvance(val_str) - 2, 2)
            p.setPen(QColor("#fff") if bar_w > 30 else QColor("#555"))
            p.drawText(text_x, y, bar_area - (text_x - label_w - margin), row_h,
                       Qt.AlignVCenter | Qt.AlignLeft, val_str)


class GraspOutcomeWidget(QWidget):
    """Row of coloured circles for last N grasp outcomes."""

    COLORS = {
        ("GRABBED", "PROPER"): "#4caf50",
        ("GRABBED", "WEAK"):   "#ffeb3b",
        ("SLIPPED", ""):       "#ff9800",
        ("NO_GRAB", ""):       "#f44336",
    }
    DEFAULT_COLOR = "#9e9e9e"

    def __init__(self):
        super().__init__()
        self.outcomes: list = []
        self.setFixedHeight(26)

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
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.outcomes:
            p.setPen(QColor("#aaa"))
            p.drawText(self.rect(), Qt.AlignCenter, "No grasps yet")
            return
        r = 9
        gap = 4
        x = gap
        for item in self.outcomes:
            color = QColor(self._color(item))
            p.setBrush(color)
            p.setPen(QColor("#555"))
            p.drawEllipse(x, (self.height() - r * 2) // 2, r * 2, r * 2)
            x += r * 2 + gap


class HarvestResultWidget(QWidget):
    """Large SUCCESS/PARTIAL/SLIPPED/FAIL banner + session tally."""

    RESULT_MAP = {
        ("GRABBED", "PROPER"): ("SUCCESS", "#2e7d32", "#e8f5e9"),
        ("GRABBED", "WEAK"):   ("PARTIAL", "#e65100", "#fff3e0"),
        ("SLIPPED", "WEAK"):   ("SLIPPED", "#f57f17", "#fffde7"),
        ("SLIPPED", ""):       ("SLIPPED", "#f57f17", "#fffde7"),
        ("NO_GRAB", ""):       ("FAIL",    "#b71c1c", "#ffebee"),
    }
    DEFAULT = ("—", "#455a64", "#eceff1")

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.banner = QLabel("—")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setFont(QFont("Helvetica", 18, QFont.Bold))
        self.banner.setFixedHeight(48)
        self.banner.setStyleSheet(
            "background: #455a64; color: #eceff1; border-radius: 8px; letter-spacing: 3px;")
        layout.addWidget(self.banner)

        tally_row = QHBoxLayout()
        tally_row.setSpacing(4)
        self._tally_labels = {}
        for key, label, color in [
            ("grabbed", "Grabbed", "#4caf50"),
            ("slipped", "Slipped", "#ff9800"),
            ("miss",    "Miss",    "#f44336"),
            ("total",   "Total",   "#90a4ae"),
        ]:
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setSpacing(1)
            cl.setContentsMargins(2, 2, 2, 2)
            count = QLabel("0")
            count.setAlignment(Qt.AlignCenter)
            count.setFont(QFont("Courier", 12, QFont.Bold))
            count.setStyleSheet(f"color: {color};")
            cl.addWidget(count)
            name_lbl = QLabel(label)
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet("font-size: 8pt; color: #666;")
            cl.addWidget(name_lbl)
            tally_row.addWidget(cell)
            self._tally_labels[key] = count
        layout.addLayout(tally_row)

    def update_result(self, history: list):
        if not history:
            self.banner.setText("—")
            self.banner.setStyleSheet(
                "background: #455a64; color: #eceff1; border-radius: 8px; letter-spacing: 3px;")
            for k in self._tally_labels:
                self._tally_labels[k].setText("0")
            return

        last = history[-1]
        outcome = last.get("outcome", "")
        end = last.get("end", "")
        key = (outcome, end)
        if key not in self.RESULT_MAP:
            key = (outcome, "")
        label, bg, fg = self.RESULT_MAP.get(key, self.DEFAULT)
        self.banner.setText(label)
        self.banner.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 8px; letter-spacing: 3px;")

        grabbed = slipped = miss = 0
        for item in history:
            o = item.get("outcome", "")
            if o == "GRABBED":
                grabbed += 1
            elif o == "SLIPPED":
                slipped += 1
            elif o == "NO_GRAB":
                miss += 1
        self._tally_labels["grabbed"].setText(str(grabbed))
        self._tally_labels["slipped"].setText(str(slipped))
        self._tally_labels["miss"].setText(str(miss))
        self._tally_labels["total"].setText(str(grabbed + slipped + miss))


# ──────────────────────────────────── plugin ─────────────────────────────────

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
        self.joint_vel_history = [deque(maxlen=200) for _ in range(6)]
        self.time_data = deque(maxlen=200)
        self.elapsed_time = 0.0

        self._build_ui()
        context.add_widget(self._widget)

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_displays)
        self._timer.start(50)  # 20 Hz

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
            "background: #e8f5e9; border-radius: 4px;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Robot program state
        self.robot_state_label = QLabel("Program: Unknown")
        self.robot_state_label.setStyleSheet("font-size: 10pt; padding: 3px;")
        layout.addWidget(self.robot_state_label)

        # Emergency stop
        stop_btn = QPushButton("EMERGENCY STOP")
        stop_btn.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold; "
            "font-size: 11pt; padding: 10px;")
        stop_btn.clicked.connect(self._handle_stop)
        layout.addWidget(stop_btn)

        # ── Motion phase + reacquire result ──────────────────────────────
        phase_row = QHBoxLayout()

        phase_group = QGroupBox("Motion Phase")
        phase_inner = QVBoxLayout(phase_group)
        self.phase_label = QLabel("IDLE")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setFont(QFont("Helvetica", 13, QFont.Bold))
        self.phase_label.setStyleSheet(
            "background: #455a64; color: #eceff1; padding: 6px; "
            "border-radius: 6px; letter-spacing: 2px;")
        phase_inner.addWidget(self.phase_label)
        phase_row.addWidget(phase_group, stretch=3)

        reacq_group = QGroupBox("Reacquire")
        reacq_inner = QVBoxLayout(reacq_group)
        self.reacq_label = QLabel("—")
        self.reacq_label.setAlignment(Qt.AlignCenter)
        self.reacq_label.setFont(QFont("Helvetica", 13, QFont.Bold))
        self.reacq_label.setStyleSheet(
            "background: #455a64; color: #eceff1; padding: 6px; border-radius: 6px;")
        reacq_inner.addWidget(self.reacq_label)
        phase_row.addWidget(reacq_group, stretch=2)

        layout.addLayout(phase_row)

        # ── Last harvest result ──────────────────────────────────────────
        result_group = QGroupBox("Last Harvest Result")
        result_inner = QVBoxLayout(result_group)
        result_inner.setContentsMargins(4, 4, 4, 4)
        self.harvest_result = HarvestResultWidget()
        result_inner.addWidget(self.harvest_result)
        layout.addWidget(result_group)

        # ── Fruit score bar chart ────────────────────────────────────────
        score_group = QGroupBox("Fruit Score Components")
        score_inner = QVBoxLayout(score_group)
        score_inner.setContentsMargins(4, 4, 4, 4)
        self.score_bar = ScoreBarWidget()
        score_inner.addWidget(self.score_bar)
        layout.addWidget(score_group)

        # ── Grasp history dots ───────────────────────────────────────────
        grasp_group = QGroupBox("Grasp History (last 15)")
        grasp_inner = QVBoxLayout(grasp_group)
        grasp_inner.setContentsMargins(4, 4, 4, 4)
        legend_row = QHBoxLayout()
        for ltext, lcolor in [("Proper", "#4caf50"), ("Weak", "#ffeb3b"),
                               ("Slipped", "#ff9800"), ("No-grab", "#f44336")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {lcolor}; font-size: 12pt;")
            legend_row.addWidget(dot)
            l = QLabel(ltext)
            l.setStyleSheet("font-size: 8pt;")
            legend_row.addWidget(l)
        legend_row.addStretch()
        grasp_inner.addLayout(legend_row)
        self.grasp_dots = GraspOutcomeWidget()
        grasp_inner.addWidget(self.grasp_dots)
        layout.addWidget(grasp_group)

        # ── Motion commands ──────────────────────────────────────────────
        motion_group = QGroupBox("Motion")
        motion_layout = QGridLayout(motion_group)
        motion_layout.setSpacing(4)

        for i, (label, cmd, color) in enumerate([
            ("Home",    "home",    "#4caf50"),
            ("Dropoff", "dropoff", "#2196f3"),
            ("Execute", "execute", "#ff9800"),
            ("Clear",   "clear",   "#9e9e9e"),
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
            "font-size: 9pt; padding: 3px; background: #f3e5f5; border-radius: 4px;")
        motion_layout.addWidget(self.calib_result_label, 3, 0, 1, 2)
        layout.addWidget(motion_group)

        # ── Gripper ──────────────────────────────────────────────────────
        gripper_group = QGroupBox("Gripper")
        gripper_layout = QHBoxLayout(gripper_group)
        for label, cmd in [("Open", "open"), ("Close", "close")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, c=cmd: self._send_cmd(c))
            gripper_layout.addWidget(btn)
        layout.addWidget(gripper_group)

        # ── Velocity scale ───────────────────────────────────────────────
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

        # ── Manual goal ──────────────────────────────────────────────────
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

        for i, (lbl, w) in enumerate([("qw", self.qw_in), ("qx", self.qx_in),
                                       ("qy", self.qy_in), ("qz", self.qz_in)]):
            goal_layout.addWidget(QLabel(lbl), i, 2)
            w.setMaximumWidth(50)
            goal_layout.addWidget(w, i, 3)

        send_btn = QPushButton("Send Goal")
        send_btn.setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;")
        send_btn.clicked.connect(self._handle_send_goal)
        goal_layout.addWidget(send_btn, 4, 0, 1, 4)
        layout.addWidget(goal_group)

        # ── Capture ──────────────────────────────────────────────────────
        capture_group = QGroupBox("Capture")
        capture_layout = QHBoxLayout(capture_group)
        cap_btn = QPushButton("Start (10s)")
        cap_btn.clicked.connect(lambda: self._send_cmd("capture 10"))
        capture_layout.addWidget(cap_btn)
        cap_stop = QPushButton("Stop")
        cap_stop.clicked.connect(lambda: self._send_cmd("capture_stop"))
        capture_layout.addWidget(cap_stop)
        layout.addWidget(capture_group)

        # ── Monitoring ───────────────────────────────────────────────────

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
        for i, name in enumerate(["Pan", "Lift", "Elbow", "W1", "W2", "W3"]):
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

        # ── Graphs ───────────────────────────────────────────────────────
        if PYQTGRAPH_AVAILABLE:
            jcolors = ['#e91e63', '#9c27b0', '#3f51b5', '#00bcd4', '#009688', '#ff9800']
            jnames = ["Pan", "Lift", "Elbow", "W1", "W2", "W3"]

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

            joint_graph_group = QGroupBox("Joint Position History")
            jg_layout = QVBoxLayout(joint_graph_group)
            self.joint_plot = PlotWidget()
            self.joint_plot.setBackground('w')
            self.joint_plot.setLabel('left', 'Pos (rad)')
            self.joint_plot.showGrid(x=True, y=True, alpha=0.3)
            self.joint_plot.setMaximumHeight(120)
            self.joint_curves = [self.joint_plot.plot(pen=mkPen(color=c, width=1.5)) for c in jcolors]
            jg_layout.addWidget(self.joint_plot)
            layout.addWidget(joint_graph_group)

            jvel_graph_group = QGroupBox("Joint Velocity History (rad/s)")
            jvg_layout = QVBoxLayout(jvel_graph_group)
            self.joint_vel_plot = PlotWidget()
            self.joint_vel_plot.setBackground('w')
            self.joint_vel_plot.setLabel('left', 'Vel (rad/s)')
            self.joint_vel_plot.showGrid(x=True, y=True, alpha=0.3)
            self.joint_vel_plot.setMaximumHeight(120)
            self.joint_vel_plot.addLegend(offset=(5, 5))
            self.joint_vel_curves = [
                self.joint_vel_plot.plot(pen=mkPen(color=c, width=1.5), name=n)
                for c, n in zip(jcolors, jnames)
            ]
            jvg_layout.addWidget(self.joint_vel_plot)
            layout.addWidget(jvel_graph_group)

        layout.addStretch(1)

        scroll.setWidget(panel)
        outer = QVBoxLayout(self._widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ----------------------------------------------------------- Updates
    def _update_displays(self):
        ros = self._bridge
        gi = ros.goal_info_data

        # Robot state
        if ros.robot_running:
            self.robot_state_label.setText("Program: RUNNING")
            self.robot_state_label.setStyleSheet(
                "font-size: 10pt; padding: 3px; background: #c8e6c9; color: #2e7d32;")
        else:
            self.robot_state_label.setText("Program: STOPPED")
            self.robot_state_label.setStyleSheet(
                "font-size: 10pt; padding: 3px; background: #ffccbc; color: #bf360c;")

        # Motion phase badge
        phase = gi.get("motion_phase", "IDLE") if gi else "IDLE"
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
            f"background: {bg}; color: {fg}; padding: 6px; border-radius: 6px; "
            f"letter-spacing: 2px; font-size: 13pt; font-weight: bold;")

        # Reacquire result badge
        reacq = gi.get("reacquire_result", "") if gi else ""
        reacq_styles = {
            "OK":    ("OK",    "#1b5e20", "#e8f5e9"),
            "NUDGE": ("NUDGE", "#e65100", "#fff3e0"),
            "FAIL":  ("FAIL",  "#b71c1c", "#ffebee"),
            "":      ("—",     "#455a64", "#eceff1"),
        }
        r_text, r_bg, r_fg = reacq_styles.get(reacq, reacq_styles[""])
        self.reacq_label.setText(r_text)
        self.reacq_label.setStyleSheet(
            f"background: {r_bg}; color: {r_fg}; padding: 6px; border-radius: 6px; "
            f"font-size: 13pt; font-weight: bold;")

        # Harvest result + session tally
        grasp_history = gi.get("grasp_history", []) if gi else []
        self.harvest_result.update_result(grasp_history)

        # Grasp history dots
        self.grasp_dots.update_outcomes(grasp_history)

        # Fruit score bar chart
        self.score_bar.update_scores(ros.fruit_score_data)

        # Forces
        forces = ros.gripper_force_data
        for i, (force, label) in enumerate(zip(forces, self.force_labels)):
            label.setText(f"{force:.3f} N")
            color = "#ff1744" if abs(force) > 0.20 else "#ffc107" if abs(force) > 0.10 else "#00e676"
            label.setStyleSheet(
                f"background: #263238; color: {color}; padding: 4px; border-radius: 3px;")

        # Joints
        if ros.joint_state_data and len(ros.joint_state_data.position) >= 6:
            for i, pos in enumerate(ros.joint_state_data.position[:6]):
                self.joint_labels[i].setText(f"{pos:.3f}")

        # Goals
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
                f"Latest: {latest[0]:.3f}, {latest[1]:.3f}, {latest[2]:.3f}"
                if latest else "Latest: --")

            self.goal_list_widget.clear()
            for i, g in enumerate(gi.get("goals", [])):
                self.goal_list_widget.addItem(f"#{i+1}: {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}")

            self.current_velocity_label.setText(f"Velocity: {gi.get('velocity_scale', 5.0):.1f}x")

        # Calibration result
        calib = ros.calib_check_result
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
                    self.force_curves[i].setData(
                        list(self.time_data)[-len(self.force_history[i]):],
                        list(self.force_history[i]))

            if ros.joint_state_data and len(ros.joint_state_data.position) >= 6:
                vels = list(ros.joint_state_data.velocity) if ros.joint_state_data.velocity else [0.0] * 6
                for i, pos in enumerate(ros.joint_state_data.position[:6]):
                    self.joint_history[i].append(pos)
                    if self.time_data and self.joint_history[i]:
                        self.joint_curves[i].setData(
                            list(self.time_data)[-len(self.joint_history[i]):],
                            list(self.joint_history[i]))
                for i in range(6):
                    v = vels[i] if i < len(vels) else 0.0
                    self.joint_vel_history[i].append(v)
                    if self.time_data and self.joint_vel_history[i]:
                        self.joint_vel_curves[i].setData(
                            list(self.time_data)[-len(self.joint_vel_history[i]):],
                            list(self.joint_vel_history[i]))

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
            f"background: {bg}; border-radius: 4px;")

    # --------------------------------------------------------- Lifecycle
    def shutdown_plugin(self):
        self._timer.stop()

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value("velocity_scale", self.velocity_slider.value())

    def restore_settings(self, plugin_settings, instance_settings):
        val = instance_settings.value("velocity_scale", 50)
        self.velocity_slider.setValue(int(val))
