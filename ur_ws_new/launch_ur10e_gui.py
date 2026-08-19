#!/usr/bin/env python3
"""Small GUI front end for the UR10e launcher."""
import shlex
import subprocess
import sys
import json
import math
import re
import importlib.util
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


WS = Path(__file__).resolve().parent
REPO_ROOT = WS.parent
LAUNCHER = REPO_ROOT / "bin" / "launch_ur10e"
VISION_DIR = WS / "src" / "ur10e_curobo" / "ur10e_curobo" / "vision"
PROFILE_DIR = VISION_DIR / "calibration_profiles"
GRIPPER_PROFILE_SOURCE = (
    WS / "src" / "ur10e_curobo" / "ur10e_curobo" / "gripper_profiles.py")
GRIPPER_CALIBRATION_FILE = (
    Path.home() / ".config" / "datepalm" / "gripper_calibration.json")


def _gripper_defaults():
    spec = importlib.util.spec_from_file_location(
        "launcher_gripper_profiles", GRIPPER_PROFILE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.default_environment_calibrations()


class GripperCalibrationDialog(QtWidgets.QDialog):
    POSES = (
        ("normal_open", "NORMAL Open"),
        ("normal_closed", "NORMAL Closed"),
        ("envelop_open", "ENVELOP Open"),
        ("envelop_closed", "ENVELOP Closed"),
    )

    def __init__(self, parent=None, initial_environment="outdoor"):
        super().__init__(parent)
        self.setWindowTitle("Environment Gripper Calibration")
        self.resize(900, 850)
        self._driver_process = None
        self.defaults = _gripper_defaults()
        self.data = json.loads(json.dumps(self.defaults))
        try:
            loaded = json.loads(GRIPPER_CALIBRATION_FILE.read_text())
            for environment in ("lab", "outdoor"):
                for key, _ in self.POSES:
                    values = loaded.get(environment, {}).get(key)
                    if (isinstance(values, list) and len(values) == 12
                            and all(math.isfinite(float(v)) for v in values)):
                        self.data[environment][key] = [float(v) for v in values]
        except (OSError, ValueError, TypeError):
            pass

        self.environment_combo = QtWidgets.QComboBox()
        self.environment_combo.addItem("Lab gripper", "lab")
        self.environment_combo.addItem("Outdoor gripper", "outdoor")
        index = self.environment_combo.findData(initial_environment)
        self.environment_combo.setCurrentIndex(max(0, index))
        self.current_environment = self.environment_combo.currentData()

        self.table = QtWidgets.QTableWidget(12, 5)
        self.table.setHorizontalHeaderLabels(
            ["Joint", *[label for _, label in self.POSES]])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        for row in range(12):
            item = QtWidgets.QTableWidgetItem(f"M{row + 1}")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("NORMAL", "normal")
        self.mode_combo.addItem("ENVELOP", "envelop")
        self.live_table = QtWidgets.QTableWidget(12, 3)
        self.live_table.setHorizontalHeaderLabels(["Joint", "Position", "Degrees"])
        self.live_table.verticalHeader().setVisible(False)
        self.live_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)
        self.live_sliders = []
        self.live_spins = []
        for row in range(12):
            joint_item = QtWidgets.QTableWidgetItem(f"M{row + 1}")
            joint_item.setFlags(joint_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.live_table.setItem(row, 0, joint_item)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-36000, 36000)
            slider.setSingleStep(25)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.25)
            slider.valueChanged.connect(
                lambda value, target=spin: target.setValue(value / 100.0))
            spin.valueChanged.connect(
                lambda value, target=slider: target.setValue(round(value * 100.0)))
            slider.sliderReleased.connect(self._publish_live_pose)
            spin.editingFinished.connect(self._publish_live_pose)
            self.live_table.setCellWidget(row, 1, slider)
            self.live_table.setCellWidget(row, 2, spin)
            self.live_sliders.append(slider)
            self.live_spins.append(spin)

        start_driver_btn = QtWidgets.QPushButton("Start Gripper Driver")
        capture_btn = QtWidgets.QPushButton("Read Current Gripper Joints")
        send_btn = QtWidgets.QPushButton("Send Current Slider Pose")
        save_open_btn = QtWidgets.QPushButton("Save Current as Open")
        save_close_btn = QtWidgets.QPushButton("Save Current as Closed")
        reset_env_btn = QtWidgets.QPushButton("Reset This Environment")
        save_btn = QtWidgets.QPushButton("Save Calibration")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        save_btn.setDefault(True)

        form = QtWidgets.QFormLayout()
        form.addRow("Physical gripper", self.environment_combo)
        form.addRow("Grasp mode", self.mode_combo)
        driver_row = QtWidgets.QHBoxLayout()
        driver_row.addWidget(start_driver_btn)
        driver_row.addWidget(capture_btn)
        driver_row.addWidget(send_btn)
        save_pose_row = QtWidgets.QHBoxLayout()
        save_pose_row.addWidget(save_open_btn)
        save_pose_row.addWidget(save_close_btn)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(reset_env_btn)
        button_row.addStretch(1)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(save_btn)
        note = QtWidgets.QLabel(
            "Values are shown in degrees and saved internally in radians. "
            "Moving a slider publishes all M1–M12 values when it is released. "
            "Read Current Gripper Joints loads physical feedback into the sliders. "
            "Restart the system after saving.")
        note.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(driver_row)
        layout.addWidget(self.live_table)
        layout.addLayout(save_pose_row)
        layout.addWidget(QtWidgets.QLabel("Saved environment poses (degrees)"))
        layout.addWidget(self.table)
        layout.addWidget(note)
        layout.addLayout(button_row)

        self.environment_combo.currentIndexChanged.connect(
            self._environment_changed)
        self.mode_combo.currentIndexChanged.connect(self._load_live_open_pose)
        start_driver_btn.clicked.connect(self._start_gripper_driver)
        capture_btn.clicked.connect(self._capture_current)
        send_btn.clicked.connect(self._publish_live_pose)
        save_open_btn.clicked.connect(lambda: self._save_live_pose("open"))
        save_close_btn.clicked.connect(lambda: self._save_live_pose("closed"))
        reset_env_btn.clicked.connect(self._reset_environment)
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)
        self._load_table()
        self._load_live_open_pose()

    def _commit_table(self):
        for column, (key, _) in enumerate(self.POSES, start=1):
            values = []
            for row in range(12):
                item = self.table.item(row, column)
                try:
                    degrees = float(item.text())
                except (AttributeError, ValueError):
                    raise ValueError(f"M{row + 1} {key} is not a number")
                if not math.isfinite(degrees) or abs(degrees) > 360.0:
                    raise ValueError(
                        f"M{row + 1} {key} must be within +/-360 degrees")
                values.append(math.radians(degrees))
            self.data[self.current_environment][key] = values

    def _load_table(self):
        for column, (key, _) in enumerate(self.POSES, start=1):
            for row, radians in enumerate(
                    self.data[self.current_environment][key]):
                self.table.setItem(
                    row, column,
                    QtWidgets.QTableWidgetItem(f"{math.degrees(radians):.3f}"))

    def _environment_changed(self):
        try:
            self._commit_table()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Joint Value", str(exc))
            return
        self.current_environment = self.environment_combo.currentData()
        self._load_table()
        self._load_live_open_pose()

    def _reset_environment(self):
        answer = QtWidgets.QMessageBox.question(
            self, "Reset Gripper Calibration",
            f"Reset all four {self.current_environment} poses to the current project defaults?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.data[self.current_environment] = json.loads(json.dumps(
                self.defaults[self.current_environment]))
            self._load_table()
            self._load_live_open_pose()

    def _selected_pose_key(self, suffix):
        return f"{self.mode_combo.currentData()}_{suffix}"

    def _live_radians(self):
        return [math.radians(spin.value()) for spin in self.live_spins]

    def _set_live_radians(self, values):
        for spin, slider, value in zip(
                self.live_spins, self.live_sliders, values):
            spin.blockSignals(True)
            slider.blockSignals(True)
            degrees = math.degrees(float(value))
            spin.setValue(degrees)
            slider.setValue(round(degrees * 100.0))
            slider.blockSignals(False)
            spin.blockSignals(False)

    def _load_live_open_pose(self):
        key = self._selected_pose_key("open")
        self._set_live_radians(self.data[self.current_environment][key])

    def _capture_current(self):
        command = (
            f"source {shlex.quote(str(WS / 'install/setup.bash'))} && "
            "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-6} && "
            "ros2 topic echo --once /gripper/joint_states --field position")
        try:
            result = subprocess.run(
                ["bash", "-lc", command], capture_output=True, text=True,
                timeout=4.0, check=True)
            values = [float(value) for value in re.findall(
                r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                result.stdout)]
            if len(values) != 12:
                raise ValueError(
                    f"expected 12 joint positions, received {len(values)}")
        except (subprocess.SubprocessError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could Not Read Gripper",
                f"Start the gripper driver and try again.\n\n{exc}")
            return
        self._set_live_radians(values)

    def _publish_live_pose(self):
        values = self._live_radians()
        payload = ", ".join(f"{value:.9f}" for value in values)
        command = (
            f"source {shlex.quote(str(WS / 'install/setup.bash'))} && "
            "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-6} && "
            "ros2 topic pub --once /gripper/target_joint "
            "std_msgs/msg/Float32MultiArray "
            f"\"{{data: [{payload}]}}\"")
        try:
            subprocess.Popen(
                ["bash", "-lc", command], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could Not Command Gripper", str(exc))

    def _driver_is_running(self):
        command = (
            f"source {shlex.quote(str(WS / 'install/setup.bash'))} && "
            "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-6} && "
            "ros2 node list --no-daemon")
        try:
            result = subprocess.run(
                ["bash", "-lc", command], capture_output=True, text=True,
                timeout=3.0, check=True)
        except subprocess.SubprocessError:
            return False
        return any("delto_3f_driver" in line for line in result.stdout.splitlines())

    def _start_gripper_driver(self):
        if self._driver_is_running():
            QtWidgets.QMessageBox.information(
                self, "Gripper Driver", "The gripper driver is already running.")
            return
        log_path = Path("/tmp/ur10e_gripper_calibration_driver.log")
        command = (
            f"source {shlex.quote(str(WS / 'install/setup.bash'))} && "
            "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-6} && "
            "ros2 launch delto_3f_driver delto_3f_bringup.launch.py "
            "launch_rviz:=false")
        try:
            log_handle = log_path.open("a")
            self._driver_process = subprocess.Popen(
                ["bash", "-lc", command], stdout=log_handle,
                stderr=subprocess.STDOUT, start_new_session=True)
            log_handle.close()
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Could Not Start Gripper Driver", str(exc))
            return
        QtWidgets.QMessageBox.information(
            self, "Gripper Driver Starting",
            f"Driver started for ROS domain ${{ROS_DOMAIN_ID:-6}}.\n"
            f"Wait a few seconds, then read the current joints.\n\nLog: {log_path}")

    def _write_calibration(self):
        GRIPPER_CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        GRIPPER_CALIBRATION_FILE.write_text(
            json.dumps(self.data, indent=2) + "\n")

    def _save_live_pose(self, suffix):
        try:
            self._commit_table()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Joint Value", str(exc))
            return
        key = self._selected_pose_key(suffix)
        self.data[self.current_environment][key] = self._live_radians()
        self._load_table()
        self._write_calibration()
        QtWidgets.QMessageBox.information(
            self, "Gripper Pose Saved",
            f"Saved current sliders as {self.current_environment.upper()} "
            f"{key.replace('_', ' ').upper()}.")

    def _save(self):
        try:
            self._commit_table()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Joint Value", str(exc))
            return
        self._write_calibration()
        QtWidgets.QMessageBox.information(
            self, "Calibration Saved",
            f"Saved Lab and Outdoor gripper poses to:\n{GRIPPER_CALIBRATION_FILE}\n\n"
            "Restart the robot system to apply them.")
        self.accept()


class LaunchDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Launch UR10e")
        self.setMinimumWidth(560)
        self.settings = QtCore.QSettings("DatePalm", "UR10eLauncher")

        self.real_radio = QtWidgets.QRadioButton("Real robot")
        self.fake_radio = QtWidgets.QRadioButton("Fake hardware")
        self.real_radio.setChecked(True)
        self.hardware_group = QtWidgets.QButtonGroup(self)
        self.hardware_group.addButton(self.real_radio)
        self.hardware_group.addButton(self.fake_radio)

        self.robot_profile_combo = QtWidgets.QComboBox()
        self.robot_profile_combo.addItem("Old robot", "old")
        self._set_combo_data(
            self.robot_profile_combo,
            self.settings.value("robot_profile", "old"),
        )

        self.environment_combo = QtWidgets.QComboBox()
        self.environment_combo.addItem("Outdoor / field", "outdoor")
        self.environment_combo.addItem("Lab", "lab")
        self._set_combo_data(
            self.environment_combo,
            self.settings.value("environment", "outdoor"),
        )

        self.gripper_profile_combo = QtWidgets.QComboBox()
        self.gripper_profile_combo.addItem("Old gripper", "old")
        self.gripper_profile_combo.addItem("New DG-3F-M gripper", "new")
        self._set_combo_data(
            self.gripper_profile_combo,
            self.settings.value("gripper_profile", "old"),
        )
        self.use_gripper_cb = QtWidgets.QCheckBox("Use gripper driver/control")
        self.use_gripper_cb.setChecked(
            self.settings.value("use_gripper", "true") != "false")
        self.use_gripper_cb.setToolTip(
            "Unchecked: do not launch/control the Delto gripper. Open/close commands become no-ops."
        )

        self.main_cb = QtWidgets.QCheckBox("Main cuRobo control + RViz panel")
        self.vision_cb = QtWidgets.QCheckBox("Vision")
        self.teleop_cb = QtWidgets.QCheckBox("Teleop")
        self.gui_cb = QtWidgets.QCheckBox("Desktop GUI")
        self.calibrate_cb = QtWidgets.QCheckBox("Grasp force calibration")
        self.main_cb.setChecked(True)
        self.vision_cb.setChecked(True)

        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItem("ZED X One + ZED Mini depth", "zed_mini")
        self.camera_combo.addItem("ZED X Mini only", "zedx_mini")
        self.camera_combo.addItem("ZED stereo only", "stereo")
        self.camera_combo.addItem("ZED X One + Livox LiDAR depth", "lidar")
        self.camera_combo.addItem("No vision", "none")
        self._set_combo_data(
            self.camera_combo,
            self.settings.value("camera_mode", "zedx_mini"),
        )

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("ChArUco", "charuco")
        self.target_combo.addItem("Chessboard", "chessboard")

        self.res_combo = QtWidgets.QComboBox()
        self._populate_resolution_combo()

        self.command_edit = QtWidgets.QLineEdit()
        self.command_edit.setReadOnly(True)

        harvest_btn = QtWidgets.QPushButton("Harvest Preset")
        lab_btn = QtWidgets.QPushButton("Lab Preset")
        start_btn = QtWidgets.QPushButton("Start System")
        copy_btn = QtWidgets.QPushButton("Copy Command")
        close_btn = QtWidgets.QPushButton("Close")
        start_btn.setDefault(True)

        hardware_box = self._group_box("Hardware", self._vbox(self.real_radio, self.fake_radio))
        profile_layout = QtWidgets.QFormLayout()
        profile_layout.addRow("Robot", self.robot_profile_combo)
        profile_layout.addRow("Environment", self.environment_combo)
        profile_layout.addRow("Gripper enabled", self.use_gripper_cb)
        profile_layout.addRow("Gripper", self.gripper_profile_combo)
        profile_box = self._group_box("Profiles", profile_layout)
        node_box = self._group_box(
            "Nodes",
            self._vbox(
                self.main_cb,
                self.vision_cb,
                self.teleop_cb,
                self.gui_cb,
                self.calibrate_cb,
            ),
        )

        camera_layout = QtWidgets.QFormLayout()
        camera_layout.addRow("Camera / depth", self.camera_combo)
        camera_box = self._group_box("Vision", camera_layout)

        hand_eye_layout = QtWidgets.QFormLayout()
        hand_eye_layout.addRow("Target", self.target_combo)
        hand_eye_layout.addRow("Resolution", self.res_combo)
        hand_eye_box = self._group_box("Hand-Eye", hand_eye_layout)

        generate_board_btn = QtWidgets.QPushButton("Generate Board")
        start_hand_eye_btn = QtWidgets.QPushButton("Start Hand-Eye")
        start_extrinsic_btn = QtWidgets.QPushButton("Start Extrinsic")
        open_profiles_btn = QtWidgets.QPushButton("Open Profile Folder")
        gripper_calibration_btn = QtWidgets.QPushButton("Gripper Calibration")
        generate_board_btn.setToolTip("Generate the selected ChArUco/chessboard print target")
        start_hand_eye_btn.setToolTip("Launch UR bringup + main/RViz + hand-eye calibration")
        start_extrinsic_btn.setToolTip("Launch ZED One to ZED X Mini extrinsic calibration")
        open_profiles_btn.setToolTip("Open the camera calibration profile YAML folder")
        gripper_calibration_btn.setToolTip(
            "Set NORMAL/ENVELOP open and closed M1-M12 poses separately for Lab and Outdoor")
        calibration_actions_layout = QtWidgets.QGridLayout()
        calibration_actions_layout.addWidget(generate_board_btn, 0, 0)
        calibration_actions_layout.addWidget(start_hand_eye_btn, 0, 1)
        calibration_actions_layout.addWidget(start_extrinsic_btn, 1, 0)
        calibration_actions_layout.addWidget(open_profiles_btn, 1, 1)
        calibration_actions_layout.addWidget(gripper_calibration_btn, 2, 0, 1, 2)
        calibration_actions_box = self._group_box("Calibration Actions", calibration_actions_layout)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(harvest_btn)
        preset_row.addWidget(lab_btn)
        preset_row.addStretch(1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(copy_btn)
        action_row.addStretch(1)
        action_row.addWidget(close_btn)
        action_row.addWidget(start_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(preset_row)
        layout.addWidget(hardware_box)
        layout.addWidget(profile_box)
        layout.addWidget(node_box)
        layout.addWidget(camera_box)
        layout.addWidget(hand_eye_box)
        layout.addWidget(calibration_actions_box)
        layout.addWidget(QtWidgets.QLabel("Command"))
        layout.addWidget(self.command_edit)
        layout.addLayout(action_row)

        for widget in (
            self.real_radio,
            self.fake_radio,
            self.robot_profile_combo,
            self.environment_combo,
            self.gripper_profile_combo,
            self.use_gripper_cb,
            self.main_cb,
            self.vision_cb,
            self.teleop_cb,
            self.gui_cb,
            self.calibrate_cb,
            self.camera_combo,
            self.target_combo,
            self.res_combo,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self.update_command)
            else:
                widget.toggled.connect(self.update_command)

        self.camera_combo.currentIndexChanged.connect(self._sync_camera_choice)
        self.camera_combo.currentIndexChanged.connect(self._sync_hand_eye_resolution)
        self.camera_combo.currentIndexChanged.connect(self._remember_camera_mode)
        self.vision_cb.toggled.connect(self._sync_vision_enabled)
        self.robot_profile_combo.currentIndexChanged.connect(self._remember_profiles)
        self.environment_combo.currentIndexChanged.connect(self._remember_profiles)
        self.gripper_profile_combo.currentIndexChanged.connect(self._on_gripper_profile_changed)
        self.use_gripper_cb.toggled.connect(self._remember_profiles)
        self.use_gripper_cb.toggled.connect(self._sync_gripper_enabled)
        harvest_btn.clicked.connect(self.apply_harvest_preset)
        lab_btn.clicked.connect(self.apply_lab_preset)
        generate_board_btn.clicked.connect(self.generate_board)
        start_hand_eye_btn.clicked.connect(self.start_hand_eye)
        start_extrinsic_btn.clicked.connect(self.start_extrinsic)
        open_profiles_btn.clicked.connect(self.open_profile_folder)
        gripper_calibration_btn.clicked.connect(self.open_gripper_calibration)
        start_btn.clicked.connect(self.start_system)
        copy_btn.clicked.connect(self.copy_command)
        close_btn.clicked.connect(self.reject)

        self._sync_gripper_enabled()
        self._sync_hand_eye_resolution()
        self.update_command()

    @staticmethod
    def _vbox(*widgets):
        layout = QtWidgets.QVBoxLayout()
        for widget in widgets:
            layout.addWidget(widget)
        return layout

    @staticmethod
    def _set_combo_data(combo, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _group_box(title, layout):
        box = QtWidgets.QGroupBox(title)
        box.setLayout(layout)
        return box

    def apply_harvest_preset(self):
        self.real_radio.setChecked(True)
        self._set_combo_data(self.environment_combo, "outdoor")
        self.use_gripper_cb.setChecked(True)
        self.main_cb.setChecked(True)
        self.vision_cb.setChecked(True)
        self.teleop_cb.setChecked(False)
        self.gui_cb.setChecked(False)
        self.calibrate_cb.setChecked(False)
        self.camera_combo.setCurrentIndex(self.camera_combo.findData("zedx_mini"))
        self.update_command()

    def apply_lab_preset(self):
        self.real_radio.setChecked(True)
        self._set_combo_data(self.environment_combo, "lab")
        self.use_gripper_cb.setChecked(True)
        self.main_cb.setChecked(True)
        self.vision_cb.setChecked(True)
        self.teleop_cb.setChecked(False)
        self.gui_cb.setChecked(False)
        self.calibrate_cb.setChecked(False)
        self.camera_combo.setCurrentIndex(self.camera_combo.findData("stereo"))
        self.update_command()

    def _sync_camera_choice(self):
        if self.camera_combo.currentData() == "none":
            self.vision_cb.setChecked(False)
        else:
            self.vision_cb.setChecked(True)
        self.update_command()

    def _sync_vision_enabled(self):
        self.camera_combo.setEnabled(self.vision_cb.isChecked())
        if not self.vision_cb.isChecked():
            self.camera_combo.setCurrentIndex(self.camera_combo.findData("none"))
        elif self.camera_combo.currentData() == "none":
            self.camera_combo.setCurrentIndex(self.camera_combo.findData("zedx_mini"))
        self.update_command()

    def _sync_gripper_enabled(self):
        self.gripper_profile_combo.setEnabled(self.use_gripper_cb.isChecked())
        self.update_command()

    def _on_gripper_profile_changed(self):
        if not self.use_gripper_cb.isChecked():
            self.use_gripper_cb.setChecked(True)
        self._remember_profiles()

    def _populate_resolution_combo(self, mini_only=False):
        current = self.res_combo.currentData() if hasattr(self, "res_combo") else None
        self.res_combo.blockSignals(True)
        self.res_combo.clear()
        if mini_only:
            self.res_combo.addItem("HD1080 hand-eye (ZED X Mini)", "hd1080")
        else:
            self.res_combo.addItem("QHD+ hand-eye", "qhdplus")
            self.res_combo.addItem("4K hand-eye", "4k")
            if current in ("qhdplus", "4k"):
                self._set_combo_data(self.res_combo, current)
        self.res_combo.blockSignals(False)

    def _sync_hand_eye_resolution(self):
        self._populate_resolution_combo(self.camera_combo.currentData() == "zedx_mini")
        self.update_command()

    def _remember_profiles(self):
        self.settings.setValue("robot_profile", self.robot_profile_combo.currentData())
        self.settings.setValue("environment", self.environment_combo.currentData())
        self.settings.setValue("gripper_profile", self.gripper_profile_combo.currentData())
        self.settings.setValue(
            "use_gripper",
            "true" if self.use_gripper_cb.isChecked() else "false",
        )
        self.update_command()

    def _remember_camera_mode(self):
        self.settings.setValue("camera_mode", self.camera_combo.currentData())
        self.update_command()

    def base_args(self):
        args = [
            f"robot_{self.robot_profile_combo.currentData()}",
            self.environment_combo.currentData(),
            f"gripper_{self.gripper_profile_combo.currentData()}",
        ]
        if self.fake_radio.isChecked():
            args.append("fake")
        args.append("gripper_on" if self.use_gripper_cb.isChecked() else "gripper_off")
        return args

    def selected_camera_arg(self):
        camera_mode = self.camera_combo.currentData()
        if camera_mode in ("zed_mini", "zedx_mini", "lidar"):
            return camera_mode
        return None

    def build_args(self):
        args = self.base_args()
        if self.main_cb.isChecked():
            args.append("main")

        camera_arg = self.selected_camera_arg()
        if self.vision_cb.isChecked() and camera_arg is not None:
            args.append("vision")
            args.append(camera_arg)

        if self.teleop_cb.isChecked():
            args.append("teleop")
        if self.gui_cb.isChecked():
            args.append("gui")
        if self.calibrate_cb.isChecked():
            args.append("calibrate")
        return args

    def update_command(self):
        self.command_edit.setText(" ".join(["launch_ur10e"] + self.build_args()))

    def copy_command(self):
        QtWidgets.QApplication.clipboard().setText(self.command_edit.text())

    def launch_args(self, args):
        if not LAUNCHER.exists():
            QtWidgets.QMessageBox.critical(
                self,
                "Launcher Missing",
                f"Could not find {LAUNCHER}",
            )
            return False

        if not args:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing Selected",
                "Select at least one node or preset before launching.",
            )
            return False

        self._remember_profiles()
        self._remember_camera_mode()
        subprocess.Popen([str(LAUNCHER)] + args)
        return True

    def start_system(self):
        if self.launch_args(self.build_args()):
            self.accept()

    def open_gripper_calibration(self):
        dialog = GripperCalibrationDialog(
            self, self.environment_combo.currentData())
        dialog.exec_()

    def generate_board(self):
        target = self.target_combo.currentData()
        filename = (
            "charuco_7x24_30mm_22mm_dict5x5_100.png"
            if target == "charuco"
            else "chessboard_8x11_30mm.png"
        )
        output_path = VISION_DIR / filename
        cmd = (
            f"source {shlex.quote(str(WS / 'install/setup.bash'))} && "
            f"ros2 run ur10e_curobo hand_eye --target {target} "
            f"--generate-board {shlex.quote(str(output_path))}"
        )
        try:
            result = subprocess.run(
                ["bash", "-lc", cmd],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            QtWidgets.QMessageBox.critical(
                self,
                "Board Generation Timeout",
                "The calibration board command did not finish within 30 seconds.",
            )
            return

        if result.returncode != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "Board Generation Failed",
                (result.stderr or result.stdout or "No output").strip(),
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Board Generated",
            f"Saved calibration board:\n{output_path}",
        )
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(output_path)))

    def start_hand_eye(self):
        args = self.base_args()
        camera_arg = self.selected_camera_arg()
        if camera_arg is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Camera Required",
                "Choose a camera mode before starting hand-eye calibration.",
            )
            return
        args.append("main")
        args.append(camera_arg)
        args.extend([
            "hand_eye",
            self.target_combo.currentData(),
            "hd1080" if camera_arg == "zedx_mini" else self.res_combo.currentData(),
        ])
        if self.launch_args(args):
            self.accept()

    def start_extrinsic(self):
        args = self.base_args()
        args.extend(["zed_mini", "extrinsic"])
        if self.launch_args(args):
            self.accept()

    def open_profile_folder(self):
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(PROFILE_DIR)))


def main():
    app = QtWidgets.QApplication(sys.argv)
    dialog = LaunchDialog()
    dialog.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
