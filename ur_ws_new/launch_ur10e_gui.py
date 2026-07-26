#!/usr/bin/env python3
"""Small GUI front end for the UR10e launcher."""
import shlex
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


WS = Path(__file__).resolve().parent
REPO_ROOT = WS.parent
LAUNCHER = REPO_ROOT / "bin" / "launch_ur10e"
VISION_DIR = WS / "src" / "ur10e_curobo" / "ur10e_curobo" / "vision"
PROFILE_DIR = VISION_DIR / "calibration_profiles"


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
        self.robot_profile_combo.addItem("New robot", "new")
        self._set_combo_data(
            self.robot_profile_combo,
            self.settings.value("robot_profile", "old"),
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
        generate_board_btn.setToolTip("Generate the selected ChArUco/chessboard print target")
        start_hand_eye_btn.setToolTip("Launch UR bringup + main/RViz + hand-eye calibration")
        start_extrinsic_btn.setToolTip("Launch ZED One to ZED X Mini extrinsic calibration")
        open_profiles_btn.setToolTip("Open the camera calibration profile YAML folder")
        calibration_actions_layout = QtWidgets.QGridLayout()
        calibration_actions_layout.addWidget(generate_board_btn, 0, 0)
        calibration_actions_layout.addWidget(start_hand_eye_btn, 0, 1)
        calibration_actions_layout.addWidget(start_extrinsic_btn, 1, 0)
        calibration_actions_layout.addWidget(open_profiles_btn, 1, 1)
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
        self.gripper_profile_combo.currentIndexChanged.connect(self._on_gripper_profile_changed)
        self.use_gripper_cb.toggled.connect(self._remember_profiles)
        self.use_gripper_cb.toggled.connect(self._sync_gripper_enabled)
        harvest_btn.clicked.connect(self.apply_harvest_preset)
        lab_btn.clicked.connect(self.apply_lab_preset)
        generate_board_btn.clicked.connect(self.generate_board)
        start_hand_eye_btn.clicked.connect(self.start_hand_eye)
        start_extrinsic_btn.clicked.connect(self.start_extrinsic)
        open_profiles_btn.clicked.connect(self.open_profile_folder)
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
