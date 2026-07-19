#!/usr/bin/env python3
"""Small GUI front end for the UR10e launcher."""
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


WS = Path(__file__).resolve().parent
REPO_ROOT = WS.parent
LAUNCHER = REPO_ROOT / "bin" / "launch_ur10e"


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

        self.main_cb = QtWidgets.QCheckBox("Main cuRobo control + RViz panel")
        self.vision_cb = QtWidgets.QCheckBox("Vision")
        self.teleop_cb = QtWidgets.QCheckBox("Teleop")
        self.gui_cb = QtWidgets.QCheckBox("Desktop GUI")
        self.calibrate_cb = QtWidgets.QCheckBox("Grasp force calibration")
        self.hand_eye_cb = QtWidgets.QCheckBox("Hand-eye calibration")
        self.main_cb.setChecked(True)
        self.vision_cb.setChecked(True)

        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItem("ZED X One + ZED Mini depth", "zed_mini")
        self.camera_combo.addItem("ZED stereo only", "stereo")
        self.camera_combo.addItem("ZED X One + Livox LiDAR depth", "lidar")
        self.camera_combo.addItem("No vision", "none")

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("ChArUco", "charuco")
        self.target_combo.addItem("Chessboard", "chessboard")

        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItem("QHD+ hand-eye", "qhdplus")
        self.res_combo.addItem("4K hand-eye", "4k")

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
                self.hand_eye_cb,
            ),
        )

        camera_layout = QtWidgets.QFormLayout()
        camera_layout.addRow("Camera / depth", self.camera_combo)
        camera_box = self._group_box("Vision", camera_layout)

        hand_eye_layout = QtWidgets.QFormLayout()
        hand_eye_layout.addRow("Target", self.target_combo)
        hand_eye_layout.addRow("Resolution", self.res_combo)
        hand_eye_box = self._group_box("Hand-Eye", hand_eye_layout)

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
        layout.addWidget(QtWidgets.QLabel("Command"))
        layout.addWidget(self.command_edit)
        layout.addLayout(action_row)

        for widget in (
            self.real_radio,
            self.fake_radio,
            self.robot_profile_combo,
            self.gripper_profile_combo,
            self.main_cb,
            self.vision_cb,
            self.teleop_cb,
            self.gui_cb,
            self.calibrate_cb,
            self.hand_eye_cb,
            self.camera_combo,
            self.target_combo,
            self.res_combo,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self.update_command)
            else:
                widget.toggled.connect(self.update_command)

        self.camera_combo.currentIndexChanged.connect(self._sync_camera_choice)
        self.vision_cb.toggled.connect(self._sync_vision_enabled)
        self.hand_eye_cb.toggled.connect(self._sync_hand_eye_enabled)
        self.robot_profile_combo.currentIndexChanged.connect(self._remember_profiles)
        self.gripper_profile_combo.currentIndexChanged.connect(self._remember_profiles)
        harvest_btn.clicked.connect(self.apply_harvest_preset)
        lab_btn.clicked.connect(self.apply_lab_preset)
        start_btn.clicked.connect(self.start_system)
        copy_btn.clicked.connect(self.copy_command)
        close_btn.clicked.connect(self.reject)

        self._sync_hand_eye_enabled()
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
        self.main_cb.setChecked(True)
        self.vision_cb.setChecked(True)
        self.teleop_cb.setChecked(False)
        self.gui_cb.setChecked(False)
        self.calibrate_cb.setChecked(False)
        self.hand_eye_cb.setChecked(False)
        self.camera_combo.setCurrentIndex(self.camera_combo.findData("zed_mini"))
        self.update_command()

    def apply_lab_preset(self):
        self.real_radio.setChecked(True)
        self.main_cb.setChecked(True)
        self.vision_cb.setChecked(True)
        self.teleop_cb.setChecked(False)
        self.gui_cb.setChecked(False)
        self.calibrate_cb.setChecked(False)
        self.hand_eye_cb.setChecked(False)
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
            self.camera_combo.setCurrentIndex(self.camera_combo.findData("zed_mini"))
        self.update_command()

    def _sync_hand_eye_enabled(self):
        enabled = self.hand_eye_cb.isChecked()
        self.target_combo.setEnabled(enabled)
        self.res_combo.setEnabled(enabled)
        self.update_command()

    def _remember_profiles(self):
        self.settings.setValue("robot_profile", self.robot_profile_combo.currentData())
        self.settings.setValue("gripper_profile", self.gripper_profile_combo.currentData())
        self.update_command()

    def build_args(self):
        args = []
        args.append(f"robot_{self.robot_profile_combo.currentData()}")
        args.append(f"gripper_{self.gripper_profile_combo.currentData()}")
        if self.fake_radio.isChecked():
            args.append("fake")
        if self.main_cb.isChecked():
            args.append("main")

        camera_mode = self.camera_combo.currentData()
        if self.vision_cb.isChecked() and camera_mode != "none":
            args.append("vision")
            if camera_mode in ("zed_mini", "lidar"):
                args.append(camera_mode)

        if self.teleop_cb.isChecked():
            args.append("teleop")
        if self.gui_cb.isChecked():
            args.append("gui")
        if self.calibrate_cb.isChecked():
            args.append("calibrate")
        if self.hand_eye_cb.isChecked():
            args.append("hand_eye")
            args.append(self.target_combo.currentData())
            args.append(self.res_combo.currentData())
        return args

    def update_command(self):
        self.command_edit.setText(" ".join(["launch_ur10e"] + self.build_args()))

    def copy_command(self):
        QtWidgets.QApplication.clipboard().setText(self.command_edit.text())

    def start_system(self):
        if not LAUNCHER.exists():
            QtWidgets.QMessageBox.critical(
                self,
                "Launcher Missing",
                f"Could not find {LAUNCHER}",
            )
            return

        args = self.build_args()
        if not args:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing Selected",
                "Select at least one node or preset before launching.",
            )
            return

        self._remember_profiles()
        subprocess.Popen([str(LAUNCHER)] + args)
        self.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    dialog = LaunchDialog()
    dialog.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
