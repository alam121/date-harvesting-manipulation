#include "rviz_ur10e_panel/ur10e_panel.hpp"

#include <QGroupBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QTabWidget>
#include <QFont>
#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unistd.h>
#include <sys/types.h>

#include <rclcpp/qos.hpp>
#include <rviz_common/display_context.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace rviz_ur10e_panel
{

namespace
{

constexpr std::array<const char *, 6> kJointNames = {
  "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
};

int panelJointIndex(const std::string & name)
{
  for (size_t i = 0; i < kJointNames.size(); ++i) {
    const std::string expected = kJointNames[i];
    if (name == expected ||
      (name.size() > expected.size() &&
      name.compare(name.size() - expected.size(), expected.size(), expected) == 0))
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

double metricScore(double value, double excellent, double poor)
{
  if (!std::isfinite(value)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (value <= excellent) {
    return 100.0;
  }
  if (value >= poor) {
    return 0.0;
  }
  return 100.0 * (poor - value) / (poor - excellent);
}

const char * ratingName(double score)
{
  if (score >= 90.0) return "Excellent";
  if (score >= 75.0) return "Good";
  if (score >= 60.0) return "Fair";
  if (score >= 40.0) return "Poor";
  return "Unstable";
}

std::string unescapeJsonString(const std::string & s)
{
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\' || i + 1 >= s.size()) {
      out.push_back(s[i]);
      continue;
    }
    const char c = s[++i];
    if (c == 'n') out.push_back('\n');
    else if (c == 't') out.push_back('\t');
    else out.push_back(c);
  }
  return out;
}

std::string jsonStringValue(const std::string & data, const char * key)
{
  const std::string needle = std::string("\"") + key + "\"";
  auto pos = data.find(needle);
  if (pos == std::string::npos) return "";
  auto colon = data.find(':', pos);
  auto q1 = data.find('"', colon + 1);
  if (colon == std::string::npos || q1 == std::string::npos) return "";
  std::string raw;
  bool escaped = false;
  for (size_t i = q1 + 1; i < data.size(); ++i) {
    const char c = data[i];
    if (escaped) {
      raw.push_back('\\');
      raw.push_back(c);
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == '"') {
      return unescapeJsonString(raw);
    } else {
      raw.push_back(c);
    }
  }
  return "";
}

}  // namespace

UR10ePanel::UR10ePanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * scroll = new QScrollArea(this);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

  auto * container = new QWidget();
  auto * layout = new QVBoxLayout(container);
  layout->setSpacing(6);
  layout->setContentsMargins(6, 6, 6, 6);

  // Status
  status_label_ = new QLabel("Ready");
  status_label_->setStyleSheet(
    "color: #2d6a4f; font-weight: bold; padding: 4px; "
    "background: #e8f5e9; border-radius: 4px;");
  status_label_->setWordWrap(true);
  layout->addWidget(status_label_);

  // Robot state
  robot_state_label_ = new QLabel("Program: Unknown");
  robot_state_label_->setStyleSheet("font-size: 10pt; padding: 2px;");
  layout->addWidget(robot_state_label_);

  // Active robot type + environment (from Python config, via /robot_config_info)
  config_label_ = new QLabel("Robot: \xe2\x80\x94  |  Env: \xe2\x80\x94");
  config_label_->setStyleSheet(
    "font-size: 10pt; font-weight: bold; padding: 2px; color: #1565c0;");
  layout->addWidget(config_label_);

  // Motion phase + reacquire result (side by side)
  auto * phase_row = new QHBoxLayout();

  auto * phase_group = new QGroupBox("Motion Phase");
  auto * phase_inner = new QVBoxLayout(phase_group);
  phase_label_ = new QLabel("IDLE");
  phase_label_->setAlignment(Qt::AlignCenter);
  phase_label_->setFont(QFont("Helvetica", 13, QFont::Bold));
  phase_label_->setStyleSheet(
    "background: #455a64; color: #eceff1; padding: 6px; border-radius: 6px;");
  phase_inner->addWidget(phase_label_);
  phase_row->addWidget(phase_group, 3);

  auto * reacq_group = new QGroupBox("Reacquire");
  auto * reacq_inner = new QVBoxLayout(reacq_group);
  reacq_label_ = new QLabel("\xe2\x80\x94");  // em dash
  reacq_label_->setAlignment(Qt::AlignCenter);
  reacq_label_->setFont(QFont("Helvetica", 13, QFont::Bold));
  reacq_label_->setStyleSheet(
    "background: #455a64; color: #eceff1; padding: 6px; border-radius: 6px;");
  reacq_inner->addWidget(reacq_label_);
  phase_row->addWidget(reacq_group, 2);

  layout->addLayout(phase_row);

  // Tabbed controls to avoid a long scroll. Status + Motion Phase stay above the tabs
  // and EMERGENCY STOP stays just below them, so both are visible from every tab.
  auto * tabs = new QTabWidget();
  auto * motion_tab = new QWidget();
  auto * motion_tab_layout = new QVBoxLayout(motion_tab);
  motion_tab_layout->setSpacing(6);
  auto * goal_tab = new QWidget();
  auto * goal_tab_layout = new QVBoxLayout(goal_tab);
  goal_tab_layout->setSpacing(6);
  auto * monitor_tab = new QWidget();
  auto * monitor_tab_layout = new QVBoxLayout(monitor_tab);
  monitor_tab_layout->setSpacing(6);
  auto * settings_tab = new QWidget();
  auto * settings_tab_layout = new QVBoxLayout(settings_tab);
  settings_tab_layout->setSpacing(6);
  auto * heat_tab = new QWidget();
  auto * heat_tab_layout = new QVBoxLayout(heat_tab);
  heat_tab_layout->setSpacing(6);
  auto * camera_tab = new QWidget();
  auto * camera_tab_layout = new QVBoxLayout(camera_tab);
  camera_tab_layout->setSpacing(6);
  tabs->addTab(motion_tab, "Motion");
  tabs->addTab(goal_tab, "Goal");
  tabs->addTab(monitor_tab, "Monitor");
  tabs->addTab(camera_tab, "Camera");
  tabs->addTab(heat_tab, "Heat");
  tabs->addTab(settings_tab, "Settings");

  // Last harvest result banner + session tally
  auto * harvest_group = new QGroupBox("Last Harvest Result");
  auto * harvest_vlayout = new QVBoxLayout(harvest_group);

  harvest_banner_ = new QLabel("\xe2\x80\x94");
  harvest_banner_->setAlignment(Qt::AlignCenter);
  harvest_banner_->setFont(QFont("Helvetica", 18, QFont::Bold));
  harvest_banner_->setFixedHeight(48);
  harvest_banner_->setStyleSheet(
    "background: #455a64; color: #eceff1; border-radius: 8px;");
  harvest_vlayout->addWidget(harvest_banner_);

  auto * tally_row = new QHBoxLayout();
  struct TallyDef { QLabel ** lbl; const char * name; const char * color; };
  TallyDef tallies[] = {
    {&grab_count_label_, "Grabbed", "#4caf50"},
    {&slip_count_label_, "Slipped", "#ff9800"},
    {&miss_count_label_, "Miss",    "#f44336"},
    {&total_count_label_,"Total",   "#90a4ae"},
  };
  for (auto & t : tallies) {
    auto * cell = new QWidget();
    auto * cl = new QVBoxLayout(cell);
    cl->setSpacing(1);
    cl->setContentsMargins(2, 2, 2, 2);
    *t.lbl = new QLabel("0");
    (*t.lbl)->setAlignment(Qt::AlignCenter);
    (*t.lbl)->setFont(QFont("Courier", 12, QFont::Bold));
    (*t.lbl)->setStyleSheet(QString("color: %1;").arg(t.color));
    cl->addWidget(*t.lbl);
    auto * name_lbl = new QLabel(t.name);
    name_lbl->setAlignment(Qt::AlignCenter);
    name_lbl->setStyleSheet("font-size: 8pt; color: #666;");
    cl->addWidget(name_lbl);
    tally_row->addWidget(cell);
  }
  harvest_vlayout->addLayout(tally_row);
  monitor_tab_layout->addWidget(harvest_group);

  // Emergency Stop
  auto * stop_btn = new QPushButton("EMERGENCY STOP");
  stop_btn->setStyleSheet(
    "background-color: #d32f2f; color: white; font-weight: bold; "
    "font-size: 11pt; padding: 10px;");
  connect(stop_btn, &QPushButton::clicked, this, &UR10ePanel::onStop);
  layout->addWidget(stop_btn);

  layout->addWidget(tabs, 1);

  // Motion Commands
  auto * motion_group = new QGroupBox("Motion");
  auto * motion_layout = new QGridLayout(motion_group);
  motion_layout->setSpacing(4);

  auto * home_btn = new QPushButton("Home");
  home_btn->setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;");
  connect(home_btn, &QPushButton::clicked, this, &UR10ePanel::onHome);
  motion_layout->addWidget(home_btn, 0, 0);

  auto * drop_btn = new QPushButton("Dropoff");
  drop_btn->setStyleSheet("background-color: #2196f3; color: white; font-weight: bold;");
  connect(drop_btn, &QPushButton::clicked, this, &UR10ePanel::onDropoff);
  motion_layout->addWidget(drop_btn, 0, 1);

  auto * set_home_btn = new QPushButton("Set Home = Current");
  set_home_btn->setStyleSheet("background-color: #00897b; color: white; font-weight: bold;");
  set_home_btn->setToolTip("Save the current robot joint position as the active HOME preset");
  connect(set_home_btn, &QPushButton::clicked, this, &UR10ePanel::onSetHomeCurrent);
  motion_layout->addWidget(set_home_btn, 1, 0);

  auto * set_dropoff_btn = new QPushButton("Set Dropoff = Current");
  set_dropoff_btn->setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;");
  set_dropoff_btn->setToolTip(
    "Save the current robot joint position as the active DROPOFF preset");
  connect(set_dropoff_btn, &QPushButton::clicked, this, &UR10ePanel::onSetDropoffCurrent);
  motion_layout->addWidget(set_dropoff_btn, 1, 1);

  auto * exec_btn = new QPushButton("Execute");
  exec_btn->setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;");
  connect(exec_btn, &QPushButton::clicked, this, &UR10ePanel::onExecute);
  motion_layout->addWidget(exec_btn, 2, 0);

  auto * clear_btn = new QPushButton("Clear");
  clear_btn->setStyleSheet("background-color: #9e9e9e; color: white;");
  connect(clear_btn, &QPushButton::clicked, this, &UR10ePanel::onClear);
  motion_layout->addWidget(clear_btn, 2, 1);

  auto * check_calib_btn = new QPushButton("Check Calibration (trunk)");
  check_calib_btn->setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;");
  check_calib_btn->setToolTip("Check hand-eye calibration using the detected trunk as the reference");
  connect(check_calib_btn, &QPushButton::clicked, this, &UR10ePanel::onCheckCalibration);
  motion_layout->addWidget(check_calib_btn, 3, 0, 1, 2);

  auto * sub_btn = new QPushButton("Subscribe (S)");
  sub_btn->setStyleSheet("background-color: #7b1fa2; color: white; font-weight: bold;");
  connect(sub_btn, &QPushButton::clicked, this, &UR10ePanel::onSubscribe);
  motion_layout->addWidget(sub_btn, 4, 0);

  auto * sub_multi_btn = new QPushButton("Sub Multi (M)");
  sub_multi_btn->setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;");
  connect(sub_multi_btn, &QPushButton::clicked, this, &UR10ePanel::onSubscribeMulti);
  motion_layout->addWidget(sub_multi_btn, 4, 1);

  auto * multi_goal_label = new QLabel("Multi goals");
  multi_goal_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  multi_goal_count_spin_ = new QSpinBox();
  multi_goal_count_spin_->setRange(1, 10);
  multi_goal_count_spin_->setValue(3);
  multi_goal_count_spin_->setToolTip("Number of top-scored goals to queue with Sub Multi");
  motion_layout->addWidget(multi_goal_label, 5, 0);
  motion_layout->addWidget(multi_goal_count_spin_, 5, 1);

  calib_result_label_ = new QLabel("—");
  calib_result_label_->setWordWrap(true);
  calib_result_label_->setStyleSheet(
    "font-size: 9pt; padding: 3px; background: #f3e5f5; border-radius: 4px;");
  motion_layout->addWidget(calib_result_label_, 6, 0, 1, 2);

  debug_preview_cb_ = new QCheckBox("Debug Plan Preview");
  debug_preview_cb_->setChecked(true);
  debug_preview_cb_->setStyleSheet("font-weight: bold; font-size: 10pt; padding: 4px;");
  debug_preview_cb_->setToolTip("Show full plan in RViz before executing");
  connect(debug_preview_cb_, &QCheckBox::stateChanged, this, &UR10ePanel::onDebugPreviewChanged);
  motion_layout->addWidget(debug_preview_cb_, 7, 0, 1, 2);

  reachability_cloud_cb_ = new QCheckBox("Reachability Cloud");
  reachability_cloud_cb_->setChecked(true);
  reachability_cloud_cb_->setStyleSheet("font-weight: bold; font-size: 10pt; padding: 4px;");
  reachability_cloud_cb_->setToolTip(
    "Show/hide the sampled green/yellow reachability cloud and LiDAR scan preview in RViz");
  connect(
    reachability_cloud_cb_, &QCheckBox::stateChanged,
    this, &UR10ePanel::onReachabilityCloudChanged);
  motion_layout->addWidget(reachability_cloud_cb_, 8, 0, 1, 2);

  zone_overlay_btn_ = new QPushButton("Zone Overlay: OFF");
  zone_overlay_btn_->setCheckable(true);
  zone_overlay_btn_->setStyleSheet("background-color: #607d8b; color: white; font-weight: bold;");
  zone_overlay_btn_->setToolTip("Show/hide date side-classification zones on the vision display");
  connect(zone_overlay_btn_, &QPushButton::clicked, this, &UR10ePanel::onZoneOverlayToggle);
  motion_layout->addWidget(zone_overlay_btn_, 9, 0, 1, 2);

  plan_confirm_btn_ = new QPushButton("Confirm Plan");
  plan_confirm_btn_->setStyleSheet(
    "background-color: #4caf50; color: white; font-weight: bold; "
    "font-size: 11pt; padding: 8px;");
  connect(plan_confirm_btn_, &QPushButton::clicked, this, &UR10ePanel::onPlanConfirm);
  plan_confirm_btn_->setVisible(false);
  motion_layout->addWidget(plan_confirm_btn_, 10, 0);

  plan_cancel_btn_ = new QPushButton("Cancel Plan");
  plan_cancel_btn_->setStyleSheet(
    "background-color: #d32f2f; color: white; font-weight: bold; "
    "font-size: 11pt; padding: 8px;");
  connect(plan_cancel_btn_, &QPushButton::clicked, this, &UR10ePanel::onPlanCancel);
  plan_cancel_btn_->setVisible(false);
  motion_layout->addWidget(plan_cancel_btn_, 10, 1);

  motion_tab_layout->addWidget(motion_group);

  // Side-home moves — test/calibrate home_left/home_right per robot profile.
  // Published to /ui_command and handled node-side (run_side_home).
  auto * side_home_row = new QHBoxLayout();
  auto * home_left_btn = new QPushButton("Home Left");
  home_left_btn->setStyleSheet("background-color: #00897b; color: white; font-weight: bold;");
  home_left_btn->setToolTip("Move to the stored home_left joint config (active robot profile)");
  connect(home_left_btn, &QPushButton::clicked, this, &UR10ePanel::onHomeLeft);
  side_home_row->addWidget(home_left_btn);
  auto * home_right_btn = new QPushButton("Home Right");
  home_right_btn->setStyleSheet("background-color: #00897b; color: white; font-weight: bold;");
  home_right_btn->setToolTip("Move to the stored home_right joint config (active robot profile)");
  connect(home_right_btn, &QPushButton::clicked, this, &UR10ePanel::onHomeRight);
  side_home_row->addWidget(home_right_btn);
  motion_tab_layout->addLayout(side_home_row);

  // Gripper
  auto * gripper_group = new QGroupBox("Gripper");
  auto * gripper_layout = new QHBoxLayout(gripper_group);

  auto * open_btn = new QPushButton("Open");
  connect(open_btn, &QPushButton::clicked, this, &UR10ePanel::onGripperOpen);
  gripper_layout->addWidget(open_btn);

  auto * close_btn = new QPushButton("Close");
  connect(close_btn, &QPushButton::clicked, this, &UR10ePanel::onGripperClose);
  gripper_layout->addWidget(close_btn);

  motion_tab_layout->addWidget(gripper_group);

  // Velocity Scale
  auto * vel_group = new QGroupBox("Velocity Scale");
  auto * vel_layout = new QVBoxLayout(vel_group);
  vel_layout->setSpacing(4);

  auto * slider_row = new QHBoxLayout();
  velocity_slider_ = new QSlider(Qt::Horizontal);
  velocity_slider_->setRange(10, 100);
  velocity_slider_->setValue(50);
  connect(velocity_slider_, &QSlider::valueChanged, this, &UR10ePanel::onVelocitySliderChanged);
  slider_row->addWidget(velocity_slider_);

  velocity_value_label_ = new QLabel("5.0x");
  velocity_value_label_->setFont(QFont("Courier", 10, QFont::Bold));
  velocity_value_label_->setStyleSheet("color: #1976d2;");
  slider_row->addWidget(velocity_value_label_);
  vel_layout->addLayout(slider_row);

  auto * preset_row = new QHBoxLayout();
  struct Preset { const char * name; double value; };
  Preset presets[] = {{"Slow", 1.0}, {"Med", 3.0}, {"Fast", 5.0}, {"Max", 8.0}};
  for (auto & p : presets) {
    auto * btn = new QPushButton(p.name);
    btn->setMaximumHeight(24);
    btn->setProperty("preset_value", p.value);
    connect(btn, &QPushButton::clicked, this, &UR10ePanel::onVelocityPreset);
    preset_row->addWidget(btn);
  }
  vel_layout->addLayout(preset_row);

  auto * apply_btn = new QPushButton("Apply");
  apply_btn->setStyleSheet("background-color: #1976d2; color: white; font-weight: bold;");
  connect(apply_btn, &QPushButton::clicked, this, &UR10ePanel::onApplyVelocity);
  vel_layout->addWidget(apply_btn);

  settings_tab_layout->addWidget(vel_group);

  // Manual Goal
  auto * goal_group = new QGroupBox("Manual Goal");
  auto * goal_layout = new QGridLayout(goal_group);
  goal_layout->setSpacing(4);

  x_in_ = new QLineEdit("0.30");  x_in_->setMaximumWidth(70);
  y_in_ = new QLineEdit("0.00");  y_in_->setMaximumWidth(70);
  z_in_ = new QLineEdit("0.20");  z_in_->setMaximumWidth(70);
  qw_in_ = new QLineEdit("1.0");  qw_in_->setMaximumWidth(55);
  qx_in_ = new QLineEdit("0.0");  qx_in_->setMaximumWidth(55);
  qy_in_ = new QLineEdit("0.0");  qy_in_->setMaximumWidth(55);
  qz_in_ = new QLineEdit("0.0");  qz_in_->setMaximumWidth(55);

  goal_layout->addWidget(new QLabel("X"), 0, 0); goal_layout->addWidget(x_in_, 0, 1);
  goal_layout->addWidget(new QLabel("Y"), 1, 0); goal_layout->addWidget(y_in_, 1, 1);
  goal_layout->addWidget(new QLabel("Z"), 2, 0); goal_layout->addWidget(z_in_, 2, 1);
  goal_layout->addWidget(new QLabel("qw"), 0, 2); goal_layout->addWidget(qw_in_, 0, 3);
  goal_layout->addWidget(new QLabel("qx"), 1, 2); goal_layout->addWidget(qx_in_, 1, 3);
  goal_layout->addWidget(new QLabel("qy"), 2, 2); goal_layout->addWidget(qy_in_, 2, 3);
  goal_layout->addWidget(new QLabel("qz"), 3, 2); goal_layout->addWidget(qz_in_, 3, 3);

  auto * send_btn = new QPushButton("Move Directly");
  send_btn->setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;");
  send_btn->setToolTip("Plan once from the current robot pose directly to this manual goal");
  connect(send_btn, &QPushButton::clicked, this, &UR10ePanel::onSendGoal);
  goal_layout->addWidget(send_btn, 4, 0, 1, 4);

  goal_tab_layout->addWidget(goal_group);

  // Safe-zone controls. These mirror the interactive-marker right-click menu so the
  // operator can control the box even when RViz marker interaction is awkward.
  auto * safe_zone_group = new QGroupBox("Safe Zone");
  auto * safe_zone_layout = new QGridLayout(safe_zone_group);
  safe_zone_layout->setSpacing(4);

  auto * safe_enable_btn = new QPushButton("Enable");
  safe_enable_btn->setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;");
  safe_enable_btn->setToolTip("Enable safe-zone enforcement using the visible box");
  connect(safe_enable_btn, &QPushButton::clicked, this, [this]() {
    publishCmd("safe_zone_enable");
  });
  safe_zone_layout->addWidget(safe_enable_btn, 0, 0);

  auto * safe_disable_btn = new QPushButton("Disable");
  safe_disable_btn->setStyleSheet("background-color: #757575; color: white; font-weight: bold;");
  safe_disable_btn->setToolTip("Disable safe-zone enforcement");
  connect(safe_disable_btn, &QPushButton::clicked, this, [this]() {
    publishCmd("safe_zone_disable");
  });
  safe_zone_layout->addWidget(safe_disable_btn, 0, 1);

  auto * safe_snap_btn = new QPushButton("Snap TCP");
  safe_snap_btn->setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;");
  safe_snap_btn->setToolTip("Center a compact safe zone around the current TCP");
  connect(safe_snap_btn, &QPushButton::clicked, this, [this]() {
    publishCmd("safe_zone_snap");
  });
  safe_zone_layout->addWidget(safe_snap_btn, 1, 0);

  auto * safe_snap_deep_btn = new QPushButton("Snap Outdoor");
  safe_snap_deep_btn->setStyleSheet("background-color: #ef6c00; color: white; font-weight: bold;");
  safe_snap_deep_btn->setToolTip("Build a deeper outdoor safe zone around the current TCP");
  connect(safe_snap_deep_btn, &QPushButton::clicked, this, [this]() {
    publishCmd("safe_zone_snap_deep");
  });
  safe_zone_layout->addWidget(safe_snap_deep_btn, 1, 1);

  goal_tab_layout->addWidget(safe_zone_group);

  // Capture
  auto * capture_group = new QGroupBox("Capture");
  auto * capture_layout = new QHBoxLayout(capture_group);

  auto * cap_btn = new QPushButton("Start (10s)");
  connect(cap_btn, &QPushButton::clicked, this, &UR10ePanel::onCapture);
  capture_layout->addWidget(cap_btn);

  auto * cap_stop = new QPushButton("Stop");
  connect(cap_stop, &QPushButton::clicked, this, &UR10ePanel::onCaptureStop);
  capture_layout->addWidget(cap_stop);

  goal_tab_layout->addWidget(capture_group);

  // Stability recorder
  auto * stability_group = new QGroupBox("Robot Stability / Tracking");
  auto * stability_layout = new QVBoxLayout(stability_group);

  stability_record_btn_ = new QPushButton("Record Stability / Tracking");
  stability_record_btn_->setStyleSheet(
    "background-color: #3949ab; color: white; font-weight: bold; "
    "font-size: 10pt; padding: 7px;");
  stability_record_btn_->setToolTip(
    "Record joint motion, trajectory tracking error, and TCP wrench.\n"
    "Moving samples are rated for trajectory tracking; settled samples are "
    "rated for stability.\n"
    "Press again to stop, calculate a rating, and save the CSV.");
  connect(
    stability_record_btn_, &QPushButton::clicked,
    this, &UR10ePanel::onStabilityRecord);
  stability_layout->addWidget(stability_record_btn_);

  stability_result_label_ = new QLabel("Not recorded");
  stability_result_label_->setWordWrap(true);
  stability_result_label_->setStyleSheet(
    "font-size: 9pt; padding: 5px; background: #e8eaf6; "
    "color: #283593; border-radius: 4px;");
  stability_layout->addWidget(stability_result_label_);

  monitor_tab_layout->addWidget(stability_group);

  // Lidar Scan
  auto * lidar_group = new QGroupBox("Lidar Scan");
  auto * lidar_layout = new QVBoxLayout(lidar_group);
  auto * lidar_scan_btn = new QPushButton("Lidar Scan");
  lidar_scan_btn->setStyleSheet(
    "background-color: #00796b; color: white; font-weight: bold; "
    "font-size: 10pt; padding: 6px;");
  lidar_scan_btn->setToolTip(
    "Sweep arm around tree (half-circle) and record /livox/lidar + /livox/imu to bag.\n"
    "Bag saved to ~/lidar_scans/. Waypoints configured in config.py LidarScan.");
  connect(lidar_scan_btn, &QPushButton::clicked, this, &UR10ePanel::onLidarScan);
  lidar_layout->addWidget(lidar_scan_btn);

  lidar_scan_preview_cb_ = new QCheckBox("LiDAR Scan Preview");
  lidar_scan_preview_cb_->setChecked(true);
  lidar_scan_preview_cb_->setStyleSheet("font-weight: bold; font-size: 10pt; padding: 4px;");
  lidar_scan_preview_cb_->setToolTip("Show/hide the S1..S7 horizontal scan arc preview in RViz");
  connect(
    lidar_scan_preview_cb_, &QCheckBox::stateChanged,
    this, &UR10ePanel::onLidarScanPreviewChanged);
  lidar_layout->addWidget(lidar_scan_preview_cb_);

  goal_tab_layout->addWidget(lidar_group);

  // Camera recording
  auto * camera_group = new QGroupBox("Camera Feed Recording");
  auto * camera_layout = new QGridLayout(camera_group);
  camera_layout->setSpacing(6);

  auto * preset_label = new QLabel("Exposure Preset");
  preset_label->setStyleSheet("font-weight: bold; color: #37474f;");
  preset_label->setAlignment(Qt::AlignCenter);
  camera_layout->addWidget(preset_label, 0, 0, 1, 2);

  auto * lab_preset_btn = new QPushButton("Lab");
  lab_preset_btn->setStyleSheet("background-color: #455a64; color: white; font-weight: bold;");
  lab_preset_btn->setToolTip("Use lab camera settings: auto exposure / auto gain");
  connect(lab_preset_btn, &QPushButton::clicked, this, &UR10ePanel::onCameraPresetLab);
  camera_layout->addWidget(lab_preset_btn, 1, 0);

  auto * outdoor_preset_btn = new QPushButton("Outdoor");
  outdoor_preset_btn->setStyleSheet("background-color: #ef6c00; color: white; font-weight: bold;");
  outdoor_preset_btn->setToolTip("Use outdoor camera settings: manual low exposure and low gain");
  connect(outdoor_preset_btn, &QPushButton::clicked, this, &UR10ePanel::onCameraPresetOutdoor);
  camera_layout->addWidget(outdoor_preset_btn, 1, 1);

  auto * auto_exposure_cb = new QCheckBox("Auto Exposure");
  auto_exposure_cb->setChecked(true);
  auto_exposure_cb->setToolTip("When checked, the ZED controls exposure/gain automatically");
  camera_layout->addWidget(auto_exposure_cb, 2, 0, 1, 2);

  auto * hdr_cb = new QCheckBox("HDR");
  hdr_cb->setChecked(true);
  hdr_cb->setToolTip("Toggle ZED HDR. Some cameras apply this only after camera refresh.");
  camera_layout->addWidget(hdr_cb, 3, 0, 1, 2);

  auto * exposure_label = new QLabel("Exposure");
  exposure_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  camera_layout->addWidget(exposure_label, 4, 0);

  auto * exposure_spin = new QSpinBox();
  exposure_spin->setRange(0, 100);
  exposure_spin->setValue(8);
  exposure_spin->setEnabled(false);
  exposure_spin->setToolTip("Manual exposure, 0-100. Try 5-12 outdoors.");
  camera_layout->addWidget(exposure_spin, 4, 1);

  auto * gain_label = new QLabel("Gain");
  gain_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  camera_layout->addWidget(gain_label, 5, 0);

  auto * gain_spin = new QSpinBox();
  gain_spin->setRange(0, 100);
  gain_spin->setValue(0);
  gain_spin->setEnabled(false);
  gain_spin->setToolTip("Manual gain, 0-100. Keep low outdoors to avoid noise.");
  camera_layout->addWidget(gain_spin, 5, 1);

  connect(auto_exposure_cb, &QCheckBox::toggled, this,
    [exposure_spin, gain_spin](bool checked) {
      exposure_spin->setEnabled(!checked);
      gain_spin->setEnabled(!checked);
    });

  auto * apply_camera_settings_btn = new QPushButton("Apply Exposure");
  apply_camera_settings_btn->setStyleSheet(
    "background-color: #00897b; color: white; font-weight: bold;");
  apply_camera_settings_btn->setToolTip("Apply the exposure controls to the live ZED camera");
  connect(apply_camera_settings_btn, &QPushButton::clicked, this,
    [this, auto_exposure_cb, hdr_cb, exposure_spin, gain_spin]() {
      std::ostringstream cmd;
      cmd << "camera_settings auto=" << (auto_exposure_cb->isChecked() ? 1 : 0)
          << " hdr=" << (hdr_cb->isChecked() ? 1 : 0)
          << " exposure=" << exposure_spin->value()
          << " gain=" << gain_spin->value();
      publishCmd(cmd.str());
    });
  camera_layout->addWidget(apply_camera_settings_btn, 6, 0, 1, 2);

  auto * model_label = new QLabel("YOLO Model");
  model_label->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
  camera_layout->addWidget(model_label, 7, 0);

  camera_model_combo_ = new QComboBox();
  camera_model_combo_->setToolTip("Models found in ur_ws_new/src/zed_date_detector/models");
  const QDir model_dir("/home/datepalm2/manipulatorsdatepalm/ur_ws_new/src/zed_date_detector/models");
  const QStringList filters = {"*.engine", "*.pt", "*.onnx"};
  const auto files = model_dir.entryInfoList(filters, QDir::Files, QDir::Name);
  for (const QFileInfo & file : files) {
    camera_model_combo_->addItem(file.fileName(), file.absoluteFilePath());
  }
  if (camera_model_combo_->count() == 0) {
    camera_model_combo_->addItem("No models found", "");
    camera_model_combo_->setEnabled(false);
  }
  camera_layout->addWidget(camera_model_combo_, 7, 1);

  auto * apply_model_btn = new QPushButton("Apply Model");
  apply_model_btn->setStyleSheet("background-color: #5e35b1; color: white; font-weight: bold;");
  apply_model_btn->setToolTip("Restart vision with the selected model from the models folder");
  connect(apply_model_btn, &QPushButton::clicked, this,
    [this]() {
      if (!camera_model_combo_ || camera_model_combo_->count() == 0) {
        return;
      }
      const QString model_path = camera_model_combo_->currentData().toString();
      if (model_path.isEmpty()) {
        return;
      }
      publishCmd(std::string("camera_model path=") + model_path.toStdString());
    });
  camera_layout->addWidget(apply_model_btn, 8, 0, 1, 2);

  auto * snapshot_btn = new QPushButton("Save Image");
  snapshot_btn->setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;");
  snapshot_btn->setToolTip("Save the latest raw /vision/raw frame to ~/camera_recordings");
  connect(snapshot_btn, &QPushButton::clicked, this, &UR10ePanel::onCameraSnapshot);
  camera_layout->addWidget(snapshot_btn, 9, 0, 1, 2);

  auto * video_start_btn = new QPushButton("Start Video");
  video_start_btn->setStyleSheet("background-color: #1565c0; color: white; font-weight: bold;");
  video_start_btn->setToolTip("Start recording raw /vision/raw to ~/camera_recordings");
  connect(video_start_btn, &QPushButton::clicked, this, &UR10ePanel::onCameraVideoStart);
  camera_layout->addWidget(video_start_btn, 10, 0);

  auto * video_stop_btn = new QPushButton("Stop Video");
  video_stop_btn->setStyleSheet("background-color: #c62828; color: white; font-weight: bold;");
  video_stop_btn->setToolTip("Stop the current camera video recording");
  connect(video_stop_btn, &QPushButton::clicked, this, &UR10ePanel::onCameraVideoStop);
  camera_layout->addWidget(video_stop_btn, 10, 1);

  auto * camera_refresh_btn = new QPushButton("Refresh Camera");
  camera_refresh_btn->setStyleSheet("background-color: #607d8b; color: white;");
  connect(camera_refresh_btn, &QPushButton::clicked, this, &UR10ePanel::onRefreshCamera);
  camera_layout->addWidget(camera_refresh_btn, 11, 0, 1, 2);

  camera_status_label_ = new QLabel("Camera: waiting for /camera_status");
  camera_status_label_->setWordWrap(true);
  camera_status_label_->setStyleSheet(
    "font-size: 9pt; color: #263238; background: #eceff1; padding: 6px; "
    "border-radius: 4px;");
  camera_layout->addWidget(camera_status_label_, 12, 0, 1, 2);

  auto * camera_note = new QLabel("Saved to ~/camera_recordings");
  camera_note->setStyleSheet("font-size: 9pt; color: #455a64; padding: 3px;");
  camera_note->setAlignment(Qt::AlignCenter);
  camera_layout->addWidget(camera_note, 13, 0, 1, 2);

  camera_tab_layout->addWidget(camera_group);
  camera_tab_layout->addStretch(1);

  // Grasp Feedback
  auto * grasp_group = new QGroupBox("Grasp Feedback");
  auto * grasp_layout = new QHBoxLayout(grasp_group);

  auto * success_btn = new QPushButton("Success (Y)");
  success_btn->setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;");
  connect(success_btn, &QPushButton::clicked, this, &UR10ePanel::onGraspSuccess);
  grasp_layout->addWidget(success_btn);

  auto * fail_btn = new QPushButton("Fail (N)");
  fail_btn->setStyleSheet("background-color: #f44336; color: white; font-weight: bold;");
  connect(fail_btn, &QPushButton::clicked, this, &UR10ePanel::onGraspFail);
  grasp_layout->addWidget(fail_btn);

  monitor_tab_layout->addWidget(grasp_group);

  // System
  auto * sys_group = new QGroupBox("System");
  auto * sys_layout = new QGridLayout(sys_group);
  sys_layout->setSpacing(4);

  auto * refresh_main_btn = new QPushButton("Refresh Main");
  refresh_main_btn->setStyleSheet("background-color: #0288d1; color: white;");
  connect(refresh_main_btn, &QPushButton::clicked, this, &UR10ePanel::onRefreshMain);
  sys_layout->addWidget(refresh_main_btn, 0, 0);

  auto * refresh_cam_btn = new QPushButton("Refresh Camera");
  refresh_cam_btn->setStyleSheet("background-color: #0288d1; color: white;");
  connect(refresh_cam_btn, &QPushButton::clicked, this, &UR10ePanel::onRefreshCamera);
  sys_layout->addWidget(refresh_cam_btn, 0, 1);

  auto * exit_btn = new QPushButton("Exit");
  exit_btn->setStyleSheet("background-color: #b71c1c; color: white; font-weight: bold;");
  connect(exit_btn, &QPushButton::clicked, this, &UR10ePanel::onExit);
  sys_layout->addWidget(exit_btn, 1, 0, 1, 2);

  settings_tab_layout->addWidget(sys_group);

  // Joint Positions
  auto * joint_group = new QGroupBox("Joint Positions (rad)");
  auto * joint_layout = new QGridLayout(joint_group);
  joint_layout->setSpacing(4);

  const char * joint_names[] = {"Pan", "Lift", "Elbow", "W1", "W2", "W3"};
  for (int i = 0; i < 6; i++) {
    joint_layout->addWidget(new QLabel(joint_names[i]), i / 3, (i % 3) * 2);
    joint_labels_[i] = new QLabel("0.000");
    joint_labels_[i]->setFont(QFont("Courier", 9, QFont::Bold));
    joint_labels_[i]->setStyleSheet(
      "background: #37474f; color: #64b5f6; padding: 3px; border-radius: 3px;");
    joint_layout->addWidget(joint_labels_[i], i / 3, (i % 3) * 2 + 1);
  }
  monitor_tab_layout->addWidget(joint_group);

  // Gripper Forces
  auto * force_group = new QGroupBox("Gripper Forces");
  auto * force_layout = new QHBoxLayout(force_group);

  const char * force_names[] = {"Left", "Center", "Right"};
  for (int i = 0; i < 3; i++) {
    auto * w = new QWidget();
    auto * vl = new QVBoxLayout(w);
    vl->setSpacing(2);
    auto * lbl = new QLabel(force_names[i]);
    lbl->setAlignment(Qt::AlignCenter);
    vl->addWidget(lbl);
    force_labels_[i] = new QLabel("0.00 N");
    force_labels_[i]->setAlignment(Qt::AlignCenter);
    force_labels_[i]->setFont(QFont("Courier", 10, QFont::Bold));
    force_labels_[i]->setStyleSheet(
      "background: #263238; color: #00e676; padding: 4px; border-radius: 4px;");
    vl->addWidget(force_labels_[i]);
    force_layout->addWidget(w);
  }
  monitor_tab_layout->addWidget(force_group);

  // ---------------- Heat / Thermal tab ----------------
  // No per-joint temperature is exposed by the driver; we show per-joint current
  // (/joint_states.effort, mapped from UR actual_current) as a heat/load proxy, plus the
  // tool-flange temperature (the only real degC available) and the gripper motor current.
  auto * heat_note = new QLabel(
    "Real per-joint temperature (\xC2\xB0""C) from /joint_temperatures when the driver "
    "publishes it; per-joint current (A) shown as a load proxy. Tool flange reports real "
    "\xC2\xB0""C.");
  heat_note->setWordWrap(true);
  heat_note->setStyleSheet("color: #90a4ae; font-size: 9pt; padding: 2px;");
  heat_tab_layout->addWidget(heat_note);

  auto * jtemp_group = new QGroupBox("Joint Temperature (\xC2\xB0""C)");
  auto * jtemp_layout = new QGridLayout(jtemp_group);
  jtemp_layout->setSpacing(4);
  const char * jtemp_names[] = {"Pan", "Lift", "Elbow", "W1", "W2", "W3"};
  for (int i = 0; i < 6; i++) {
    jtemp_layout->addWidget(new QLabel(jtemp_names[i]), i / 3, (i % 3) * 2);
    joint_temp_labels_[i] = new QLabel("-- \xC2\xB0""C");
    joint_temp_labels_[i]->setFont(QFont("Courier", 9, QFont::Bold));
    joint_temp_labels_[i]->setStyleSheet(
      "background: #37474f; color: #b0bec5; padding: 3px; border-radius: 3px;");
    jtemp_layout->addWidget(joint_temp_labels_[i], i / 3, (i % 3) * 2 + 1);
  }
  heat_tab_layout->addWidget(jtemp_group);

  auto * jcur_group = new QGroupBox("Joint Current / Load (A)");
  auto * jcur_layout = new QGridLayout(jcur_group);
  jcur_layout->setSpacing(4);
  const char * jheat_names[] = {"Pan", "Lift", "Elbow", "W1", "W2", "W3"};
  for (int i = 0; i < 6; i++) {
    jcur_layout->addWidget(new QLabel(jheat_names[i]), i / 3, (i % 3) * 2);
    joint_current_labels_[i] = new QLabel("0.00 A");
    joint_current_labels_[i]->setFont(QFont("Courier", 9, QFont::Bold));
    joint_current_labels_[i]->setStyleSheet(
      "background: #37474f; color: #b0bec5; padding: 3px; border-radius: 3px;");
    jcur_layout->addWidget(joint_current_labels_[i], i / 3, (i % 3) * 2 + 1);
  }
  heat_tab_layout->addWidget(jcur_group);

  auto * tooltemp_group = new QGroupBox("Tool Flange Temperature");
  auto * tooltemp_layout = new QHBoxLayout(tooltemp_group);
  tool_temp_label_ = new QLabel("-- \xC2\xB0""C");
  tool_temp_label_->setAlignment(Qt::AlignCenter);
  tool_temp_label_->setFont(QFont("Courier", 14, QFont::Bold));
  tool_temp_label_->setStyleSheet(
    "background: #263238; color: #b0bec5; padding: 6px; border-radius: 4px;");
  tooltemp_layout->addWidget(tool_temp_label_);
  heat_tab_layout->addWidget(tooltemp_group);

  auto * ghmot_group = new QGroupBox("Gripper Motor Current (heat proxy)");
  auto * ghmot_layout = new QHBoxLayout(ghmot_group);
  const char * ghmot_names[] = {"Left", "Center", "Right"};
  for (int i = 0; i < 3; i++) {
    auto * w = new QWidget();
    auto * vl = new QVBoxLayout(w);
    vl->setSpacing(2);
    auto * lbl = new QLabel(ghmot_names[i]);
    lbl->setAlignment(Qt::AlignCenter);
    vl->addWidget(lbl);
    gripper_heat_labels_[i] = new QLabel("0.00");
    gripper_heat_labels_[i]->setAlignment(Qt::AlignCenter);
    gripper_heat_labels_[i]->setFont(QFont("Courier", 10, QFont::Bold));
    gripper_heat_labels_[i]->setStyleSheet(
      "background: #263238; color: #b0bec5; padding: 4px; border-radius: 4px;");
    vl->addWidget(gripper_heat_labels_[i]);
    ghmot_layout->addWidget(w);
  }
  heat_tab_layout->addWidget(ghmot_group);

  // Goals info
  auto * goals_group = new QGroupBox("Goals");
  auto * goals_layout = new QVBoxLayout(goals_group);

  auto * goals_row = new QHBoxLayout();
  goal_count_label_ = new QLabel("Goals: 0");
  goal_count_label_->setStyleSheet("color: #1976d2; font-weight: bold;");
  goals_row->addWidget(goal_count_label_);

  current_velocity_label_ = new QLabel("Velocity: 5.0x");
  current_velocity_label_->setStyleSheet("color: #ff5722; font-weight: bold;");
  goals_row->addWidget(current_velocity_label_);
  goals_row->addStretch();
  goals_layout->addLayout(goals_row);

  latest_goal_label_ = new QLabel("Latest: --");
  latest_goal_label_->setFont(QFont("Courier", 8));
  latest_goal_label_->setWordWrap(true);
  goals_layout->addWidget(latest_goal_label_);

  goal_coords_label_ = new QLabel("");
  goal_coords_label_->setFont(QFont("Courier", 8));
  goal_coords_label_->setWordWrap(true);
  goal_coords_label_->setStyleSheet("color: #7b1fa2;");
  goals_layout->addWidget(goal_coords_label_);

  home_joints_label_ = new QLabel("HOME: --");
  home_joints_label_->setFont(QFont("Courier", 8));
  home_joints_label_->setWordWrap(true);
  home_joints_label_->setStyleSheet("color: #00695c;");
  goals_layout->addWidget(home_joints_label_);

  // Queue current robot pose as a goal, then run the whole queue in sequence.
  auto * add_current_btn = new QPushButton("Add Current Pos as Goal (G)");
  add_current_btn->setStyleSheet("background-color: #00897b; color: white; font-weight: bold;");
  add_current_btn->setToolTip(
    "Append the robot's current joint posture to the goal queue.\n"
    "Move the arm, press again to add more, then Execute to run them in order.");
  connect(add_current_btn, &QPushButton::clicked, this, &UR10ePanel::onAddCurrentGoal);
  goals_layout->addWidget(add_current_btn);

  auto * queue_run_row = new QHBoxLayout();
  auto * exec_queue_btn = new QPushButton("Execute Queue (moves)");
  exec_queue_btn->setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;");
  exec_queue_btn->setToolTip(
    "Move through all queued goals in order as plain Cartesian moves "
    "(no grasp behavior)");
  connect(exec_queue_btn, &QPushButton::clicked, this, &UR10ePanel::onExecuteMoves);
  queue_run_row->addWidget(exec_queue_btn);

  auto * clear_queue_btn = new QPushButton("Clear Queue");
  clear_queue_btn->setStyleSheet("background-color: #9e9e9e; color: white;");
  clear_queue_btn->setToolTip("Remove all queued goals");
  connect(clear_queue_btn, &QPushButton::clicked, this, &UR10ePanel::onClear);
  queue_run_row->addWidget(clear_queue_btn);
  goals_layout->addLayout(queue_run_row);

  auto * queue_reuse_row = new QHBoxLayout();
  auto * reuse_queue_btn = new QPushButton("Reuse Last Queue");
  reuse_queue_btn->setStyleSheet("background-color: #5e35b1; color: white; font-weight: bold;");
  reuse_queue_btn->setToolTip("Replace the current queue with the last cleared or executed queue");
  connect(reuse_queue_btn, &QPushButton::clicked, this, &UR10ePanel::onReuseLastGoalQueue);
  queue_reuse_row->addWidget(reuse_queue_btn);

  auto * goal_home_btn = new QPushButton("Go Home");
  goal_home_btn->setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;");
  goal_home_btn->setToolTip("Move to the active HOME joint posture");
  connect(goal_home_btn, &QPushButton::clicked, this, &UR10ePanel::onHome);
  queue_reuse_row->addWidget(goal_home_btn);
  goals_layout->addLayout(queue_reuse_row);

  auto * gripper_queue_row = new QHBoxLayout();
  auto * queue_open_btn = new QPushButton("Add Open");
  queue_open_btn->setStyleSheet("background-color: #43a047; color: white;");
  queue_open_btn->setToolTip("Append an OPEN gripper action at the current end of the queue");
  connect(queue_open_btn, &QPushButton::clicked, this, &UR10ePanel::onQueueGripperOpen);
  gripper_queue_row->addWidget(queue_open_btn);

  auto * queue_close_btn = new QPushButton("Add Close");
  queue_close_btn->setStyleSheet("background-color: #e53935; color: white;");
  queue_close_btn->setToolTip("Append a CLOSE gripper action at the current end of the queue");
  connect(queue_close_btn, &QPushButton::clicked, this, &UR10ePanel::onQueueGripperClose);
  gripper_queue_row->addWidget(queue_close_btn);
  goals_layout->addLayout(gripper_queue_row);

  goal_tab_layout->addWidget(goals_group);

  layout->addStretch(1);
  motion_tab_layout->addStretch(1);
  goal_tab_layout->addStretch(1);
  monitor_tab_layout->addStretch(1);
  heat_tab_layout->addStretch(1);
  settings_tab_layout->addStretch(1);

  scroll->setWidget(container);

  auto * outer = new QVBoxLayout(this);
  outer->setContentsMargins(0, 0, 0, 0);
  outer->addWidget(scroll);

  // Update timer
  update_timer_ = new QTimer(this);
  connect(update_timer_, &QTimer::timeout, this, &UR10ePanel::updateDisplay);
  update_timer_->start(100);
}

UR10ePanel::~UR10ePanel()
{
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (stability_csv_.is_open()) {
      stability_csv_.flush();
      stability_csv_.close();
    }
    if (trajectory_csv_.is_open()) {
      trajectory_csv_.flush();
      trajectory_csv_.close();
    }
  }
  if (node_) {
    node_.reset();
  }
}

void UR10ePanel::onInitialize()
{
  setupRos();
  // Install keyboard event filter on top-level RViz window
  if (window()) {
    window()->installEventFilter(this);
  }
}

bool UR10ePanel::eventFilter(QObject * obj, QEvent * event)
{
  if (event->type() == QEvent::KeyPress) {
    // Don't intercept keys when typing in text fields
    auto * focus = QApplication::focusWidget();
    if (qobject_cast<QLineEdit *>(focus)) {
      return false;
    }
    auto * ke = static_cast<QKeyEvent *>(event);
    // Plan confirmation intercept
    if (plan_waiting_confirm_) {
      if (ke->key() == Qt::Key_Return || ke->key() == Qt::Key_Enter) {
        onPlanConfirm(); return true;
      }
      if (ke->key() == Qt::Key_K) {
        onPlanCancel(); return true;
      }
    }
    switch (ke->key()) {
      case Qt::Key_H: onHome(); return true;
      case Qt::Key_D: onDropoff(); return true;
      case Qt::Key_S: onSubscribe(); return true;
      case Qt::Key_M: onSubscribeMulti(); return true;
      case Qt::Key_E: onExecute(); return true;
      case Qt::Key_G: onAddCurrentGoal(); return true;
      case Qt::Key_O: onGripperOpen(); return true;
      case Qt::Key_C: onGripperClose(); return true;
      case Qt::Key_U: onUpdateVoxel(); return true;
      case Qt::Key_Y: onGraspSuccess(); return true;
      case Qt::Key_N: onGraspFail(); return true;
    }
  }
  return rviz_common::Panel::eventFilter(obj, event);
}

void UR10ePanel::setupRos()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  cmd_pub_ = node_->create_publisher<std_msgs::msg::String>("/ui_command", 10);
  overlay_cmd_pub_ = node_->create_publisher<std_msgs::msg::String>("/vision/overlay_command", 10);

  auto goal_qos = rclcpp::QoS(1)
    .reliability(rclcpp::ReliabilityPolicy::BestEffort)
    .durability(rclcpp::DurabilityPolicy::Volatile);
  goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/manual_goal_pose", goal_qos);

  stop_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/emergency_stop", 10);

  joint_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [this](sensor_msgs::msg::JointState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (size_t i = 0; i < msg->name.size(); ++i) {
        const int joint_index = panelJointIndex(msg->name[i]);
        if (joint_index < 0) {
          continue;
        }
        if (i < msg->position.size()) {
          joint_positions_[joint_index] = msg->position[i];
        }
        if (i < msg->velocity.size()) {
          joint_velocities_[joint_index] = msg->velocity[i];
        }
        if (i < msg->effort.size()) {
          // Driver maps UR actual_current (A) onto joint effort — used as a heat/load proxy.
          joint_efforts_[joint_index] = msg->effort[i];
        }
      }
      if (stability_recording_) {
        appendStabilitySampleLocked(std::chrono::steady_clock::now());
      }
    });

  force_sub_ = node_->create_subscription<std_msgs::msg::Float32MultiArray>(
    "/gripper/force", 10,
    [this](std_msgs::msg::Float32MultiArray::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (size_t i = 0; i < std::min(msg->data.size(), size_t(3)); i++) {
        gripper_forces_[i] = msg->data[i];
      }
    });

  tool_data_sub_ = node_->create_subscription<ur_msgs::msg::ToolDataMsg>(
    "/io_and_status_controller/tool_data", 10,
    [this](ur_msgs::msg::ToolDataMsg::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      tool_temperature_ = msg->tool_temperature;
      have_tool_temp_ = true;
    });

  joint_temp_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/joint_temperatures", 10,
    [this](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (size_t i = 0; i < std::min(msg->data.size(), size_t(6)); i++) {
        joint_temperatures_[i] = msg->data[i];
      }
      have_joint_temps_ = true;
    });

  running_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
    "/io_and_status_controller/robot_program_running", 10,
    [this](std_msgs::msg::Bool::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      robot_running_ = msg->data;
    });

  camera_status_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/camera_status", 10,
    [this](std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      camera_status_text_ = msg->data;
    });

  goal_info_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/goal_info", 10,
    [this](std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      // Simple JSON parsing for goal_count, velocity_scale, latest_goal
      auto data = msg->data;
      // Extract goal_count
      auto pos = data.find("\"goal_count\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        if (colon != std::string::npos) {
          goal_count_ = std::atoi(data.c_str() + colon + 1);
        }
      }
      // Extract velocity_scale
      pos = data.find("\"velocity_scale\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        if (colon != std::string::npos) {
          velocity_scale_ = std::atof(data.c_str() + colon + 1);
        }
      }
      // Extract latest_goal
      pos = data.find("\"latest_goal\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        auto quote1 = data.find('"', colon + 1);
        auto quote2 = data.find('"', quote1 + 1);
        if (quote1 != std::string::npos && quote2 != std::string::npos) {
          latest_goal_ = data.substr(quote1 + 1, quote2 - quote1 - 1);
        }
      }
      const auto goal_display = jsonStringValue(data, "goal_display");
      if (!goal_display.empty()) {
        goal_coords_str_ = goal_display;
      } else if (data.find("\"goal_display\": \"\"") != std::string::npos) {
        goal_coords_str_.clear();
      }
      const auto home_display = jsonStringValue(data, "home_joints_display");
      if (!home_display.empty()) {
        home_joints_str_ = home_display;
      }
      // Extract plan_waiting_confirm
      pos = data.find("\"plan_waiting_confirm\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        if (colon != std::string::npos) {
          auto val_start = data.find_first_not_of(" ", colon + 1);
          plan_waiting_confirm_ = (data.substr(val_start, 4) == "true");
        }
      }
      // Extract debug_plan_preview
      pos = data.find("\"debug_plan_preview\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        if (colon != std::string::npos) {
          auto val_start = data.find_first_not_of(" ", colon + 1);
          debug_plan_preview_ = (data.substr(val_start, 4) == "true");
        }
      }
      // Extract reachability_cloud_enabled
      pos = data.find("\"reachability_cloud_enabled\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        if (colon != std::string::npos) {
          auto val_start = data.find_first_not_of(" ", colon + 1);
          reachability_cloud_enabled_ = (data.substr(val_start, 4) == "true");
        }
      }
      // Extract lidar_scan_preview_enabled
      pos = data.find("\"lidar_scan_preview_enabled\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        if (colon != std::string::npos) {
          auto val_start = data.find_first_not_of(" ", colon + 1);
          lidar_scan_preview_enabled_ = (data.substr(val_start, 4) == "true");
        }
      }
      // Extract motion_phase
      pos = data.find("\"motion_phase\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        auto q1 = data.find('"', colon + 1);
        auto q2 = data.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
          motion_phase_ = data.substr(q1 + 1, q2 - q1 - 1);
      }
      // Extract reacquire_result
      pos = data.find("\"reacquire_result\"");
      if (pos != std::string::npos) {
        auto colon = data.find(':', pos);
        auto q1 = data.find('"', colon + 1);
        auto q2 = data.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
          reacquire_result_ = data.substr(q1 + 1, q2 - q1 - 1);
      }
      // Parse grasp_history array
      pos = data.find("\"grasp_history\"");
      if (pos != std::string::npos) {
        auto bracket = data.find('[', pos);
        if (bracket != std::string::npos) {
          int grabbed = 0, slipped = 0, miss = 0;
          std::string last_outcome, last_end;
          size_t search = bracket;
          while (true) {
            auto ob = data.find('{', search);
            if (ob == std::string::npos) break;
            auto cb = data.find('}', ob);
            if (cb == std::string::npos) break;
            std::string item = data.substr(ob, cb - ob + 1);
            std::string outcome, end;
            auto op = item.find("\"outcome\"");
            if (op != std::string::npos) {
              auto oc = item.find(':', op);
              auto oq1 = item.find('"', oc + 1);
              auto oq2 = item.find('"', oq1 + 1);
              if (oq1 != std::string::npos && oq2 != std::string::npos)
                outcome = item.substr(oq1 + 1, oq2 - oq1 - 1);
            }
            auto ep = item.find("\"end\"");
            if (ep != std::string::npos) {
              auto ec = item.find(':', ep);
              auto eq1 = item.find('"', ec + 1);
              auto eq2 = item.find('"', eq1 + 1);
              if (eq1 != std::string::npos && eq2 != std::string::npos)
                end = item.substr(eq1 + 1, eq2 - eq1 - 1);
            }
            if (outcome == "GRABBED") grabbed++;
            else if (outcome == "SLIPPED") slipped++;
            else if (outcome == "NO_GRAB") miss++;
            last_outcome = outcome;
            last_end = end;
            search = cb + 1;
          }
          grab_count_ = grabbed;
          slip_count_ = slipped;
          miss_count_ = miss;
          last_outcome_ = last_outcome;
          last_end_ = last_end;
        }
      }
      // Extract goals array for coordinate display when older nodes do not publish
      // the formatted goal_display string.
      pos = data.find("\"goals\"");
      if (goal_display.empty() && pos != std::string::npos) {
        auto bracket = data.find('[', pos);
        if (bracket != std::string::npos) {
          // Find matching outer bracket
          auto end = data.find("]]", bracket);
          if (end != std::string::npos) {
            goal_coords_str_ = data.substr(bracket, end - bracket + 2);
          } else {
            // Single or no goals: "[]"
            auto single_end = data.find(']', bracket);
            if (single_end != std::string::npos) {
              goal_coords_str_ = data.substr(bracket, single_end - bracket + 1);
            }
          }
        }
      }
    });

  vel_scale_sub_ = node_->create_subscription<std_msgs::msg::Float32>(
    "/velocity_scale", 10,
    [this](std_msgs::msg::Float32::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      velocity_scale_ = msg->data;
    });

  calib_check_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/calib_check_result", 10,
    [this](std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      calib_check_result_ = msg->data;
    });

  config_sub_ = node_->create_subscription<std_msgs::msg::String>(
    "/robot_config_info", 10,
    [this](std_msgs::msg::String::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      robot_config_text_ = msg->data;
    });

  wrench_sub_ = node_->create_subscription<geometry_msgs::msg::WrenchStamped>(
    "/force_torque_sensor_broadcaster/wrench", 10,
    [this](geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      tcp_wrench_[0] = msg->wrench.force.x;
      tcp_wrench_[1] = msg->wrench.force.y;
      tcp_wrench_[2] = msg->wrench.force.z;
      tcp_wrench_[3] = msg->wrench.torque.x;
      tcp_wrench_[4] = msg->wrench.torque.y;
      tcp_wrench_[5] = msg->wrench.torque.z;
      have_wrench_ = true;
    });

  controller_state_sub_ =
    node_->create_subscription<control_msgs::msg::JointTrajectoryControllerState>(
    "/scaled_joint_trajectory_controller/controller_state", 10,
    [this](control_msgs::msg::JointTrajectoryControllerState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      const auto & errors = !msg->error.positions.empty() ?
        msg->error.positions : msg->desired.positions;
      for (size_t i = 0; i < msg->joint_names.size(); ++i) {
        const int joint_index = panelJointIndex(msg->joint_names[i]);
        if (joint_index < 0) {
          continue;
        }
        if (!msg->error.positions.empty() && i < msg->error.positions.size()) {
          tracking_errors_[joint_index] = msg->error.positions[i];
          have_tracking_error_ = true;
        } else if (
          i < errors.size() && i < msg->actual.positions.size())
        {
          tracking_errors_[joint_index] = errors[i] - msg->actual.positions[i];
          have_tracking_error_ = true;
        }
      }
      if (stability_recording_) {
        writeFollowedTrajectoryLocked(*msg, std::chrono::steady_clock::now());
      }
    });

  trajectory_command_sub_ =
    node_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    "/scaled_joint_trajectory_controller/joint_trajectory", 10,
    [this](trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      if (stability_recording_) {
        writeComputedTrajectoryLocked(*msg, std::chrono::steady_clock::now());
      }
    });
}

void UR10ePanel::publishCmd(const std::string & cmd)
{
  if (!cmd_pub_) return;
  auto msg = std_msgs::msg::String();
  msg.data = cmd;
  cmd_pub_->publish(msg);
  status_label_->setText(QString::fromStdString("Sent: " + cmd));
}

void UR10ePanel::publishOverlayCmd(const std::string & cmd)
{
  if (!overlay_cmd_pub_) return;
  auto msg = std_msgs::msg::String();
  msg.data = cmd;
  overlay_cmd_pub_->publish(msg);
  status_label_->setText(QString::fromStdString("Vision overlay: " + cmd));
}

void UR10ePanel::publishStop()
{
  if (!stop_pub_) return;
  auto msg = std_msgs::msg::Bool();
  msg.data = true;
  stop_pub_->publish(msg);
  status_label_->setText("EMERGENCY STOP SENT");
  status_label_->setStyleSheet(
    "color: white; font-weight: bold; padding: 4px; "
    "background: #d32f2f; border-radius: 4px;");
}

void UR10ePanel::publishGoal(double x, double y, double z,
                              double qw, double qx, double qy, double qz)
{
  if (!goal_pub_ || !node_) return;
  double qmag = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
  if (qmag < 0.01) return;
  qw /= qmag; qx /= qmag; qy /= qmag; qz /= qmag;

  auto msg = geometry_msgs::msg::PoseStamped();
  msg.header.frame_id = "base_link";
  msg.header.stamp = node_->get_clock()->now();
  msg.pose.position.x = x;
  msg.pose.position.y = y;
  msg.pose.position.z = z;
  msg.pose.orientation.w = qw;
  msg.pose.orientation.x = qx;
  msg.pose.orientation.y = qy;
  msg.pose.orientation.z = qz;
  goal_pub_->publish(msg);
}

void UR10ePanel::onStop() { publishStop(); }
void UR10ePanel::onHome() { publishCmd("home"); }
void UR10ePanel::onHomeLeft() { publishCmd("home_left"); }
void UR10ePanel::onHomeRight() { publishCmd("home_right"); }
void UR10ePanel::onSetHomeCurrent()
{
  auto reply = QMessageBox::question(
    this,
    "Set Home",
    "Use the robot's current joint position as the new HOME for this running node?",
    QMessageBox::Yes | QMessageBox::No,
    QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    publishCmd("set_home_current");
  }
}
void UR10ePanel::onSetDropoffCurrent()
{
  auto reply = QMessageBox::question(
    this,
    "Set Dropoff",
    "Use the robot's current joint position as the new DROPOFF for this running node?",
    QMessageBox::Yes | QMessageBox::No,
    QMessageBox::No);
  if (reply == QMessageBox::Yes) {
    publishCmd("set_dropoff_current");
  }
}
void UR10ePanel::onDropoff() { publishCmd("dropoff"); }
void UR10ePanel::onExecute() { publishCmd("execute"); }
void UR10ePanel::onExecuteMoves() { publishCmd("execute_moves"); }
void UR10ePanel::onAddCurrentGoal() { publishCmd("add_current_goal"); }
void UR10ePanel::onReuseLastGoalQueue() { publishCmd("reuse_last_goal_queue"); }
void UR10ePanel::onQueueGripperOpen() { publishCmd("queue_gripper_open"); }
void UR10ePanel::onQueueGripperClose() { publishCmd("queue_gripper_close"); }
void UR10ePanel::onClear() { publishCmd("clear"); }
void UR10ePanel::onCheckCalibration() { publishCmd("check_calibration"); }
void UR10ePanel::onGripperOpen() { publishCmd("open"); }
void UR10ePanel::onGripperClose() { publishCmd("close"); }
void UR10ePanel::onCapture() { publishCmd("capture 10"); }
void UR10ePanel::onCaptureStop() { publishCmd("capture_stop"); }
void UR10ePanel::onSubscribe() { publishCmd("subscribe"); }
void UR10ePanel::onSubscribeMulti()
{
  const int count = multi_goal_count_spin_ ? multi_goal_count_spin_->value() : 3;
  publishCmd("subscribe_multi " + std::to_string(count));
}
void UR10ePanel::onUpdateVoxel() { publishCmd("update_voxel"); }
void UR10ePanel::onExit()
{
  auto reply = QMessageBox::question(
    this, "Exit System",
    "Shut down the entire system (main, vision, RViz)?",
    QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
  if (reply != QMessageBox::Yes) return;

  // 1. Signal all ROS nodes to shut down cleanly
  publishCmd("exit");

  // 2. Small delay so "exit" message is published before we die
  QTimer::singleShot(400, this, []() {
    // Kill the entire process group — takes RViz and all child processes with it
    ::kill(-::getpgid(0), SIGTERM);
  });
}
void UR10ePanel::onRefreshMain() { publishCmd("refresh_main"); }
void UR10ePanel::onRefreshCamera() { publishCmd("refresh_camera"); }
void UR10ePanel::onCameraPresetLab() { publishCmd("camera_preset_lab"); }
void UR10ePanel::onCameraPresetOutdoor() { publishCmd("camera_preset_outdoor"); }
void UR10ePanel::onCameraSnapshot() { publishCmd("camera_snapshot"); }
void UR10ePanel::onCameraVideoStart() { publishCmd("camera_video_start"); }
void UR10ePanel::onCameraVideoStop() { publishCmd("camera_video_stop"); }
void UR10ePanel::onGraspSuccess() { publishCmd("grasp_success"); }
void UR10ePanel::onGraspFail() { publishCmd("grasp_fail"); }
void UR10ePanel::onPlanConfirm() { publishCmd("plan_confirm"); }
void UR10ePanel::onPlanCancel() { publishCmd("plan_cancel"); }
void UR10ePanel::onLidarScan() { publishCmd("lidar_scan"); }
void UR10ePanel::onStabilityRecord()
{
  if (stability_recording_) {
    stopStabilityRecording();
  } else {
    startStabilityRecording();
  }
}

void UR10ePanel::startStabilityRecording()
{
  const QString output_dir = QDir::homePath() + "/ur10e_stability";
  if (!QDir().mkpath(output_dir)) {
    QMessageBox::critical(
      this, "Stability Recording",
      "Could not create output directory:\n" + output_dir);
    return;
  }

  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&now_time, &local_time);
  std::ostringstream filename;
  filename << output_dir.toStdString() << "/stability_"
           << std::put_time(&local_time, "%Y%m%d_%H%M%S") << ".csv";
  std::ostringstream trajectory_filename;
  trajectory_filename << output_dir.toStdString() << "/trajectory_comparison_"
                      << std::put_time(&local_time, "%Y%m%d_%H%M%S") << ".csv";

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    stability_csv_.open(filename.str(), std::ios::out | std::ios::trunc);
    if (!stability_csv_.is_open()) {
      QMessageBox::critical(
        this, "Stability Recording",
        QString::fromStdString("Could not open:\n" + filename.str()));
      return;
    }
    trajectory_csv_.open(
      trajectory_filename.str(), std::ios::out | std::ios::trunc);
    if (!trajectory_csv_.is_open()) {
      stability_csv_.close();
      QMessageBox::critical(
        this, "Stability Recording",
        QString::fromStdString(
          "Could not open:\n" + trajectory_filename.str()));
      return;
    }
    stability_csv_ <<
      "elapsed_s,"
      "pan_pos_rad,lift_pos_rad,elbow_pos_rad,wrist1_pos_rad,wrist2_pos_rad,wrist3_pos_rad,"
      "pan_vel_rad_s,lift_vel_rad_s,elbow_vel_rad_s,wrist1_vel_rad_s,wrist2_vel_rad_s,wrist3_vel_rad_s,"
      "pan_error_rad,lift_error_rad,elbow_error_rad,wrist1_error_rad,wrist2_error_rad,wrist3_error_rad,"
      "force_x_N,force_y_N,force_z_N,torque_x_Nm,torque_y_Nm,torque_z_Nm\n";
    stability_csv_ << std::fixed << std::setprecision(8);
    trajectory_csv_ <<
      "row_type,recording_elapsed_s,trajectory_id,waypoint_index,"
      "trajectory_time_s,"
      "computed_pan_rad,computed_lift_rad,computed_elbow_rad,"
      "computed_wrist1_rad,computed_wrist2_rad,computed_wrist3_rad,"
      "followed_pan_rad,followed_lift_rad,followed_elbow_rad,"
      "followed_wrist1_rad,followed_wrist2_rad,followed_wrist3_rad,"
      "error_pan_rad,error_lift_rad,error_elbow_rad,"
      "error_wrist1_rad,error_wrist2_rad,error_wrist3_rad,"
      "computed_pan_vel_rad_s,computed_lift_vel_rad_s,"
      "computed_elbow_vel_rad_s,computed_wrist1_vel_rad_s,"
      "computed_wrist2_vel_rad_s,computed_wrist3_vel_rad_s,"
      "followed_pan_vel_rad_s,followed_lift_vel_rad_s,"
      "followed_elbow_vel_rad_s,followed_wrist1_vel_rad_s,"
      "followed_wrist2_vel_rad_s,followed_wrist3_vel_rad_s\n";
    trajectory_csv_ << std::fixed << std::setprecision(8);
    stability_csv_path_ = filename.str();
    trajectory_csv_path_ = trajectory_filename.str();
    stability_sample_count_ = 0;
    trajectory_id_ = 0;
    stability_window_.clear();
    stability_start_time_ = std::chrono::steady_clock::now();
    stability_recording_ = true;
  }

  stability_record_btn_->setText("Stop & Rate");
  stability_record_btn_->setStyleSheet(
    "background-color: #c62828; color: white; font-weight: bold; "
    "font-size: 10pt; padding: 7px;");
  stability_result_label_->setText("Recording... press again to stop and rate.");
  stability_result_label_->setStyleSheet(
    "font-size: 9pt; padding: 5px; background: #ffebee; "
    "color: #b71c1c; border-radius: 4px;");
  status_label_->setText(
    QString::fromStdString(
      "Recording stability and trajectory data: " + stability_csv_path_));
}

void UR10ePanel::stopStabilityRecording()
{
  StabilityResult result;
  std::string csv_path;
  std::string trajectory_csv_path;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!stability_recording_) {
      return;
    }
    stability_recording_ = false;
    result = calculateStabilityLocked();
    csv_path = stability_csv_path_;
    trajectory_csv_path = trajectory_csv_path_;

    if (stability_csv_.is_open()) {
      stability_csv_ << "\n# result_mode," <<
        (result.trajectory_tracking ? "trajectory_tracking" :
        "stationary_stability") << "\n";
      stability_csv_ << "# rating_score," << result.score << "\n";
      stability_csv_ << "# rating," <<
        (result.valid ? ratingName(result.score) : "Insufficient data") << "\n";
      stability_csv_ << "# rating_window_s,2.0\n";
      stability_csv_ << "# rating_samples," << result.sample_count << "\n";
      stability_csv_ << "# velocity_rms_rad_s," << result.velocity_rms << "\n";
      stability_csv_ << "# position_jitter_rms_rad," <<
        result.position_jitter_rms << "\n";
      stability_csv_ << "# tracking_error_rms_rad," <<
        result.tracking_error_rms << "\n";
      stability_csv_ << "# tracking_error_max_rad," <<
        result.tracking_error_max << "\n";
      stability_csv_ << "# force_noise_rms_N," << result.force_noise_rms << "\n";
      stability_csv_ << "# torque_noise_rms_Nm," << result.torque_noise_rms << "\n";
      stability_csv_.flush();
      stability_csv_.close();
    }
    if (trajectory_csv_.is_open()) {
      trajectory_csv_.flush();
      trajectory_csv_.close();
    }
  }

  stability_record_btn_->setText("Record Stability / Tracking");
  stability_record_btn_->setStyleSheet(
    "background-color: #3949ab; color: white; font-weight: bold; "
    "font-size: 10pt; padding: 7px;");

  if (!result.valid) {
    stability_result_label_->setText(
      QString("Insufficient data (%1 samples).\nStability CSV: %2\nTrajectory CSV: %3")
      .arg(result.sample_count)
      .arg(QString::fromStdString(csv_path))
      .arg(QString::fromStdString(trajectory_csv_path)));
    stability_result_label_->setStyleSheet(
      "font-size: 9pt; padding: 5px; background: #fff8e1; "
      "color: #e65100; border-radius: 4px;");
    status_label_->setText("Stability recording stopped: insufficient rating data");
    return;
  }

  const QString rating = QString::fromUtf8(ratingName(result.score));
  const QString tracking_text = std::isfinite(result.tracking_error_rms) ?
    QString("%1 deg").arg(result.tracking_error_rms * 180.0 / M_PI, 0, 'f', 3) :
    QString("n/a");
  if (result.trajectory_tracking) {
    const QString tracking_max_text =
      std::isfinite(result.tracking_error_max) ?
      QString("%1 deg").arg(
        result.tracking_error_max * 180.0 / M_PI, 0, 'f', 3) :
      QString("n/a");
    stability_result_label_->setText(
      QString("Trajectory Tracking: %1 - %2/100\n"
              "Position error RMS: %3 | Max: %4\n"
              "Motion velocity RMS: %5 rad/s\n"
              "Stability CSV: %6\nTrajectory CSV: %7")
      .arg(rating)
      .arg(result.score, 0, 'f', 1)
      .arg(tracking_text)
      .arg(tracking_max_text)
      .arg(result.velocity_rms, 0, 'f', 4)
      .arg(QString::fromStdString(csv_path))
      .arg(QString::fromStdString(trajectory_csv_path)));
  } else {
    stability_result_label_->setText(
      QString("Stationary Stability: %1 - %2/100\n"
              "Velocity RMS: %3 rad/s | Position jitter: %4 deg\n"
              "Tracking RMS: %5\nStability CSV: %6\nTrajectory CSV: %7")
      .arg(rating)
      .arg(result.score, 0, 'f', 1)
      .arg(result.velocity_rms, 0, 'f', 4)
      .arg(result.position_jitter_rms * 180.0 / M_PI, 0, 'f', 3)
      .arg(tracking_text)
      .arg(QString::fromStdString(csv_path))
      .arg(QString::fromStdString(trajectory_csv_path)));
  }

  const char * bg = result.score >= 75.0 ? "#e8f5e9" :
    (result.score >= 60.0 ? "#fff8e1" : "#ffebee");
  const char * fg = result.score >= 75.0 ? "#2e7d32" :
    (result.score >= 60.0 ? "#e65100" : "#c62828");
  stability_result_label_->setStyleSheet(
    QString("font-size: 9pt; padding: 5px; background: %1; "
            "color: %2; border-radius: 4px;").arg(bg).arg(fg));
  status_label_->setText(
    QString("%1: %2 (%3/100), saved %4")
    .arg(result.trajectory_tracking ? "Trajectory tracking" :
      "Stationary stability")
    .arg(rating)
    .arg(result.score, 0, 'f', 1)
    .arg(QString::fromStdString(csv_path)));
}

void UR10ePanel::appendStabilitySampleLocked(
  const std::chrono::steady_clock::time_point & now)
{
  if (!stability_csv_.is_open()) {
    return;
  }

  StabilitySample sample;
  sample.elapsed_s =
    std::chrono::duration<double>(now - stability_start_time_).count();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (size_t i = 0; i < 6; ++i) {
    sample.positions[i] = joint_positions_[i];
    sample.velocities[i] = joint_velocities_[i];
    sample.tracking_errors[i] = have_tracking_error_ ? tracking_errors_[i] : nan;
    sample.wrench[i] = have_wrench_ ? tcp_wrench_[i] : nan;
  }

  stability_csv_ << sample.elapsed_s;
  for (double value : sample.positions) stability_csv_ << ',' << value;
  for (double value : sample.velocities) stability_csv_ << ',' << value;
  for (double value : sample.tracking_errors) stability_csv_ << ',' << value;
  for (double value : sample.wrench) stability_csv_ << ',' << value;
  stability_csv_ << '\n';

  stability_window_.push_back(sample);
  while (
    !stability_window_.empty() &&
    sample.elapsed_s - stability_window_.front().elapsed_s > 2.0)
  {
    stability_window_.pop_front();
  }
  ++stability_sample_count_;
}

void UR10ePanel::writeComputedTrajectoryLocked(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const std::chrono::steady_clock::time_point & now)
{
  if (!trajectory_csv_.is_open()) {
    return;
  }

  ++trajectory_id_;
  const double elapsed_s =
    std::chrono::duration<double>(now - stability_start_time_).count();
  const double nan = std::numeric_limits<double>::quiet_NaN();

  for (size_t point_index = 0; point_index < msg.points.size(); ++point_index) {
    const auto & point = msg.points[point_index];
    std::array<double, 6> computed_positions;
    std::array<double, 6> computed_velocities;
    computed_positions.fill(nan);
    computed_velocities.fill(nan);

    for (size_t i = 0; i < msg.joint_names.size(); ++i) {
      const int joint_index = panelJointIndex(msg.joint_names[i]);
      if (joint_index < 0) {
        continue;
      }
      if (i < point.positions.size()) {
        computed_positions[joint_index] = point.positions[i];
      }
      if (i < point.velocities.size()) {
        computed_velocities[joint_index] = point.velocities[i];
      }
    }

    const double trajectory_time_s =
      static_cast<double>(point.time_from_start.sec) +
      static_cast<double>(point.time_from_start.nanosec) * 1e-9;
    trajectory_csv_ << "planned_waypoint," << elapsed_s << ','
                    << trajectory_id_ << ',' << point_index << ','
                    << trajectory_time_s;
    for (double value : computed_positions) trajectory_csv_ << ',' << value;
    for (size_t i = 0; i < 12; ++i) trajectory_csv_ << ',' << nan;
    for (double value : computed_velocities) trajectory_csv_ << ',' << value;
    for (size_t i = 0; i < 6; ++i) trajectory_csv_ << ',' << nan;
    trajectory_csv_ << '\n';
  }
}

void UR10ePanel::writeFollowedTrajectoryLocked(
  const control_msgs::msg::JointTrajectoryControllerState & msg,
  const std::chrono::steady_clock::time_point & now)
{
  if (!trajectory_csv_.is_open()) {
    return;
  }

  const double elapsed_s =
    std::chrono::duration<double>(now - stability_start_time_).count();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::array<double, 6> computed_positions;
  std::array<double, 6> followed_positions;
  std::array<double, 6> position_errors;
  std::array<double, 6> computed_velocities;
  std::array<double, 6> followed_velocities;
  computed_positions.fill(nan);
  followed_positions.fill(nan);
  position_errors.fill(nan);
  computed_velocities.fill(nan);
  followed_velocities.fill(nan);

  const auto & reference_positions = !msg.reference.positions.empty() ?
    msg.reference.positions : msg.desired.positions;
  const auto & feedback_positions = !msg.feedback.positions.empty() ?
    msg.feedback.positions : msg.actual.positions;
  const auto & reference_velocities = !msg.reference.velocities.empty() ?
    msg.reference.velocities : msg.desired.velocities;
  const auto & feedback_velocities = !msg.feedback.velocities.empty() ?
    msg.feedback.velocities : msg.actual.velocities;

  for (size_t i = 0; i < msg.joint_names.size(); ++i) {
    const int joint_index = panelJointIndex(msg.joint_names[i]);
    if (joint_index < 0) {
      continue;
    }
    if (i < reference_positions.size()) {
      computed_positions[joint_index] = reference_positions[i];
    }
    if (i < feedback_positions.size()) {
      followed_positions[joint_index] = feedback_positions[i];
    }
    if (i < msg.error.positions.size()) {
      position_errors[joint_index] = msg.error.positions[i];
    } else if (
      i < reference_positions.size() && i < feedback_positions.size())
    {
      position_errors[joint_index] =
        reference_positions[i] - feedback_positions[i];
    }
    if (i < reference_velocities.size()) {
      computed_velocities[joint_index] = reference_velocities[i];
    }
    if (i < feedback_velocities.size()) {
      followed_velocities[joint_index] = feedback_velocities[i];
    }
  }

  trajectory_csv_ << "followed_sample," << elapsed_s << ','
                  << trajectory_id_ << ",-1," << nan;
  for (double value : computed_positions) trajectory_csv_ << ',' << value;
  for (double value : followed_positions) trajectory_csv_ << ',' << value;
  for (double value : position_errors) trajectory_csv_ << ',' << value;
  for (double value : computed_velocities) trajectory_csv_ << ',' << value;
  for (double value : followed_velocities) trajectory_csv_ << ',' << value;
  trajectory_csv_ << '\n';
}

UR10ePanel::StabilityResult UR10ePanel::calculateStabilityLocked() const
{
  StabilityResult result;
  result.sample_count = stability_window_.size();
  if (stability_window_.size() < 10) {
    return result;
  }

  double velocity_sq_sum = 0.0;
  size_t velocity_count = 0;
  double error_sq_sum = 0.0;
  size_t error_count = 0;
  double tracking_error_max = 0.0;
  std::array<double, 6> position_sum{};
  std::array<double, 6> wrench_sum{};
  std::array<size_t, 6> wrench_count{};

  for (const auto & sample : stability_window_) {
    for (size_t i = 0; i < 6; ++i) {
      position_sum[i] += sample.positions[i];
      if (std::isfinite(sample.velocities[i])) {
        velocity_sq_sum += sample.velocities[i] * sample.velocities[i];
        ++velocity_count;
      }
      if (std::isfinite(sample.tracking_errors[i])) {
        error_sq_sum += sample.tracking_errors[i] * sample.tracking_errors[i];
        tracking_error_max = std::max(
          tracking_error_max, std::abs(sample.tracking_errors[i]));
        ++error_count;
      }
      if (std::isfinite(sample.wrench[i])) {
        wrench_sum[i] += sample.wrench[i];
        ++wrench_count[i];
      }
    }
  }

  double position_variance_sum = 0.0;
  std::array<double, 6> wrench_variance{};
  for (const auto & sample : stability_window_) {
    for (size_t i = 0; i < 6; ++i) {
      const double position_mean =
        position_sum[i] / static_cast<double>(stability_window_.size());
      const double position_delta = sample.positions[i] - position_mean;
      position_variance_sum += position_delta * position_delta;

      if (wrench_count[i] > 0 && std::isfinite(sample.wrench[i])) {
        const double wrench_mean =
          wrench_sum[i] / static_cast<double>(wrench_count[i]);
        const double wrench_delta = sample.wrench[i] - wrench_mean;
        wrench_variance[i] += wrench_delta * wrench_delta;
      }
    }
  }

  result.velocity_rms = velocity_count > 0 ?
    std::sqrt(velocity_sq_sum / static_cast<double>(velocity_count)) :
    std::numeric_limits<double>::quiet_NaN();
  result.position_jitter_rms = std::sqrt(
    position_variance_sum /
    static_cast<double>(stability_window_.size() * 6));
  result.tracking_error_rms = error_count > 0 ?
    std::sqrt(error_sq_sum / static_cast<double>(error_count)) :
    std::numeric_limits<double>::quiet_NaN();
  result.tracking_error_max = error_count > 0 ?
    tracking_error_max : std::numeric_limits<double>::quiet_NaN();

  double force_variance_sum = 0.0;
  double torque_variance_sum = 0.0;
  size_t force_axes = 0;
  size_t torque_axes = 0;
  for (size_t i = 0; i < 6; ++i) {
    if (wrench_count[i] == 0) {
      continue;
    }
    const double axis_variance =
      wrench_variance[i] / static_cast<double>(wrench_count[i]);
    if (i < 3) {
      force_variance_sum += axis_variance;
      ++force_axes;
    } else {
      torque_variance_sum += axis_variance;
      ++torque_axes;
    }
  }
  result.force_noise_rms = force_axes > 0 ?
    std::sqrt(force_variance_sum / static_cast<double>(force_axes)) :
    std::numeric_limits<double>::quiet_NaN();
  result.torque_noise_rms = torque_axes > 0 ?
    std::sqrt(torque_variance_sum / static_cast<double>(torque_axes)) :
    std::numeric_limits<double>::quiet_NaN();

  // A moving robot should be judged by how closely it follows the controller
  // reference. Joint travel and velocity are expected motion, not instability.
  result.trajectory_tracking =
    std::isfinite(result.velocity_rms) && result.velocity_rms > 0.030;
  if (result.trajectory_tracking) {
    result.valid = error_count > 0;
    result.score = result.valid ?
      metricScore(result.tracking_error_rms, 0.00175, 0.0175) : 0.0;
    return result;
  }

  const std::array<double, 5> scores = {
    metricScore(result.velocity_rms, 0.002, 0.030),
    metricScore(result.position_jitter_rms, 0.00035, 0.0035),
    metricScore(result.tracking_error_rms, 0.00175, 0.0175),
    metricScore(result.force_noise_rms, 0.5, 5.0),
    metricScore(result.torque_noise_rms, 0.03, 0.5)
  };
  const std::array<double, 5> weights = {0.35, 0.25, 0.25, 0.10, 0.05};
  double weighted_score = 0.0;
  double active_weight = 0.0;
  for (size_t i = 0; i < scores.size(); ++i) {
    if (std::isfinite(scores[i])) {
      weighted_score += scores[i] * weights[i];
      active_weight += weights[i];
    }
  }

  result.valid = active_weight >= 0.50;
  result.score = result.valid ? weighted_score / active_weight : 0.0;
  return result;
}

void UR10ePanel::onDebugPreviewChanged(int state)
{
  publishCmd(state == Qt::Checked ? "set_debug_preview true" : "set_debug_preview false");
}

void UR10ePanel::onReachabilityCloudChanged(int state)
{
  publishCmd(state == Qt::Checked ?
    "set_reachability_cloud true" : "set_reachability_cloud false");
}

void UR10ePanel::onLidarScanPreviewChanged(int state)
{
  publishCmd(state == Qt::Checked ?
    "set_lidar_scan_preview true" : "set_lidar_scan_preview false");
}

void UR10ePanel::onZoneOverlayToggle()
{
  const bool enabled = zone_overlay_btn_ && zone_overlay_btn_->isChecked();
  if (zone_overlay_btn_) {
    zone_overlay_btn_->setText(enabled ? "Zone Overlay: ON" : "Zone Overlay: OFF");
    zone_overlay_btn_->setStyleSheet(enabled
      ? "background-color: #009688; color: white; font-weight: bold;"
      : "background-color: #607d8b; color: white; font-weight: bold;");
  }
  publishOverlayCmd(enabled ? "classification_zones true" : "classification_zones false");
}

void UR10ePanel::onSendGoal()
{
  bool ok = true;
  double x = x_in_->text().toDouble(&ok); if (!ok) return;
  double y = y_in_->text().toDouble(&ok); if (!ok) return;
  double z = z_in_->text().toDouble(&ok); if (!ok) return;
  double qw = qw_in_->text().toDouble(&ok); if (!ok) return;
  double qx = qx_in_->text().toDouble(&ok); if (!ok) return;
  double qy = qy_in_->text().toDouble(&ok); if (!ok) return;
  double qz = qz_in_->text().toDouble(&ok); if (!ok) return;
  publishGoal(x, y, z, qw, qx, qy, qz);
  status_label_->setText(QString("Direct manual goal sent: (%1, %2, %3)").arg(x, 0, 'f', 3).arg(y, 0, 'f', 3).arg(z, 0, 'f', 3));
  status_label_->setStyleSheet(
    "color: #2d6a4f; font-weight: bold; padding: 4px; "
    "background: #e8f5e9; border-radius: 4px;");
}

void UR10ePanel::onVelocitySliderChanged(int value)
{
  double scale = value / 10.0;
  velocity_value_label_->setText(QString("%1x").arg(scale, 0, 'f', 1));
}

void UR10ePanel::onApplyVelocity()
{
  double scale = velocity_slider_->value() / 10.0;
  std::ostringstream ss;
  ss << "set_velocity_scale " << std::fixed << std::setprecision(2) << scale;
  publishCmd(ss.str());
}

void UR10ePanel::onVelocityPreset()
{
  auto * btn = qobject_cast<QPushButton *>(sender());
  if (!btn) return;
  double value = btn->property("preset_value").toDouble();
  velocity_slider_->setValue(static_cast<int>(value * 10));
}

void UR10ePanel::updateDisplay()
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if (!robot_config_text_.empty()) {
    config_label_->setText(QString::fromStdString(robot_config_text_));
  }
  if (camera_status_label_ && !camera_status_text_.empty()) {
    camera_status_label_->setText(QString::fromStdString(camera_status_text_));
  }

  // Motion phase badge
  {
    const char * bg = "#455a64", * fg = "#eceff1";
    if      (motion_phase_ == "APPROACH")    { bg = "#1565c0"; fg = "#e3f2fd"; }
    else if (motion_phase_ == "REACQUIRE")  { bg = "#6a1b9a"; fg = "#f3e5f5"; }
    else if (motion_phase_ == "FINAL")      { bg = "#e65100"; fg = "#fff3e0"; }
    else if (motion_phase_ == "REVERSING")  { bg = "#558b2f"; fg = "#f1f8e9"; }
    else if (motion_phase_ == "DROPOFF")    { bg = "#00838f"; fg = "#e0f7fa"; }
    else if (motion_phase_ == "HOME")       { bg = "#2e7d32"; fg = "#e8f5e9"; }
    else if (motion_phase_ == "LIDAR_SCAN") { bg = "#00796b"; fg = "#e0f2f1"; }
    phase_label_->setText(QString::fromStdString(motion_phase_.empty() ? "IDLE" : motion_phase_));
    phase_label_->setStyleSheet(QString(
      "background: %1; color: %2; padding: 6px; border-radius: 6px; "
      "font-size: 13pt; font-weight: bold;").arg(bg).arg(fg));
  }

  // Reacquire badge
  {
    const char * txt = "\xe2\x80\x94", * bg = "#455a64", * fg = "#eceff1";
    if      (reacquire_result_ == "OK")         { txt = "OK";         bg = "#1b5e20"; fg = "#e8f5e9"; }
    else if (reacquire_result_ == "SEARCH")     { txt = "SEARCH\xe2\x80\xa6"; bg = "#1565c0"; fg = "#e3f2fd"; }
    else if (reacquire_result_ == "NUDGING")    { txt = "NUDGING\xe2\x80\xa6"; bg = "#f57f17"; fg = "#fffde7"; }
    else if (reacquire_result_ == "NUDGE OK")   { txt = "NUDGE OK";  bg = "#33691e"; fg = "#f1f8e9"; }
    else if (reacquire_result_ == "NUDGE FAIL") { txt = "NUDGE FAIL"; bg = "#b71c1c"; fg = "#ffebee"; }
    reacq_label_->setText(txt);
    reacq_label_->setStyleSheet(QString(
      "background: %1; color: %2; padding: 6px; border-radius: 6px; "
      "font-size: 13pt; font-weight: bold;").arg(bg).arg(fg));
  }

  // Harvest result banner + tally
  {
    const char * txt = "\xe2\x80\x94", * bg = "#455a64", * fg = "#eceff1";
    if (!last_outcome_.empty()) {
      if      (last_outcome_ == "GRABBED" && last_end_ == "PROPER")
        { txt = "SUCCESS"; bg = "#2e7d32"; fg = "#e8f5e9"; }
      else if (last_outcome_ == "GRABBED" && last_end_ == "WEAK")
        { txt = "PARTIAL"; bg = "#e65100"; fg = "#fff3e0"; }
      else if (last_outcome_ == "SLIPPED")
        { txt = "SLIPPED"; bg = "#f57f17"; fg = "#fffde7"; }
      else if (last_outcome_ == "NO_GRAB")
        { txt = "FAIL";    bg = "#b71c1c"; fg = "#ffebee"; }
    }
    harvest_banner_->setText(txt);
    harvest_banner_->setStyleSheet(QString(
      "background: %1; color: %2; border-radius: 8px; "
      "font-size: 18pt; font-weight: bold;").arg(bg).arg(fg));
    grab_count_label_->setText(QString::number(grab_count_));
    slip_count_label_->setText(QString::number(slip_count_));
    miss_count_label_->setText(QString::number(miss_count_));
    total_count_label_->setText(QString::number(grab_count_ + slip_count_ + miss_count_));
  }

  // Robot state
  robot_state_label_->setText(robot_running_ ? "Program: Running" : "Program: Stopped");
  robot_state_label_->setStyleSheet(
    robot_running_ ?
    "font-size: 10pt; padding: 2px; color: #2d6a4f;" :
    "font-size: 10pt; padding: 2px; color: #d32f2f;");

  // Joint positions
  for (int i = 0; i < 6; i++) {
    joint_labels_[i]->setText(QString::number(joint_positions_[i], 'f', 3));
  }

  // Gripper forces
  for (int i = 0; i < 3; i++) {
    force_labels_[i]->setText(QString("%1 N").arg(gripper_forces_[i], 0, 'f', 2));
  }

  // Heat tab: real per-joint temperature (degC) when published, colour-graded.
  for (int i = 0; i < 6; i++) {
    if (have_joint_temps_) {
      const double tc = joint_temperatures_[i];
      joint_temp_labels_[i]->setText(QString("%1 \xC2\xB0""C").arg(tc, 0, 'f', 1));
      const char * col = (tc > 60.0) ? "#ff5252" : (tc > 45.0) ? "#ffb300" : "#69f0ae";
      joint_temp_labels_[i]->setStyleSheet(
        QString("background: #37474f; color: %1; padding: 3px; border-radius: 3px;").arg(col));
    }
  }

  // Heat tab: per-joint current (A) as a heat/load proxy, colour-graded by magnitude.
  for (int i = 0; i < 6; i++) {
    const double a = std::abs(joint_efforts_[i]);
    joint_current_labels_[i]->setText(QString("%1 A").arg(joint_efforts_[i], 0, 'f', 2));
    const char * col = (a > 6.0) ? "#ff5252" : (a > 3.0) ? "#ffb300" : "#b0bec5";
    joint_current_labels_[i]->setStyleSheet(
      QString("background: #37474f; color: %1; padding: 3px; border-radius: 3px;").arg(col));
  }

  // Tool flange temperature (real degC, the only joint-side temperature the driver exposes).
  if (have_tool_temp_) {
    const char * tcol = (tool_temperature_ > 60.0) ? "#ff5252"
      : (tool_temperature_ > 45.0) ? "#ffb300" : "#69f0ae";
    tool_temp_label_->setText(QString("%1 \xC2\xB0""C").arg(tool_temperature_, 0, 'f', 1));
    tool_temp_label_->setStyleSheet(
      QString("background: #263238; color: %1; padding: 6px; border-radius: 4px;").arg(tcol));
  }

  // Gripper motor current (heat proxy) mirrors the force values.
  for (int i = 0; i < 3; i++) {
    const double g = std::abs(gripper_forces_[i]);
    gripper_heat_labels_[i]->setText(QString::number(gripper_forces_[i], 'f', 2));
    const char * gcol = (g > 6.0) ? "#ff5252" : (g > 3.0) ? "#ffb300" : "#b0bec5";
    gripper_heat_labels_[i]->setStyleSheet(
      QString("background: #263238; color: %1; padding: 4px; border-radius: 4px;").arg(gcol));
  }

  // Goals
  goal_count_label_->setText(QString("Goals: %1").arg(goal_count_));
  current_velocity_label_->setText(QString("Velocity: %1x").arg(velocity_scale_, 0, 'f', 1));
  if (!latest_goal_.empty()) {
    latest_goal_label_->setText(QString("Latest: %1").arg(
      QString::fromStdString(latest_goal_)));
  }
  if (!home_joints_str_.empty()) {
    home_joints_label_->setText(QString("HOME: %1").arg(
      QString::fromStdString(home_joints_str_)));
  }
  // Plan confirm/cancel visibility
  plan_confirm_btn_->setVisible(plan_waiting_confirm_);
  plan_cancel_btn_->setVisible(plan_waiting_confirm_);

  if (!calib_check_result_.empty()) {
    calib_result_label_->setText(QString::fromStdString(calib_check_result_));
    if (calib_check_result_.rfind("GOOD", 0) == 0) {
      calib_result_label_->setStyleSheet(
        "font-size: 10pt; padding: 4px; background: #e8f5e9; color: #2e7d32; border-radius: 4px;");
    } else if (calib_check_result_.rfind("ACCEPTABLE", 0) == 0) {
      calib_result_label_->setStyleSheet(
        "font-size: 10pt; padding: 4px; background: #fff8e1; color: #e65100; border-radius: 4px;");
    } else if (calib_check_result_.rfind("POOR", 0) == 0 || calib_check_result_.rfind("FAIL", 0) == 0) {
      calib_result_label_->setStyleSheet(
        "font-size: 10pt; padding: 4px; background: #ffebee; color: #c62828; border-radius: 4px;");
    } else {
      calib_result_label_->setStyleSheet(
        "font-size: 10pt; padding: 4px; background: #f3e5f5; border-radius: 4px;");
    }
  }

  // Sync debug preview checkbox
  debug_preview_cb_->blockSignals(true);
  debug_preview_cb_->setChecked(debug_plan_preview_);
  debug_preview_cb_->blockSignals(false);

  // Sync reachability cloud checkbox
  reachability_cloud_cb_->blockSignals(true);
  reachability_cloud_cb_->setChecked(reachability_cloud_enabled_);
  reachability_cloud_cb_->blockSignals(false);

  // Sync LiDAR scan preview checkbox
  lidar_scan_preview_cb_->blockSignals(true);
  lidar_scan_preview_cb_->setChecked(lidar_scan_preview_enabled_);
  lidar_scan_preview_cb_->blockSignals(false);

  // Show goal coordinates
  if (!goal_coords_str_.empty() && goal_coords_str_ != "[]") {
    if (!goal_coords_str_.empty() && goal_coords_str_[0] != '[') {
      goal_coords_label_->setText(QString::fromStdString(goal_coords_str_));
      return;
    }
    // Format: [[x,y,z],[x,y,z],...] → readable lines
    QString coords_text;
    // Simple parsing: split by ],[
    std::string s = goal_coords_str_;
    int goal_num = 1;
    size_t p = 0;
    while ((p = s.find('[', p)) != std::string::npos) {
      auto close = s.find(']', p);
      if (close == std::string::npos) break;
      std::string inner = s.substr(p + 1, close - p - 1);
      // Skip if inner contains another bracket (outer bracket)
      if (inner.find('[') != std::string::npos) { p = close + 1; continue; }
      if (!inner.empty()) {
        if (!coords_text.isEmpty()) coords_text += "\n";
        coords_text += QString("G%1: %2").arg(goal_num++).arg(QString::fromStdString(inner));
      }
      p = close + 1;
    }
    goal_coords_label_->setText(coords_text);
  } else {
    goal_coords_label_->setText("");
  }
}

void UR10ePanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("velocity_slider", velocity_slider_->value());
  config.mapSetValue("reachability_cloud", reachability_cloud_cb_->isChecked() ? 1 : 0);
  config.mapSetValue("lidar_scan_preview", lidar_scan_preview_cb_->isChecked() ? 1 : 0);
}

void UR10ePanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  int val;
  if (config.mapGetInt("velocity_slider", &val)) {
    velocity_slider_->setValue(val);
  }
  if (config.mapGetInt("reachability_cloud", &val)) {
    reachability_cloud_enabled_ = (val != 0);
    if (reachability_cloud_cb_) {
      reachability_cloud_cb_->setChecked(reachability_cloud_enabled_);
    }
  }
  if (config.mapGetInt("lidar_scan_preview", &val)) {
    lidar_scan_preview_enabled_ = (val != 0);
    if (lidar_scan_preview_cb_) {
      lidar_scan_preview_cb_->setChecked(lidar_scan_preview_enabled_);
    }
  }
}

}  // namespace rviz_ur10e_panel

PLUGINLIB_EXPORT_CLASS(rviz_ur10e_panel::UR10ePanel, rviz_common::Panel)
