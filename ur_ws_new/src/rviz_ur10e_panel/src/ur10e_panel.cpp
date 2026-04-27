#include "rviz_ur10e_panel/ur10e_panel.hpp"

#include <QGroupBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFont>
#include <QApplication>
#include <QMessageBox>

#include <cmath>
#include <csignal>
#include <sstream>
#include <unistd.h>
#include <sys/types.h>

#include <rclcpp/qos.hpp>
#include <rviz_common/display_context.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace rviz_ur10e_panel
{

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
  layout->addWidget(harvest_group);

  // Emergency Stop
  auto * stop_btn = new QPushButton("EMERGENCY STOP");
  stop_btn->setStyleSheet(
    "background-color: #d32f2f; color: white; font-weight: bold; "
    "font-size: 11pt; padding: 10px;");
  connect(stop_btn, &QPushButton::clicked, this, &UR10ePanel::onStop);
  layout->addWidget(stop_btn);

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

  auto * exec_btn = new QPushButton("Execute");
  exec_btn->setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;");
  connect(exec_btn, &QPushButton::clicked, this, &UR10ePanel::onExecute);
  motion_layout->addWidget(exec_btn, 1, 0);

  auto * clear_btn = new QPushButton("Clear");
  clear_btn->setStyleSheet("background-color: #9e9e9e; color: white;");
  connect(clear_btn, &QPushButton::clicked, this, &UR10ePanel::onClear);
  motion_layout->addWidget(clear_btn, 1, 1);

  auto * check_calib_btn = new QPushButton("Check Calibration (trunk)");
  check_calib_btn->setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;");
  check_calib_btn->setToolTip("Check hand-eye calibration using the detected trunk as the reference");
  connect(check_calib_btn, &QPushButton::clicked, this, &UR10ePanel::onCheckCalibration);
  motion_layout->addWidget(check_calib_btn, 2, 0, 1, 2);

  auto * sub_btn = new QPushButton("Subscribe (S)");
  sub_btn->setStyleSheet("background-color: #7b1fa2; color: white; font-weight: bold;");
  connect(sub_btn, &QPushButton::clicked, this, &UR10ePanel::onSubscribe);
  motion_layout->addWidget(sub_btn, 3, 0);

  auto * sub_multi_btn = new QPushButton("Sub Multi (M)");
  sub_multi_btn->setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;");
  connect(sub_multi_btn, &QPushButton::clicked, this, &UR10ePanel::onSubscribeMulti);
  motion_layout->addWidget(sub_multi_btn, 3, 1);

  calib_result_label_ = new QLabel("—");
  calib_result_label_->setWordWrap(true);
  calib_result_label_->setStyleSheet(
    "font-size: 9pt; padding: 3px; background: #f3e5f5; border-radius: 4px;");
  motion_layout->addWidget(calib_result_label_, 4, 0, 1, 2);

  debug_preview_cb_ = new QCheckBox("Debug Plan Preview");
  debug_preview_cb_->setChecked(true);
  debug_preview_cb_->setStyleSheet("font-weight: bold; font-size: 10pt; padding: 4px;");
  debug_preview_cb_->setToolTip("Show full plan in RViz before executing");
  connect(debug_preview_cb_, &QCheckBox::stateChanged, this, &UR10ePanel::onDebugPreviewChanged);
  motion_layout->addWidget(debug_preview_cb_, 5, 0, 1, 2);

  plan_confirm_btn_ = new QPushButton("Confirm Plan");
  plan_confirm_btn_->setStyleSheet(
    "background-color: #4caf50; color: white; font-weight: bold; "
    "font-size: 11pt; padding: 8px;");
  connect(plan_confirm_btn_, &QPushButton::clicked, this, &UR10ePanel::onPlanConfirm);
  plan_confirm_btn_->setVisible(false);
  motion_layout->addWidget(plan_confirm_btn_, 6, 0);

  plan_cancel_btn_ = new QPushButton("Cancel Plan");
  plan_cancel_btn_->setStyleSheet(
    "background-color: #d32f2f; color: white; font-weight: bold; "
    "font-size: 11pt; padding: 8px;");
  connect(plan_cancel_btn_, &QPushButton::clicked, this, &UR10ePanel::onPlanCancel);
  plan_cancel_btn_->setVisible(false);
  motion_layout->addWidget(plan_cancel_btn_, 6, 1);

  layout->addWidget(motion_group);

  // Gripper
  auto * gripper_group = new QGroupBox("Gripper");
  auto * gripper_layout = new QHBoxLayout(gripper_group);

  auto * open_btn = new QPushButton("Open");
  connect(open_btn, &QPushButton::clicked, this, &UR10ePanel::onGripperOpen);
  gripper_layout->addWidget(open_btn);

  auto * close_btn = new QPushButton("Close");
  connect(close_btn, &QPushButton::clicked, this, &UR10ePanel::onGripperClose);
  gripper_layout->addWidget(close_btn);

  layout->addWidget(gripper_group);

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

  layout->addWidget(vel_group);

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

  auto * send_btn = new QPushButton("Send Goal");
  send_btn->setStyleSheet("background-color: #0277bd; color: white; font-weight: bold;");
  connect(send_btn, &QPushButton::clicked, this, &UR10ePanel::onSendGoal);
  goal_layout->addWidget(send_btn, 4, 0, 1, 4);

  layout->addWidget(goal_group);

  // Capture
  auto * capture_group = new QGroupBox("Capture");
  auto * capture_layout = new QHBoxLayout(capture_group);

  auto * cap_btn = new QPushButton("Start (10s)");
  connect(cap_btn, &QPushButton::clicked, this, &UR10ePanel::onCapture);
  capture_layout->addWidget(cap_btn);

  auto * cap_stop = new QPushButton("Stop");
  connect(cap_stop, &QPushButton::clicked, this, &UR10ePanel::onCaptureStop);
  capture_layout->addWidget(cap_stop);

  layout->addWidget(capture_group);

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

  layout->addWidget(grasp_group);

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

  layout->addWidget(sys_group);

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
  layout->addWidget(joint_group);

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
  layout->addWidget(force_group);

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

  layout->addWidget(goals_group);

  layout->addStretch(1);
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

  auto goal_qos = rclcpp::QoS(1)
    .reliability(rclcpp::ReliabilityPolicy::BestEffort)
    .durability(rclcpp::DurabilityPolicy::Volatile);
  goal_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
    "/external_goal_pose", goal_qos);

  stop_pub_ = node_->create_publisher<std_msgs::msg::Bool>("/emergency_stop", 10);

  joint_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [this](sensor_msgs::msg::JointState::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (size_t i = 0; i < std::min(msg->position.size(), size_t(6)); i++) {
        joint_positions_[i] = msg->position[i];
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

  running_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
    "/io_and_status_controller/robot_program_running", 10,
    [this](std_msgs::msg::Bool::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      robot_running_ = msg->data;
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
      // Extract goals array for coordinate display
      pos = data.find("\"goals\"");
      if (pos != std::string::npos) {
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
}

void UR10ePanel::publishCmd(const std::string & cmd)
{
  if (!cmd_pub_) return;
  auto msg = std_msgs::msg::String();
  msg.data = cmd;
  cmd_pub_->publish(msg);
  status_label_->setText(QString::fromStdString("Sent: " + cmd));
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
void UR10ePanel::onDropoff() { publishCmd("dropoff"); }
void UR10ePanel::onExecute() { publishCmd("execute"); }
void UR10ePanel::onClear() { publishCmd("clear"); }
void UR10ePanel::onCheckCalibration() { publishCmd("check_calibration"); }
void UR10ePanel::onGripperOpen() { publishCmd("open"); }
void UR10ePanel::onGripperClose() { publishCmd("close"); }
void UR10ePanel::onCapture() { publishCmd("capture 10"); }
void UR10ePanel::onCaptureStop() { publishCmd("capture_stop"); }
void UR10ePanel::onSubscribe() { publishCmd("subscribe"); }
void UR10ePanel::onSubscribeMulti() { publishCmd("subscribe_multi"); }
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
void UR10ePanel::onGraspSuccess() { publishCmd("grasp_success"); }
void UR10ePanel::onGraspFail() { publishCmd("grasp_fail"); }
void UR10ePanel::onPlanConfirm() { publishCmd("plan_confirm"); }
void UR10ePanel::onPlanCancel() { publishCmd("plan_cancel"); }

void UR10ePanel::onDebugPreviewChanged(int state)
{
  publishCmd(state == Qt::Checked ? "set_debug_preview true" : "set_debug_preview false");
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
  status_label_->setText(QString("Goal sent: (%1, %2, %3)").arg(x, 0, 'f', 3).arg(y, 0, 'f', 3).arg(z, 0, 'f', 3));
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

  // Motion phase badge
  {
    const char * bg = "#455a64", * fg = "#eceff1";
    if      (motion_phase_ == "APPROACH")  { bg = "#1565c0"; fg = "#e3f2fd"; }
    else if (motion_phase_ == "REACQUIRE") { bg = "#6a1b9a"; fg = "#f3e5f5"; }
    else if (motion_phase_ == "FINAL")     { bg = "#e65100"; fg = "#fff3e0"; }
    else if (motion_phase_ == "REVERSING") { bg = "#558b2f"; fg = "#f1f8e9"; }
    else if (motion_phase_ == "DROPOFF")   { bg = "#00838f"; fg = "#e0f7fa"; }
    else if (motion_phase_ == "HOME")      { bg = "#2e7d32"; fg = "#e8f5e9"; }
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

  // Goals
  goal_count_label_->setText(QString("Goals: %1").arg(goal_count_));
  current_velocity_label_->setText(QString("Velocity: %1x").arg(velocity_scale_, 0, 'f', 1));
  if (!latest_goal_.empty()) {
    latest_goal_label_->setText(QString("Latest: %1").arg(
      QString::fromStdString(latest_goal_)));
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

  // Show goal coordinates
  if (!goal_coords_str_.empty() && goal_coords_str_ != "[]") {
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
}

void UR10ePanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  int val;
  if (config.mapGetInt("velocity_slider", &val)) {
    velocity_slider_->setValue(val);
  }
}

}  // namespace rviz_ur10e_panel

PLUGINLIB_EXPORT_CLASS(rviz_ur10e_panel::UR10ePanel, rviz_common::Panel)
