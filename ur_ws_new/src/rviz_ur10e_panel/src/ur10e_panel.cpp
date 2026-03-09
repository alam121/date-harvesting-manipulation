#include "rviz_ur10e_panel/ur10e_panel.hpp"

#include <QGroupBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFont>
#include <QApplication>

#include <cmath>
#include <sstream>

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

  auto * sub_btn = new QPushButton("Subscribe (S)");
  sub_btn->setStyleSheet("background-color: #7b1fa2; color: white; font-weight: bold;");
  connect(sub_btn, &QPushButton::clicked, this, &UR10ePanel::onSubscribe);
  motion_layout->addWidget(sub_btn, 2, 0);

  auto * sub_multi_btn = new QPushButton("Sub Multi (M)");
  sub_multi_btn->setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;");
  connect(sub_multi_btn, &QPushButton::clicked, this, &UR10ePanel::onSubscribeMulti);
  motion_layout->addWidget(sub_multi_btn, 2, 1);

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
void UR10ePanel::onGripperOpen() { publishCmd("open"); }
void UR10ePanel::onGripperClose() { publishCmd("close"); }
void UR10ePanel::onCapture() { publishCmd("capture 10"); }
void UR10ePanel::onCaptureStop() { publishCmd("capture_stop"); }
void UR10ePanel::onSubscribe() { publishCmd("subscribe"); }
void UR10ePanel::onSubscribeMulti() { publishCmd("subscribe_multi"); }
void UR10ePanel::onUpdateVoxel() { publishCmd("update_voxel"); }
void UR10ePanel::onExit() { publishCmd("exit"); }
void UR10ePanel::onRefreshMain() { publishCmd("refresh_main"); }
void UR10ePanel::onRefreshCamera() { publishCmd("refresh_camera"); }
void UR10ePanel::onGraspSuccess() { publishCmd("grasp_success"); }
void UR10ePanel::onGraspFail() { publishCmd("grasp_fail"); }

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
