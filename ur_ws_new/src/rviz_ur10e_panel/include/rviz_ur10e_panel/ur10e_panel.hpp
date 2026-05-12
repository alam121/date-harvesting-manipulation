#ifndef RVIZ_UR10E_PANEL_HPP
#define RVIZ_UR10E_PANEL_HPP

#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>
#include <QKeyEvent>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace rviz_ur10e_panel
{

class UR10ePanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit UR10ePanel(QWidget * parent = nullptr);
  ~UR10ePanel() override;

  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;
  bool eventFilter(QObject * obj, QEvent * event) override;

private Q_SLOTS:
  void onStop();
  void onHome();
  void onDropoff();
  void onExecute();
  void onClear();
  void onCheckCalibration();
  void onGripperOpen();
  void onGripperClose();
  void onSendGoal();
  void onVelocitySliderChanged(int value);
  void onApplyVelocity();
  void onVelocityPreset();
  void onCapture();
  void onCaptureStop();
  void onSubscribe();
  void onSubscribeMulti();
  void onUpdateVoxel();
  void onExit();
  void onRefreshMain();
  void onRefreshCamera();
  void onGraspSuccess();
  void onGraspFail();
  void onDebugPreviewChanged(int state);
  void onPlanConfirm();
  void onPlanCancel();
  void updateDisplay();

private:
  void publishCmd(const std::string & cmd);
  void publishStop();
  void publishGoal(double x, double y, double z,
                   double qw, double qx, double qy, double qz);
  void setupRos();

  // ROS
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stop_pub_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr force_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr running_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_info_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr vel_scale_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr calib_check_sub_;

  // Data
  std::mutex data_mutex_;
  double joint_positions_[6] = {};
  double gripper_forces_[3] = {};
  bool robot_running_ = false;
  double velocity_scale_ = 5.0;
  int goal_count_ = 0;
  std::string latest_goal_;
  std::string goal_coords_str_;
  std::string calib_check_result_;
  bool plan_waiting_confirm_ = false;
  bool debug_plan_preview_ = true;
  std::string motion_phase_ = "IDLE";
  std::string reacquire_result_;
  std::string last_outcome_;
  std::string last_end_;
  int grab_count_ = 0;
  int slip_count_ = 0;
  int miss_count_ = 0;

  // UI
  QLabel * status_label_;
  QLabel * robot_state_label_;
  QLabel * velocity_value_label_;
  QLabel * current_velocity_label_;
  QLabel * goal_count_label_;
  QLabel * latest_goal_label_;
  QLabel * goal_coords_label_;
  QLabel * calib_result_label_;
  QSlider * velocity_slider_;
  QLabel * joint_labels_[6];
  QLabel * force_labels_[3];
  QLineEdit * x_in_, * y_in_, * z_in_;
  QLineEdit * qw_in_, * qx_in_, * qy_in_, * qz_in_;
  QSpinBox * multi_goal_count_spin_;
  QCheckBox * debug_preview_cb_;
  QPushButton * plan_confirm_btn_;
  QPushButton * plan_cancel_btn_;
  QTimer * update_timer_;
  QLabel * phase_label_;
  QLabel * reacq_label_;
  QLabel * harvest_banner_;
  QLabel * grab_count_label_;
  QLabel * slip_count_label_;
  QLabel * miss_count_label_;
  QLabel * total_count_label_;
};

}  // namespace rviz_ur10e_panel

#endif
