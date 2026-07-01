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

#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <ur_msgs/msg/tool_data_msg.hpp>

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
  void onHomeLeft();
  void onHomeRight();
  void onSetHomeCurrent();
  void onDropoff();
  void onExecute();
  void onExecuteMoves();
  void onAddCurrentGoal();
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
  void onCameraSnapshot();
  void onCameraVideoStart();
  void onCameraVideoStop();
  void onGraspSuccess();
  void onGraspFail();
  void onDebugPreviewChanged(int state);
  void onReachabilityCloudChanged(int state);
  void onLidarScanPreviewChanged(int state);
  void onZoneOverlayToggle();
  void onPlanConfirm();
  void onPlanCancel();
  void onLidarScan();
  void onStabilityRecord();
  void updateDisplay();

private:
  struct StabilitySample
  {
    double elapsed_s = 0.0;
    std::array<double, 6> positions{};
    std::array<double, 6> velocities{};
    std::array<double, 6> tracking_errors{};
    std::array<double, 6> wrench{};
  };

  struct StabilityResult
  {
    double score = 0.0;
    double velocity_rms = std::numeric_limits<double>::quiet_NaN();
    double position_jitter_rms = std::numeric_limits<double>::quiet_NaN();
    double tracking_error_rms = std::numeric_limits<double>::quiet_NaN();
    double tracking_error_max = std::numeric_limits<double>::quiet_NaN();
    double force_noise_rms = std::numeric_limits<double>::quiet_NaN();
    double torque_noise_rms = std::numeric_limits<double>::quiet_NaN();
    size_t sample_count = 0;
    bool trajectory_tracking = false;
    bool valid = false;
  };

  void publishCmd(const std::string & cmd);
  void publishOverlayCmd(const std::string & cmd);
  void publishStop();
  void publishGoal(double x, double y, double z,
                   double qw, double qx, double qy, double qz);
  void setupRos();
  void startStabilityRecording();
  void stopStabilityRecording();
  void appendStabilitySampleLocked(
    const std::chrono::steady_clock::time_point & now);
  void writeComputedTrajectoryLocked(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const std::chrono::steady_clock::time_point & now);
  void writeFollowedTrajectoryLocked(
    const control_msgs::msg::JointTrajectoryControllerState & msg,
    const std::chrono::steady_clock::time_point & now);
  StabilityResult calculateStabilityLocked() const;

  // ROS
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr overlay_cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr stop_pub_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr force_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr running_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_info_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr vel_scale_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr calib_check_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr config_sub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;
  rclcpp::Subscription<control_msgs::msg::JointTrajectoryControllerState>::SharedPtr
    controller_state_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr
    trajectory_command_sub_;
  rclcpp::Subscription<ur_msgs::msg::ToolDataMsg>::SharedPtr tool_data_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_temp_sub_;

  // Data
  std::mutex data_mutex_;
  double joint_positions_[6] = {};
  double joint_velocities_[6] = {};
  double joint_efforts_[6] = {};   // per-joint current (A) from /joint_states.effort — heat proxy
  double tracking_errors_[6] = {};
  double tcp_wrench_[6] = {};
  double gripper_forces_[3] = {};
  double tool_temperature_ = 0.0;  // tool/wrist flange temperature (degC)
  bool have_tool_temp_ = false;
  double joint_temperatures_[6] = {};  // per-joint temperature (degC) from /joint_temperatures
  bool have_joint_temps_ = false;
  bool have_tracking_error_ = false;
  bool have_wrench_ = false;
  bool robot_running_ = false;
  double velocity_scale_ = 5.0;
  int goal_count_ = 0;
  std::string latest_goal_;
  std::string goal_coords_str_;
  std::string calib_check_result_;
  std::string robot_config_text_;
  bool plan_waiting_confirm_ = false;
  bool debug_plan_preview_ = true;
  bool reachability_cloud_enabled_ = true;
  bool lidar_scan_preview_enabled_ = true;
  std::string motion_phase_ = "IDLE";
  std::string reacquire_result_;
  std::string last_outcome_;
  std::string last_end_;
  int grab_count_ = 0;
  int slip_count_ = 0;
  int miss_count_ = 0;

  // Stability recording
  bool stability_recording_ = false;
  std::chrono::steady_clock::time_point stability_start_time_;
  std::deque<StabilitySample> stability_window_;
  std::ofstream stability_csv_;
  std::ofstream trajectory_csv_;
  std::string stability_csv_path_;
  std::string trajectory_csv_path_;
  size_t stability_sample_count_ = 0;
  size_t trajectory_id_ = 0;

  // UI
  QLabel * status_label_;
  QLabel * robot_state_label_;
  QLabel * config_label_;
  QLabel * velocity_value_label_;
  QLabel * current_velocity_label_;
  QLabel * goal_count_label_;
  QLabel * latest_goal_label_;
  QLabel * goal_coords_label_;
  QLabel * calib_result_label_;
  QSlider * velocity_slider_;
  QLabel * joint_labels_[6];
  QLabel * force_labels_[3];
  QLabel * joint_current_labels_[6];
  QLabel * joint_temp_labels_[6];
  QLabel * gripper_heat_labels_[3];
  QLabel * tool_temp_label_;
  QLineEdit * x_in_, * y_in_, * z_in_;
  QLineEdit * qw_in_, * qx_in_, * qy_in_, * qz_in_;
  QSpinBox * multi_goal_count_spin_;
  QCheckBox * debug_preview_cb_;
  QCheckBox * reachability_cloud_cb_;
  QCheckBox * lidar_scan_preview_cb_;
  QPushButton * zone_overlay_btn_;
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
  QPushButton * stability_record_btn_;
  QLabel * stability_result_label_;
};

}  // namespace rviz_ur10e_panel

#endif
