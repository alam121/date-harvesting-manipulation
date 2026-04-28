#ifndef GRIPPER_PANEL_HPP
#define GRIPPER_PANEL_HPP

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QPaintEvent>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

namespace rviz_ur10e_panel
{

class GripperWidget : public QWidget
{
  Q_OBJECT
public:
  explicit GripperWidget(QWidget * parent = nullptr);
  void setForces(float l, float c, float r);
  void setPhase(const std::string & phase);
  void setClosureData(bool stopped_early, int first_contact, int total_steps, int closure_step);

protected:
  void paintEvent(QPaintEvent *) override;

private:
  float    forces_[3]    = {0, 0, 0};
  std::string phase_     = "IDLE";
  bool     closed_       = false;
  bool     stopped_early_= false;
  int      first_contact_= -1;
  int      total_steps_  = 10;
  int      closure_step_ = -1;
};

class GripperPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit GripperPanel(QWidget * parent = nullptr);
  ~GripperPanel() override;
  void onInitialize() override;

private Q_SLOTS:
  void updateDisplay();

private:
  void setupRos();

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr force_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr             info_sub_;

  std::mutex  mutex_;
  float       forces_[3]     = {0,0,0};
  std::string phase_         = "IDLE";
  bool        stopped_early_ = false;
  int         first_contact_ = -1;
  int         total_steps_   = 10;
  int         closure_step_  = -1;

  GripperWidget * gw_;
  QLabel        * state_lbl_;
  QLabel        * quality_lbl_;
  QTimer        * timer_;
};

}  // namespace rviz_ur10e_panel
#endif
