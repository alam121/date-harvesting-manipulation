#ifndef RVIZ_UR10E_LOG_PANEL_HPP
#define RVIZ_UR10E_LOG_PANEL_HPP

#include <QWidget>
#include <QPlainTextEdit>
#include <QTimer>
#include <QVBoxLayout>

#include <deque>
#include <mutex>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rcl_interfaces/msg/log.hpp>

namespace rviz_ur10e_panel
{

class LogPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit LogPanel(QWidget * parent = nullptr);
  ~LogPanel() override;

  void onInitialize() override;

private Q_SLOTS:
  void flushLogs();

private:
  void setupRos();
  bool shouldShow(const std::string & node_name) const;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<rcl_interfaces::msg::Log>::SharedPtr rosout_sub_;

  std::mutex log_mutex_;
  std::deque<std::string> pending_logs_;

  QPlainTextEdit * log_view_;
  QTimer * flush_timer_;

  static constexpr int MAX_LINES = 500;
};

}  // namespace rviz_ur10e_panel

#endif
