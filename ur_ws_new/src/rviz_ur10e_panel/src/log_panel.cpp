#include "rviz_ur10e_panel/log_panel.hpp"

#include <QFont>
#include <QScrollBar>
#include <QTextCharFormat>

#include <chrono>
#include <iomanip>
#include <sstream>

#include <rviz_common/display_context.hpp>
#include <pluginlib/class_list_macros.hpp>

namespace rviz_ur10e_panel
{

LogPanel::LogPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(2, 2, 2, 2);

  log_view_ = new QPlainTextEdit(this);
  log_view_->setReadOnly(true);
  log_view_->setMaximumBlockCount(MAX_LINES);
  log_view_->setFont(QFont("Monospace", 8));
  log_view_->setStyleSheet(
    "QPlainTextEdit { background: #1e1e1e; color: #cccccc; }");
  log_view_->setLineWrapMode(QPlainTextEdit::WidgetWidth);
  layout->addWidget(log_view_);

  flush_timer_ = new QTimer(this);
  connect(flush_timer_, &QTimer::timeout, this, &LogPanel::flushLogs);
  flush_timer_->start(200);
}

LogPanel::~LogPanel()
{
  if (node_) {
    node_.reset();
  }
}

void LogPanel::onInitialize()
{
  setupRos();
}

void LogPanel::setupRos()
{
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  rosout_sub_ = node_->create_subscription<rcl_interfaces::msg::Log>(
    "/rosout", rclcpp::QoS(100).best_effort(),
    [this](rcl_interfaces::msg::Log::SharedPtr msg) {
      if (!shouldShow(msg->name)) return;

      // Format timestamp
      time_t sec = static_cast<time_t>(msg->stamp.sec);
      auto t = std::localtime(&sec);
      std::ostringstream ss;
      ss << std::put_time(t, "%H:%M:%S");

      // Severity prefix and color
      const char * sev;
      const char * color;
      switch (msg->level) {
        case rcl_interfaces::msg::Log::DEBUG:
          sev = "DBG"; color = "#666666"; break;
        case rcl_interfaces::msg::Log::INFO:
          sev = "INF"; color = "#cccccc"; break;
        case rcl_interfaces::msg::Log::WARN:
          sev = "WRN"; color = "#ff9800"; break;
        case rcl_interfaces::msg::Log::ERROR:
          sev = "ERR"; color = "#f44336"; break;
        case rcl_interfaces::msg::Log::FATAL:
          sev = "FTL"; color = "#ff0000"; break;
        default:
          sev = "???"; color = "#cccccc"; break;
      }

      // Short node name (strip leading /)
      std::string node_short = msg->name;
      if (!node_short.empty() && node_short[0] == '/') {
        node_short = node_short.substr(1);
      }
      // Truncate long node names
      if (node_short.size() > 20) {
        node_short = node_short.substr(0, 18) + "..";
      }

      std::ostringstream line;
      line << "<span style='color:" << color << "'>"
           << "[" << ss.str() << "] [" << sev << "] [" << node_short << "] "
           << msg->msg
           << "</span>";

      std::lock_guard<std::mutex> lock(log_mutex_);
      pending_logs_.push_back(line.str());
      // Limit buffer
      while (pending_logs_.size() > 50) {
        pending_logs_.pop_front();
      }
    });
}

bool LogPanel::shouldShow(const std::string & name) const
{
  // Show logs from our nodes + UR driver nodes
  static const char * filters[] = {
    "ur10e", "vision", "curobo", "moveit",
    "ur_ros2_control", "robot_state_publisher",
    "controller_manager", "joint_trajectory_controller",
    "scaled_joint_trajectory_controller",
    "io_and_status_controller",
  };
  for (auto & f : filters) {
    if (name.find(f) != std::string::npos) return true;
  }
  return false;
}

void LogPanel::flushLogs()
{
  std::deque<std::string> batch;
  {
    std::lock_guard<std::mutex> lock(log_mutex_);
    batch.swap(pending_logs_);
  }

  if (batch.empty()) return;

  bool at_bottom = log_view_->verticalScrollBar()->value() >=
                   log_view_->verticalScrollBar()->maximum() - 10;

  for (auto & line : batch) {
    log_view_->appendHtml(QString::fromStdString(line));
  }

  if (at_bottom) {
    log_view_->verticalScrollBar()->setValue(
      log_view_->verticalScrollBar()->maximum());
  }
}

}  // namespace rviz_ur10e_panel

PLUGINLIB_EXPORT_CLASS(rviz_ur10e_panel::LogPanel, rviz_common::Panel)
