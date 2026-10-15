#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "ArgusCapture.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

class ZedXOneNode : public rclcpp::Node {
 public:
  ZedXOneNode() : Node("zedxone_node") {
    auto qos = rclcpp::QoS(1).best_effort();
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("/zedxone/image_raw", qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/zedxone/camera_info", qos);

    oc::ArgusCameraConfig config;
    config.mDeviceId = 0;
    config.mWidth = 1920;
    config.mHeight = 1080;
    config.mFPS = 30;
    config.verbose_level = 1;
    config.hdr = false;

    oc::ARGUS_STATE state = oc::ARGUS_STATE::CAPTURE_TIMEOUT;
    for (int attempt = 1; attempt <= 10; ++attempt) {
      state = camera_.openCamera(config);
      if (state == oc::ARGUS_STATE::OK) break;
      RCLCPP_WARN(get_logger(), "Camera open attempt %d/10 failed (%s), retrying in 3s...",
                  attempt, ARGUS_STATE2str(state).c_str());
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
    if (state != oc::ARGUS_STATE::OK) {
      RCLCPP_FATAL(get_logger(), "Failed to open ZED X One camera after 10 attempts: %s",
                   ARGUS_STATE2str(state).c_str());
      throw std::runtime_error("ZED X One camera open failed");
    }

    width_ = camera_.getWidth();
    height_ = camera_.getHeight();
    RCLCPP_INFO(get_logger(), "ZED X One opened: %dx%d @ 30fps", width_, height_);

    // Calibration from /usr/local/zed/settings/SN57931814.conf [LEFT_CAM_FHD]
    camera_info_.header.frame_id = "zed2_left_camera_frame";
    camera_info_.width = width_;
    camera_info_.height = height_;
    camera_info_.distortion_model = "plumb_bob";
    camera_info_.d = {-0.0137106, -0.0304117, 0.000282395, -0.000401063, 0.00817325};
    camera_info_.k = {738.615, 0.0, 934.32, 0.0, 738.284, 582.52, 0.0, 0.0, 1.0};
    camera_info_.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    camera_info_.p = {738.615, 0.0, 934.32, 0.0, 0.0, 738.284, 582.52, 0.0, 0.0, 0.0, 1.0, 0.0};

    capture_thread_ = std::thread(&ZedXOneNode::captureLoop, this);
  }

  ~ZedXOneNode() {
    running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    camera_.closeCamera();
  }

 private:
  void captureLoop() {
    const size_t buf_size = static_cast<size_t>(width_) * height_ * 4;
    std::vector<uint8_t> buffer(buf_size);

    auto last_frame = std::chrono::steady_clock::now();
    const auto freeze_timeout = std::chrono::seconds(5);

    while (running_ && rclcpp::ok()) {
      if (!camera_.isNewFrame()) {
        // Detect frozen camera — if no frame for 5s, reopen
        if (std::chrono::steady_clock::now() - last_frame > freeze_timeout) {
          RCLCPP_WARN(get_logger(), "Camera frozen, reopening...");
          camera_.closeCamera();
          std::this_thread::sleep_for(std::chrono::seconds(3));
          oc::ArgusCameraConfig config;
          config.mDeviceId = 0;
          config.mWidth = 1920;
          config.mHeight = 1080;
          config.mFPS = 30;
          config.verbose_level = 1;
          config.hdr = false;
          for (int i = 1; i <= 5; ++i) {
            auto s = camera_.openCamera(config);
            if (s == oc::ARGUS_STATE::OK) { last_frame = std::chrono::steady_clock::now(); break; }
            RCLCPP_WARN(get_logger(), "Reopen attempt %d/5 failed (%s)", i, ARGUS_STATE2str(s).c_str());
            std::this_thread::sleep_for(std::chrono::seconds(3));
          }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        continue;
      }

      last_frame = std::chrono::steady_clock::now();
      std::memcpy(buffer.data(), camera_.getPixels(), buf_size);
      auto now = get_clock()->now();

      sensor_msgs::msg::Image img_msg;
      img_msg.header.stamp = now;
      img_msg.header.frame_id = "zed2_left_camera_frame";
      img_msg.height = height_;
      img_msg.width = width_;
      img_msg.encoding = "bgra8";
      img_msg.is_bigendian = false;
      img_msg.step = width_ * 4;
      img_msg.data.assign(buffer.begin(), buffer.end());
      image_pub_->publish(img_msg);

      camera_info_.header.stamp = now;
      info_pub_->publish(camera_info_);
    }
  }

  oc::ArgusBayerCapture camera_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  sensor_msgs::msg::CameraInfo camera_info_;
  std::thread capture_thread_;
  std::atomic<bool> running_{true};
  int width_ = 0, height_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ZedXOneNode>());
  rclcpp::shutdown();
  return 0;
}
