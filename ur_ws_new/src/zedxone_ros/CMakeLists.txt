cmake_minimum_required(VERSION 3.8)
project(zedxone_ros)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

find_library(ARGUSCAPTURE_LIB arguscapture REQUIRED HINTS /usr/local/lib)
find_library(DRM_LIB drm REQUIRED)
find_library(X11_LIB X11 REQUIRED)
include_directories(/home/datepalm2/manipulatorsdatepalm/zedx-one-capture/lib/include)

add_executable(zedxone_node src/zedxone_node.cpp)
target_link_libraries(zedxone_node ${ARGUSCAPTURE_LIB} ${DRM_LIB} ${X11_LIB})
ament_target_dependencies(zedxone_node rclcpp sensor_msgs)

install(TARGETS zedxone_node DESTINATION lib/${PROJECT_NAME})

ament_package()
