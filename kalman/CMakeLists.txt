cmake_minimum_required(VERSION 3.10)

project(kalman)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
add_executable(kalman main.cpp KalmanFilter.cpp KalmanFilterNode.cpp UnscentedKalmanFilter.cpp)
# add_definitions(-DFPS_RATE=2)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(kalman rclcpp std_msgs)