#pragma once

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <vector>

#include "KalmanFilter.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

/**
 * @brief
 * ROS2节点：负责接收篮球检测结果，进行卡尔曼滤波插帧，并周期性发布平滑轨迹。
 *
 * 输入话题：ball_position（Float32MultiArray，每5个float为一组[id, x, y, z,
 * t]） 输出话题：ball_trajectory（Float32MultiArray，插帧后轨迹，格式同上）
 */
class KalmanFilterNode : public rclcpp::Node {
   public:
    /**
     * @brief 构造函数，初始化订阅、发布和定时器。
     */
    KalmanFilterNode();

   private:
    /**
     * @brief 球位置消息回调，处理新观测并维护卡尔曼滤波器。
     * @param msg 输入的Float32MultiArray消息，每5个float为一组[id, x, y, z, t]
     */
    void ballPositionCallback(
        const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg);

    /**
     * @brief 新建卡尔曼滤波器并发布初始轨迹
     * @param ballId 球ID
     * @param x y z 观测位置
     * @param frameNum 当前帧号
     */
    void createNewFilter(int ballId, float x, float y, float z, float frameNum);

    /**
     * @brief 对丢失帧进行插值预测并发布轨迹
     * @param ballId 球ID
     * @param lastFrame 上一帧号
     * @param currentFrame 当前帧号
     */
    void interpolateFrames(int ballId, float lastFrame, float currentFrame);

    /**
     * @brief 发布当前球的轨迹消息
     * @param ballId 球ID
     * @param frameNum 帧号
     */
    void publishTrajectory(int ballId, float frameNum);

    float lastFrameNum = 0;
    bool firstCall = true;

    std::vector<float> lastMeasurement;
    std::vector<float> lastVelocity;

    rclcpp::Time lastMsgTime = this->get_clock()->now();

    std::unordered_map<int, std::unique_ptr<KalmanFilter>> m_kalmanFiltersMap;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
        m_subscription;  ///< 订阅检测结果
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
        m_publisher;  ///< 发布插帧轨迹
};
