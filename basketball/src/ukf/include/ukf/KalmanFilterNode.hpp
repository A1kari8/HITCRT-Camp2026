#pragma once

#include <cstdio>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>

#include "std_msgs/msg/float32_multi_array.hpp"
#include "ukf/Constant.hpp"
#include "ukf/KalmanFilter.hpp"
#include "ukf/UnscentedKalmanFilter.hpp"

#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

/**
 * @brief
 * ROS2节点：负责接收篮球检测结果，进行卡尔曼滤波插帧，并周期性发布平滑轨迹。
 *
 * 输入话题：ball_position（Float32MultiArray，每5个float为一组[id, x, y, z,
 * t]） 输出话题：ball_trajectory（Float32MultiArray，插帧后轨迹，格式同上）
 *
 */
template <typename FilterType>
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

    void fpsCallback(
        const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg);

    /**
     * @brief 新建卡尔曼滤波器并发布初始轨迹
     * @param ballId 球ID
     * @param x y z 观测位置
     * @param frameNum 当前帧号
     */
    void createNewFilter(int ballId, float x, float y, float z, int frameNum);

    /**
     * @brief 对丢失帧进行插值预测并发布轨迹
     * @param ballId 球ID
     * @param lastFrame 上一帧号
     * @param currentFrame 当前帧号
     */
    void interpolateFrames(int ballId, int lastFrame, int currentFrame);

    /**
     * @brief 发布当前球的轨迹消息
     * @param ballId 球ID
     * @param frameNum 帧号
     */
    void publishTrajectory(int ballId, int frameNum);

    void publish_points(float posX, float posY, float posZ, int ballId);

    int lastFrameNum = 0;
    bool isFirstMessage = true;
    bool isSecondMessage = true;

    float dt_frame;
    float dt_origin;

    std::unordered_map<int, std::unique_ptr<FilterType>> m_kalmanFiltersMap;

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
        m_subscription;  ///< 订阅检测结果

    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
        m_publisher;  ///< 发布插帧轨迹

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
        m_fpsSubscription;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
        m_markerPublisher;  ///< 发布RViz可视化Marker
};

template <typename FilterType>
KalmanFilterNode<FilterType>::KalmanFilterNode()
    : rclcpp::Node("kalman_filter_node") {
    rclcpp::QoS qos_profile(10000);
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    m_subscription =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "ball_position", qos_profile,
            std::bind(&KalmanFilterNode<FilterType>::ballPositionCallback, this,
                      std::placeholders::_1));
    m_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "ball_trajectory", qos_profile);

    m_fpsSubscription =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "ball_fps", qos_profile,
            std::bind(&KalmanFilterNode<FilterType>::fpsCallback, this,
                      std::placeholders::_1));

    m_markerPublisher =
        this->create_publisher<visualization_msgs::msg::Marker>(
            "ball_marker", qos_profile);

    dt_frame =
        (1.0f / 29.9f) / static_cast<float>(FPS_RATE);  // 视频帧间隔 / 插帧倍数
}

template <typename FilterType>
void KalmanFilterNode<FilterType>::fpsCallback(
    const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {
    if (msg->data.size() < 1) return;

    dt_frame = (1.0f / static_cast<float>(msg->data[0])) /
               static_cast<float>(FPS_RATE);  // 视频帧间隔 / 插帧倍数
    dt_origin = 1.0f / static_cast<float>(msg->data[0]);
    printf("[INFO] dt_frame: %.6f, FPS: %.2f\n", dt_frame,
           static_cast<float>(msg->data[0]));
}

template <typename FilterType>
void KalmanFilterNode<FilterType>::ballPositionCallback(
    const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {
    if (msg->data.size() < 5) return;

    int ballId = static_cast<int>(msg->data[0]);
    float posX = msg->data[1];
    float posY = msg->data[2];
    float posZ = msg->data[3];
    int frameNum = static_cast<int>(msg->data[4]);

    printf("[RECV] ball_position: [%d, %.4f, %.4f, %.4f, %d]\n", ballId, posX,
           posY, posZ, frameNum);

    // 新球出现，创建新滤波器
    if (m_kalmanFiltersMap.count(ballId) == 0) {
        createNewFilter(ballId, posX, posY, posZ, frameNum);
        return;
    }

    if (m_kalmanFiltersMap[ballId]->m_firstMessageReceived) {
        if (isFirstMessage) {
            lastFrameNum = frameNum;
            isFirstMessage = false;
        }
        m_kalmanFiltersMap[ballId]->m_firstMessageReceived = false;
        return;
    } else if (m_kalmanFiltersMap[ballId]->m_secondMessageReceived) {
        if (isSecondMessage) {
            lastFrameNum = frameNum;
            isSecondMessage = false;
        }
        m_kalmanFiltersMap[ballId]->setVelocity(
            (posX - m_kalmanFiltersMap[ballId]->getState()[0]) / dt_origin,
            (posY - m_kalmanFiltersMap[ballId]->getState()[1]) / dt_origin,
            (posZ - m_kalmanFiltersMap[ballId]->getState()[2]) / dt_origin
        );
        m_kalmanFiltersMap[ballId]->m_secondMessageReceived = false;
    }

    // 丢帧插值
    if (frameNum - lastFrameNum > 1) {
        printf("\033[33m[WARN] detect lose\033[0m\n");
        interpolateFrames(ballId, lastFrameNum, frameNum);
    }
    lastFrameNum = frameNum;

    m_kalmanFiltersMap[ballId]->predict(dt_frame);
    m_kalmanFiltersMap[ballId]->update(posX, posY, posZ);
    publishTrajectory(ballId, frameNum);

    // 插帧
    for (int i = 1; i < FPS_RATE; ++i) {
        m_kalmanFiltersMap[ballId]->predict(dt_frame);
        publishTrajectory(ballId, frameNum);
    }
}

template <typename FilterType>
void KalmanFilterNode<FilterType>::createNewFilter(int ballId, float x, float y,
                                                   float z, int frameNum) {
    m_kalmanFiltersMap[ballId] =
        std::make_unique<FilterType>(x, y, z, dt_frame);
    publishTrajectory(ballId, frameNum);
}

template <typename FilterType>
void KalmanFilterNode<FilterType>::interpolateFrames(int ballId, int lastFrame,
                                                     int currentFrame) {
    for (int frame = lastFrame + 1; frame < currentFrame; ++frame) {
        for (int i = 0; i < FPS_RATE; ++i) {
            m_kalmanFiltersMap[ballId]->predict(dt_frame);
            publishTrajectory(ballId, frame);
        }
    }
}

template <typename FilterType>
void KalmanFilterNode<FilterType>::publishTrajectory(int ballId, int frameNum) {
    auto state = m_kalmanFiltersMap[ballId]->getState();
    std_msgs::msg::Float32MultiArray msg;
    msg.data = {static_cast<float>(ballId), state[0], state[1], state[2],
                static_cast<float>(frameNum)};
    printf("[SEND] ball_trajectory: [%d, %.4f, %.4f, %.4f, %d]\n", ballId,
           state[0], state[1], state[2], frameNum);
    m_publisher->publish(msg);
    publish_points(state[0], state[1], state[2], ballId);
}

template <typename FilterType>
void KalmanFilterNode<FilterType>::publish_points(float posX, float posY, float posZ, int ballId) {
        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "points";
        marker.id = this->now().nanoseconds();
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = posX;
        marker.pose.position.y = posZ;
        marker.pose.position.z = -posY;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;

        marker.color.r = (ballId % 3 == 0) ? 1.0 : 0.0;
        marker.color.g = (ballId % 3 == 1) ? 1.0 : 0.0;
        marker.color.b = (ballId % 3 == 2) ? 1.0 : 0.0;
        marker.color.a = 1.0;

        m_markerPublisher->publish(marker);
}