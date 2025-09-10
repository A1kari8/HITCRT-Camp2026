#include "KalmanFilterNode.hpp"

#include <vector>

#include "KalmanFilter.hpp"
#include "UnscentedKalmanFilter.hpp"

#define FPS_RATE 12


/**
 * @briefvoid KalmanFilterNode::interpolateFrames(int ballId,
                                                     float lastFrame,
                                                     float currentFrame) {
    for (float frame = lastFrame + 1; frame < currentFrame; ++frame) {
        m_kalmanFiltersMap[ballId]->predict(dt_frame * FPS_RATE);
        publishTrajectory(ballId, frame);
    }
}阅、发布和定时器。
 */
KalmanFilterNode::KalmanFilterNode()
    : rclcpp::Node("kalman_filter_node") {
    rclcpp::QoS qos_profile(100);
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    m_subscription =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "ball_position", qos_profile,
            std::bind(&KalmanFilterNode::ballPositionCallback, this,
                      std::placeholders::_1));
    m_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "ball_trajectory", qos_profile);
    dt_frame = (1.0f / 20.0f) / FPS_RATE;  // 视频帧间隔 / 插帧倍数
}

/**
 * @brief 球位置消息回调，处理新观测并维护卡尔曼滤波器。
 * @param msg 输入的Float32MultiArray消息，每5个float为一组[id, x, y, z, t]
 */
void KalmanFilterNode::ballPositionCallback(
    const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {
    if (msg->data.size() < 5) return;

    rclcpp::Time current_time = this->get_clock()->now();
    rclcpp::Duration delta = current_time - lastMsgTime;
    double dt = delta.seconds();
    lastMsgTime = current_time;

    int ballId = static_cast<int>(msg->data[0]);
    float posX = msg->data[1];
    float posY = msg->data[2];
    float posZ = msg->data[3];
    float frameNum = msg->data[4];

    printf("[RECV] ball_position: [%d, %.4f, %.4f, %.4f, %.4f]\n", ballId, posX,
           posY, posZ, frameNum);

    // 新球出现，创建新滤波器
    if (m_kalmanFiltersMap.count(ballId) == 0) {
        createNewFilter(ballId, posX, posY, posZ, frameNum);
        return;
    }

    // 第一次观测，初始化测量和速度
    if (m_kalmanFiltersMap[ballId]->firstCall) {
        lastFrameNum = frameNum;
        m_kalmanFiltersMap[ballId]->lastMeasurement = {posX, posY, posZ};
        m_kalmanFiltersMap[ballId]->lastVelocity = {0.0f, 0.0f, 0.0f};
        m_kalmanFiltersMap[ballId]->firstCall = false;
        return;
    } else if (m_kalmanFiltersMap[ballId]->secondCall) {
        // 第二帧，初始化速度
        lastFrameNum = frameNum;
        m_kalmanFiltersMap[ballId]->lastVelocity = {
            static_cast<float>(
                (posX - m_kalmanFiltersMap[ballId]->lastMeasurement[0]) / dt),
            static_cast<float>(
                (posY - m_kalmanFiltersMap[ballId]->lastMeasurement[1]) / dt),
            static_cast<float>(
                (posZ - m_kalmanFiltersMap[ballId]->lastMeasurement[2]) / dt)};
        m_kalmanFiltersMap[ballId]->setStateVelocity(
            m_kalmanFiltersMap[ballId]->lastVelocity[0],
            m_kalmanFiltersMap[ballId]->lastVelocity[1],
            m_kalmanFiltersMap[ballId]->lastVelocity[2]);
        m_kalmanFiltersMap[ballId]->lastMeasurement = {posX, posY, posZ};
        m_kalmanFiltersMap[ballId]->secondCall = false;
        return;
    }

    // 丢帧插值
    if (frameNum - lastFrameNum > 1) {
        interpolateFrames(ballId, lastFrameNum, frameNum);
    }
    lastFrameNum = frameNum;

    m_kalmanFiltersMap[ballId]->update(posX, posY, posZ);
    // m_kalmanFiltersMap[ballId]->setStatePos(posX, posY, posZ);

    // if (!firstCall) {
    //     m_kalmanFiltersMap[ballId]->setStateVelocity(
    //         (posX - lastMeasurement[0]) / dt, (posY - lastMeasurement[1]) /
    //         dt, (posZ - lastMeasurement[2]) / dt);

    //     m_kalmanFiltersMap[ballId]->setStateAcceleration(
    //         (((posX - lastMeasurement[0]) / dt) - lastVelocity[0]) / dt,
    //         (((posY - lastMeasurement[1]) / dt) - lastVelocity[1]) / dt,
    //         (((posZ - lastMeasurement[2]) / dt) - lastVelocity[2]) / dt
    //     );

    //     lastVelocity = {
    //         static_cast<float>((posX - lastMeasurement[0]) / dt),
    //         static_cast<float>((posY - lastMeasurement[1]) / dt),
    //         static_cast<float>((posZ - lastMeasurement[2]) / dt)
    //     };

    // }

    // 更新速度
    if (!m_kalmanFiltersMap[ballId]->firstCall) {
        m_kalmanFiltersMap[ballId]->lastVelocity = {
            static_cast<float>(
                (posX - m_kalmanFiltersMap[ballId]->lastMeasurement[0]) / dt),
            static_cast<float>(
                (posY - m_kalmanFiltersMap[ballId]->lastMeasurement[1]) / dt),
            static_cast<float>(
                (posZ - m_kalmanFiltersMap[ballId]->lastMeasurement[2]) / dt)};
    }

    m_kalmanFiltersMap[ballId]->lastMeasurement = {posX, posY, posZ};

    // 卡尔曼滤波 predict-update
    m_kalmanFiltersMap[ballId]->predict(dt_frame);
    m_kalmanFiltersMap[ballId]->update(posX, posY, posZ);
    publishTrajectory(ballId, frameNum);

    // 插帧平滑
    for (int i = 0; i < FPS_RATE; ++i) {
        m_kalmanFiltersMap[ballId]->predict(dt_frame);
        publishTrajectory(ballId, frameNum);
    }
}

/**
 * @brief 新建卡尔曼滤波器并发布初始轨迹
 * @param ballId 球ID
 * @param x y z 观测位置
 * @param frameNum 当前帧号
 */
void KalmanFilterNode::createNewFilter(int ballId, float x, float y,
                                                   float z, float frameNum) {
    m_kalmanFiltersMap[ballId] = std::make_unique<FILTER>(x, y, z, dt_frame);
    publishTrajectory(ballId, frameNum);
}

/**
 * @brief 对丢失帧进行插值预测并发布轨迹
 * @param ballId 球ID
 * @param lastFrame 上一帧号
 * @param currentFrame 当前帧号
 */
void KalmanFilterNode::interpolateFrames(int ballId,
                                                     float lastFrame,
                                                     float currentFrame) {
    for (float frame = lastFrame + 1; frame < currentFrame; ++frame) {
        for (int i = 0; i < FPS_RATE; ++i) {
            m_kalmanFiltersMap[ballId]->predict(dt_frame);
            publishTrajectory(ballId, frame);
        }
    }
}

/**
 * @brief 发布当前球的轨迹消息
 * @param ballId 球ID
 * @param frameNum 帧号
 */
void KalmanFilterNode::publishTrajectory(int ballId,
                                                     float frameNum) {
    auto state = m_kalmanFiltersMap[ballId]->getState();
    std_msgs::msg::Float32MultiArray msg;
    msg.data = {static_cast<float>(ballId), state[0], state[1], state[2],
                frameNum};
    printf("[SEND] ball_trajectory: [%d, %.4f, %.4f, %.4f, %.4f]\n", ballId,
           state[0], state[1], state[2], frameNum);
    m_publisher->publish(msg);
}