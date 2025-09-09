
#include "KalmanFilterNode.hpp"

#include <vector>

#ifndef FPS_RATE
#define FPS_RATE 4  // fallback
#endif

KalmanFilterNode::KalmanFilterNode() : rclcpp::Node("kalman_filter_node") {
    rclcpp::QoS qos_profile(100);
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

    m_subscription =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "ball_position", qos_profile,
            std::bind(&KalmanFilterNode::ballPositionCallback, this,
                      std::placeholders::_1));
    m_publisher = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "ball_trajectory", qos_profile);
}

void KalmanFilterNode::ballPositionCallback(
    const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {
    if (msg->data.size() < 5) return;

    rclcpp::Time current_time = this->get_clock()->now();
    rclcpp::Duration delta = current_time - lastMsgTime;
    double dt = delta.seconds();  // 秒为单位
    lastMsgTime = current_time;

    int ballId = static_cast<int>(msg->data[0]);
    float posX = msg->data[1];
    float posY = msg->data[2];
    float posZ = msg->data[3];
    float frameNum = msg->data[4];

    if (firstCall) {
        lastFrameNum = frameNum;
        lastMeasurement = {posX, posY, posZ};
        lastVelocity = {0.0f, 0.0f, 0.0f};
        firstCall = false;
    }

    printf("[RECV] ball_position: [%d, %.4f, %.4f, %.4f, %.4f]\n", ballId, posX,
           posY, posZ, frameNum);

    if (m_kalmanFiltersMap.count(ballId) == 0) {
        createNewFilter(ballId, posX, posY, posZ, frameNum);
        return;
    }

    if (frameNum - lastFrameNum > 1) {
        interpolateFrames(ballId, lastFrameNum, frameNum);
    }
    lastFrameNum = frameNum;

    m_kalmanFiltersMap[ballId]->update(posX, posY, posZ);
    m_kalmanFiltersMap[ballId]->setStatePos(posX, posY, posZ);
    if (!firstCall) {
        m_kalmanFiltersMap[ballId]->setStateVelocity(
            (posX - lastMeasurement[0]) / dt, (posY - lastMeasurement[1]) / dt,
            (posZ - lastMeasurement[2]) / dt);

        m_kalmanFiltersMap[ballId]->setStateAcceleration(
            (((posX - lastMeasurement[0]) / dt) - lastVelocity[0]) / dt,
            (((posY - lastMeasurement[1]) / dt) - lastVelocity[1]) / dt,
            (((posZ - lastMeasurement[2]) / dt) - lastVelocity[2]) / dt
        );

        lastVelocity = {
            static_cast<float>((posX - lastMeasurement[0]) / dt),
            static_cast<float>((posY - lastMeasurement[1]) / dt),
            static_cast<float>((posZ - lastMeasurement[2]) / dt)
        };
        
    }

    lastMeasurement = {posX, posY, posZ};

    m_kalmanFiltersMap[ballId]->predict();
    m_kalmanFiltersMap[ballId]->update(posX, posY, posZ);
    publishTrajectory(ballId, frameNum);

    for (int i = 0; i < FPS_RATE; ++i) {
        m_kalmanFiltersMap[ballId]->predict();
        publishTrajectory(ballId, frameNum);
    }
}

void KalmanFilterNode::createNewFilter(int ballId, float x, float y, float z,
                                       float frameNum) {
    m_kalmanFiltersMap[ballId] = std::make_unique<KalmanFilter>(x, y, z);
    publishTrajectory(ballId, frameNum);
}

void KalmanFilterNode::interpolateFrames(int ballId, float lastFrame,
                                         float currentFrame) {
    for (int frame = lastFrame + 1; frame < currentFrame; ++frame) {
        for (int i = 0; i < FPS_RATE + 1; ++i) {
            m_kalmanFiltersMap[ballId]->predict();
            publishTrajectory(ballId, frame);
        }
    }
}

void KalmanFilterNode::publishTrajectory(int ballId, float frameNum) {
    auto state = m_kalmanFiltersMap[ballId]->getState();
    std_msgs::msg::Float32MultiArray msg;
    msg.data = {static_cast<float>(ballId), state[0], state[1], state[2],
                frameNum};
    printf("[SEND] ball_trajectory: [%d, %.4f, %.4f, %.4f, %.4f]\n", ballId,
           state[0], state[1], state[2], frameNum);
    m_publisher->publish(msg);
}
