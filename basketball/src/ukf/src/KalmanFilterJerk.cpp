#include "ukf/KalmanFilterJerk.hpp"

#include <vector>

KalmanFilterJerk::KalmanFilterJerk(float x, float y, float z, float dt) {
    updateTransitionMatrix(dt);
    updateProcessNoise(dt);

    // 初始化观测矩阵 H (3x12) - 只观测位置
    m_observation.setZero();
    m_observation.block<3,3>(0,0) = Eigen::Matrix3f::Identity();

    // 初始化观测噪声矩阵 R
    m_observationNoise.setZero();
    float sigma_pos = 0.01f; // 位置测量误差标准差（1cm视觉定位误差）
    for (int i = 0; i < 3; ++i) {
        m_observationNoise(i, i) = sigma_pos * sigma_pos;
    }

    // 状态初始化
    m_state.setZero();
    m_state(0) = x;
    m_state(1) = y;
    m_state(2) = z;

    // 协方差初始化 - 为不同状态变量设置合理的初始不确定性
    m_covariance.setIdentity();
    for (int i = 0; i < 3; ++i) {
        m_covariance(i, i) = 0.1f;     // 位置初始方差 (10cm)
        m_covariance(i + 3, i + 3) = 1.0f;   // 速度初始方差 (1m/s)
        m_covariance(i + 6, i + 6) = 4.0f;   // 加速度初始方差 (2m/s²)
        m_covariance(i + 9, i + 9) = 16.0f;  // jerk初始方差 (4m/s³)
    }
}

void KalmanFilterJerk::predict(float dt) {
    updateTransitionMatrix(dt);
    // 状态预测：x = F * x
    m_state = m_transition * m_state;
    // 协方差预测：P = F * P * F^T + Q
    m_covariance =
        m_transition * m_covariance * m_transition.transpose() + m_processNoise;
}

void KalmanFilterJerk::update(float x, float y, float z) {
    // 构造观测向量
    Eigen::Vector3f measurement;
    measurement << x, y, z;
    // 计算创新（观测-预测）
    Eigen::Vector3f innovation = measurement - m_observation * m_state;
    // 创新协方差
    Eigen::Matrix3f S =
        m_observation * m_covariance * m_observation.transpose() +
        m_observationNoise;
    // 卡尔曼增益
    Eigen::Matrix<float, 12, 3> K =
        m_covariance * m_observation.transpose() * S.inverse();
    // 更新状态
    m_state = m_state + K * innovation;
    // 更新协方差
    m_covariance =
        (Eigen::Matrix<float, 12, 12>::Identity() - K * m_observation) *
        m_covariance;
}

std::vector<float> KalmanFilterJerk::getState() const {
    return std::vector<float>{m_state(0), m_state(1), m_state(2)};
}

void KalmanFilterJerk::setVelocity(float vx, float vy, float vz) {
    m_state(3) = vx;
    m_state(4) = vy;
    m_state(5) = vz;
}

void KalmanFilterJerk::updateTransitionMatrix(float dt) {
    m_transition.setIdentity();

    // 位置 ← 速度
    m_transition.block<3,3>(0,3) = Eigen::Matrix3f::Identity() * dt;

    // 位置 ← 加速度
    m_transition.block<3,3>(0,6) = Eigen::Matrix3f::Identity() * 0.5f * dt * dt;

    // 位置 ← jerk
    m_transition.block<3,3>(0,9) = Eigen::Matrix3f::Identity() * (1.0f/6.0f) * dt * dt * dt;

    // 速度 ← 加速度
    m_transition.block<3,3>(3,6) = Eigen::Matrix3f::Identity() * dt;

    // 速度 ← jerk
    m_transition.block<3,3>(3,9) = Eigen::Matrix3f::Identity() * 0.5f * dt * dt;

    // 加速度 ← jerk
    m_transition.block<3,3>(6,9) = Eigen::Matrix3f::Identity() * dt;
}

void KalmanFilterJerk::updateProcessNoise(float dt) {
    // 简化的过程噪声模型
    // 基于位置、速度、加速度、jerk的噪声
    m_processNoise.setZero();

    float sigma_pos = 0.01f;  // 位置过程噪声
    float sigma_vel = 0.1f;   // 速度过程噪声
    float sigma_acc = 1.0f;   // 加速度过程噪声
    float sigma_jerk = 5.0f;  // jerk过程噪声

    // 为每个维度设置对角过程噪声
    for (int i = 0; i < 3; ++i) {
        m_processNoise(i, i) = sigma_pos * sigma_pos * dt;       // 位置噪声
        m_processNoise(i + 3, i + 3) = sigma_vel * sigma_vel * dt; // 速度噪声
        m_processNoise(i + 6, i + 6) = sigma_acc * sigma_acc * dt; // 加速度噪声
        m_processNoise(i + 9, i + 9) = sigma_jerk * sigma_jerk * dt; // jerk噪声
    }
}