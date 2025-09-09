#include "KalmanFilter.hpp"

#include <vector>

#include "Eigen/src/Core/Matrix.h"

/**
 * @brief 构造函数，初始化卡尔曼滤波状态和各类矩阵。
 * @param x 初始x坐标
 * @param y 初始y坐标
 * @param z 初始z坐标
 * @param dt 状态转移步长
 */
KalmanFilter::KalmanFilter(float x, float y, float z, float dt) {
    // 状态转移矩阵，三维匀加速
    m_transition << 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0, 0, 1, 0, 0, dt, 0,
        0, 0.5 * dt * dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0, 0, 1,
        0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1;

    // 过程噪声协方差
    float sigma_a = 1.0f;  // 加速度标准差，可根据实际情况调整

    m_processNoise = Eigen::Matrix<float, 9, 9>::Zero();

    // 单轴过程噪声块
    Eigen::Matrix3f q_block;
    q_block << 0.25f * dt * dt * dt * dt, 0.5f * dt * dt * dt, dt * dt,
        0.5f * dt * dt * dt, dt * dt, dt, dt * dt, dt, 1.0f;

    // 将 q_block 放入三个子块中（x, y, z 方向）
    m_processNoise.block<3, 3>(0, 0) = sigma_a * sigma_a * q_block;  // x方向
    m_processNoise.block<3, 3>(3, 3) = sigma_a * sigma_a * q_block;  // y方向
    m_processNoise.block<3, 3>(6, 6) = sigma_a * sigma_a * q_block;  // z方向

    // 观测矩阵，仅观测位置
    m_observation << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0;

    // 观测噪声协方差
    float sigma_x = 0.05f;  // 单位：米
    float sigma_y = 0.05f;
    float sigma_z = 0.1f;

    m_observationNoise << sigma_x * sigma_x, 0, 0, 0, sigma_y * sigma_y, 0, 0,
        0, sigma_z * sigma_z;

    // 状态初始化，速度为0
    m_state.setZero();
    m_state(0) = x;
    m_state(1) = y;
    m_state(2) = z;
    // m_state(8) = -9.8f;

    // 协方差初始化为单位阵
    m_covariance = Eigen::Matrix<float, 9, 9>::Identity();
}

/**
 * @brief 仅用模型预测推进状态。
 */
void KalmanFilter::predict() {
    // 状态预测：x = F * x
    m_state = m_transition * m_state;
    // 协方差预测：P = F * P * F^T + Q
    m_covariance =
        m_transition * m_covariance * m_transition.transpose() + m_processNoise;
}

/**
 * @brief 用新观测更新状态。
 * @param x 新观测x
 * @param y 新观测y
 * @param z 新观测z
 */
void KalmanFilter::update(float x, float y, float z) {
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
    Eigen::Matrix<float, 9, 3> K =
        m_covariance * m_observation.transpose() * S.inverse();
    // 更新状态
    m_state = m_state + K * innovation;
    // 更新协方差
    m_covariance =
        (Eigen::Matrix<float, 9, 9>::Identity() - K * m_observation) *
        m_covariance;
}

/**
 * @brief 获取当前状态的三维位置。
 * @param x 输出x
 * @param y 输出y
 * @param z 输出z
 */
std::vector<float> KalmanFilter::getState() const {
    return std::vector<float>{m_state(0), m_state(1), m_state(2)};
}

// /**
//  * @brief 预测未来若干步的轨迹。
//  * @param xyz 输出，每步[x, y, z]
//  * @param steps 预测步数
//  */
// void KalmanFilter::predictTrajectory(std::vector<float>& xyz, int steps)
// const {
//     // 拷贝当前状态
//     Eigen::Matrix<float, 6, 1> state_pred = m_state;
//     for (int i = 0; i < steps; ++i) {
//         // 预测一步
//         state_pred = m_transition * state_pred;
//         // 存储预测位置
//         xyz.push_back(state_pred(0));
//         xyz.push_back(state_pred(1));
//         xyz.push_back(state_pred(2));
//     }
// }
