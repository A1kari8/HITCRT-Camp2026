#include "ukf/KalmanFilter.hpp"

#include <vector>

KalmanFilter::KalmanFilter(float x, float y, float z, float dt) {
    // 状态转移矩阵，三维匀加速
    // m_transition << 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0, 0, 1, 0, 0, dt, 0,
    //     0, 0.5 * dt * dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0, 0, 1,
    //     0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt,
    //     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    //     0, 0, 1;
    updateTransitionMatrix(dt);

    // 过程噪声协方差（优化：减小sigma_a）
    updateProcessNoise(dt);


    // 观测矩阵
    m_observation << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0;

    float sigma_obs_xy = 0.0000000001f;  // x/y 方向误差（单位：米）
    float sigma_obs_z  = 0.0000000001f;  // z 方向误差（高度误差更大）

    m_observationNoise = Eigen::Matrix3f::Zero();
    m_observationNoise(0,0) = sigma_obs_xy * sigma_obs_xy;
    m_observationNoise(1,1) = sigma_obs_xy * sigma_obs_xy;
    m_observationNoise(2,2) = sigma_obs_z  * sigma_obs_z;

    // 初始化过程噪声矩阵 Q_
    // m_processNoise.setZero();
    // float sigma_a = 0.7; // 加速度噪声标准差（过程噪声来自于学长的篮球技术）
    // for (int i = 0; i < 3; ++i) {
    //     m_processNoise(i, i) = 0.25 * sigma_a * sigma_a;       // 位置噪声
    //     m_processNoise(i + 3, i + 3) = sigma_a * sigma_a;      // 速度噪声
    //     m_processNoise(i + 6, i + 6) = sigma_a * sigma_a;      // 加速度噪声
    // }

    // // 初始化观测噪声矩阵 R_
    // m_observationNoise.setZero();
    // float sigma_pos = 0.0001; // 位置测量误差标准差（视觉定位）
    // for (int i = 0; i < 3; ++i) {
    //     m_observationNoise(i, i) = sigma_pos * sigma_pos;
    // }



    // 状态初始化
    m_state.setZero();
    m_state(0) = x;
    m_state(1) = y;
    m_state(2) = z;

    // 协方差初始化为单位阵
    m_covariance = Eigen::Matrix<float, 9, 9>::Identity();
    m_covariance *= 1e-3;
}

void KalmanFilter::predict(float dt) {
    // 过程噪声协方差（优化：减小sigma_a）
    // updateProcessNoise(dt);

    updateTransitionMatrix(dt);
    // 状态预测：x = F * x
    m_state = m_transition * m_state;
    // 协方差预测：P = F * P * F^T + Q
    m_covariance =
        m_transition * m_covariance * m_transition.transpose() + m_processNoise;
}

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

std::vector<float> KalmanFilter::getState() const {
    return std::vector<float>{m_state(0), m_state(1), m_state(2)};
}

void KalmanFilter::setVelocity(float vx, float vy, float vz) {
    m_state(3) = vx;
    m_state(4) = vy;
    m_state(5) = vz;
}

void KalmanFilter::updateTransitionMatrix(float dt) {
    m_transition.setIdentity();

    // 位置 ← 速度
    m_transition.block<3,3>(0,3) = Eigen::Matrix3f::Identity() * dt;

    // 位置 ← 加速度
    m_transition.block<3,3>(0,6) = Eigen::Matrix3f::Identity() * 0.5f * dt * dt;

    // 速度 ← 加速度
    m_transition.block<3,3>(3,6) = Eigen::Matrix3f::Identity() * dt;
}

void KalmanFilter::updateProcessNoise(float dt) {
    float sigma_a_xy = 4.0f;  // x/y方向加速度标准差（单位：m/s²）
    float sigma_a_z  = 4.0f;  // z方向加速度标准差（跳跃/落地更剧烈）

    Eigen::Matrix3f q_xy, q_z;
    q_xy << 0.25f * dt * dt * dt * dt, 0.5f * dt * dt * dt, dt * dt,
            0.5f * dt * dt * dt,       dt * dt,             dt,
            dt * dt,                   dt,                  1.0f;

    q_z = q_xy;  // 同结构，但乘不同 sigma

    m_processNoise.setZero();
    m_processNoise.block<3,3>(0,0) = sigma_a_xy * sigma_a_xy * q_xy;  // x方向
    m_processNoise.block<3,3>(3,3) = sigma_a_xy * sigma_a_xy * q_xy;  // y方向
    m_processNoise.block<3,3>(6,6) = sigma_a_z  * sigma_a_z  * q_z;   // z方向
}

