#include "ukf/UnscentedKalmanFilterJerk.hpp"

#include <vector>

UnscentedKalmanFilterJerk::UnscentedKalmanFilterJerk(float x, float y, float z,
                                                     float dt) {
    // UKF参数初始化
    L = 12;  // 状态维度
    numSigmaPoints = 2 * L + 1;  // 25个sigma点
    alpha = sqrt(3.0f);
    beta = 2.0f;
    kappa = 3.0f - L;

    // 状态转移矩阵初始化 (12x12)
    updateTransitionMatrix(dt);

    // 观测矩阵初始化 (3x12) - 只观测位置
    m_observation.setZero();
    m_observation.block<3,3>(0,0) = Eigen::Matrix3f::Identity();

    // 协方差初始化
    m_covariance.setIdentity();
    for (int i = 0; i < 3; ++i) {
        m_covariance(i, i) = 0.1f;     // 位置初始方差 (10cm)
        m_covariance(i + 3, i + 3) = 1.0f;   // 速度初始方差 (1m/s)
        m_covariance(i + 6, i + 6) = 4.0f;   // 加速度初始方差 (2m/s²)
        m_covariance(i + 9, i + 9) = 16.0f;  // jerk初始方差 (4m/s³)
    }

    // 过程噪声初始化
    updateProcessNoise(dt);

    // 观测噪声初始化
    m_observationNoise.setIdentity();
    m_observationNoise *= 0.00001f;  // 1cm观测误差

    // sigma点和权重初始化
    sigmaPoints = Eigen::Matrix<float, 12, Eigen::Dynamic>(12, numSigmaPoints);
    weightsMean = Eigen::VectorXf(numSigmaPoints);
    weightsCovariance = Eigen::VectorXf(numSigmaPoints);

    // 状态初始化
    m_state.setZero();
    m_state(0) = x;
    m_state(1) = y;
    m_state(2) = z;
}

void UnscentedKalmanFilterJerk::updateTransitionMatrix(float dt) {
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

void UnscentedKalmanFilterJerk::updateProcessNoise(float dt) {
    // 简化的过程噪声模型
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

void UnscentedKalmanFilterJerk::predict(float dt) {
    // 更新状态转移矩阵和过程噪声
    updateTransitionMatrix(dt);
    updateProcessNoise(dt);

    // 计算sigma点
    Eigen::Matrix<float, 12, 12> sqrtCovariance = m_covariance.llt().matrixL();
    float lambda = alpha * alpha * (L + kappa) - L;

    sigmaPoints.col(0) = m_state;
    for (int i = 0; i < L; ++i) {
        sigmaPoints.col(i + 1) =
            m_state + sqrt((L + lambda)) * sqrtCovariance.col(i);
        sigmaPoints.col(i + 1 + L) =
            m_state - sqrt((L + lambda)) * sqrtCovariance.col(i);
    }

    // 预测sigma点
    for (int i = 0; i < numSigmaPoints; ++i) {
        sigmaPoints.col(i) = m_transition * sigmaPoints.col(i);
    }

    // 计算权重
    weightsMean.resize(numSigmaPoints);
    weightsCovariance.resize(numSigmaPoints);
    weightsMean(0) = lambda / (L + lambda);
    weightsCovariance(0) = weightsMean(0) + (1 - alpha * alpha + beta);
    for (int i = 1; i < numSigmaPoints; ++i) {
        weightsMean(i) = 1 / (2 * (L + lambda));
        weightsCovariance(i) = weightsMean(i);
    }

    // 计算预测均值
    m_state.setZero();
    for (int i = 0; i < numSigmaPoints; ++i) {
        m_state += weightsMean(i) * sigmaPoints.col(i);
    }

    // 计算预测协方差
    m_covariance.setZero();
    for (int i = 0; i < numSigmaPoints; ++i) {
        Eigen::Matrix<float, 12, 1> diff = sigmaPoints.col(i) - m_state;
        m_covariance += weightsCovariance(i) * diff * diff.transpose();
    }
    m_covariance += m_processNoise;
}

void UnscentedKalmanFilterJerk::update(float x, float y, float z) {
    // 计算预测观测均值
    Eigen::Matrix<float, 3, 1> predictedMeasurement =
        Eigen::Matrix<float, 3, 1>::Zero();

    for (int i = 0; i < numSigmaPoints; ++i) {
        Eigen::Matrix<float, 3, 1> meas = m_observation * sigmaPoints.col(i);
        predictedMeasurement += weightsMean(i) * meas;
    }

    // 计算预测观测协方差和交叉协方差
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    Eigen::Matrix<float, 12, 3> crossCovariance =
        Eigen::Matrix<float, 12, 3>::Zero();

    for (int i = 0; i < numSigmaPoints; ++i) {
        Eigen::Matrix<float, 3, 1> meas = m_observation * sigmaPoints.col(i);
        Eigen::Matrix<float, 3, 1> measDiff = meas - predictedMeasurement;
        S += weightsCovariance(i) * measDiff * measDiff.transpose();

        Eigen::Matrix<float, 12, 1> stateDiff = sigmaPoints.col(i) - m_state;
        crossCovariance +=
            weightsCovariance(i) * stateDiff * measDiff.transpose();
    }
    S += m_observationNoise;

    // 计算卡尔曼增益
    Eigen::Matrix<float, 12, 3> K = crossCovariance * S.inverse();

    // 更新状态和协方差
    Eigen::Matrix<float, 3, 1> measurement;
    measurement << x, y, z;
    Eigen::Matrix<float, 3, 1> innovation = measurement - predictedMeasurement;

    m_state += K * innovation;
    m_covariance -= K * S * K.transpose();
}

std::vector<float> UnscentedKalmanFilterJerk::getState() const {
    return std::vector<float>{m_state(0), m_state(1), m_state(2)};
}

void UnscentedKalmanFilterJerk::setVelocity(float vx, float vy, float vz) {
    m_state(3) = vx;
    m_state(4) = vy;
    m_state(5) = vz;
}