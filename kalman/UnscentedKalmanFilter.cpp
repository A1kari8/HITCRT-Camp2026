#include "UnscentedKalmanFilter.hpp"

#include <vector>

UnscentedKalmanFilter::UnscentedKalmanFilter(float x, float y, float z,
                                             float dt) {
    // UKF参数初始化
    L = 9;  // 状态维度
    numSigmaPoints = 2 * L + 1;
    alpha = sqrt(3.0f);
    beta = 2.0f;
    kappa = 3.0f - L;

    // 协方差、噪声初始化
    m_covariance = Eigen::Matrix<float, 9, 9>::Identity();
    m_covariance.diagonal() << 0.01f, 0.01f, 0.01f, 1.0f, 1.0f, 1.0f, 0.1f, 0.1f, 0.1f;
    m_processNoise = Eigen::Matrix<float, 9, 9>::Identity();
    m_processNoise.diagonal() << 0.001f, 0.001f, 0.001f, 0.1f, 0.1f, 0.1f, 0.01f, 0.01f, 0.01f;
    m_observationNoise = Eigen::Matrix3f::Identity() * 0.0001f;

    // 状态转移矩阵和观测矩阵初始化
    m_transition << 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0, 0, 1, 0, 0, dt, 0,
        0, 0.5 * dt * dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0, 0, 1,
        0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1;

    m_observation << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0;

    // sigma点和权重初始化
    sigmaPoints = Eigen::Matrix<float, 9, Eigen::Dynamic>(9, numSigmaPoints);
    weightsMean = Eigen::VectorXf(numSigmaPoints);
    weightsCovariance = Eigen::VectorXf(numSigmaPoints);

    m_state.setZero();
    m_state(0) = x;
    m_state(1) = y;
    m_state(2) = z;

    // 初始观测
    lastMeasurement = {0.0f, 0.0f, 0.0f};
}

void UnscentedKalmanFilter::predict(float dt) {
    // 更新状态转移矩阵
    m_transition << 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0,
        0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0,
        0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt * dt,
        0, 0, 0, 1, 0, 0, dt, 0, 0,
        0, 0, 0, 0, 1, 0, 0, dt, 0,
        0, 0, 0, 0, 0, 1, 0, 0, dt,
        0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1;

    // 计算sigma点
    Eigen::Matrix<float, 9, 9> sqrtCovariance = m_covariance.llt().matrixL();
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

    // 计算预测均值和协方差
    weightsMean.resize(numSigmaPoints);
    weightsCovariance.resize(numSigmaPoints);
    weightsMean(0) = lambda / (L + lambda);
    weightsCovariance(0) = weightsMean(0) + (1 - alpha * alpha + beta);
    for (int i = 1; i < numSigmaPoints; ++i) {
        weightsMean(i) = 1 / (2 * (L + lambda));
        weightsCovariance(i) = weightsMean(i);
    }

    m_state.setZero();
    for (int i = 0; i < numSigmaPoints; ++i) {
        m_state += weightsMean(i) * sigmaPoints.col(i);
    }

    m_covariance.setZero();
    for (int i = 0; i < numSigmaPoints; ++i) {
        Eigen::Matrix<float, 9, 1> diff = sigmaPoints.col(i) - m_state;
        m_covariance += weightsCovariance(i) * diff * diff.transpose();
    }
    m_covariance += m_processNoise;
}

void UnscentedKalmanFilter::update(float x, float y, float z) {
    // 更新最后测量值
    lastMeasurement = {x, y, z};

    // 计算预测观测均值和协方差
    Eigen::Matrix<float, 3, 1> predictedMeasurement =
        Eigen::Matrix<float, 3, 1>::Zero();
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    Eigen::Matrix<float, 9, 3> crossCovariance =
        Eigen::Matrix<float, 9, 3>::Zero();

    for (int i = 0; i < numSigmaPoints; ++i) {
        Eigen::Matrix<float, 3, 1> meas = m_observation * sigmaPoints.col(i);
        predictedMeasurement += weightsMean(i) * meas;
    }

    for (int i = 0; i < numSigmaPoints; ++i) {
        Eigen::Matrix<float, 3, 1> meas = m_observation * sigmaPoints.col(i);
        Eigen::Matrix<float, 3, 1> measDiff = meas - predictedMeasurement;
        S += weightsCovariance(i) * measDiff * measDiff.transpose();

        Eigen::Matrix<float, 9, 1> stateDiff = sigmaPoints.col(i) - m_state;
        crossCovariance +=
            weightsCovariance(i) * stateDiff * measDiff.transpose();
    }
    S += m_observationNoise;

    // 计算卡尔曼增益
    Eigen::Matrix<float, 9, 3> K = crossCovariance * S.inverse();

    // 更新状态和协方差
    Eigen::Matrix<float, 3, 1> measurement;
    measurement << x, y, z;
    Eigen::Matrix<float, 3, 1> innovation = measurement - predictedMeasurement;

    m_state += K * innovation;
    m_covariance -= K * S * K.transpose();
}

void UnscentedKalmanFilter::setStatePos(float x, float y, float z) {
    m_state(0) = x;
    m_state(1) = y;
    m_state(2) = z;
}

void UnscentedKalmanFilter::setStateVelocity(float vx, float vy, float vz) {
    m_state(3) = vx;
    m_state(4) = vy;
    m_state(5) = vz;
}

void UnscentedKalmanFilter::setStateAcceleration(float ax, float ay, float az) {
    m_state(6) = ax;
    m_state(7) = ay;
    m_state(8) = az;
}

std::vector<float> UnscentedKalmanFilter::getState() const {
    return std::vector<float>{m_state(0), m_state(1), m_state(2)};
}