#pragma once
#include <Eigen/Dense>

#include "ukf/Constant.hpp"

class UnscentedKalmanFilter {
   public:
    UnscentedKalmanFilter(float x, float y, float z,
                          float dt = 1.0f / (29.9f * FPS_RATE));

    void predict(float dt);
    void update(float x, float y, float z);

    std::vector<float> getState() const;
    void setVelocity(float vx, float vy, float vz);

    bool m_firstMessageReceived = true;
    bool m_secondMessageReceived = true;

   private:
    Eigen::Matrix<float, 9, 1> m_state;
    Eigen::Matrix<float, 9, 9> m_covariance;
    Eigen::Matrix<float, 9, 9> m_processNoise;
    Eigen::Matrix<float, 3, 9> m_observation;
    Eigen::Matrix<float, 3, 3> m_observationNoise;
    Eigen::Matrix<float, 9, 9> m_transition;  ///< 状态转移矩阵

    float alpha = 1e-3;
    float beta = 2.0f;
    float kappa = 0.0f;
    int L = 9;  // 状态维度
    int numSigmaPoints = 2 * L + 1;
    Eigen::Matrix<float, 9, 19> sigmaPoints;  // 19 = 2*L + 1
    Eigen::VectorXf weightsMean;
    Eigen::VectorXf weightsCovariance;
};
