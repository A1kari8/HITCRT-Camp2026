#pragma once
#include <Eigen/Dense>

#include "KalmanFilter.hpp"

class UnscentedKalmanFilter {
   public:
    UnscentedKalmanFilter(float x, float y, float z,
                          float dt = 1.0f / (21.0f * FPS_RATE));

    void predict(float dt);
    void update(float x, float y, float z);

    std::vector<float> lastMeasurement;
    std::vector<float> lastVelocity;

    bool firstCall = true;
    bool secondCall = true;

    virtual std::vector<float> getState() const;

    void setStatePos(float x, float y, float z);
    void setStateVelocity(float vx, float vy, float vz);
    void setStateAcceleration(float ax, float ay, float az);
    // std::vector<float> getState() const;
   private:
    Eigen::Matrix<float, 9, 1> m_state;
    Eigen::Matrix<float, 9, 9> m_covariance;
    Eigen::Matrix<float, 9, 9> m_processNoise;
    Eigen::Matrix<float, 3, 9> m_observation;
    Eigen::Matrix<float, 3, 3> m_observationNoise;
    Eigen::Matrix<float, 9, 9> m_transition;    ///< 状态转移矩阵

    float alpha = 1e-3;
    float beta = 2.0f;
    float kappa = 0.0f;
    int L = 9;  // 状态维度
    int numSigmaPoints = 2 * L + 1;
    Eigen::Matrix<float, 9, 19> sigmaPoints;  // 19 = 2*L + 1
    Eigen::VectorXf weightsMean;
    Eigen::VectorXf weightsCovariance;
};
