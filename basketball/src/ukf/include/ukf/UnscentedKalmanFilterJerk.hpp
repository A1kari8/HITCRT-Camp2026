#pragma once
#include <Eigen/Dense>
#include <vector>

#include "ukf/Constant.hpp"

/**
 * @brief 12维无迹卡尔曼滤波器（包含jerk）
 * 状态向量：[x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz]
 */
class UnscentedKalmanFilterJerk {
   public:
    /**
     * @brief 初始化
     * @param x 初始x坐标
     * @param y 初始y坐标
     * @param z 初始z坐标
     * @param dt 状态转移步长
     */
    UnscentedKalmanFilterJerk(float x, float y, float z,
                              float dt = 1.0f / (29.8f * FPS_RATE));

    /**
     * @brief 仅用预测状态。
     */
    void predict(float dt);

    /**
     * @brief 用新观测更新状态。
     * @param x 新观测x
     * @param y 新观测y
     * @param z 新观测z
     */
    void update(float x, float y, float z);

    /**
     * @brief 获取当前状态的三维位置。
     */
    std::vector<float> getState() const;

    void setVelocity(float vx, float vy, float vz);

    void updateTransitionMatrix(float dt);
    void updateProcessNoise(float dt);

    bool m_firstMessageReceived = true;
    bool m_secondMessageReceived = true;

   private:
    int L;  // 状态维度 (12)
    int numSigmaPoints;  // sigma点数量 (25)
    float alpha, beta, kappa;  // UKF参数

    Eigen::Matrix<float, 12, 1> m_state;       ///< 状态向量[x, y, z, vx, vy, vz, ax, ay, az, jx, jy, jz]
    Eigen::Matrix<float, 12, 12> m_covariance; ///< 协方差矩阵
    Eigen::Matrix<float, 12, 12> m_transition; ///< 状态转移矩阵
    Eigen::Matrix<float, 12, 12> m_processNoise;  ///< 过程噪声协方差
    Eigen::Matrix<float, 3, 12> m_observation;   ///< 观测矩阵
    Eigen::Matrix3f m_observationNoise;         ///< 观测噪声协方差

    // UKF相关变量
    Eigen::Matrix<float, 12, Eigen::Dynamic> sigmaPoints;
    Eigen::VectorXf weightsMean;
    Eigen::VectorXf weightsCovariance;
};