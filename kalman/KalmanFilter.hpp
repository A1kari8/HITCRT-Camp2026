#pragma once
#include <Eigen/Dense>
#include <vector>

#include "Eigen/src/Core/Matrix.h"

#ifndef FPS_RATE
#define FPS_RATE 4  // fallback
#endif

/**
 * @brief 通用6维卡尔曼滤波器，支持三维位置与速度估计。
 */
class KalmanFilter {
   public:
    /**
     * @brief 构造函数，初始化卡尔曼滤波状态。
     * @param x 初始x坐标
     * @param y 初始y坐标
     * @param z 初始z坐标
     * @param dt 状态转移步长，默认1/20秒
     */
    KalmanFilter(float x, float y, float z,
                 float dt = 1.0f / (30.0f * FPS_RATE));

    /**
     * @brief 仅用模型预测推进状态。
     */
    void predict();

    /**
     * @brief 用新观测更新状态。
     * @param x 新观测x
     * @param y 新观测y
     * @param z 新观测z
     */
    void update(float x, float y, float z);

    /**
     * @brief 获取当前状态的三维位置。
     * @param x 输出x
     * @param y 输出y
     * @param z 输出z
     */
    std::vector<float> getState() const;

    void setStatePos(float x, float y, float z);
    void setStateVelocity(float vx, float vy, float vz);
    void setStateAcceleration(float ax, float ay, float az);

    // /**
    //  * @brief 预测未来若干步的轨迹。
    //  * @param xyz 输出，每步[x, y, z]
    //  * @param steps 预测步数
    //  */
    // void predictTrajectory(std::vector<float>& xyz, int steps) const;
    
   private:
    Eigen::Matrix<float, 9, 1> m_state;  ///< 状态向量[x, y, z, vx, vy, vz]
    Eigen::Matrix<float, 9, 9> m_covariance;    ///< 协方差矩阵
    Eigen::Matrix<float, 9, 9> m_transition;    ///< 状态转移矩阵
    Eigen::Matrix<float, 9, 9> m_processNoise;  ///< 过程噪声协方差
    Eigen::Matrix<float, 3, 9> m_observation;   ///< 观测矩阵
    Eigen::Matrix3f m_observationNoise;         ///< 观测噪声协方差
};
