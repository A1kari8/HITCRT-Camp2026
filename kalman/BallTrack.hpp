#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>

/**
 * @brief 单球卡尔曼滤波轨迹管理类。封装状态、协方差、观测矩阵等，支持状态预测、更新与轨迹插值。
 */
class BallTrack {
public:
    /**
     * @brief 构造函数，初始化卡尔曼滤波状态。
     * @param id 球的唯一ID
     * @param posX 初始x坐标
     * @param posY 初始y坐标
     * @param posZ 初始z坐标
     * @param videoTime 初始视频时间戳
     * @param transition 状态转移矩阵
     * @param processNoise 过程噪声协方差
     * @param observation 观测矩阵
     * @param observationNoise 观测噪声协方差
     * @param wallTime 初始wall time
     */
    BallTrack(int id, float posX, float posY, float posZ, float videoTime,
              const Eigen::Matrix<float, 6, 6>& transition,
              const Eigen::Matrix<float, 6, 6>& processNoise,
              const Eigen::Matrix<float, 3, 6>& observation,
              const Eigen::Matrix3f& observationNoise,
              rclcpp::Time wallTime);

    /**
     * @brief 用新观测更新卡尔曼状态。
     * @param posX 新观测x
     * @param posY 新观测y
     * @param posZ 新观测z
     * @param videoTime 新观测视频时间戳
     * @param wallTime 新观测wall time
     */
    void update(float posX, float posY, float posZ, float videoTime, rclcpp::Time wallTime);

    /**
     * @brief 预测轨迹到当前时刻，插值生成若干点。
     * @param data 输出：每个点为[id, x, y, z, t]
     * @param dt 采样间隔
     * @param interpNum 插值细分数
     * @param now 当前wall time
     */
    void predictToNow(std::vector<float>& data, float dt, int interpNum, rclcpp::Time now);

    /**
     * @brief 距离上次观测的wall time秒数。
     * @param now 当前wall time
     * @return 距离上次观测的秒数
     */
    double timeSinceLastObs(rclcpp::Time now) const;

    /**
     * @brief 获取球的唯一ID。
     */
    int id() const;

private:
    int m_id; ///< 球ID
    Eigen::Matrix<float, 6, 1> m_state; ///< 状态向量[x, y, z, vx, vy, vz]
    Eigen::Matrix<float, 6, 6> m_covariance; ///< 协方差矩阵
    float m_lastVideoTime; ///< 上次观测视频时间戳
    rclcpp::Time m_lastWallTime; ///< 上次观测wall time
    Eigen::Matrix<float, 6, 6> m_transition; ///< 状态转移矩阵
    Eigen::Matrix<float, 6, 6> m_processNoise; ///< 过程噪声
    Eigen::Matrix<float, 3, 6> m_observation; ///< 观测矩阵
    Eigen::Matrix3f m_observationNoise; ///< 观测噪声
};
