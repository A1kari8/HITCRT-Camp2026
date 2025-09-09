
#include "BallTrack.hpp"

/**
 * @brief BallTrack 构造函数，初始化卡尔曼滤波状态。
 */
/**
 * @brief BallTrack 构造函数，初始化卡尔曼滤波状态。
 * @param id 球ID
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
BallTrack::BallTrack(int id, float posX, float posY, float posZ, float videoTime,
                                         const Eigen::Matrix<float, 6, 6>& transition,
                                         const Eigen::Matrix<float, 6, 6>& processNoise,
                                         const Eigen::Matrix<float, 3, 6>& observation,
                                         const Eigen::Matrix3f& observationNoise,
                                         rclcpp::Time wallTime)
        : m_id(id), m_lastVideoTime(videoTime), m_lastWallTime(wallTime),
            m_transition(transition), m_processNoise(processNoise), m_observation(observation), m_observationNoise(observationNoise)
{
        // 状态向量初始化为观测值，速度分量为0
        m_state.setZero();
        m_state(0) = posX; // 初始x
        m_state(1) = posY; // 初始y
        m_state(2) = posZ; // 初始z
        // 协方差初始化为单位阵
        m_covariance = Eigen::Matrix<float, 6, 6>::Identity();
}
/**
 * @brief 用新观测更新卡尔曼状态。
 * @param posX 新观测x
 * @param posY 新观测y
 * @param posZ 新观测z
 * @param videoTime 新观测视频时间戳
 * @param wallTime 新观测wall time
 */
void BallTrack::update(float posX, float posY, float posZ, float videoTime, rclcpp::Time wallTime) {
    // 步骤1：状态预测（时间推进）
    m_state = m_transition * m_state; // 用状态转移矩阵推进到当前时刻
    m_covariance = m_transition * m_covariance * m_transition.transpose() + m_processNoise; // 协方差预测

    // 步骤2：构造观测向量
    Eigen::Vector3f measurement;
    measurement << posX, posY, posZ; // 新的观测位置

    // 步骤3：计算创新（观测与预测的差值）
    Eigen::Vector3f innovation = measurement - m_observation * m_state;

    // 步骤4：计算创新协方差
    Eigen::Matrix3f innovationCov = m_observation * m_covariance * m_observation.transpose() + m_observationNoise;

    // 步骤5：计算卡尔曼增益
    Eigen::Matrix<float, 6, 3> kalmanGain = m_covariance * m_observation.transpose() * innovationCov.inverse();

    // 步骤6：用观测修正状态
    m_state = m_state + kalmanGain * innovation;

    // 步骤7：更新协方差
    m_covariance = (Eigen::Matrix<float, 6, 6>::Identity() - kalmanGain * m_observation) * m_covariance;

    // 步骤8：更新时间戳
    m_lastVideoTime = videoTime;
    m_lastWallTime = wallTime;
}
/**
 * @brief 预测轨迹到当前时刻，插值生成若干点。
 * @param data 输出：每个点为[id, x, y, z, t]
 * @param dt 采样间隔
 * @param interpNum 插值细分数
 * @param now 当前wall time
 */
void BallTrack::predictToNow(std::vector<float>& data, float dt, int interpNum, rclcpp::Time now) {
    // 步骤1：计算距离上次观测的wall time
    double wallDt = (now - m_lastWallTime).seconds();
    if (wallDt < 1.0) {
        // 步骤2：计算插值区间的起止视频时间
        float lastVideoTime = m_lastVideoTime;
        double nowVideoTime = lastVideoTime + wallDt;
        // 步骤3：计算插值步数，保证轨迹平滑
        int totalSteps = std::max(1, int((nowVideoTime - lastVideoTime) / (dt / (interpNum + 1))));
        // 步骤4：初始化预测状态和协方差
        Eigen::Matrix<float, 6, 1> statePred = m_state;
        Eigen::Matrix<float, 6, 6> covariancePred = m_covariance;
        float t = lastVideoTime;
        double step = (nowVideoTime - lastVideoTime) / totalSteps;
        // 步骤5：逐步插值预测
        for (int i = 0; i < totalSteps; ++i) {
            t += step;
            // 预测一步，推进状态
            statePred = m_transition * statePred;
            covariancePred = m_transition * covariancePred * m_transition.transpose() + m_processNoise;
            // 存储插值点 [id, x, y, z, t]
            data.push_back(static_cast<float>(m_id));
            data.push_back(statePred(0));
            data.push_back(statePred(1));
            data.push_back(statePred(2));
            data.push_back(t);
        }
    }
}
/**
 * @brief 距离上次观测的wall time秒数。
 * @param now 当前wall time
 * @return 距离上次观测的秒数
 */
double BallTrack::timeSinceLastObs(rclcpp::Time now) const {
    return (now - m_lastWallTime).seconds();
}
/**
 * @brief 获取球的唯一ID。
 */
int BallTrack::id() const {
    return m_id;
}
