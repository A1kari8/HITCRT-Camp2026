#include "KalmanFilterNode.hpp"
#include "KalmanFilter.hpp"
#include "UnscentedKalmanFilter.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanFilterNode>());
    rclcpp::shutdown();
    return 0;
}
