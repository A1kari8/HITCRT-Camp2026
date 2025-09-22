#include <memory>

#include "ukf/KalmanFilterNode.hpp"
#include "ukf/KalmanFilterJerk.hpp"
#include "ukf/UnscentedKalmanFilterJerk.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanFilterNode<UnscentedKalmanFilterJerk>>());
    rclcpp::shutdown();
    return 0;
}
