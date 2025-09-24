import os
import tomllib
import rclpy
from ament_index_python.packages import get_package_share_directory

from .camera_utils import load_camera_params
from .ball_publisher import BallPublisher
from .detection_pipeline import DetectionPipeline

# 路径配置
BASE_DIR = get_package_share_directory('assets')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.toml')

# 加载配置
with open(CONFIG_PATH, 'rb') as f:
    config = tomllib.load(f)

MODEL_PATH = os.path.join(BASE_DIR, config['paths']['model_path'])
VIDEO_PATH = os.path.join(BASE_DIR, config['paths']['video_path'])


def main() -> None:
    """
    主入口
    """
    # 加载相机参数
    camera_matrix, dist_coeffs = load_camera_params()

    # 初始化ROS2
    rclpy.init()

    # 创建发布者节点
    publisher = BallPublisher()
    publisher.wait_for_subscribers(timeout_sec=10)

    # 创建检测流水线
    pipeline = DetectionPipeline(MODEL_PATH, VIDEO_PATH)

    # 使用DeepSORT进行检测和跟踪
    pipeline.process_with_deepsort(camera_matrix, dist_coeffs, publisher)

    # 清理资源
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()