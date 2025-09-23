"""
篮球位置发布模块
"""

import time
from typing import TYPE_CHECKING

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .ball_detection import BallDetection

if TYPE_CHECKING:
    pass


class BallPublisher(Node):
    """
    ROS2节点：负责发布检测到的篮球三维位置。
    """
    def __init__(self) -> None:
        super().__init__('ball_publisher')
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos = QoSProfile(depth=10000, reliability=QoSReliabilityPolicy.RELIABLE)
        self.position_publisher = self.create_publisher(Float32MultiArray, 'ball_position', qos)
        self.fps_publisher = self.create_publisher(Float32MultiArray, 'ball_fps', qos)

    def wait_for_subscribers(self, timeout_sec: int = 10) -> None:
        """
        等待订阅者连接，避免消息丢失。
        """
        start = time.time()
        while self.position_publisher.get_subscription_count() == 0:
            if time.time() - start > timeout_sec:
                self.get_logger().warn(f'No subscribers after {timeout_sec} seconds, continue anyway.')
                break
            self.get_logger().info('Waiting for subscribers to connect to ball_position...')
            time.sleep(0.2)

    def publish_position(self, ball_detection: 'BallDetection') -> None:
        """
        发布篮球的三维位置。
        """
        msg = Float32MultiArray()
        msg.data = [
            float(ball_detection.ball_id),
            float(ball_detection.x),
            float(ball_detection.y),
            float(ball_detection.z),
            float(ball_detection.frame_num)
        ]
        self.get_logger().info(f"[SEND] ball_position: {[round(x, 4) for x in msg.data]}")
        self.position_publisher.publish(msg)

    def publish_fps(self, fps: float) -> None:
        """
        发布视频FPS信息。
        """
        msg = Float32MultiArray()
        msg.data = [fps]
        self.fps_publisher.publish(msg)