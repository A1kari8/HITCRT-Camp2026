
"""
篮球轨迹可视化ROS2节点
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import os
from typing import Dict
from ament_index_python.packages import get_package_share_directory

from .trajectory import TrajectoryPoint, BallTrajectory
from .camera_utils import load_camera_params
from .video_processor import VideoProcessor

BASE_DIR = get_package_share_directory('assets')
VIDEO_PATH = os.path.join(BASE_DIR, 'test5', 'rgb.mp4')
OUTPUT_PATH = os.path.join(BASE_DIR, 'output_with_trajectory.mp4')


class TrajectoryVisualizer(Node):
    """
    ROS2节点：订阅C++端发布的ball_trajectory话题，收集多球三维轨迹，并将轨迹投影叠加到原视频上。
    """
    def __init__(self) -> None:
        super().__init__('trajectory_visualizer')
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        # 使用可靠QoS策略，防止消息丢失
        qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ball_trajectory',
            self.on_trajectory_received,
            qos)

        # 轨迹缓存：{ball_id: BallTrajectory}
        self.ball_trajectories: Dict[int, BallTrajectory] = {}

        # 相机参数
        self.camera_matrix, self.dist_coeffs = load_camera_params()

        # 视频处理器
        self.video_processor = VideoProcessor(VIDEO_PATH, OUTPUT_PATH)

        # 绘制模式配置
        self.declare_parameter('trail_length', 170)  # 轨迹长度（最近n个点）
        self.declare_parameter('enable_color_fade', True)  # 是否启用颜色渐变（近深远浅）
        self.declare_parameter('enable_size_variation', True)  # 是否启用大小变化（近大远小）
        self.declare_parameter('base_radius', 30)  # 基准半径
        self.declare_parameter('max_trail_length', 300)  # 最大轨迹长度限制
        self.declare_parameter('enable_interpolation_color', False)  # 是否启用插帧颜色区分

        # 状态变量
        self.last_frame_num = None
        self.first_message_received = True
        self.processing_finished = False
        self.timeout_timer = None

    def check_timeout(self) -> None:
        """检查超时，如果5秒内无新消息，则结束处理。"""
        self.processing_finished = True

    def on_trajectory_received(self, msg: Float32MultiArray) -> None:
        """
        轨迹消息回调：解析并缓存每个球的三维轨迹点。
        :param msg: ROS2 Float32MultiArray，每5个float为[ball_id, x, y, z, frame_num]
        """
        if not msg.data:
            return

        # 重置超时定时器
        if self.timeout_timer is None:
            self.timeout_timer = self.create_timer(5.0, self.check_timeout)
        else:
            self.timeout_timer.cancel()
            self.timeout_timer = self.create_timer(5.0, self.check_timeout)

        self.get_logger().info(f"[RECV] ball_trajectory: {[round(x, 4) for x in msg.data]}")

        # 解析消息
        ball_id = int(msg.data[0])
        pos_x, pos_y, pos_z = msg.data[1], msg.data[2], msg.data[3]
        frame_num = int(msg.data[4])
        point = TrajectoryPoint(ball_id, pos_x, pos_y, pos_z, frame_num)

        # 首次接收消息时，快进视频
        if self.first_message_received:
            self.get_logger().info(f"First message received, fast-forwarding video to frame {frame_num}...")
            self.video_processor.fast_forward_to_frame(frame_num)
            self.first_message_received = False

        # 添加轨迹点
        if ball_id not in self.ball_trajectories:
            self.ball_trajectories[ball_id] = BallTrajectory(ball_id)
        self.ball_trajectories[ball_id].add_point(point)

        # 初始化last_frame_num
        if self.last_frame_num is None:
            self.last_frame_num = frame_num
            return

        # 如果帧号变化，处理上一帧
        if frame_num != self.last_frame_num:
            ret, frame = self.video_processor.read_frame_at(frame_num)
            if not ret:
                return
            
            # 获取当前参数值
            try:
                trail_length = self.get_parameter('trail_length').get_parameter_value().integer_value
                enable_color_fade = self.get_parameter('enable_color_fade').get_parameter_value().bool_value
                enable_size_variation = self.get_parameter('enable_size_variation').get_parameter_value().bool_value
                base_radius = self.get_parameter('base_radius').get_parameter_value().integer_value
                max_trail_length = self.get_parameter('max_trail_length').get_parameter_value().integer_value
                enable_interpolation_color = self.get_parameter('enable_interpolation_color').get_parameter_value().bool_value
            except Exception as e:
                self.get_logger().error(f"获取参数失败: {e}，使用默认值")
                trail_length = 170
                enable_color_fade = True
                enable_size_variation = True
                base_radius = 30
                max_trail_length = 300
                enable_interpolation_color = False
            
            self.get_logger().info(f"使用绘制参数: trail_length={trail_length}, color_fade={enable_color_fade}, size_var={enable_size_variation}, base_radius={base_radius}, interp_color={enable_interpolation_color}")
            
            self.video_processor.draw_trajectory_on_frame(
                frame, self.ball_trajectories, self.last_frame_num,
                self.camera_matrix, self.dist_coeffs,
                trail_length=trail_length,
                enable_color_fade=enable_color_fade,
                enable_size_variation=enable_size_variation,
                base_radius=base_radius,
                max_trail_length=max_trail_length,
                enable_interpolation_color=enable_interpolation_color
            )
            self.video_processor.write_frame(frame)

        self.last_frame_num = frame_num

    def flush_remaining_frames(self) -> None:
        """
        在节点关闭时补写所有剩余帧，防止视频损坏或丢失帧。
        """
        # 获取当前参数值
        try:
            trail_length = self.get_parameter('trail_length').get_parameter_value().integer_value
            enable_color_fade = self.get_parameter('enable_color_fade').get_parameter_value().bool_value
            enable_size_variation = self.get_parameter('enable_size_variation').get_parameter_value().bool_value
            base_radius = self.get_parameter('base_radius').get_parameter_value().integer_value
            max_trail_length = self.get_parameter('max_trail_length').get_parameter_value().integer_value
            enable_interpolation_color = self.get_parameter('enable_interpolation_color').get_parameter_value().bool_value
        except Exception as e:
            self.get_logger().error(f"获取参数失败: {e}，使用默认值")
            trail_length = 170
            enable_color_fade = True
            enable_size_variation = True
            base_radius = 30
            max_trail_length = 300
            enable_interpolation_color = False
        
        self.video_processor.flush_remaining_frames(
            self.ball_trajectories, self.camera_matrix, self.dist_coeffs,
            trail_length=trail_length,
            enable_color_fade=enable_color_fade,
            enable_size_variation=enable_size_variation,
            base_radius=base_radius,
            max_trail_length=max_trail_length,
            enable_interpolation_color=enable_interpolation_color
        )

    def shutdown(self) -> None:
        """释放资源。"""
        self.video_processor.release()


def main() -> None:
    """
    ROS2主入口：收集轨迹，轨迹收集完毕后自动绘制并保存视频。
    """
    rclpy.init()
    node = TrajectoryVisualizer()
    try:
        while rclpy.ok() and not node.processing_finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    # 轨迹收集完毕后，自动绘制并保存视频
    # 补写最后一帧，防止最后一帧未写入
    node.flush_remaining_frames()
    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
