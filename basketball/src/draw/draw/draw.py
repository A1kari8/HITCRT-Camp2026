
"""
篮球轨迹可视化ROS2节点
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import os
import tomllib
from typing import Dict
from ament_index_python.packages import get_package_share_directory
import time
from colorama import Fore, Style

from .trajectory import TrajectoryPoint, BallTrajectory
from .camera_utils import load_camera_params
from .video_processor import VideoProcessor

import cv2

BASE_DIR = get_package_share_directory('assets')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.toml')

# 加载配置
with open(CONFIG_PATH, 'rb') as f:
    config = tomllib.load(f)

VIDEO_PATH = os.path.join(BASE_DIR, config['paths']['video_path'])
OUTPUT_PATH = os.path.join(BASE_DIR, config['paths']['output_path'])


class TrajectoryVisualizer(Node):
    """
    ROS2节点：订阅C++端发布的ball_trajectory话题，收集多球三维轨迹，并将轨迹投影叠加到原视频上。
    """
    def __init__(self) -> None:
        super().__init__('trajectory_visualizer')
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        # 使用可靠QoS策略，防止消息丢失
        qos = QoSProfile(depth=10000, reliability=QoSReliabilityPolicy.RELIABLE)
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
        
        # 获取视频信息
        self.video_total_frames = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.video_processor.fps
        self.video_duration = self.video_total_frames / self.video_fps if self.video_fps > 0 else 0

        # 绘制模式配置
        self.declare_parameter('trail_length', config['draw']['trail_length'])  # 轨迹长度（最近n个点）
        self.declare_parameter('enable_color_fade', config['draw']['enable_color_fade'])  # 是否启用颜色渐变（近深远浅）
        self.declare_parameter('enable_size_variation', config['draw']['enable_size_variation'])  # 是否启用大小变化（近大远小）
        self.declare_parameter('base_radius', config['draw']['base_radius'])  # 基准半径
        self.declare_parameter('max_trail_length', config['draw']['max_trail_length'])  # 最大轨迹长度限制
        self.declare_parameter('enable_interpolation_color', config['draw']['enable_interpolation_color'])  # 是否启用插帧颜色区分
        self.declare_parameter('enable_real_time_display', config['draw']['enable_real_time_display'])  # 是否启用实时轨迹显示

        # 如果启用实时显示，创建OpenCV窗口
        if self.get_parameter('enable_real_time_display').get_parameter_value().bool_value:
            cv2.namedWindow('Real-time Trajectory', cv2.WINDOW_NORMAL)

        # FPS计算变量
        self.fps_frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.start_time = time.time()  # 记录节点启动时间

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
        :param msg: ROS2 Float32MultiArray,每5个float为[ball_id, x, y, z, frame_num]
        """
        if not msg.data:
            return

        # 重置超时定时器
        if self.timeout_timer is None:
            self.timeout_timer = self.create_timer(5.0, self.check_timeout)
        else:
            self.timeout_timer.cancel()
            self.timeout_timer = self.create_timer(5.0, self.check_timeout)

        print(f"[RECV] {[round(x, 4) for x in msg.data]}")

        # 更新FPS（每处理一个点就算一帧）
        self.fps_frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.fps_start_time
        if time_diff >= 1.0:
            self.current_fps = self.fps_frame_count / time_diff
            self.fps_frame_count = 0
            self.fps_start_time = current_time

        # 解析消息
        ball_id = int(msg.data[0])
        pos_x, pos_y, pos_z = msg.data[1], msg.data[2], msg.data[3]
        frame_num = int(msg.data[4])
        point = TrajectoryPoint(ball_id, pos_x, pos_y, pos_z, frame_num)

        # 首次接收消息时，快进视频
        if self.first_message_received:
            print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} First message received, fast-forwarding to frame {frame_num}")
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
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to get parameters: {e}, using defaults")
                trail_length = 170
                enable_color_fade = True
                enable_size_variation = True
                base_radius = 30
                max_trail_length = 300
                enable_interpolation_color = False            # self.get_logger().info(f"使用绘制参数: trail_length={trail_length}, color_fade={enable_color_fade}, size_var={enable_size_variation}, base_radius={base_radius}, interp_color={enable_interpolation_color}")
            
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
            
            # 在帧上绘制实时FPS
            cv2.putText(frame, f'FPS: {self.current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.video_processor.write_frame(frame)

            # 如果启用实时显示，更新OpenCV窗口
            if self.get_parameter('enable_real_time_display').get_parameter_value().bool_value:
                # 在帧上绘制FPS
                cv2.putText(frame, f'FPS: {self.current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Real-time Trajectory', frame)
                cv2.waitKey(1)

        self.last_frame_num = frame_num

    def flush_remaining_frames(self) -> None:
        """
        在节点关闭时补写所有剩余帧，防止视频损坏或丢失帧。
        """
        # 计算平均FPS
        total_frames = sum(len(traj.points) for traj in self.ball_trajectories.values())
        total_time = time.time() - self.start_time
        average_fps = total_frames / total_time if total_time > 0 else 0.0
        
        # 获取当前参数值
        try:
            trail_length = self.get_parameter('trail_length').get_parameter_value().integer_value
            enable_color_fade = self.get_parameter('enable_color_fade').get_parameter_value().bool_value
            enable_size_variation = self.get_parameter('enable_size_variation').get_parameter_value().bool_value
            base_radius = self.get_parameter('base_radius').get_parameter_value().integer_value
            max_trail_length = self.get_parameter('max_trail_length').get_parameter_value().integer_value
            enable_interpolation_color = self.get_parameter('enable_interpolation_color').get_parameter_value().bool_value
        except Exception as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to get parameters: {e}, using defaults")
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
            enable_interpolation_color=enable_interpolation_color,
            fps=average_fps
        )

    def shutdown(self) -> None:
        """释放资源。"""
        # 计算并输出平均FPS
        total_frames = sum(len(traj.points) for traj in self.ball_trajectories.values())
        average_fps = total_frames / self.video_duration if self.video_duration > 0 else 0.0
        print(f"[INFO] 绘制结束，总平均帧率: {average_fps:.2f} FPS (总点数: {total_frames}, 视频时长: {self.video_duration:.2f}s)")
        
        self.video_processor.release()
        # 销毁实时显示窗口
        if self.get_parameter('enable_real_time_display').get_parameter_value().bool_value:
            cv2.destroyAllWindows()


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
