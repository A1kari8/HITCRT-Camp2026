"""
视频处理模块
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from ament_index_python.packages import get_package_share_directory

from .trajectory import TrajectoryPoint, BallTrajectory


class VideoProcessor:
    """
    处理视频读取、写入和轨迹绘制。
    """
    def __init__(self, video_path: str, output_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (self.width, self.height))

        # 轨迹可视化配色
        self.ball_colors: Dict[int, Tuple[int, int, int]] = {}
        self.color_list = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]

    def fast_forward_to_frame(self, target_frame: int) -> None:
        """
        快进视频到指定帧。
        """
        for _ in range(target_frame - 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.out.write(frame)

    def read_frame_at(self, frame_num: int) -> Tuple[bool, np.ndarray]:
        """
        读取指定帧号的帧。
        """
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        while current_frame < frame_num and self.cap.isOpened():
            ret, frame = self.cap.read()
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        return ret, frame

    def draw_trajectory_on_frame(
        self,
        frame: np.ndarray,
        trajectories: Dict[int, BallTrajectory],
        current_frame_num: int,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        trail_length: int = 170,
        enable_color_fade: bool = True,
        enable_size_variation: bool = True,
        base_radius: int = 30,
        max_trail_length: int = 300,
        enable_interpolation_color: bool = False
    ) -> None:
        """
        在帧上绘制轨迹。
        :param trail_length: 轨迹长度（最近n个点）
        :param enable_color_fade: 是否启用颜色渐变（近深远浅）
        :param enable_size_variation: 是否启用大小变化（近大远小）
        :param base_radius: 基准半径
        :param max_trail_length: 最大轨迹长度限制
        :param enable_interpolation_color: 是否启用插帧颜色区分
        """
        for ball_id, trajectory in trajectories.items():
            color = self.ball_colors.get(ball_id)
            if color is None:
                color = self.color_list[ball_id % len(self.color_list)]
                self.ball_colors[ball_id] = color

            recent_points = trajectory.get_recent_points(min(trail_length, max_trail_length), current_frame_num)

            for i, point in enumerate(recent_points):
                # 投影三维点到二维
                world_point = np.array([[0, 0, 0]], dtype=np.float32)
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.array([[point.pos_x], [point.pos_y], [point.pos_z]], dtype=np.float32)
                imgpt, _ = cv2.projectPoints(world_point, rvec, tvec, camera_matrix, dist_coeffs)
                u, v = float(imgpt[0][0][0]), float(imgpt[0][0][1])

                if 0 <= u < self.width and 0 <= v < self.height:
                    # 根据配置决定点大小
                    if enable_size_variation:
                        radius = max(1, int(base_radius / (point.pos_z * 1.1)))
                    else:
                        radius = base_radius // 10  # 默认固定大小

                    # 根据配置决定颜色
                    base_color = color
                    if enable_interpolation_color and point.is_interpolated:
                        # 插帧点使用反色
                        base_color = tuple(255 - c for c in color)
                    
                    if enable_color_fade:
                        alpha = (i + 1) / len(recent_points) if recent_points else 1.0
                        color_fade = tuple(int(c * alpha) for c in base_color)
                    else:
                        color_fade = base_color

                    cv2.circle(frame, (int(u), int(v)), radius, color_fade, -1)

    def write_frame(self, frame: np.ndarray) -> None:
        """
        写入帧到输出视频。
        """
        self.out.write(frame)

    def flush_remaining_frames(
        self,
        trajectories: Dict[int, BallTrajectory],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        trail_length: int = 170,
        enable_color_fade: bool = True,
        enable_size_variation: bool = True,
        base_radius: int = 30,
        max_trail_length: int = 300,
        enable_interpolation_color: bool = False,
        fps: Optional[float] = None
    ) -> None:
        """
        补写剩余帧。
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # 获取当前帧号（从1开始）
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                self.draw_trajectory_on_frame(frame, trajectories, current_frame, camera_matrix, dist_coeffs,
                                             trail_length=trail_length,
                                             enable_color_fade=enable_color_fade, 
                                             enable_size_variation=enable_size_variation, 
                                             base_radius=base_radius, 
                                             max_trail_length=max_trail_length,
                                             enable_interpolation_color=enable_interpolation_color)
                
                # 绘制平均FPS
                if fps is not None:
                    cv2.putText(frame, f'Avg FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.out.write(frame)
            else:
                break

    def release(self) -> None:
        """
        释放资源。
        """
        self.cap.release()
        self.out.release()
