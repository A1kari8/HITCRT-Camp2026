"""
篮球检测和定位模块
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class BallDetection:
    """
    表示检测到的篮球及其三维位置信息。
    """
    def __init__(self, ball_id: int, x: float, y: float, z: float, x2d: float, y2d: float, frame_num: int) -> None:
        self.ball_id = ball_id
        self.x = x
        self.y = y
        self.z = z
        self.x2d = x2d
        self.y2d = y2d
        self.frame_num = frame_num

    @staticmethod
    def calculate_3d_position(
        center_x: float,
        center_y: float,
        radius: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        ball_radius_m: float = 0.123
    ) -> Optional[Tuple[float, float, float]]:
        """
        使用 PnP 算法计算篮球的三维位置。

        Args:
            center_x: 检测框中心 x 坐标
            center_y: 检测框中心 y 坐标
            radius: 检测框半径
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            ball_radius_m: 篮球半径（米）

        Returns:
            三维位置 (x, y, z) 或 None（如果计算失败）
        """
        # 定义篮球的3D模型点（球心和球面上的点）
        object_points = np.array([
            [0, 0, 0],  # 球心
            [ball_radius_m, 0, 0],
            [-ball_radius_m, 0, 0],
            [0, ball_radius_m, 0],
            [0, -ball_radius_m, 0]
        ], dtype=np.float32)

        # 定义对应的2D图像点
        image_points = np.array([
            [center_x, center_y],  # 球心投影
            [center_x + radius, center_y],
            [center_x - radius, center_y],
            [center_x, center_y + radius],
            [center_x, center_y - radius]
        ], dtype=np.float32)

        # 使用 PnP 求解
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            return float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
        return None

    @staticmethod
    def project_to_image(
        position_3d: Tuple[float, float, float],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Tuple[float, float]:
        """
        将三维点投影到图像平面。

        Args:
            position_3d: 三维位置 (x, y, z)
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数

        Returns:
            图像坐标 (u, v)
        """
        point_3d = np.array([position_3d], dtype=np.float32)
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.array([[position_3d[0]], [position_3d[1]], [position_3d[2]]], dtype=np.float32)

        img_points, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
        u, v = float(img_points[0][0][0]), float(img_points[0][0][1])
        return u, v