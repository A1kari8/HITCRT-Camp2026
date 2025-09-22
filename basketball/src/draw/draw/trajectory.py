"""
轨迹数据结构模块
"""

from typing import List


class TrajectoryPoint:
    """
    表示轨迹中的一个点，包含球ID、位置和帧号。
    """
    def __init__(self, ball_id: int, pos_x: float, pos_y: float, pos_z: float, frame_num: int, is_interpolated: bool = False) -> None:
        self.ball_id = ball_id
        self.frame_num = frame_num
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.is_interpolated = is_interpolated


class BallTrajectory:
    """
    表示单个球的轨迹，包含多个轨迹点。
    """
    def __init__(self, ball_id: int):
        self.ball_id = ball_id
        self.points: List[TrajectoryPoint] = []

    def add_point(self, point: TrajectoryPoint) -> None:
        # 检查是否已经有相同frame_num的点，如果有则标记为插帧点
        has_same_frame = any(p.frame_num == point.frame_num for p in self.points)
        if has_same_frame:
            point.is_interpolated = True
        self.points.append(point)

    def get_recent_points(self, trail_length: int, current_frame: int) -> List[TrajectoryPoint]:
        """
        获取最近的轨迹点，只保留当前帧之前的点。
        """
        recent_points = [p for p in self.points[-trail_length:] if p.frame_num < current_frame]
        return recent_points