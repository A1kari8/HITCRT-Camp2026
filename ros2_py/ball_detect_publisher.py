
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, '..', 'assets', 'test3', 'rgb.mp4')
output_path = os.path.join(BASE_DIR, '..', 'output_with_trajectory.mp4')

class Frame2Draw():
    def __init__(self, ballId,posX, posY, posZ, frameNum) -> None:
        self.ballId = ballId
        self.frameNum = frameNum
        self.posX = posX
        self.posY = posY
        self.posZ = posZ


class BallDetectPublisher(Node):
    """
    ROS2节点：订阅C++端发布的ball_trajectory话题，收集多球三维轨迹，并可将轨迹投影叠加到原视频上。
    """
    def __init__(self) -> None:
        super().__init__('ball_detect_publisher')
        # 启动日志可选，已移除
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        # 使用可靠QoS策略，防止消息丢失
        qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ball_trajectory',
            self.trajectory_callback,
            qos)
        # 轨迹缓存：{id: [(x, y, z, t), ...]}
        # 所有帧的轨迹缓存：{ballId: [Frame2Draw, ...]}
        self.tracks = {}  # type: Dict[int, List[Frame2Draw]]
        # 轨迹可视化配色
        self.color_map = {}  # type: Dict[int, Tuple[int, int, int]]
        self.color_list = [
            (255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
            (128,0,0), (0,128,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)
        ]
        # 读取相机内参
        self.camera_matrix, self.dist_coeffs = self._load_camera_params()
        # 视频相关
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (self.width, self.height))
        # self.idx = 0
        self.lastFrame = None
        self.firstResv = True

    def _load_camera_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取相机标定参数（内参和畸变），返回相机矩阵和畸变系数。
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CALIB_PATH = os.path.join(BASE_DIR, '..', 'assets', 'calibration.json')
        with open(CALIB_PATH, 'r') as f:
            params = json.load(f)
        camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
        dist_coeffs = np.array(params['dist_coeffs'], dtype=np.float32).reshape(-1, 1)
        return camera_matrix, dist_coeffs



    def trajectory_callback(self, msg: Float32MultiArray) -> None:
        """
        轨迹消息回调：解析并缓存每个球的三维轨迹点。
        :param msg: ROS2 Float32MultiArray，每5个float为[id, x, y, z, t]
        """
        if not msg.data:
            return
        print(f"[RECV] ball_trajectory: {[round(x,4) for x in msg.data]}")
        frame2Draw = Frame2Draw(int(msg.data[0]),msg.data[1], msg.data[2], msg.data[3], int(msg.data[4]))

        if self.firstResv:
            for _ in range(1,int(frame2Draw.frameNum)):
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.out.write(frame)
                # self.idx += 1
            
            self.firstResv = False


        # 累计轨迹点（先append，保证同帧所有点都能被画出来）
        if frame2Draw.ballId not in self.tracks:
            self.tracks[frame2Draw.ballId] = []
        self.tracks[frame2Draw.ballId].append(frame2Draw)

        # 首次赋值
        if self.lastFrame is None:
            self.lastFrame = frame2Draw.frameNum
            return

        # 如果frameNum变化，说明新的一帧，写入上一帧
        if frame2Draw.frameNum != self.lastFrame and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                return
            self._draw_tracks_on_frame(frame, self.width, self.height)
            self.out.write(frame)
            # self.idx += 1
        self.lastFrame = frame2Draw.frameNum


    def _draw_tracks_on_frame(
        self,
        frame: np.ndarray,
        width: int,
        height: int
    ) -> None:
        """
        在单帧上绘制所有球的历史轨迹点，并连线。
        """
        for ballId, frame2Draws in self.tracks.items():
            color = self.color_map.get(ballId)
            if color is None:
                color = self.color_list[ballId % len(self.color_list)]
                self.color_map[ballId] = color
            for frame2Draw in frame2Draws:
                # 正确投影：三维点为相机坐标系下球心，投影球心(0,0,0)
                X = np.array([[0, 0, 0]], dtype=np.float32)
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.array([[frame2Draw.posX], [frame2Draw.posY], [frame2Draw.posZ]], dtype=np.float32)
                imgpt, _ = cv2.projectPoints(X, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                u, v = float(imgpt[0][0][0]), float(imgpt[0][0][1])
                # 调试输出
                # print(f"[DRAW][frame={frame2Draw.frameNum}] 3D=({frame2Draw.posX:.4f},{frame2Draw.posY:.4f},{frame2Draw.posZ:.4f}) -> pixel=({u:.1f},{v:.1f}) cam_mat={self.camera_matrix.flatten().tolist()} dist={self.dist_coeffs.flatten().tolist()}")
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame, (int(u), int(v)), 5, color, -1)



    def flush_last_frame(self):
        """
        在节点关闭时补写最后一帧，防止最后一帧未写入导致视频损坏。
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self._draw_tracks_on_frame(frame, self.width, self.height)
                self.out.write(frame)


def main() -> None:
    """
    ROS2主入口：收集轨迹，轨迹收集完毕后自动绘制并保存视频。
    """
    rclpy.init()
    node = BallDetectPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # 轨迹收集完毕后，自动绘制并保存视频
    # 补写最后一帧，防止最后一帧未写入
    node.flush_last_frame()
    node.cap.release()
    node.out.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
