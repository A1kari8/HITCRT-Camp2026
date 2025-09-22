
import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from typing import Tuple, Dict, List, Any
from collections import defaultdict



# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best-blur-video.pt')
VIDEO_PATH = os.path.join(BASE_DIR, '..', 'assets', 'test3', 'rgb.mp4')


def load_camera_params() -> Tuple[np.ndarray, np.ndarray]:
    """
    读取相机标定参数（内参和畸变），返回相机矩阵和畸变系数。
    """
    calib_path = os.path.join(BASE_DIR, '..', 'assets', 'calibration.json')
    with open(calib_path, 'r') as f:
        params = json.load(f)
    camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(params['dist_coeffs'], dtype=np.float32).reshape(-1, 1)
    return camera_matrix, dist_coeffs

# ROS2 节点定义
import time


class Ball:
    def __init__(self, ball_id: int, x: float, y: float, z: float, x2d: float, y2d: float, frame_num: int) -> None:
        self.id = ball_id
        self.x = x
        self.y = y
        self.z = z
        self.x2d = x2d
        self.y2d = y2d
        self.frameNum = frame_num


class BallPublisher(Node):
    """
    ROS2节点：负责发布检测到的篮球三维位置，支持简单track管理。
    """
    def __init__(self) -> None:
        super().__init__('ball_publisher')
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        qos = QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE)
        self.publisher_ = self.create_publisher(Float32MultiArray, 'ball_position', qos)

    def wait_for_subscribers(self, timeout_sec: int = 10) -> None:
        """
        等待订阅者连接，避免消息丢失。
        """
        start = time.time()
        while self.publisher_.get_subscription_count() == 0:
            if time.time() - start > timeout_sec:
                self.get_logger().warn(f'No subscribers after {timeout_sec} seconds, continue anyway.')
                break
            self.get_logger().info('Waiting for subscribers to connect to ball_position...')
            time.sleep(0.2)

    def publish_positions(self, ball: Ball) -> None:
        """
        发布三维位置。
        """
        msg = Float32MultiArray()
        msg.data = [ball.id, ball.x, ball.y, ball.z, ball.frameNum]
        print(f"[SEND] ball_position: {[round(x,4) for x in msg.data]}")
        self.publisher_.publish(msg)





def detect_and_publish(
    video_path: str,
    model_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    node: BallPublisher
) -> None:
    """
    主检测与发布流程：逐帧检测篮球，三维定位，track管理并通过ROS2发布。
    集成光流算法辅助识别运动模糊的篮球。
    """

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    yolo_input_size = 640
    frame_count = 0
    track_history = defaultdict(list)

    debug_dir = os.path.join(BASE_DIR, 'debug_frames')
    os.makedirs(debug_dir, exist_ok=True)

    # 光流相关变量
    prev_frame = None
    prev_gray = None

    while cap.isOpened():
        frame_count += 1
        success, frame = cap.read()
        if not success:
            break

        # 转换为灰度图用于光流
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = None
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # 计算运动幅度
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = mag > 2.0  # 运动阈值

            # 可视化光流（可选）
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            phase = cv2.phase(flow[..., 0], flow[..., 1])
            hsv[..., 0] = (phase / (2 * np.pi) * 179).astype(np.uint8)
            mag_norm = (mag / np.max(mag) * 255).astype(np.uint8) if np.max(mag) > 0 else np.zeros_like(mag, dtype=np.uint8)
            hsv[..., 2] = mag_norm
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 在运动区域增强帧（简单锐化）
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced_frame = cv2.filter2D(frame, -1, kernel)
            frame = np.where(motion_mask[..., None], enhanced_frame, frame)

        # 更新上一帧
        prev_gray = gray.copy()

        # 检测与跟踪
        result = model.track(frame, persist=True, imgsz=yolo_input_size, conf=0.5, iou=0.1, max_det=2, tracker=os.path.join(BASE_DIR, 'bytetrack.yaml'))[0]
        
        if not result.boxes:
            continue

        xywhBoxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [2]
        track_ids = [1 if tid == 1 else 2 for tid in track_ids]  # id为1的球id就是1，其他全变成2


        for xywhBox, track_id in zip(xywhBoxes, track_ids):
            x, y, w, h = xywhBox
            radius = h / 2.0
            x = x + w / 2.0 - radius
            x = float(x)
            y = float(y)

            print(f"[DETECT] id={track_id}, x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}, frame={frame_count}")
            track = track_history[track_id]
            track.append((x, y))

            # id为1的球id就是1，其他全变成2
            # ball_id = 1 if track_id == 1 else 2

            if radius > 5:
                BALL_RADIUS_M = 0.123
                object_points = np.array([
                    [0, 0, 0],
                    [ BALL_RADIUS_M, 0, 0],
                    [-BALL_RADIUS_M, 0, 0],
                    [0,  BALL_RADIUS_M, 0],
                    [0, -BALL_RADIUS_M, 0]
                ], dtype=np.float32)
                image_points = np.array([
                    [x, y],
                    [x + radius, y],
                    [x - radius, y],
                    [x, y + radius],
                    [x, y - radius]
                ], dtype=np.float32)
                # 打印solvePnP输入
                print(f"[PnP] img center=({x:.1f},{y:.1f}), r={radius:.1f}, frame={frame_count}")
                print(f"[PnP] object_points(m): {object_points.tolist()}")
                print(f"[PnP] image_points(px): {image_points.tolist()}")
                success_pnp, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                if success_pnp:
                    # 打印三维点
                    print(f"[PnP] tvec(m): {tvec.ravel().tolist()}")
                    # 正确投影三维球心到像素
                    X = np.array([[0, 0, 0]], dtype=np.float32)  # 球心在自身坐标系原点
                    imgpt, _ = cv2.projectPoints(X, rvec, tvec, camera_matrix, dist_coeffs)
                    u, v = float(imgpt[0][0][0]), float(imgpt[0][0][1])
                    print(f"[PnP] 球心投影像素: ({u:.1f},{v:.1f})")
                    node.publish_positions(Ball(track_id, tvec[0][0], tvec[1][0], tvec[2][0], x, y, frame_count))
    cap.release()


def main() -> None:
    """
    主入口：加载参数，初始化ROS2节点，检测并发布。
    """
    camera_matrix, dist_coeffs = load_camera_params()
    rclpy.init()
    node = BallPublisher()
    node.wait_for_subscribers(timeout_sec=10)
    detect_and_publish(VIDEO_PATH, MODEL_PATH, camera_matrix, dist_coeffs, node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()