"""
检测
"""

import os
import cv2
import torch
import timm
import numpy as np
import tomllib
import time
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv
from colorama import Fore, Style
from typing import TYPE_CHECKING

from ament_index_python.packages import get_package_share_directory
from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor
from trackers.core.deepsort.tracker import DeepSORTTracker

from .ball_detection import BallDetection

if TYPE_CHECKING:
    from .ball_publisher import BallPublisher


class DetectionPipeline:
    """
    篮球检测跟踪
    """
    def __init__(self, model_path: str, video_path: str):
        self.model = YOLO(model_path)
        self.video_path = video_path
        
        # 加载配置
        BASE_DIR = get_package_share_directory('assets')
        CONFIG_PATH = os.path.join(BASE_DIR, 'config.toml')
        with open(CONFIG_PATH, 'rb') as f:
            self.config = tomllib.load(f)
        
        # 如果启用深度定位，加载深度视频
        self.enable_depth = self.config['yolo'].get('enable_depth_based_positioning', False)
        self.depth_cap = None
        if self.enable_depth:
            depth_path = os.path.join(BASE_DIR, self.config['paths']['depth_path'])
            self.depth_cap = cv2.VideoCapture(depth_path)
            if not self.depth_cap.isOpened():
                print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} 无法打开深度视频: {depth_path}")
                self.enable_depth = False

    def process_with_yolo_tracking(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        publisher: 'BallPublisher'
    ) -> None:
        """
        使用 YOLO 内置跟踪
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        track_history = defaultdict(list)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] 视频FPS: {fps}")
        publisher.publish_fps(fps)

        tracker_config = os.path.join(get_package_share_directory('assets'), 'mytracker.yaml')

        while cap.isOpened():
            success, frame = cap.read()
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if not success:
                print("视频读取完毕或出错")
                break

            # 使用YOLO跟踪
            result = self.model.track(
                frame,
                imgsz=self.config['yolo']['yolo_input_size'],
                conf=self.config['yolo']['conf_yolo_tracking'],
                iou=self.config['yolo']['iou'],
                max_det=self.config['yolo']['max_det'],
                tracker=tracker_config
            )[0]

            if not result.boxes:
                continue

            xywh_boxes = result.boxes.xywh.cpu().numpy()  # type: ignore
            track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [2]  # type: ignore
            # 标准化ID
            track_ids = [1 if tid == 1 else 2 for tid in track_ids]

            for xywh_box, track_id in zip(xywh_boxes, track_ids):
                x, y, w, h = xywh_box
                radius = float(h / 2.0)
                center_x = float(x)
                center_y = float(y)

                print(f"[DETECT] id={track_id}, x={center_x:.1f}, y={center_y:.1f}, w={w:.1f}, h={h:.1f}, frame={frame_count}")

                track_history[track_id].append((center_x, center_y))

                if radius > 5:
                    # 计算三维位置
                    position_3d = BallDetection.calculate_3d_position(
                        center_x, center_y, radius, camera_matrix, dist_coeffs
                    )

                    if position_3d:
                        x_3d, y_3d, z_3d = position_3d
                        ball_detection = BallDetection(
                            track_id, x_3d, y_3d, z_3d, center_x, center_y, frame_count
                        )
                        publisher.publish_position(ball_detection)

        cap.release()

    def process_with_deepsort(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        publisher: 'BallPublisher'
    ) -> None:
        """
        使用 DeepSORT
        """
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        fps_list = []

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} 视频FPS: {fps}")
        publisher.publish_fps(fps)

        # 初始化 DeepSORT
        timm_model = timm.create_model("resnetblur50", pretrained=False)
        state_dict = torch.load(
            os.path.join(get_package_share_directory('assets'), 'pytorch_model.bin'),
            map_location="cpu"
        )
        timm_model.load_state_dict(state_dict)

        feature_extractor = DeepSORTFeatureExtractor(
            timm_model,
            device=self.config['yolo']['device'],
            input_size=tuple(self.config['yolo']['feature_extractor_input_size'])
        )

        tracker = DeepSORTTracker(
            feature_extractor=feature_extractor,
            device=self.config['yolo']['device'],
            lost_track_buffer=self.config['yolo']['lost_track_buffer'],
            appearance_weight=self.config['yolo']['appearance_weight']
        )

        while cap.isOpened():
            success, frame = cap.read()
            frame_start = time.time()
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 读取深度帧
            depth_frame = None
            if self.depth_cap is not None:
                depth_success, depth_frame = self.depth_cap.read()
                if depth_success and depth_frame is not None:
                    # 确保深度帧是单通道
                    if len(depth_frame.shape) == 3:
                        depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                else:
                    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} 深度帧读取失败，帧 {frame_count}")
                    depth_frame = None

            if not success:
                print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} 视频读取完毕或出错")
                break

            # 检测
            result = self.model.predict(source=frame, conf=self.config['yolo']['conf_deepsort'], iou=self.config['yolo']['iou'], max_det=self.config['yolo']['max_det'])[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update(detections, frame)

            xyxy_boxes = detections.xyxy
            track_ids = detections.tracker_id

            if len(xyxy_boxes) == 0 or track_ids is None:
                continue

            for xyxy, track_id in zip(xyxy_boxes, track_ids):
                x1, y1, x2, y2 = xyxy
                center_x = (x2 - x1) / 2 + x1
                center_y = (y2 - y1) / 2 + y1
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                radius = float(max(w, h) / 2.0)

                # 标准化跟踪 ID
                if track_id != 0:
                    track_id = 1

                if radius > 2:
                    # 计算三维位置
                    position_3d = BallDetection.calculate_3d_position(
                        center_x, center_y, radius, camera_matrix, dist_coeffs,
                        enable_depth=self.enable_depth, depth_frame=depth_frame
                    )

                    if position_3d:
                        x_3d, y_3d, z_3d = position_3d
                        ball_detection = BallDetection(
                            track_id, x_3d, y_3d, z_3d, center_x, center_y, frame_count
                        )
                        publisher.publish_position(ball_detection)

            frame_end = time.time()
            processing_time = frame_end - frame_start
            if processing_time > 0:
                fps = 1.0 / processing_time
                fps_list.append(fps)

        cap.release()

        if fps_list:
            average_fps = sum(fps_list) / len(fps_list)
            print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} 总平均帧率: {average_fps:.2f} FPS")
        else:
            print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} 没有处理任何帧，无法计算平均帧率")