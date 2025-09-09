from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(BASE_DIR, 'best-blur-video.pt'))

VIDEO_PATH = os.path.join(BASE_DIR, '..', 'assets', 'test2', 'rgb.mp4')
# Perform tracking with the model
results = model.track(
    VIDEO_PATH, 
    show=True,
    conf=0.1, 
    iou=0.1,
    save=True,
    tracker=os.path.join(BASE_DIR, 'bytetrack.yaml')
)