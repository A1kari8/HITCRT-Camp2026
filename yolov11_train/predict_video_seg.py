import cv2
import os
from ultralytics import YOLO

model = YOLO('best-seg1s.pt') 

video_path = '../assets/test1/rgb.mp4'
cap = cv2.VideoCapture(video_path)

output_dir = 'output_frames_seg'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(
        frame,
        stream=True,
        iou=0.05,
        conf=0.001,
        imgsz=640,
        max_det=3,
        task='segment'
    )

    for result in results:
       
        annotated_frame = result.plot()

        save_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(save_path, annotated_frame)

        frame_count += 1

cap.release()
cv2.destroyAllWindows()
