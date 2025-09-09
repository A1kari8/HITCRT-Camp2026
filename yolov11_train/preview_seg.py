import os
import cv2
import numpy as np

image_dir = 'video_data_seg/train/images/'         
label_dir = 'video_data_seg/train/labels/'     

def draw_mask(image, points, color=(0, 255, 0)):
    """绘制多边形掩码"""
    pts = points.reshape((-1, 1, 2)).astype(int)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

def preview_dataset():
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue

        image_path = os.path.join(image_dir, filename.replace('.txt', '.jpg'))  # 或 .png
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(image_path):
            print(f"图像不存在：{image_path}")
            continue

        img = cv2.imread(image_path)
        h_img, w_img = img.shape[:2]

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = np.array(coords).reshape(-1, 2)
            # 将归一化坐标转换为像素坐标
            points[:, 0] *= w_img
            points[:, 1] *= h_img
            draw_mask(img, points)

        cv2.imshow('Preview', img)
        key = cv2.waitKey(0)
        if key == 27: 
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    preview_dataset()
