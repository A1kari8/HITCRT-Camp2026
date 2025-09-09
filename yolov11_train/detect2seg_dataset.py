import os
import cv2
import math
import numpy as np

image_dir = 'video_data/train/images/'         
label_dir = 'video_data/train/labels/'          
output_dir = 'video_data_seg/train/labels/'   

num_points = 24               

os.makedirs(output_dir, exist_ok=True)

def generate_ellipse_mask(xc, yc, w, h, num_points, img_w, img_h):
    """生成归一化椭圆掩码点"""
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = xc + (w / 2) * math.cos(angle)
        y = yc + (h / 2) * math.sin(angle)
        # 归一化
        x_norm = x / img_w
        y_norm = y / img_h
        points.append((x_norm, y_norm))
    return points

def convert_labels(image_path, label_path, output_path):
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    seg_lines = []
    for line in lines:
        cls, xc, yc, w, h = map(float, line.strip().split())
        xc *= w_img
        yc *= h_img
        w *= w_img
        h *= h_img

        mask_points = generate_ellipse_mask(xc, yc, w, h, num_points, w_img, h_img)
        mask_str = f"{int(cls)} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in mask_points])
        seg_lines.append(mask_str)

    with open(output_path, 'w') as f:
        f.write("\n".join(seg_lines))

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        image_name = filename.replace('.txt', '.jpg')
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, filename)
        output_path = os.path.join(output_dir, filename)
        convert_labels(image_path, label_path, output_path)

