import cv2
import numpy as np
import os
import random

# 模拟运动模糊（方向 + 强度可控）
def apply_motion_blur(image, kernel_size=15, direction='horizontal'):
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'horizontal':
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    elif direction == 'vertical':
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    elif direction == 'diagonal':
        np.fill_diagonal(kernel, 1)
    else:
        # 随机方向 fallback
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

# 对目标框区域应用模糊
def blur_bbox_region(image, bbox, direction, kernel_size):
    h, w = image.shape[:2]
    x_center, y_center, bw, bh = bbox
    x_min = int((x_center - bw / 2) * w)
    y_min = int((y_center - bh / 2) * h)
    x_max = int((x_center + bw / 2) * w)
    y_max = int((y_center + bh / 2) * h)

    # 边界检查
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return image  # 跳过空区域
    blurred_roi = apply_motion_blur(roi, kernel_size=kernel_size, direction=direction)
    image[y_min:y_max, x_min:x_max] = blurred_roi
    return image

# 主处理函数
def generate_blurred_versions(image_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = cv2.imread(img_path)
        if image is None or not os.path.exists(label_path):
            continue

        blurred_image = image.copy()

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = map(float, parts)
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            kernel_size = random.randint(15, 45)  # 随机模糊强度
            blurred_image = blur_bbox_region(blurred_image, (x, y, w, h), direction, kernel_size)

        # 保存模糊图像（保留原图）
        out_path = os.path.join(output_dir, img_file.replace('.jpg', '_blur.jpg').replace('.png', '_blur.png'))
        cv2.imwrite(out_path, blurred_image)

        print(f"✅ 模糊图生成完成：{out_path}")

import shutil

def copy_and_rename_labels(image_dir, label_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        blur_name = base_name + '_blur'
        label_file = os.path.join(label_dir, base_name + '.txt')
        new_label_file = os.path.join(output_label_dir, blur_name + '.txt')

        if os.path.exists(label_file):
            shutil.copy(label_file, new_label_file)
            print(f"✅ 标签复制完成：{new_label_file}")
        else:
            print(f"⚠️ 找不到标签文件：{label_file}")




# generate_blurred_versions(
#     image_dir='dataset3/valid/images',
#     label_dir='dataset3/valid/labels',
#     output_dir='dataset4/valid/images'
# )

# copy_and_rename_labels(
#     image_dir='dataset3/valid/images/',
#     label_dir='dataset_seg/valid/labels/',
#     output_label_dir='dataset4_seg/valid/labels/'
# )

