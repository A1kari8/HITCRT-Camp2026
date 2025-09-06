import os
import random
import shutil

def split_train_to_valid(image_dir, label_dir, valid_image_dir, valid_label_dir, ratio=0.2):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    total = len(image_files)
    num_valid = int(total * ratio)

    valid_images = random.sample(image_files, num_valid)

    for img_file in valid_images:
        # 移动图像
        src_img = os.path.join(image_dir, img_file)
        dst_img = os.path.join(valid_image_dir, img_file)
        shutil.move(src_img, dst_img)

        # 移动标签
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(valid_label_dir, label_file)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"标签缺失：{label_file}")

    print(f"{num_valid} 个移动到验证集")

split_train_to_valid(
    image_dir='dataset/train/images',
    label_dir='dataset/train/labels',
    valid_image_dir='dataset/valid/images',
    valid_label_dir='dataset/valid/labels',
    ratio=0.2 
)
