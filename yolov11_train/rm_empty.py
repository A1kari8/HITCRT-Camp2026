import os

def clean_empty_labels(label_dir, image_dir, image_exts=['.jpg', '.png']):
    removed = 0
    for file in os.listdir(label_dir):
        if not file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            os.remove(label_path)
            # 删除图片
            image_name = os.path.splitext(file)[0]
            for ext in image_exts:
                image_path = os.path.join(image_dir, image_name + ext)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Removed: {image_path}")
            print(f"Removed: {label_path}")
            removed += 1
    print(f"删除空标签及图片 {removed} 组")

clean_empty_labels('dataset2/train/labels', 'dataset2/train/images')
clean_empty_labels('dataset2/valid/labels', 'dataset2/valid/images')
