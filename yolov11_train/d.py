import os

# 设置你的图像和标签目录路径
image_dir = 'dataset2/train/images'
label_dir = 'dataset2/train/labels'

# 支持的图像扩展名
image_exts = ['.jpg', '.jpeg', '.png']

# 记录不符合要求的文件
missing_labels = []
wrong_class = []

# 遍历图像文件
for img_name in os.listdir(image_dir):
    img_base, ext = os.path.splitext(img_name)
    if ext.lower() not in image_exts:
        continue

    label_path = os.path.join(label_dir, f"{img_base}.txt")
    if not os.path.exists(label_path):
        missing_labels.append(img_name)
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 检查每行的 class_id 是否为 '0'
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0] != '0':
            wrong_class.append(img_name)
            break  # 一旦发现不合规就跳出

# 输出结果
print("✅ 检查完成")
print(f"❌ 缺少标签文件的图像: {len(missing_labels)}")
print(f"❌ 标签中存在非0类别的图像: {len(wrong_class)}")

# 可选：打印文件名
if missing_labels:
    print("\n缺少标签文件:")
    for name in missing_labels:
        print(f" - {name}")

if wrong_class:
    print("\n标签中存在非0类别:")
    for name in wrong_class:
        print(f" - {name}")

