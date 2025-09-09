from ultralytics import YOLO

model = YOLO('yolo11s-seg')

model.train(
    data='data_seg.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=64, 
    name='seg4s-blur',  
    workers=14,
    optimizer="AdamW",          
    lr0=0.001,       
    cache=True,          
    amp=True,
    # ✅ 去除 Mosaic（不适合单目标任务）
    mosaic=0.0,           # 禁用 Mosaic增强

    # ✅ 遮挡模拟（随机擦除）
    erasing=0.5,          # 50%概率进行遮挡模拟

    # ✅ 颜色扰动（HSV变换）
    hsv_h=0.015,          # 色调扰动
    hsv_s=0.7,            # 饱和度扰动
    hsv_v=0.4,            # 亮度扰动

    fliplr=0.5            # 水平翻转
)