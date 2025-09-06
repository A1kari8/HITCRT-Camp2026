from ultralytics import YOLO

model = YOLO('yolo11s-seg.pt')

model.train(
    data='data_seg.yaml', 
    epochs=200, 
    imgsz=640, 
    batch=-1, 
    name='seg1s',
    mosaic=1.0,   # Mosaic
    erasing=0.5,
    hsv_h= 0.015,
    hsv_s= 0.7,
    hsv_v= 0.4,
    flipud= 0.5,
    fliplr=0.5
)



