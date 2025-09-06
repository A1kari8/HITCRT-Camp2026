from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict(
    source='dataset/test/images/', 
    save=True,              
    conf=0.25               
)
