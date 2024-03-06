from ultralytics import YOLO

model = YOLO('best.pt')

results = model(source='./test images/ervinke.jpeg', conf=0.5, save=True)