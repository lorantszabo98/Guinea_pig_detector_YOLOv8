from ultralytics import YOLO

model = YOLO('best_n_new.pt')

# results = model(source='./test images/sarika_lujzika.jpg', conf=0.2, save=True)

results = model.track(source="https://youtu.be/eW6hheq8qfg?si=wXpYHm2e1pD6Rfnx", show=True)