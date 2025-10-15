from ultralytics import YOLO
import os

cwd = os.path.join(os.getcwd(), 'Documents')

coco_dir = os.path.join(cwd, 'coco')

save_dir = os.path.join(cwd, 'save')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

yaml_path = os.path.join(coco_dir, 'data.yaml')

model = YOLO("yolo11n-seg.pt")

results = model.train(data=yaml_path, epochs=100, imgsz=640, device=0, workers=8, project=save_dir) # device = 0 for gpu, 'cpu' for cpu