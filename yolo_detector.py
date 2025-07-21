# yolo_detector.py
from ultralytics import YOLO
from PIL import Image
import torch

# Load the pretrained YOLOv8 model (best to fine-tune on food data later)
model = YOLO('yolov8n.pt')  # Can upgrade to yolov8m or yolov8l if needed

def detect_ingredients(image_path):
    results = model(image_path)
    ingredient_names = set()
    for result in results:
        for cls in result.boxes.cls:
            class_name = model.names[int(cls)]
            ingredient_names.add(class_name)
    return list(ingredient_names)
