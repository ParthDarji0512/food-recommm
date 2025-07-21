try:
    from ultralytics import YOLO
    print("Ultralytics YOLO is installed and working!")
except ImportError as e:
    print("ERROR:", e)
