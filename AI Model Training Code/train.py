import ultralytics
from ultralytics import YOLO
import torch

print(f"Using Ultralytics v{ultralytics.__version__}")
print(f"Using torch v{torch.__version__}, CUDA available: {torch.cuda.is_available()}")

if __name__ == '__main__':
    # Load the YOLOv8n pretrained model
    model = YOLO("yolo11n.pt")

    # Train with all augmentations turned off
    model.train(
        data="conf.yaml",                     # Your dataset config
        epochs=160,
        batch=16,
        imgsz=640,
        device=0,                             # GPU 0
        project="yolov11_version",
        name="thief_detection_yolov11",
        exist_ok=True,
        workers=2,
        save=True,
        visualize=True,
    )
