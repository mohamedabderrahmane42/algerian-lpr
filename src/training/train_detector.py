"""
train_detector.py - Trains a YOLOv8 License Plate Detection Model
"""

import os
import torch
from ultralytics import YOLO
from src.config import PROJECT_ROOT, YOLO_BASE

def main():
    # 1. Provide absolute path to the dataset yaml
    yaml_path = PROJECT_ROOT / "data" / "roboflow_dataset" / "data.yaml"
    
    if not yaml_path.exists():
        print(f"[ERROR] Cannot find dataset at {yaml_path}")
        return

    # 2. Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Load the YOLOv8-nano base model
    # (If we already have a fine-tuned model, we could start from there instead,
    #  but yolov8n.pt gives a fresh, stable base for this new dataset).
    print(f"Loading Base Model: yolov8s.pt")
    model = YOLO("yolov8s.pt")  # Auto-downloads YOLOv8-Small

    # 4. Train
    print("\n--- Starting YOLOv8 Training ---")
    results = model.train(
        data=str(yaml_path),
        epochs=150,                 # Bumped to 150 epochs for maximum performance
        imgsz=640,                  # Image size 640x640
        batch=-1,                   # Auto-batch-size (maximizes GPU memory safely)
        name='detector_algerian500_v150',# New run folder name
        device=0 if device == "cuda" else "cpu",
        workers=0,                  # 0 prevents dataloader multiprocessing issues on Windows
        patience=25                 # Stop early if no improvement for 25 epochs
    )

    print("\n--- Training Complete ---")
    print(f"Best weights saved to: runs/detect/detector_algerian500_v150/weights/best.pt")

if __name__ == '__main__':
    main()
