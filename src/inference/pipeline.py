"""
pipeline.py  – Batch inference pipeline for Algerian License Plates
"""

import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple

from src.config import MATRICULES, OUTPUT_DIR, CROPS_DIR, YOLO_BEST, OCR_MODEL, NUM_CLASSES, IMG_W, IMG_H
from src.models.crnn import CRNN
from src.utils.formatters import decode_crnn, format_algerian

# Confidence thresholds
CONF_THRESHOLD = 0.50
OCR_CONF_THRESHOLD = 0.30
BBOX_PADDING = 12

def load_models(device) -> Tuple[object, nn.Module]:
    from ultralytics import YOLO
    
    print(f"Loading YOLO model from: {YOLO_BEST}")
    yolo_model = YOLO(YOLO_BEST)

    print(f"Loading CRNN from: {OCR_MODEL}")
    crnn = CRNN(NUM_CLASSES).to(device)
    ckpt = torch.load(OCR_MODEL, map_location=device)
    crnn.load_state_dict(ckpt['model_state_dict'])
    crnn.eval()

    return yolo_model, crnn

def read_plate_crnn(img_crop: np.ndarray, crnn_model: nn.Module, device: str) -> Tuple[str, str]:
    """Given a plate crop, return (raw_digits, formatted_plate_string)."""
    # Force Grayscale
    if len(img_crop.shape) == 3:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_crop = clahe.apply(img_crop)
    img_crop = cv2.resize(img_crop, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    
    img_tensor = torch.FloatTensor(img_crop).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = crnn_model(img_tensor)
        preds_np = preds[:, 0, :].cpu().numpy()
        
    raw_digits = decode_crnn(preds_np)
    formatted = format_algerian(raw_digits)
    return raw_digits, formatted

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Load models
    try:
        yolo_model, crnn = load_models(device)
    except Exception as e:
        print(f"\n[ERROR] Could not load models. Did you train them? Error: {e}")
        return

    # Prepare I/O
    if not MATRICULES.exists():
        print(f"[ERROR] Input directory not found: {MATRICULES}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

    # Get valid images
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = [f for f in os.listdir(MATRICULES) if f.lower().endswith(valid_exts)]
    print(f"Processing {len(files)} images from {MATRICULES} …\n")

    results_data = []

    t0 = time.time()
    processed = 0
    detected = 0

    for fname in files:
        img_path = str(MATRICULES / fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        processed += 1
        h_img, w_img = img.shape[:2]

        results = yolo_model(img, verbose=False)

        for result in results:
            for box in result.boxes:
                det_conf = float(box.conf[0])
                if det_conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Apply padding, clamped to image bounds
                x1p = max(0, x1 - BBOX_PADDING)
                y1p = max(0, y1 - BBOX_PADDING)
                x2p = min(w_img, x2 + BBOX_PADDING)
                y2p = min(h_img, y2 + BBOX_PADDING)

                crop = img[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                detected += 1
                
                # Perform OCR
                raw_text, formatted_plate = read_plate_crnn(crop, crnn, device)
                
                # Log results
                results_data.append({
                    'image': fname,
                    'plate': formatted_plate,
                    'raw': raw_text,
                    'confidence': f"{det_conf:.2f}"
                })
                
                # Save crop
                cv2.imwrite(str(CROPS_DIR / f"plate_{formatted_plate}_{fname}"), crop)

        if processed % 20 == 0:
            print(f"  Processed {processed}/{len(files)} images, {detected} detections so far...")

    # Save to CSV
    if results_data:
        df = pd.DataFrame(results_data)
        csv_path = str(OUTPUT_DIR / "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Processed {len(results_data)} plates. Results saved to {csv_path}")
    else:
        print("\n[INFO] No plates detected in any images.")

if __name__ == '__main__':
    main()
