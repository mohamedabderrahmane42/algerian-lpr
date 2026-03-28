"""
test_single.py – Interactive script to test the model on a single image.
"""

import sys
import cv2
import torch
import numpy as np

from src.config import OUTPUT_DIR
from src.inference.pipeline import load_models, read_plate_crnn

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.test_single path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Load models
    try:
        yolo_model, crnn = load_models(device)
    except Exception as e:
        print(f"\n[ERROR] Models missing: {e}")
        return

    # Process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    h_img, w_img = img.shape[:2]
    results = yolo_model(img, verbose=False)
    
    found = False
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
                
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Extract crop with generous padding for robust OCR
            pad = 25
            crop = img[max(0, y1-pad):min(h_img, y2+pad), max(0, x1-pad):min(w_img, x2+pad)]
            
            raw_text, plate = read_plate_crnn(crop, crnn, device)

            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 4)
            label = f" {plate} "
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(img, (x1, y1-th-20), (x1+tw+8, y1), (0, 200, 0), -1)
            cv2.putText(img, label, (x1+4, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
            print(f"Detected: {plate} (raw: {raw_text}, conf={conf:.2f})")

    if not found:
        print("No plates detected in the image.")
        return

    # Save output
    if w_img > 1400:
        scale = 1400 / w_img
        img = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        
    out_path = OUTPUT_DIR / f"test_{image_path.split('/')[-1].split('\\')[-1]}"
    cv2.imwrite(str(out_path), img)
    print(f"\nSaved visualization to: {out_path}")


if __name__ == '__main__':
    main()
