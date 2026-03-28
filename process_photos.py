"""
process_photos.py  –  Algerian License Plate Pipeline
======================================================
Two-stage pipeline:
  1. YOLOv8 detects the plate bounding box in each vehicle photo.
  2. CRNN (trained via train_ocr.py) reads the plate text from the crop.
     Falls back to EasyOCR if the CRNN model is not found.

Output
------
  output/cropped_plates/   – every detected plate crop
  output/results.csv       – one row per detection with formatted plate text
"""

import os
import re
import cv2
import csv
import time
import logging
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT  = r"c:\Users\Mohamed\Desktop\projects\one"

MODEL_PATHS = [
    os.path.join(PROJECT, r"runs\detect\license_plate_det3\weights\best.pt"),
    os.path.join(PROJECT, r"runs\detect\license_plate_det2\weights\best.pt"),
    os.path.join(PROJECT, r"runs\detect\license_plate_det\weights\best.pt"),
]
OCR_MODEL_PATH = os.path.join(PROJECT, "ocr_model.pt")
INPUT_DIR      = os.path.join(PROJECT, "Matricules")
OUT_DIR        = os.path.join(PROJECT, "output")
CROP_DIR       = os.path.join(OUT_DIR,  "cropped_plates")

# Detection hyper-params
CONF_THRESHOLD = 0.35   # skip YOLO detections below this confidence
BBOX_PADDING   = 12     # pixels to add around each detected bbox

# CRNN hyper-params (defaults; overridden by checkpoint if available)
IMG_HEIGHT  = 32
IMG_WIDTH   = 128
CHARSET     = '0123456789'
BLANK_IDX   = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1

logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ──────────────────────────────────────────────
# CRNN architecture  (must match train_ocr.py)
# ──────────────────────────────────────────────
class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.rnn    = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(hidden * 2, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        T, B, H = out.shape
        return self.linear(out.view(T * B, H)).view(T, B, -1)


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def _block(cin, cout, pool_k=(2, 2), pool_s=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_k, pool_s),
            )

        self.cnn = nn.Sequential(
            _block(  1,  64),
            _block( 64, 128),
            _block(128, 256, (2,1),(2,1)),
            _block(256, 256, (2,1),(2,1)),
            _block(256, 512, (2,1),(2,1)),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes),
        )

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2).permute(2, 0, 1)   # (T, B, C)
        out  = self.rnn(feat)
        return torch.log_softmax(out, dim=2)


# ──────────────────────────────────────────────
# Image pre-processing
# ──────────────────────────────────────────────
def preprocess_for_crnn(crop_bgr: np.ndarray,
                         h: int = IMG_HEIGHT,
                         w: int = IMG_WIDTH) -> torch.Tensor:
    """
    BGR crop  →  float32 tensor (1, 1, h, w)  ready for the CRNN.
    Pipeline:
      1. Upscale if tiny (helps OCR)
      2. Grayscale
      3. CLAHE contrast enhancement
      4. Resize to (h, w)
      5. Normalise [0,1]
    """
    # Upscale very small crops
    ch, cw = crop_bgr.shape[:2]
    if cw < 100 or ch < 20:
        scale = max(100 / cw, 20 / ch, 1.0)
        crop_bgr = cv2.resize(crop_bgr,
                              (int(cw * scale), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    gray  = cv2.resize(gray, (w, h))
    gray  = gray.astype(np.float32) / 255.0
    return torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)


# ──────────────────────────────────────────────
# CRNN decoder
# ──────────────────────────────────────────────
def ctc_greedy_decode(log_probs: np.ndarray, blank: int) -> str:
    indices = log_probs.argmax(axis=1)
    decoded, prev = [], -1
    for idx in indices:
        if idx != prev and idx != blank:
            decoded.append(int(idx))
        prev = idx
    return ''.join(CHARSET[i] for i in decoded if i < len(CHARSET))


# ──────────────────────────────────────────────
# Plate text formatting
# ──────────────────────────────────────────────
# Valid Algerian wilaya codes: 01–58
WILAYAS = {str(i).zfill(2) for i in range(1, 59)}


def _try_format(d: str):
    """
    Try to format a digit string as an Algerian plate.
    Returns formatted string if last 2 digits are a valid wilaya, else None.
    Handles:
      NNNNN-WWW-WW  (5+3+2 = 10)  new format since 2004
      NNNN-NNNN-WW  (4+4+2 = 10)  older format
      NNNNN-NNNN-WW (5+4+2 = 11)  rare extended format
    """
    n = len(d)
    if n < 6:
        return None
    wil = d[-2:]
    if wil not in WILAYAS:
        return None
    mid = d[:-2]           # everything except wilaya
    ml  = len(mid)
    # try right-hand group of 3 then 4
    for rg in (3, 4):
        if ml > rg:
            left  = mid[:-rg]
            right = mid[-rg:]
            return f"{left} {right} {wil}"
    # only one group left
    return f"{mid} {wil}"


def format_algerian(digits: str) -> str:
    """
    Format a raw OCR digit string as a printed Algerian plate.

    Strategy:
      1. Try the digit string directly.
      2. If it fails (no valid wilaya), and the string is 11 chars, try
         removing each digit one at a time — return the first 10-digit
         result whose last 2 digits are a valid wilaya code.
      3. Fallback: standard 5+3+2 split (no wilaya check).
    """
    d = re.sub(r'[^0-9]', '', digits)

    # Direct attempt
    result = _try_format(d)
    if result:
        return result

    # Error-correction: try removing one digit (handles extra OCR character)
    if len(d) == 11:
        for i in range(len(d)):
            candidate = d[:i] + d[i+1:]
            result = _try_format(candidate)
            if result:
                return result

    # Hard fallback — split mechanically
    n = len(d)
    if n >= 10:
        return f"{d[:5]} {d[5:8]} {d[8:10]}"
    if n == 9:
        return f"{d[:4]} {d[4:7]} {d[7:]}"
    return d


def is_valid_algerian(digits: str) -> bool:
    """Accept digit strings that plausibly encode an Algerian plate."""
    d = re.sub(r'[^0-9]', '', digits)
    return 8 <= len(d) <= 12


# ──────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────
def load_crnn(path: str, device: torch.device):
    """Load the CRNN from a checkpoint file.  Returns (model, None) or (None, error_msg)."""
    if not os.path.exists(path):
        return None, f"CRNN model not found at {path}. Run train_ocr.py first."
    ckpt = torch.load(path, map_location=device)
    nc   = ckpt.get('num_classes', NUM_CLASSES)
    model = CRNN(nc).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, None


def load_easyocr_fallback():
    """Lazy-load EasyOCR as fallback."""
    try:
        import easyocr
        print("  Loading EasyOCR fallback …")
        return easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except ImportError:
        return None


# ──────────────────────────────────────────────
# Read plate text
# ──────────────────────────────────────────────
def read_plate_crnn(crop_bgr: np.ndarray,
                    model: CRNN,
                    device: torch.device) -> tuple[str, str]:
    """
    Run CRNN on a BGR plate crop.
    Returns (raw_digits, formatted_plate_text).
    """
    tensor = preprocess_for_crnn(crop_bgr).to(device)     # (1,1,h,w)
    with torch.no_grad():
        log_probs = model(tensor)                          # (T,1,C)
    decoded = ctc_greedy_decode(log_probs[:, 0, :].cpu().numpy(), BLANK_IDX)
    formatted = format_algerian(decoded) if is_valid_algerian(decoded) else decoded
    return decoded, formatted


def read_plate_easyocr(crop_bgr: np.ndarray, reader) -> tuple[str, float]:
    """EasyOCR fallback.  Returns (formatted_text, confidence)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE + upscale for better OCR
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    h, w  = gray.shape
    if w < 200:
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    detections = reader.readtext(gray, allowlist='0123456789 ')
    best_text, best_conf = '', 0.0
    for _, text, conf in detections:
        clean = re.sub(r'[^0-9]', '', text)
        if len(clean) >= 8 and float(conf) > best_conf:
            best_text = format_algerian(clean) if is_valid_algerian(clean) else clean
            best_conf = float(conf)
    return best_text, best_conf


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main():
    os.makedirs(CROP_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Load YOLO detection model ──
    yolo_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            yolo_path = p
            break
    if yolo_path is None:
        print("ERROR: No YOLO model found. Run train_model.py first.")
        return
    print(f"Loading YOLO model from: {yolo_path}")
    yolo = YOLO(yolo_path)

    # ── Load CRNN OCR model ──
    crnn, crnn_err = load_crnn(OCR_MODEL_PATH, device)
    if crnn_err:
        print(f"WARNING: {crnn_err}")
        print("Falling back to EasyOCR.")
    else:
        print(f"CRNN OCR model loaded from: {OCR_MODEL_PATH}")

    # EasyOCR fallback (only load if needed)
    easy_reader = None
    if crnn is None:
        easy_reader = load_easyocr_fallback()
        if easy_reader is None:
            print("ERROR: Neither CRNN nor EasyOCR is available. Aborting.")
            return

    # ── Discover images ──
    exts  = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)]
    if not files:
        print(f"No images found in {INPUT_DIR}")
        return
    print(f"Processing {len(files)} images from {INPUT_DIR} …\n")

    # ── Run pipeline ──
    csv_path = os.path.join(OUT_DIR, "results.csv")
    t0 = time.time()
    processed = detected = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "det_confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "raw_digits",
            "formatted_plate",
            "ocr_method",
        ])

        for idx, fname in enumerate(files):
            img_path = os.path.join(INPUT_DIR, fname)
            img      = cv2.imread(img_path)
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            results = yolo(img, verbose=False)
            processed += 1

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

                    # Save crop
                    crop_name = f"crop_{idx}_{detected}_{fname}"
                    cv2.imwrite(os.path.join(CROP_DIR, crop_name), crop)

                    # OCR
                    if crnn is not None:
                        raw, formatted = read_plate_crnn(crop, crnn, device)
                        method = "CRNN"
                    else:
                        formatted, _ = read_plate_easyocr(crop, easy_reader)
                        raw = re.sub(r'[^0-9]', '', formatted)
                        method = "EasyOCR"

                    writer.writerow([
                        fname,
                        f"{det_conf:.3f}",
                        x1, y1, x2, y2,
                        raw,
                        formatted,
                        method,
                    ])

            if processed % 20 == 0:
                print(f"  Processed {processed}/{len(files)} images, "
                      f"{detected} detections so far …")

    elapsed = time.time() - t0
    print(f"\nDone! {processed} images processed, {detected} plates detected "
          f"in {elapsed:.1f}s")
    print(f"Results → {csv_path}")
    print(f"Crops   → {CROP_DIR}")


if __name__ == '__main__':
    main()
