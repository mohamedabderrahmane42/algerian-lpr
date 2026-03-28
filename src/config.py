import os
from pathlib import Path

# ── Project Root ──────────────────────────────
# src/config.py -> parent is src -> parent is root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data"
RAW_PLATES  = DATA_DIR / "License_Plates" / "License_Plates_of_Algeria_Dataset-master"
MATRICULES  = DATA_DIR / "Matricules"
YOLO_OUTPUT = DATA_DIR / "yolo_dataset"

# ── Model Weights ─────────────────────────────
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OCR_MODEL   = WEIGHTS_DIR / "ocr_model.pt"
YOLO_BASE   = WEIGHTS_DIR / "yolov8n.pt"

# The trained YOLOv8 model - prioritizes the newest run from the Roboflow dataset
YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "detector_algerian500_v150" / "weights" / "best.pt"

if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "detector_algerian500" / "weights" / "best.pt"
if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "license_plate_det3" / "weights" / "best.pt"
if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "license_plate_det2" / "weights" / "best.pt"
if not YOLO_BEST.exists():
    YOLO_BEST = PROJECT_ROOT / "runs" / "detect" / "license_plate_det" / "weights" / "best.pt"

# ── Outputs ───────────────────────────────────
OUTPUT_DIR  = PROJECT_ROOT / "output"
CROPS_DIR   = OUTPUT_DIR / "cropped_plates"
TEST_RES    = OUTPUT_DIR / "test_results"
CSV_PATH    = OUTPUT_DIR / "results.csv"

# ── OCR Params ────────────────────────────────
CHARSET   = '0123456789'
BLANK_IDX = len(CHARSET)
NUM_CLASSES = len(CHARSET) + 1
IMG_H, IMG_W = 32, 128
