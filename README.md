# Algerian License Plate Detection

End-to-end pipeline to detect and read Algerian license plates from vehicle photos using a custom YOLOv8 detector and a CRNN OCR model.

## Pipeline

```
Vehicle photo  →  YOLOv8 (detect plate region)  →  CRNN (read plate text)  →  results.csv
```

**Format recognised:** `NNNNN WWW WW` (5-digit serial · 3-digit group · 2-digit wilaya code)

## Files

| File | Description |
|---|---|
| `process_photos.py` | **Main script** — runs the full pipeline on `Matricules/` |
| `train_model.py` | Train the YOLOv8 plate detector |
| `train_ocr.py` | Train the CRNN OCR model |
| `prepare_dataset.py` | Convert raw annotations to YOLO format |
| `requirements.txt` | Python dependencies |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Usage

### 1 — Prepare dataset (first time only)
```bash
python prepare_dataset.py
```

### 2 — Train YOLO detector
```bash
python train_model.py
```

### 3 — Train CRNN OCR model
```bash
python train_ocr.py
```

### 4 — Run inference on your photos
```bash
python process_photos.py
```
Results are saved to `output/results.csv` and plate crops to `output/cropped_plates/`.

## Notes

- Model weights (`*.pt`) are **not tracked by git** — train them locally or download separately.
- The `Matricules/` input folder and dataset directories are excluded from the repo (too large).
