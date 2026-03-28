# 🇩🇿 Algerian License Plate Recognition (ALPR)

<div align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-Small-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/mAP50-99.4%25-brightgreen.svg" alt="mAP 99.4%">
</div>

<br>

A production-ready pipeline for detecting and recognizing Algerian License Plates. This repository uses a highly tuned **YOLOv8s** architecture to locate the plates and a custom **CRNN** (CNN + BiLSTM + CTC) Optical Character Recognition engine to accurately transcribe the characters using intelligent wilaya-code formatting.

## 🌟 Key Features

- **Blazing Fast Detection**: Uses the state-of-the-art YOLOv8-Small network.
- **Robust OCR Engine**: Character recognition trained with Albumentations (handling blurs, shadows, and perspective shifts).
- **Smart Formatting**: Algerian plate rules are baked in. Extracts the 2-digit wilaya code dynamically to error-correct edge digits (e.g. `NNNNN WWW WW`).
- **Batch Processing Pipeline**: Quickly inferences over datasets and outputs structured CSV files with individual plate crops.
- **Clean Architecture**: Designed for modularity, extendability, and maintainability.

---

## 🏗 Directory Structure

```text
algerian-lpr/
├── data/               # [Ignored] Raw images and YOLO-formatted datasets
├── weights/            # [Ignored] Pretrained and fine-tuned `.pt` models
├── output/             # [Ignored] Cropped plates and results.csv logs
│
├── src/                # Core Application Source Code
│   ├── models/         # Neural network definitions (CRNN, BiLSTM)
│   ├── data/           # Dataset loaders and Albumentations augmentations
│   ├── training/       # YOLO and OCR model training scripts
│   ├── inference/      # Batch pipeline and single-image testing modules
│   └── utils/          # Plate formatting and OCR decoding utilities
│
└── requirements.txt    # Application dependencies
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/algerian-lpr.git
   cd algerian-lpr
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/macOS
   venv\Scripts\activate          # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models & Datasets (Google Drive):**
   The pre-trained model weights (`.pt` files) and raw image datasets are too large for Git. Download them manually from the links below and place them in their respective folders:

   * **Models (Place in `weights/` directory):**
     * [YOLOv8 Detection Model (11MB)](https://drive.google.com/drive/folders/1_2C0wZXz9646zwE1UatzZ8Q9Y4OeJ4lW?usp=sharing) — `detector_algerian500_v150.pt`
     * [CRNN Recognition Model (20MB)](https://drive.google.com/drive/folders/1_2C0wZXz9646zwE1UatzZ8Q9Y4OeJ4lW?usp=sharing) — `ocr_model.pt`
     * [YOLOv8 Base Weights (21MB)](https://drive.google.com/drive/folders/1_2C0wZXz9646zwE1UatzZ8Q9Y4OeJ4lW?usp=sharing) — `yolov8s.pt`
   * **Datasets (Place in `data/` directory):**
     * [Matricules Test Images ZIP (150MB)](https://drive.google.com/drive/folders/1_2C0wZXz9646zwE1UatzZ8Q9Y4OeJ4lW?usp=sharing) — Extract to `data/Matricules/`

---

## 🚀 Usage

The codebase is engineered to be executed as Python modules from the project root.

### 1. 📷 Inference (Reading Plates)

**Test on a Single Image:**  
Outputs a visualization to `output/test_<image_name>.jpg` along with the bounding box and transcribed OCR output.
```bash
python -m src.inference.test_single data/Matricules/algerije11.jpg
```

**Batch Pipeline:**  
Processes all images within `data/Matricules/`, generating cropped images and a consolidated `output/results.csv`.
```bash
python -m src.inference.pipeline
```

### 2. 🧠 Training (Fine-tuning the Models)

**Train the Detector (YOLOv8s):**  
Converts a Roboflow dataset and trains the YOLO detection model (runs for 150 epochs by default).
```bash
python -m src.data.prepare_yolo       # Prepare raw JSON to YOLO format (if needed)
python -m src.training.train_detector # Train the object detector
```

**Train the Reader (CRNN OCR):**  
Trains the text-recognition model using customized Albumentations.
```bash
python -m src.training.train_ocr
```

---

## 📊 Performance Metrics

| Model Component | Architecture | Epochs | Metric | Score |
| :--- | :--- | :--- | :--- | :--- |
| **Object Detection** | YOLOv8s | 150 | `mAP50` | **99.4%** |
| **Object Detection** | YOLOv8s | 150 | `mAP50-95` | **94.0%** |
| **Text Recognition** | Custom CRNN | 60 | `Accuracy` | **~93.0%** |

*To push OCR accuracy to 100%, consider using the **Hard Negative Mining** technique: Add cropped images that the CRNN previously failed to read, label them correctly, drop them into your training folder, and incrementally retrain the OCR module.*

