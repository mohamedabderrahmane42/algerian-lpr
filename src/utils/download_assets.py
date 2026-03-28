"""
download_assets.py - Utility to fetch required models and datasets from Google Drive
"""

import os
import zipfile
import subprocess
from pathlib import Path
from src.config import PROJECT_ROOT, WEIGHTS_DIR, DATA_DIR

# =====================================================================
# Configuration: REPLACE THESE IDs WITH YOUR GOOGLE DRIVE FILE IDs
# =====================================================================
# To get an ID: Right click file -> Share -> Copy Link.
# The link looks like: https://drive.google.com/file/d/THIS_IS_THE_ID/view
GDRIVE_IDS = {
    # 1. The custom OCR text-recognition model (.pt)
    "ocr_model.pt": "1aBcDeFgHiJkLmNoPqRsTuVwXyZ012345",
    
    # 2. The custom YOLOv8 detection model (.pt)
    "detector_algerian500_v150.pt": "1bCdEfGhIjKlMnOpQrStUvWxYz0123456",
    
    # 3. (Optional) The raw Matricules zip file for testing
    "matricules_dataset.zip": "1cDeFgHiJkLmNoPqRsTuVwXyZ01234567"
}
# =====================================================================


def download_file_from_google_drive(id, destination):
    """Downloads a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("[Error] gdown is not installed. Run: pip install gdown")
        return

    url = f'https://drive.google.com/uc?id={id}'
    print(f"Downloading {destination.name} from Google Drive...")
    gdown.download(url, str(destination), quiet=False)


def unzip_file(zip_path, extract_to):
    """Extracts a zip file and removes the archive."""
    print(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)


def main():
    print("--- 📥 Starting Asset Download Pipeline ---")
    
    # Ensure directories exist
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download OCR Model
    ocr_dest = WEIGHTS_DIR / "ocr_model.pt"
    if not ocr_dest.exists():
        download_file_from_google_drive(GDRIVE_IDS["ocr_model.pt"], ocr_dest)
    else:
        print(f"  [Skip] {ocr_dest.name} already exists.")

    # 2. Download YOLO Detector
    yolo_dest = WEIGHTS_DIR / "detector_algerian500_v150.pt"
    if not yolo_dest.exists():
        download_file_from_google_drive(GDRIVE_IDS["detector_algerian500_v150.pt"], yolo_dest)
    else:
        print(f"  [Skip] {yolo_dest.name} already exists.")

    # 3. Download and Extact Dataset
    dataset_zip = DATA_DIR / "matricules_dataset.zip"
    dataset_dest = DATA_DIR / "Matricules"
    if not dataset_dest.exists():
        download_file_from_google_drive(GDRIVE_IDS["matricules_dataset.zip"], dataset_zip)
        if dataset_zip.exists():
            unzip_file(dataset_zip, DATA_DIR)
    else:
        print("  [Skip] Matricules dataset already extracted.")

    print("--- ✅ Asset Download Complete! ---")
    

if __name__ == "__main__":
    main()
