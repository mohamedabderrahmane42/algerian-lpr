"""
FastAPI server for testing the Algerian License Plate Recognition pipeline.

Run:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""

import base64
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.inference.pipeline import (
    BBOX_PADDING,
    CONF_THRESHOLD,
    load_models,
    read_plate_crnn,
)

STATIC_DIR = Path(__file__).parent / "static"

_state: dict = {"yolo": None, "crnn": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[api] Loading models on {device} ...")
    yolo, crnn = load_models(device)
    _state["yolo"] = yolo
    _state["crnn"] = crnn
    _state["device"] = device
    print("[api] Models ready.")
    yield


app = FastAPI(title="Algerian License Plate Recognition", lifespan=lifespan)


def _annotate(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 4)
    text = f" {label} "
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    y_top = max(0, y1 - th - 16)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 8, y1), (0, 200, 0), -1)
    cv2.putText(img, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": _state["device"],
        "models_loaded": _state["yolo"] is not None and _state["crnn"] is not None,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    yolo = _state["yolo"]
    crnn = _state["crnn"]
    device = _state["device"]
    if yolo is None or crnn is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    h, w = img.shape[:2]
    results = yolo(img, verbose=False)

    plates = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1p = max(0, x1 - BBOX_PADDING)
            y1p = max(0, y1 - BBOX_PADDING)
            x2p = min(w, x2 + BBOX_PADDING)
            y2p = min(h, y2 + BBOX_PADDING)
            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            raw, plate = read_plate_crnn(crop, crnn, device)
            _annotate(img, x1, y1, x2, y2, plate)
            plates.append({
                "plate": plate,
                "raw": raw,
                "det_conf": round(conf, 2),
                "box": [x1, y1, x2, y2],
            })

    if w > 1400:
        scale = 1400 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return JSONResponse({
        "plates": plates,
        "annotated_image": f"data:image/jpeg;base64,{b64}",
    })
