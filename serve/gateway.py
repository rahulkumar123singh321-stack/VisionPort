
import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI(title="VisionPort Gateway")

# Detect available device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
MODEL_PATH = Path(__file__).resolve().parent / "yolov8n.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Home endpoint
@app.get("/")
def home():
    return {
        "project": "VisionPort",
        "message": "VisionPort FastAPI Gateway is running",
        "status": "success",
        "device": device,
        "available_endpoints": [
            "/health",
            "/detect",
            "/docs"
        ]
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": device,
        "model": "YOLOv8n"
    }


# Object detection endpoint
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.35
):

    # Validate image format
    if file.content_type not in [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG or PNG image"
        )

    # Validate confidence
    if not 0 <= conf <= 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1"
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

    # Run YOLO detection
    try:
        results = model.predict(
            image,
            conf=conf,
            device=device,
            verbose=False
        )

        r = results[0]

        detections = []

        if r.boxes is not None:
            for c, s, b in zip(
                r.boxes.cls,
                r.boxes.conf,
                r.boxes.xyxy
            ):
                detections.append({
                    "class": r.names[int(c)],
                    "confidence": round(float(s), 3),
                    "box_xyxy": [
                        round(float(v), 1) for v in b
                    ]
                })

        return {
            "num_detections": len(detections),
            "detections": detections,
            "timing_ms": {
                k: round(v, 2)
                for k, v in r.speed.items()
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
