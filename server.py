"""
FastAPI single-service Graph Digitizer for render.com.

Serves:
  GET  /              -> static SPA (static/index.html)
  GET  /static/*      -> JS / CSS / etc.
  GET  /api/healthz   -> liveness + active weight name
  GET  /api/info      -> bundled model name + classes
  POST /api/predict   -> multipart `file=<image>` ; returns {phase4, series, csv, annotated_png_b64}

Model + EasyOCR are lazily loaded on first /predict (saves ~30 s of cold-start
boot time). The active weight is the file at $WEIGHTS_PATH (default
weights/best.pt — bundled yolo26n in this repo).
"""

from __future__ import annotations

import base64
import os
import threading
from pathlib import Path

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

import pipeline as pl


ROOT = Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
WEIGHTS_PATH = Path(os.environ.get("WEIGHTS_PATH", "weights/best.pt"))
if not WEIGHTS_PATH.is_absolute():
    WEIGHTS_PATH = ROOT / WEIGHTS_PATH


_state: dict = {"model": None, "ocr": None}
_state_lock = threading.Lock()


def _ensure_loaded():
    """Lazy-load YOLO + EasyOCR on first /predict call. Both are slow to
    initialize (~10-20 s on render's CPU) so we don't pay that at import time —
    the dyno boots faster and health probes pass."""
    with _state_lock:
        if _state["model"] is None:
            if not WEIGHTS_PATH.exists():
                raise HTTPException(503,
                    f"model file missing: {WEIGHTS_PATH}. Did you commit weights/best.pt to the repo?")
            _state["model"] = YOLO(str(WEIGHTS_PATH))
        if _state["ocr"] is None:
            ocr_dir = os.environ.get("EASYOCR_MODULE_PATH")
            kwargs: dict = {"gpu": False, "verbose": False}
            if ocr_dir:
                # Persistent disk on render — EasyOCR caches detector + recognizer here.
                Path(ocr_dir).mkdir(parents=True, exist_ok=True)
                kwargs["model_storage_directory"] = ocr_dir
                kwargs["user_network_directory"] = ocr_dir
            _state["ocr"] = easyocr.Reader(["en"], **kwargs)
    return _state["model"], _state["ocr"]


app = FastAPI(title="Graph Digitizer", version="1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve /static/* and an index page at /
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/healthz")
def healthz():
    return {
        "ok": True,
        "model_present": WEIGHTS_PATH.exists(),
        "model_path": str(WEIGHTS_PATH.relative_to(ROOT)) if WEIGHTS_PATH.is_relative_to(ROOT) else str(WEIGHTS_PATH),
        "model_loaded": _state["model"] is not None,
        "ocr_loaded": _state["ocr"] is not None,
    }


@app.get("/api/info")
def info():
    return {
        "model_path": str(WEIGHTS_PATH.relative_to(ROOT)) if WEIGHTS_PATH.is_relative_to(ROOT) else str(WEIGHTS_PATH),
        "classes": pl.CLASSES,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (png/jpg/jpeg/webp supported)")

    model, ocr = _ensure_loaded()
    phase4 = pl.detect_and_calibrate(img, model, ocr)
    if "error" in phase4:
        return JSONResponse({
            "ok": False, "filename": file.filename,
            "phase4": phase4, "series": [], "csv": "",
            "annotated_png_b64": "",
        })

    series = pl.extract_series(img, phase4)
    csv_text = pl.series_to_csv(series)
    annotated = pl.annotate_image(img, phase4, series)

    return {
        "ok": True,
        "filename": file.filename,
        "phase4": {
            "chart_class": phase4["chart_class"],
            "plot_xyxy": phase4["plot_xyxy"],
            "plot_conf": phase4["plot_conf"],
            "used_fallback": phase4["used_fallback"],
            "x_numbers_ocr": phase4["x_numbers_ocr"],
            "y_numbers_ocr": phase4["y_numbers_ocr"],
            "x_calib": phase4["x_calib"],
            "y_calib": phase4["y_calib"],
            "legend_xyxy": phase4["legend_xyxy"],
        },
        "series": series,
        "n_series": len(series),
        "n_points": sum(len(s["points"]) for s in series),
        "csv": csv_text,
        "annotated_png_b64": base64.b64encode(annotated).decode("ascii"),
    }
