"""HTTP API for Qwen3-VL. One request runs at a time."""

import logging
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .engine import _load_model, ensure_weights, generate_text
from .settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("qwen")

app = FastAPI(title="drawing-calc qwen")
_lock = threading.Lock()


def _cuda_status() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda": False, "device": ""}
    if not torch.cuda.is_available():
        return {"cuda": False, "device": ""}
    return {"cuda": True, "device": torch.cuda.get_device_name(0)}


@app.on_event("startup")
def _startup() -> None:
    status = _cuda_status()
    logger.info("Qwen VL startup cuda=%s device=%s", status["cuda"], status["device"])
    if Settings.DOWNLOAD_ON_START:
        try:
            ensure_weights()
        except Exception as exc:
            logger.info("Qwen VL weights not ready at startup: %s", exc)
    if Settings.LOAD_ON_START:
        _load_model()


@app.get("/health")
def health() -> dict:
    body = {"status": "ok"}
    body.update(_cuda_status())
    return body


@app.post("/v1/generate")
def generate(image: UploadFile = File(...)) -> dict:
    """Decode one image and return the raw Qwen reply text."""
    raw = image.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="invalid image")
    height, width = int(decoded.shape[0]), int(decoded.shape[1])
    logger.info("Qwen VL request image=%sx%s bytes=%s", width, height, len(raw))
    with _lock:
        try:
            text = generate_text(decoded)
        except Exception as exc:
            logger.info("Qwen VL generate failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"text": text, "width": width, "height": height}
