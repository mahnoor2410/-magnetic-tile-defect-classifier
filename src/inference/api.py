"""
FastAPI inference service.

Run locally (from the project root):

    uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health        - service and model status
    POST /predict        - multipart image upload -> defect prediction
    GET  /classes         - list of class names the model can predict

Design notes:
    - The model is loaded exactly once at startup (module load time), not
      per-request, which is the standard production pattern for avoiding
      repeated disk I/O and weight-loading latency on every call.
    - Configuration (checkpoint path, model type, device) is read from
      config/config.yaml, so switching which trained model serves traffic
      requires no code changes.
"""

import logging
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config_loader import load_config
from src.inference.predictor import DefectPredictor
from src.training.utils import setup_logging

config = load_config()
setup_logging(config["logging"]["log_dir"], "api.log", config["logging"]["level"])
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Magnetic Tile Surface Defect Classifier",
    description="Classifies magnetic tile surface images into defect categories.",
    version="1.0.0",
)

# Loaded once at import time (i.e. once per server process), not per-request.
predictor = None
model_load_error = None

try:
    predictor = DefectPredictor(
        checkpoint_path=config["inference"]["checkpoint_path"],
        model_type=config["inference"]["active_model"],
        device=config["inference"]["device"],
        image_size=config["inference"]["image_size"],
    )
    logger.info("Model loaded successfully at startup.")
except Exception as exc:
    # We intentionally do not crash the process on a missing checkpoint:
    # the API should still start and report a clear health status, rather
    # than failing silently with a confusing stack trace on first request.
    model_load_error = str(exc)
    logger.error("Model failed to load at startup: %s", model_load_error)


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health")
def health_check():
    """Report whether the service is up and whether the model loaded correctly."""
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False, "error": model_load_error},
        )
    return {"status": "healthy", "model_loaded": True, "model_type": config["inference"]["active_model"]}


@app.get("/classes")
def get_classes():
    """Return the list of defect classes the model can predict."""
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")
    return {"classes": predictor.classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return the predicted defect class with
    confidence scores for every class.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Allowed types: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = predictor.predict(image_bytes)
        logger.info("Prediction for '%s': %s (confidence=%.4f)",
                     file.filename, result["predicted_class"], result["confidence"])
        return result

    except HTTPException:
        raise
    except ValueError as exc:
        # Raised by predictor.preprocess for corrupt/unreadable images.
        logger.warning("Bad request for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during prediction for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Internal error during prediction: {exc}")
