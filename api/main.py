"""
api/main.py
-----------
FastAPI prediction service for the Traffic Accident Severity Classifier.

Endpoints:
    GET  /              Service status
    GET  /health        Model health check
    GET  /model-info    Model metadata
    POST /predict       Predict severity for one accident
    POST /predict-batch Predict severity for up to 500 accidents

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:
    http://localhost:8000/docs

ReDoc:
    http://localhost:8000/redoc
"""

import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from api.schemas import (
    AccidentFeatures,
    BatchAccidentFeatures,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
    PredictionResult,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/severity_model.pkl"))
META_PATH  = Path("models/metadata.json")

# ── Global model state ────────────────────────────────────────────────────
_pipeline = None
_metadata: dict = {}

SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}


def load_model() -> None:
    """Load model from disk on startup. Raises on failure."""
    global _pipeline, _metadata

    if not MODEL_PATH.exists():
        log.error("Model file not found: %s", MODEL_PATH)
        log.error("Run: python scripts/train_model.py")
        return          # API starts but /health will report model_loaded=False

    with MODEL_PATH.open("rb") as fh:
        _pipeline = pickle.load(fh)
    log.info("Model loaded from %s", MODEL_PATH)

    if META_PATH.exists():
        _metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        log.info("Metadata loaded: accuracy=%.4f, macro_f1=%.4f",
                 _metadata.get("accuracy", 0),
                 _metadata.get("macro_f1", 0))
    else:
        log.warning("Metadata file not found: %s", META_PATH)


def features_to_dataframe(feat: AccidentFeatures) -> pd.DataFrame:
    """Convert a single AccidentFeatures instance to a one-row DataFrame."""
    return pd.DataFrame([feat.model_dump()])


def make_prediction(df: pd.DataFrame) -> list[PredictionResult]:
    """Run pipeline on a DataFrame and return PredictionResult list."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run scripts/train_model.py first.",
        )

    classes = _pipeline.classes_           # e.g. [1, 2, 3]
    y_pred  = _pipeline.predict(df)
    y_prob  = _pipeline.predict_proba(df)  # shape: (n, n_classes)

    # Map probabilities to named classes
    cls_to_idx = {int(c): i for i, c in enumerate(classes)}

    results = []
    for pred, prob_row in zip(y_pred, y_prob):
        pred_int = int(pred)
        results.append(PredictionResult(
            predicted_class=pred_int,
            predicted_label=SEVERITY_LABELS.get(pred_int, "Unknown"),
            probability_fatal=float(prob_row[cls_to_idx.get(1, 0)]),
            probability_serious=float(prob_row[cls_to_idx.get(2, 1)]),
            probability_slight=float(prob_row[cls_to_idx.get(3, 2)]),
        ))
    return results


# ── App factory ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Accident Severity Prediction API",
    description=(
        "Predict the severity of a UK road accident (Fatal / Serious / Slight) "
        "using a Random Forest classifier trained on DfT STATS19 2025 data "
        "enriched with Open-Meteo weather and road context lookup tables.\n\n"
        "**Note**: Severity codes: 1=Fatal, 2=Serious, 3=Slight."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
def on_startup():
    load_model()


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", summary="Service status")
def root():
    """Returns basic service information."""
    return {
        "service":     "Traffic Accident Severity Prediction API",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
        "model_info":  "/model-info",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
)
def health():
    """Returns whether the model is loaded and ready to serve predictions."""
    loaded  = _pipeline is not None
    version = _metadata.get("trained_at", "unknown")
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=version,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
)
def model_info():
    """Returns training metadata for the loaded model."""
    if not _metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata not available. Run train_model.py.",
        )
    return ModelInfoResponse(
        model_type=_metadata.get("model_type", "Unknown"),
        target_column=_metadata.get("target_column", "collision_severity"),
        severity_classes={str(k): v for k, v in {1: "Fatal", 2: "Serious", 3: "Slight"}.items()},
        numeric_features=_metadata.get("numeric_features", []),
        categorical_features=_metadata.get("categorical_features", []),
        n_features=_metadata.get("n_features", 0),
        dataset_size=_metadata.get("dataset_size", 0),
        accuracy=_metadata.get("accuracy"),
        macro_f1=_metadata.get("macro_f1"),
        trained_at=_metadata.get("trained_at", "unknown"),
        mlflow_run_id=_metadata.get("mlflow_run_id"),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict severity for one accident",
    status_code=status.HTTP_200_OK,
)
def predict(features: AccidentFeatures):
    """
    Predict the severity of a single accident.

    Returns the predicted class (1=Fatal, 2=Serious, 3=Slight),
    the human-readable label, and class probabilities.

    Missing input fields are imputed by the model pipeline.
    """
    try:
        df = features_to_dataframe(features)
        results = make_prediction(df)
        return PredictResponse(
            prediction=results[0],
            model_version=_metadata.get("trained_at", "unknown"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictResponse,
    summary="Predict severity for multiple accidents",
    status_code=status.HTTP_200_OK,
)
def predict_batch(body: BatchAccidentFeatures):
    """
    Predict severity for a batch of accidents (max 500 per request).

    Each record follows the same schema as /predict.
    Returns a list of predictions in the same order as the input.
    """
    try:
        df = pd.DataFrame([r.model_dump() for r in body.records])
        results = make_prediction(df)
        return BatchPredictResponse(
            predictions=results,
            count=len(results),
            model_version=_metadata.get("trained_at", "unknown"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Batch prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        )
