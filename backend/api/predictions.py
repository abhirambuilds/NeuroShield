"""Prediction endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest
from ml import predict as predict_mod

logger = logging.getLogger("neuroshield.api.predictions")
router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        if req.sample_id is not None:
            return predict_mod.predict_sample(req.sample_id, req.model)
        if req.features:
            return predict_mod.predict_features(req.features, req.model)
        raise HTTPException(
            status_code=422,
            detail="Provide either 'sample_id' or a non-empty 'features' object.",
        )
    except predict_mod.PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@router.get("/predict/schema")
def predict_schema() -> dict:
    try:
        return predict_mod.sample_feature_schema()
    except predict_mod.PredictionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/predict/samples")
def list_samples(limit: int = 50) -> dict:
    """Small catalogue of test-sample ids the UI can offer for selection."""
    try:
        ts = predict_mod._test_set()
        schema = predict_mod._schema()
        n = len(ts["y_test"])
        limit = max(1, min(limit, n))
        return {
            "total_test_samples": int(n),
            "samples": [
                {
                    "sample_id": int(i),
                    "true_label": schema["class_labels"][int(ts["y_test"][i])],
                }
                for i in range(limit)
            ],
        }
    except predict_mod.PredictionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
