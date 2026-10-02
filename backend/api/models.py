"""Model endpoints: train, status, comparison, list/metadata."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import TrainRequest
from ml import predict as predict_mod, train as train_mod

logger = logging.getLogger("neuroshield.api.models")
router = APIRouter(prefix="/api/models", tags=["models"])


def _run_training(force: bool) -> None:
    try:
        train_mod.train_all(force=force)
        predict_mod.clear_caches()  # ensure fresh artifacts are loaded
    except Exception:  # noqa: BLE001 - state captured in TRAINING_STATE
        logger.exception("Background training failed")


@router.post("/train")
def train_models(req: TrainRequest, background_tasks: BackgroundTasks) -> dict:
    if train_mod.TRAINING_STATE["status"] == "running":
        return {"status": "running", "detail": "Training already in progress."}

    if train_mod.artifacts_exist() and not req.force:
        return {
            "status": "already_trained",
            "detail": "Cached model artifacts exist. Pass force=true to retrain.",
        }

    if req.background:
        # Reset visible state immediately so the UI shows 'running'.
        train_mod.TRAINING_STATE.update(
            {"status": "running", "steps": [], "message": "Queued", "finished_at": None}
        )
        background_tasks.add_task(_run_training, req.force)
        return {"status": "running", "detail": "Training started."}

    try:
        train_mod.train_all(force=req.force)
        predict_mod.clear_caches()
        return {"status": "completed", "detail": "Training completed."}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@router.get("/train/status")
def train_status() -> dict:
    return train_mod.TRAINING_STATE


@router.get("/comparison")
def comparison() -> dict:
    if not train_mod.EVALUATION_PATH.exists():
        raise HTTPException(status_code=404, detail="Models not trained yet.")
    evals = json.loads(train_mod.EVALUATION_PATH.read_text())
    rows = [
        {
            "model": v["model"],
            "model_key": k,
            "accuracy": v["accuracy"],
            "precision": v["precision"],
            "recall": v["recall"],
            "f1": v["f1"],
            "roc_auc": v.get("roc_auc"),
            "training_time_seconds": v.get("training_time_seconds"),
            "averaging": v.get("averaging"),
        }
        for k, v in evals.items()
    ]
    return {"models": rows, "averaging": "weighted"}


@router.get("")
def list_models() -> dict:
    if not train_mod.META_PATH.exists():
        return {"trained": False, "models": []}
    meta = train_mod.load_meta()
    models = []
    for key, disp in train_mod.MODEL_DISPLAY.items():
        models.append(
            {
                "key": key,
                "name": disp,
                "artifact_exists": train_mod._model_path(key).exists(),
            }
        )
    return {
        "trained": True,
        "trained_at": meta.get("trained_at"),
        "selection_metric": meta.get("selection_metric"),
        "bayesian_config": meta.get("bayesian_config"),
        "class_labels": meta.get("class_labels"),
        "models": models,
    }
