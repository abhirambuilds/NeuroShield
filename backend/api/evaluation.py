"""Evaluation + uncertainty endpoints (chart-ready payloads)."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from ml import train as train_mod

logger = logging.getLogger("neuroshield.api.evaluation")
router = APIRouter(prefix="/api", tags=["evaluation"])


@router.get("/evaluation")
def evaluation() -> dict:
    if not train_mod.EVALUATION_PATH.exists():
        raise HTTPException(status_code=404, detail="Models not trained yet.")
    evals = json.loads(train_mod.EVALUATION_PATH.read_text())
    confusion = (
        json.loads(train_mod.CONFUSION_PATH.read_text())
        if train_mod.CONFUSION_PATH.exists()
        else {}
    )
    # Bar-chart friendly reshape.
    metric_names = ["accuracy", "precision", "recall", "f1"]
    bars = []
    for key, v in evals.items():
        bars.append({"model": v["model"], **{m: v[m] for m in metric_names}})

    # Highest-per-metric (never labelled "best overall").
    highlights = {}
    for m in metric_names:
        best_key = max(evals, key=lambda k: evals[k].get(m, 0.0))
        highlights[m] = {
            "model": evals[best_key]["model"],
            "value": round(evals[best_key][m], 4),
        }

    return {
        "averaging": "weighted",
        "metrics": evals,
        "bar_chart": bars,
        "confusion_matrices": confusion,
        "highlights": highlights,
    }


@router.get("/uncertainty")
def uncertainty() -> dict:
    if not train_mod.UNCERTAINTY_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Uncertainty analysis unavailable. Train models first.",
        )
    return json.loads(train_mod.UNCERTAINTY_SUMMARY_PATH.read_text())
