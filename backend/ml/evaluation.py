"""Model evaluation metrics with explicit, honest averaging.

For multiclass targets we report *weighted* precision/recall/F1 by default
(so class imbalance is accounted for) and always label the averaging method
in the output so the UI never silently compares incompatible metrics.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("neuroshield.evaluation")

AVERAGING = "weighted"  # documented and surfaced to the UI


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable metrics dict for one model."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(len(class_labels)))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=AVERAGING, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=AVERAGING, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=AVERAGING, zero_division=0)),
        "averaging": AVERAGING,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_labels": class_labels,
    }

    metrics["roc_auc"] = _safe_roc_auc(y_true, y_proba, labels)
    return metrics


def _safe_roc_auc(
    y_true: np.ndarray, y_proba: np.ndarray | None, labels: list[int]
) -> float | None:
    if y_proba is None:
        return None
    try:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted", labels=labels
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROC-AUC could not be computed: %s", exc)
        return None
