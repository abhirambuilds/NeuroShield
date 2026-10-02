"""Prediction service: loads persisted artifacts once and serves predictions.

Supports two input modes used by the frontend:
  * by test-sample index (uses the exact held-out, already-transformed row)
  * by raw feature mapping (transformed through the fitted preprocessor)

Model selection supports the four models plus "automatic", which picks the
model that maximises the configured selection metric on the test set.
"""
from __future__ import annotations

import functools
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR, SELECTION_METRIC
from ml import train as train_mod
from ml.uncertainty import interpret_confidence

logger = logging.getLogger("neuroshield.predict")


class PredictionError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    p = MODELS_DIR / "schema.joblib"
    if not p.exists():
        raise PredictionError("Models not trained yet.")
    return joblib.load(p)


@functools.lru_cache(maxsize=1)
def _preprocessor():
    return joblib.load(train_mod.PREPROCESSOR_PATH)


@functools.lru_cache(maxsize=1)
def _test_set() -> dict[str, Any]:
    return joblib.load(train_mod.TESTSET_PATH)


@functools.lru_cache(maxsize=8)
def _load_model(key: str):
    path = train_mod._model_path(key)
    if not path.exists():
        raise PredictionError(f"Model '{key}' is not trained.")
    return joblib.load(path)


def clear_caches() -> None:
    for fn in (_schema, _preprocessor, _test_set, _load_model):
        fn.cache_clear()


def _selection_metric_key() -> str:
    return SELECTION_METRIC if SELECTION_METRIC in {
        "accuracy", "precision", "recall", "f1"
    } else "f1"


def _auto_select() -> str:
    eval_path = train_mod.EVALUATION_PATH
    if not eval_path.exists():
        return "linear_svm"
    evals = json.loads(eval_path.read_text())
    metric = _selection_metric_key()
    best = max(evals.items(), key=lambda kv: kv[1].get(metric, 0.0))
    return best[0]


def resolve_model_key(requested: str) -> tuple[str, str]:
    """Map a UI selection to a concrete model key. Returns (key, reason)."""
    requested = (requested or "automatic").lower()
    if requested in ("auto", "automatic", ""):
        key = _auto_select()
        return key, (
            f"Automatic mode selected '{train_mod.MODEL_DISPLAY[key]}' using the "
            f"configured criterion: highest {_selection_metric_key()} on the test set."
        )
    if requested not in train_mod.MODEL_KEYS:
        raise PredictionError(f"Unknown model '{requested}'.")
    return requested, f"User-selected model: {train_mod.MODEL_DISPLAY[requested]}"


def _vectorize_features(features: dict[str, Any]) -> np.ndarray:
    """Turn a raw {column: value} mapping into a transformed feature row."""
    schema = _schema()
    cols = schema["raw_feature_columns"]
    row = {c: features.get(c, np.nan) for c in cols}
    df = pd.DataFrame([row], columns=cols)
    X = _preprocessor().transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=float)


def _format_class_label(raw_label: str) -> str:
    s = str(raw_label).strip()
    if s == "1":
        return "High Cognitive Workload"
    if s == "0":
        return "Low Cognitive Workload"
    return s


def _predict_with(model_key: str, X: np.ndarray) -> dict[str, Any]:
    schema = _schema()
    class_labels = schema["class_labels"]
    model = _load_model(model_key)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X), dtype=float)[0]
        except Exception:  # noqa: BLE001
            proba = None

    if proba is not None:
        cls_idx = int(np.argmax(proba))
        probability = float(proba[cls_idx])
    else:
        cls_idx = int(model.predict(X)[0])
        probability = None

    # Uncertainty: exact for Bayesian, derived from proba otherwise.
    if model_key == "bayesian" and hasattr(model, "predict_uncertainty"):
        u = model.predict_uncertainty(X)
        unc = float(u["uncertainty"][0])
        epistemic = float(u["epistemic_std"][0])
    elif proba is not None:
        unc = float(1.0 - probability)
        epistemic = None
    else:
        unc = None
        epistemic = None

    return {
        "prediction_index": cls_idx,
        "prediction_class": class_labels[cls_idx],
        "prediction_label": _format_class_label(class_labels[cls_idx]),
        "probability": round(probability, 4) if probability is not None else None,
        "uncertainty": round(unc, 4) if unc is not None else None,
        "epistemic_std": round(epistemic, 4) if epistemic is not None else None,
        "confidence_interpretation": (
            interpret_confidence(unc) if unc is not None else "Not available"
        ),
        "model_key": model_key,
        "model": train_mod.MODEL_DISPLAY[model_key],
        "probabilities": (
            {class_labels[i]: round(float(p), 4) for i, p in enumerate(proba)}
            if proba is not None else None
        ),
        "probabilities_breakdown": (
            [
                {
                    "class": class_labels[i],
                    "label": _format_class_label(class_labels[i]),
                    "probability": round(float(p), 4),
                    "percentage": round(float(p) * 100, 1),
                }
                for i, p in enumerate(proba)
            ]
            if proba is not None else None
        ),
    }


def predict_sample(sample_id: int, model: str = "automatic") -> dict[str, Any]:
    ts = _test_set()
    X_test = ts["X_test"]
    if sample_id < 0 or sample_id >= len(X_test):
        raise PredictionError(
            f"sample_id out of range (0..{len(X_test) - 1})."
        )
    key, reason = resolve_model_key(model)
    X = X_test[sample_id : sample_id + 1]
    out = _predict_with(key, X)
    out["selection_reason"] = reason
    out["input_mode"] = "dataset_sample"
    out["sample_id"] = int(sample_id)
    raw_true = _schema()["class_labels"][int(ts["y_test"][sample_id])]
    out["true_label"] = raw_true
    out["true_label_display"] = _format_class_label(raw_true)
    return out


def predict_features(
    features: dict[str, Any], model: str = "automatic"
) -> dict[str, Any]:
    if not features:
        raise PredictionError("No features provided.")
    key, reason = resolve_model_key(model)
    X = _vectorize_features(features)
    out = _predict_with(key, X)
    out["selection_reason"] = reason
    out["input_mode"] = "advanced_input"
    return out


def sample_feature_schema(limit: int = 12) -> dict[str, Any]:
    """Expose a manageable subset of raw features for the advanced form."""
    schema = _schema()
    numeric = schema["numeric_columns"][:limit]
    return {
        "numeric_features": numeric,
        "total_raw_features": len(schema["raw_feature_columns"]),
        "note": (
            "Only a subset of numeric features is shown. Unspecified features "
            "are imputed by the fitted preprocessor."
        ),
        "class_labels": schema["class_labels"],
    }
