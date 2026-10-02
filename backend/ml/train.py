"""Training orchestration for the four NeuroShield models.

Pipeline: load -> inspect -> preprocess -> train {LDA, Bayesian, LinearSVM,
KernelSVM} -> evaluate -> uncertainty -> persist artifacts + results.

A module-level ``TRAINING_STATE`` exposes coarse progress so the UI can render
a live training log without a websocket.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from config import KERNEL_MAX_ROWS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from ml import dataset_loader, evaluation, preprocessing, uncertainty
from ml.bayesian import BayesianLogisticRegression

logger = logging.getLogger("neuroshield.train")

MODEL_KEYS = ["lda", "bayesian", "linear_svm", "kernel_svm"]
MODEL_DISPLAY = {
    "lda": "LDA",
    "bayesian": "Bayesian Logistic Regression",
    "linear_svm": "Linear SVM",
    "kernel_svm": "Kernel SVM",
}

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
META_PATH = MODELS_DIR / "training_meta.json"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
CONFUSION_PATH = RESULTS_DIR / "confusion_matrices.json"
UNCERTAINTY_PATH = RESULTS_DIR / "bayesian_uncertainty.csv"
UNCERTAINTY_SUMMARY_PATH = RESULTS_DIR / "bayesian_uncertainty_summary.json"
EVALUATION_PATH = RESULTS_DIR / "evaluation.json"
TESTSET_PATH = MODELS_DIR / "test_set.joblib"

TRAINING_STATE: dict[str, Any] = {
    "status": "idle",  # idle | running | completed | failed
    "steps": [],
    "message": "",
    "started_at": None,
    "finished_at": None,
}


def _model_path(key: str) -> Path:
    return MODELS_DIR / f"{key}_model.joblib"


def artifacts_exist() -> bool:
    return META_PATH.exists() and PREPROCESSOR_PATH.exists()


def _set_step(name: str, status: str) -> None:
    TRAINING_STATE["steps"].append({"name": name, "status": status, "t": time.time()})
    TRAINING_STATE["message"] = f"{name}: {status}"
    logger.info("[train] %s -> %s", name, status)


def _build_models() -> dict[str, Any]:
    return {
        "lda": LinearDiscriminantAnalysis(),
        "bayesian": BayesianLogisticRegression(),
        # LinearSVC is fast + calibrated to yield probabilities without the
        # O(n^2) cost of SVC(kernel="linear", probability=True).
        "linear_svm": CalibratedClassifierCV(
            _make_linear_svc(), method="sigmoid", cv=3
        ),
        "kernel_svm": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_SEED
        ),
    }


def _make_linear_svc():
    from sklearn.svm import LinearSVC

    return LinearSVC(random_state=RANDOM_SEED, dual="auto", max_iter=5000)


def train_all(force: bool = False) -> dict[str, Any]:
    """Run the full pipeline. Returns the training metadata dict."""
    if TRAINING_STATE["status"] == "running":
        raise RuntimeError("Training already in progress.")

    if artifacts_exist() and not force:
        logger.info("Artifacts already present; skipping training.")
        return load_meta()

    TRAINING_STATE.update(
        {"status": "running", "steps": [], "message": "", "started_at": time.time(),
         "finished_at": None}
    )
    try:
        _set_step("Loading dataset", "running")
        df, source_meta = dataset_loader.load_raw()
        report = dataset_loader.inspect(df)
        target = report["target_column"]
        _set_step("Loading dataset", "done")

        _set_step("Preprocessing", "running")
        data = preprocessing.prepare(df, target)
        joblib.dump(data.preprocessor, PREPROCESSOR_PATH)
        joblib.dump(
            {
                "label_encoder": data.label_encoder,
                "feature_names": data.feature_names,
                "raw_feature_columns": data.raw_feature_columns,
                "numeric_columns": data.numeric_columns,
                "categorical_columns": data.categorical_columns,
                "class_labels": data.class_labels,
            },
            MODELS_DIR / "schema.joblib",
        )
        # Persist the test split so prediction-by-sample and the uncertainty
        # dashboard can reuse the exact held-out rows.
        joblib.dump(
            {"X_test": data.X_test, "y_test": data.y_test,
             "test_index": data.test_index},
            TESTSET_PATH,
        )
        _set_step("Preprocessing", "done")

        models = _build_models()
        comparison_rows: list[dict[str, Any]] = []
        confusion: dict[str, Any] = {}
        evaluations: dict[str, Any] = {}
        bayes_config: dict[str, Any] = {}

        for key in MODEL_KEYS:
            disp = MODEL_DISPLAY[key]
            _set_step(f"Training {disp}", "running")
            model = models[key]
            t0 = time.time()
            if key == "kernel_svm" and data.X_train.shape[0] > KERNEL_MAX_ROWS:
                # Representative stratified subset for Kernel SVM to keep O(n^2) RBF complexity tractable,
                # matching the methodology in unit_2.py. Evaluation is performed on the entire test set.
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=KERNEL_MAX_ROWS, random_state=RANDOM_SEED)
                k_idx, _ = next(sss.split(data.X_train, data.y_train))
                model.fit(data.X_train[k_idx], data.y_train[k_idx])
            else:
                model.fit(data.X_train, data.y_train)
            train_time = time.time() - t0

            y_pred = model.predict(data.X_test)
            y_proba = _safe_proba(model, data.X_test)
            metrics = evaluation.evaluate(
                data.y_test, y_pred, data.class_labels, y_proba
            )
            metrics["training_time_seconds"] = round(train_time, 3)
            metrics["model_key"] = key
            metrics["model"] = disp

            joblib.dump(model, _model_path(key))
            if key == "bayesian":
                bayes_config = getattr(model, "config_", {})
                _save_bayesian_trace(model)
                _compute_and_save_uncertainty(model, data)

            comparison_rows.append(
                {k: metrics[k] for k in
                 ["model", "accuracy", "precision", "recall", "f1",
                  "training_time_seconds"]}
            )
            confusion[key] = {
                "model": disp,
                "matrix": metrics["confusion_matrix"],
                "labels": data.class_labels,
            }
            evaluations[key] = metrics
            _set_step(f"Training {disp}", "done")

        _set_step("Evaluating models", "running")
        pd.DataFrame(comparison_rows).to_csv(COMPARISON_PATH, index=False)
        CONFUSION_PATH.write_text(json.dumps(confusion, indent=2))
        EVALUATION_PATH.write_text(json.dumps(evaluations, indent=2))
        _set_step("Evaluating models", "done")

        _set_step("Saving artifacts", "running")
        meta = {
            "trained_at": time.time(),
            "dataset": source_meta,
            "inspection": report,
            "split": data.split_meta,
            "models": list(MODEL_DISPLAY.values()),
            "bayesian_config": bayes_config,
            "selection_metric": __import__("config").SELECTION_METRIC,
            "class_labels": data.class_labels,
        }
        META_PATH.write_text(json.dumps(meta, indent=2, default=str))
        _set_step("Saving artifacts", "done")

        TRAINING_STATE.update(
            {"status": "completed", "finished_at": time.time(),
             "message": "Training completed successfully. 4 models trained."}
        )
        return meta
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        TRAINING_STATE.update(
            {"status": "failed", "finished_at": time.time(),
             "message": f"Training failed: {exc}"}
        )
        raise


def _safe_proba(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:  # noqa: BLE001
            return None
    return None


def _save_bayesian_trace(model: BayesianLogisticRegression) -> None:
    trace = getattr(model, "_trace", None)
    if trace is None:
        return
    try:
        out_path = MODELS_DIR / "bayesian_trace.nc"
        if hasattr(trace, "to_netcdf"):
            trace.to_netcdf(str(out_path))
        else:
            import arviz as az
            az.to_netcdf(trace, str(out_path))
        logger.info("Saved Bayesian trace to %s", out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save Bayesian trace: %s", exc)


def _compute_and_save_uncertainty(
    model: BayesianLogisticRegression, data: preprocessing.PreparedData
) -> None:
    proba = model.predict_proba(data.X_test)
    unc = model.predict_uncertainty(data.X_test)
    preds = np.argmax(proba, axis=1)
    summary = uncertainty.summarize_uncertainty(
        proba, unc["uncertainty"], preds, data.class_labels
    )
    summary["config"] = model.config_
    UNCERTAINTY_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(
        {
            "sample": np.arange(len(preds)),
            "prediction": [data.class_labels[i] for i in preds],
            "probability": proba.max(axis=1),
            "uncertainty": unc["uncertainty"],
            "epistemic_std": unc["epistemic_std"],
            "entropy": unc["entropy"],
        }
    ).to_csv(UNCERTAINTY_PATH, index=False)


def load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        raise FileNotFoundError("Models not trained yet.")
    return json.loads(META_PATH.read_text())
