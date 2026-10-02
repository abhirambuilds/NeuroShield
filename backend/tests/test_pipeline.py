"""End-to-end pipeline + API tests.

The full-training test needs PyMC; it is skipped automatically if PyMC is
not installed so the lighter tests still run.
"""
import importlib.util

import pytest
from fastapi.testclient import TestClient

import main
from ml import train as train_mod

HAS_PYMC = importlib.util.find_spec("pymc") is not None
client = TestClient(main.app)


def test_health_endpoint():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
def test_full_pipeline_train_save_reload_predict():
    meta = train_mod.train_all(force=False)
    assert meta["split"]["train_samples"] > 0

    # Artifacts persisted.
    for key in train_mod.MODEL_KEYS:
        assert train_mod._model_path(key).exists()
    assert train_mod.PREPROCESSOR_PATH.exists()

    # Comparison endpoint returns all four models.
    r = client.get("/api/models/comparison")
    assert r.status_code == 200
    assert len(r.json()["models"]) == 4

    # Evaluation endpoint has confusion matrices + highlights.
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    body = r.json()
    assert "confusion_matrices" in body and "highlights" in body

    # Prediction by sample works and returns uncertainty.
    r = client.post("/api/predict", json={"model": "automatic", "sample_id": 0})
    assert r.status_code == 200
    pred = r.json()
    assert pred["prediction_class"] is not None
    assert 0.0 <= pred["uncertainty"] <= 1.0

    # Bayesian predictions expose epistemic std.
    r = client.post("/api/predict", json={"model": "bayesian", "sample_id": 1})
    assert r.status_code == 200
    assert r.json()["model_key"] == "bayesian"

    # Uncertainty summary endpoint.
    r = client.get("/api/uncertainty")
    assert r.status_code == 200
    assert r.json()["max_uncertainty"] <= 1.0


def test_predict_requires_input():
    r = client.post("/api/predict", json={"model": "automatic"})
    assert r.status_code in (400, 422)
