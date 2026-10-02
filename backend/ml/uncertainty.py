"""Aggregate uncertainty analysis for the Bayesian model.

Uncertainty for a single prediction is defined as ``1 - max_class_probability``
(the posterior-mean probability of the winning class). This is continuous,
bounded in ``[0, (K-1)/K]``, and for a binary target reduces to
``min(p, 1-p)`` — hence a maximum of 0.5 — matching the original study's
reference numbers.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def summarize_uncertainty(
    probabilities: np.ndarray,
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    class_labels: list[str],
    top_k: int = 25,
    n_bins: int = 20,
) -> dict[str, Any]:
    """Build the payload for the Uncertainty dashboard."""
    probabilities = np.asarray(probabilities, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)
    predictions = np.asarray(predictions).astype(int)

    # Max class probability per sample.
    max_prob = probabilities.max(axis=1) if probabilities.ndim == 2 else probabilities

    unc_hist, unc_edges = np.histogram(uncertainties, bins=n_bins, range=(0, 1))
    prob_hist, prob_edges = np.histogram(max_prob, bins=n_bins, range=(0, 1))

    order = np.argsort(uncertainties)[::-1][:top_k]
    high_uncertainty = [
        {
            "sample": int(i),
            "prediction": class_labels[int(predictions[i])],
            "probability": round(float(max_prob[i]), 4),
            "uncertainty": round(float(uncertainties[i]), 4),
        }
        for i in order
    ]

    return {
        "n_predictions": int(len(uncertainties)),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "median_uncertainty": float(np.median(uncertainties)),
        "max_uncertainty": float(np.max(uncertainties)),
        "min_uncertainty": float(np.min(uncertainties)),
        "std_uncertainty": float(np.std(uncertainties)),
        "mean_probability": float(np.mean(max_prob)),
        "std_probability": float(np.std(max_prob)),
        "uncertainty_histogram": {
            "counts": unc_hist.tolist(),
            "bin_edges": [round(float(e), 4) for e in unc_edges],
        },
        "probability_histogram": {
            "counts": prob_hist.tolist(),
            "bin_edges": [round(float(e), 4) for e in prob_edges],
        },
        "high_uncertainty_samples": high_uncertainty,
        "definition": "uncertainty = 1 - max_class_probability",
    }


def interpret_confidence(uncertainty: float) -> str:
    """Map a scalar uncertainty to a human-readable confidence band."""
    if uncertainty < 0.10:
        return "High confidence"
    if uncertainty < 0.25:
        return "Moderate confidence"
    if uncertainty < 0.40:
        return "Low confidence"
    return "Very low confidence"
