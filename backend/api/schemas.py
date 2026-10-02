"""Pydantic request/response models for the NeuroShield API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    models_trained: bool
    version: str = "1.0.0"


class TrainRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Retrain even if cached artifacts already exist.",
    )
    background: bool = Field(
        default=True,
        description="Run training in the background and poll status.",
    )


class PredictRequest(BaseModel):
    model: str = Field(
        default="automatic",
        description="One of: automatic, lda, bayesian, linear_svm, kernel_svm.",
    )
    sample_id: int | None = Field(
        default=None, description="Index into the held-out test set."
    )
    features: dict[str, Any] | None = Field(
        default=None, description="Raw {column: value} mapping for advanced input."
    )


class MessageResponse(BaseModel):
    detail: str
