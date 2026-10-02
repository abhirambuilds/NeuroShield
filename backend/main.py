"""NeuroShield FastAPI application entrypoint.

Run locally with:  uvicorn main:app --reload
Interactive docs at /docs .
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import dataset, evaluation, models, predictions
from api.schemas import HealthResponse
from config import CORS_ORIGINS
from ml import train as train_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("neuroshield")

app = FastAPI(
    title="NeuroShield API",
    description=(
        "Probabilistic Cognitive Workload & Fatigue Analysis. Compares LDA, "
        "Bayesian Logistic Regression, Linear SVM, and Kernel SVM on Kaggle HCI data."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(evaluation.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(models_trained=train_mod.artifacts_exist())


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "name": "NeuroShield API",
        "docs": "/docs",
        "health": "/api/health",
    }
