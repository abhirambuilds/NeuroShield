"""Central configuration for the NeuroShield backend.

All tunable behaviour is driven by environment variables so nothing
sensitive (Kaggle keys) or environment-specific (paths) is hardcoded.
See ``.env.example`` for the full list.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env sitting next to this file (backend/.env) if present.
load_dotenv(Path(__file__).resolve().parent / ".env")

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"
RESULTS_DIR = BACKEND_DIR / "results"
DATA_DIR = BACKEND_DIR / "data"

for _d in (MODELS_DIR, RESULTS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset -----------------------------------------------------------------
KAGGLE_DATASET = os.getenv(
    "NEUROSHIELD_KAGGLE_DATASET",
    "libiaestvez/multimodal-dataset-for-classifying-cognitive-load",
)

# Explicit target column override (leave empty to auto-detect).
TARGET_COLUMN = os.getenv("NEUROSHIELD_TARGET_COLUMN", "").strip()

# When true, and Kaggle is unreachable, generate a small synthetic dataset so
# the app/tests can run offline. OFF by default: real runs must use Kaggle.
ALLOW_SYNTHETIC = os.getenv("NEUROSHIELD_ALLOW_SYNTHETIC", "false").lower() in (
    "1",
    "true",
    "yes",
)

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = int(os.getenv("NEUROSHIELD_RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("NEUROSHIELD_TEST_SIZE", "0.25"))

# --- Bayesian sampling (kept modest so training is feasible) -----------------
BAYES_DRAWS = int(os.getenv("NEUROSHIELD_BAYES_DRAWS", "400"))
BAYES_TUNE = int(os.getenv("NEUROSHIELD_BAYES_TUNE", "400"))
BAYES_CHAINS = int(os.getenv("NEUROSHIELD_BAYES_CHAINS", "1"))
# Cap dimensionality fed to the (expensive) Bayesian sampler via PCA.
BAYES_MAX_FEATURES = int(os.getenv("NEUROSHIELD_BAYES_MAX_FEATURES", "25"))
# Subsample rows for the Bayesian sampler to keep wall-clock reasonable.
BAYES_MAX_ROWS = int(os.getenv("NEUROSHIELD_BAYES_MAX_ROWS", "4000"))

# Subsample rows for Kernel SVM training to keep quadratic complexity tractable
# (as established in the original study / unit_2.py). Evaluated on full test set.
KERNEL_MAX_ROWS = int(os.getenv("NEUROSHIELD_KERNEL_MAX_ROWS", "5000"))

# --- Model-selection criterion (used by "Automatic" prediction mode) ---------
SELECTION_METRIC = os.getenv("NEUROSHIELD_SELECTION_METRIC", "f1")

# --- API ---------------------------------------------------------------------
CORS_ORIGINS = os.getenv(
    "NEUROSHIELD_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
