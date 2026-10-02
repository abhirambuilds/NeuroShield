"""Shared pytest fixtures/config.

Forces offline synthetic-dataset mode and tiny Bayesian sampling so the
suite runs quickly without Kaggle credentials or network access. These env
vars MUST be set before backend modules import ``config``.
"""
import os

os.environ.setdefault("NEUROSHIELD_ALLOW_SYNTHETIC", "true")
os.environ.setdefault("NEUROSHIELD_BAYES_DRAWS", "50")
os.environ.setdefault("NEUROSHIELD_BAYES_TUNE", "50")
os.environ.setdefault("NEUROSHIELD_BAYES_CHAINS", "1")
os.environ.setdefault("NEUROSHIELD_BAYES_MAX_FEATURES", "8")
os.environ.setdefault("NEUROSHIELD_BAYES_MAX_ROWS", "800")

import sys
from pathlib import Path

# Ensure `backend/` is importable as the root (matches uvicorn's cwd).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
