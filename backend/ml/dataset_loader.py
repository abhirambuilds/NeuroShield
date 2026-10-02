"""Dataset acquisition and automatic inspection for NeuroShield.

Responsibilities:
  * Download the Kaggle dataset via kagglehub / the Kaggle API (never a
    hardcoded local path).
  * Auto-detect which file(s) hold features + labels.
  * Auto-detect the target column.
  * Produce a rich inspection report (shape, dtypes, missing values,
    class distribution) that the API surfaces to the dashboard.

Nothing here computes model metrics; it only prepares the raw frame.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ALLOW_SYNTHETIC,
    DATA_DIR,
    KAGGLE_DATASET,
    RANDOM_SEED,
    TARGET_COLUMN,
)

logger = logging.getLogger("neuroshield.dataset")

_TABULAR_SUFFIXES = {".csv", ".parquet", ".tsv", ".xlsx", ".xls"}
# Column names that commonly denote the label in cognitive-load datasets.
_TARGET_HINTS = (
    "cognitive_load",
    "cognitiveload",
    "workload",
    "fatigue",
    "label",
    "target",
    "class",
    "y",
    "level",
    "state",
)


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be obtained or understood."""


def download_dataset() -> Path:
    """Return the local directory containing the Kaggle dataset files.

    Checks backend/data/ first; if absent, downloads via kagglehub (cached
    locally in ~/.cache/kagglehub) and reuses it.
    """
    # 1. Check if dataset already exists locally in project backend/data
    if list_tabular_files(DATA_DIR):
        logger.info("Found local dataset in %s", DATA_DIR)
        return DATA_DIR

    try:
        import kagglehub

        logger.info("Downloading Kaggle dataset '%s' via kagglehub", KAGGLE_DATASET)
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        logger.info("Dataset cached at %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - surface a clean error upstream
        logger.warning("kagglehub download failed: %s", exc)
        if ALLOW_SYNTHETIC:
            logger.warning("Falling back to SYNTHETIC dataset (offline mode).")
            return _write_synthetic_dataset()
        raise DatasetError(
            "Kaggle download failed. Ensure KAGGLE_USERNAME/KAGGLE_KEY are set "
            "(or ~/.kaggle/kaggle.json exists) and you have network access. "
            f"Underlying error: {exc}"
        ) from exc


def list_tabular_files(path: Path) -> list[Path]:
    files = [
        p
        for p in Path(path).rglob("*")
        if p.is_file() and p.suffix.lower() in _TABULAR_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.stat().st_size, reverse=True)


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise DatasetError(f"Unsupported file type: {path}")


def _detect_target(df: pd.DataFrame) -> str:
    """Best-effort detection of the label column."""
    if TARGET_COLUMN and TARGET_COLUMN in df.columns:
        return TARGET_COLUMN

    lowered = {c.lower(): c for c in df.columns}
    # 1) Exact/substring hint match.
    for hint in _TARGET_HINTS:
        for low, original in lowered.items():
            if low == hint or hint in low:
                return original

    # 2) Heuristic: a low-cardinality, non-continuous trailing column.
    for col in reversed(df.columns.tolist()):
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 20:
            return col

    # 3) Give up gracefully -> last column.
    return df.columns[-1]


def load_raw() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download, locate the main table, and return (dataframe, source_meta)."""
    source = download_dataset()
    files = list_tabular_files(source)
    if not files:
        raise DatasetError(f"No tabular files found under {source}")

    inspected: list[dict[str, Any]] = []
    best: tuple[pd.DataFrame, Path] | None = None
    for f in files:
        try:
            df = _read_any(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", f, exc)
            continue
        inspected.append(
            {"file": str(f), "rows": int(df.shape[0]), "columns": int(df.shape[1])}
        )
        # Prefer the file with the most columns (features live there).
        if best is None or df.shape[1] > best[0].shape[1]:
            best = (df, f)

    if best is None:
        raise DatasetError("Found tabular files but none were readable.")

    df, chosen = best
    meta = {
        "dataset_name": KAGGLE_DATASET,
        "source_dir": str(source),
        "main_file": str(chosen),
        "files_inspected": inspected,
    }
    logger.info("Selected main file %s with shape %s", chosen, df.shape)
    return df, meta


def inspect(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Produce a JSON-serialisable inspection report for the dashboard."""
    target = target or _detect_target(df)
    identifier_cols = [
        c for c in df.columns
        if c != target
        and (
            c.lower() in ("id", "subject", "participant", "subject_id", "participant_id")
            or c.lower().endswith("_id")
        )
    ]
    feature_cols = [c for c in df.columns if c != target and c not in identifier_cols]

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    class_counts = (
        df[target].value_counts(dropna=False).sort_index().to_dict()
        if target in df.columns
        else {}
    )
    class_distribution = {str(k): int(v) for k, v in class_counts.items()}

    missing_total = int(df.isna().sum().sum())
    missing_by_col = {
        str(c): int(n) for c, n in df.isna().sum().items() if n > 0
    }

    return {
        "target_column": target,
        "n_samples": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": len(feature_cols),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "categorical_columns": categorical_cols[:50],
        "identifier_columns": identifier_cols,
        "classes": [str(c) for c in class_counts.keys()],
        "n_classes": len(class_counts),
        "class_distribution": class_distribution,
        "missing_values_total": missing_total,
        "missing_by_column": dict(list(missing_by_col.items())[:50]),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def _write_synthetic_dataset() -> Path:
    """Create a small, clearly-labelled synthetic HCI dataset for offline dev.

    Mirrors the real dataset's shape (many numeric behavioural features +
    a multi-class cognitive-load target) but is entirely random. Only used
    when NEUROSHIELD_ALLOW_SYNTHETIC=true.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n, n_feat = 2000, 60
    X = rng.normal(size=(n, n_feat))
    # Weak signal so accuracy is realistically modest (like the real study).
    logits = X[:, :5] @ rng.normal(size=5) + rng.normal(scale=2.0, size=n)
    y = np.digitize(logits, np.quantile(logits, [0.33, 0.66]))
    cols = {f"hci_feat_{i:03d}": X[:, i] for i in range(n_feat)}
    cols["cognitive_load_level"] = y
    df = pd.DataFrame(cols)
    out = DATA_DIR / "synthetic_cognitive_load.csv"
    df.to_csv(out, index=False)
    logger.info("Wrote synthetic dataset to %s", out)
    return DATA_DIR


@functools.lru_cache(maxsize=1)
def get_cached_dataframe() -> pd.DataFrame:
    """Load (and cache) the main dataframe so dataset endpoints are cheap."""
    df, _meta = load_raw()
    return df


def get_page(page: int, page_size: int) -> dict[str, Any]:
    """Backend pagination for the dataset explorer table."""
    df = get_cached_dataframe()
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))
    total = len(df)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    chunk = df.iloc[start:end].copy()
    # Make values JSON-safe.
    chunk = chunk.where(pd.notna(chunk), None)
    return {
        "page": page,
        "page_size": page_size,
        "total_rows": int(total),
        "total_pages": int((total + page_size - 1) // page_size),
        "columns": [str(c) for c in df.columns],
        "rows": chunk.to_dict(orient="records"),
    }
