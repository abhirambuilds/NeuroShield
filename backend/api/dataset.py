"""Dataset endpoints: info + backend-paginated rows."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from ml import dataset_loader, train as train_mod

logger = logging.getLogger("neuroshield.api.dataset")
router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/info")
def dataset_info() -> dict:
    """Dataset name, sizes, features, missing values, class distribution.

    Prefers cached training metadata (so it reflects the exact split used).
    Falls back to a live inspection if models are not trained yet.
    """
    try:
        if train_mod.META_PATH.exists():
            meta = train_mod.load_meta()
            report = meta["inspection"]
            split = meta.get("split", {})
            return {
                "dataset_name": meta["dataset"].get("dataset_name"),
                "main_file": meta["dataset"].get("main_file"),
                "n_samples": report["n_samples"],
                "n_features": report["n_features"],
                "train_samples": split.get("train_samples"),
                "test_samples": split.get("test_samples"),
                "missing_values": report["missing_values_total"],
                "classes": report["classes"],
                "class_distribution": report["class_distribution"],
                "numeric_features": report["numeric_features"],
                "categorical_features": report["categorical_features"],
                "duplicate_rows": report["duplicate_rows"],
                "target_column": report["target_column"],
                "source": "training_metadata",
            }
        df = dataset_loader.get_cached_dataframe()
        report = dataset_loader.inspect(df)
        return {
            "dataset_name": dataset_loader.KAGGLE_DATASET,
            "n_samples": report["n_samples"],
            "n_features": report["n_features"],
            "train_samples": None,
            "test_samples": None,
            "missing_values": report["missing_values_total"],
            "classes": report["classes"],
            "class_distribution": report["class_distribution"],
            "numeric_features": report["numeric_features"],
            "categorical_features": report["categorical_features"],
            "duplicate_rows": report["duplicate_rows"],
            "target_column": report["target_column"],
            "source": "live_inspection",
        }
    except dataset_loader.DatasetError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("dataset/info failed")
        raise HTTPException(
            status_code=500, detail="Failed to load dataset information."
        ) from exc


@router.get("/rows")
def dataset_rows(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
) -> dict:
    try:
        return dataset_loader.get_page(page, page_size)
    except dataset_loader.DatasetError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("dataset/rows failed")
        raise HTTPException(status_code=500, detail="Failed to load rows.") from exc
