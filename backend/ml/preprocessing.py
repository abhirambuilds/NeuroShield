"""Preprocessing pipeline: cleaning, encoding, scaling, and a reproducible
stratified train/test split. All fitting happens on the training split only,
so there is no leakage from test into the transformers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from config import RANDOM_SEED, TEST_SIZE

logger = logging.getLogger("neuroshield.preprocess")


@dataclass
class PreparedData:
    """Container for a fully prepared dataset ready for model training."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_labels: list[str]
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    raw_feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    test_index: list[int] = field(default_factory=list)
    split_meta: dict[str, Any] = field(default_factory=dict)


def _build_column_transformer(
    numeric_cols: list[str], categorical_cols: list[str]
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _can_stratify(y: pd.Series) -> bool:
    counts = y.value_counts()
    # Every class needs >= 2 samples for a stratified split to be valid.
    return bool(len(counts) >= 2 and counts.min() >= 2)


def prepare(df: pd.DataFrame, target: str) -> PreparedData:
    """Clean, split, and transform the dataframe into model-ready arrays.

    Steps:
      1. Drop fully-empty columns and exact duplicate rows.
      2. Separate features / target; infer numeric vs categorical.
      3. Reproducible stratified train/test split (fixed seed).
      4. Fit the ColumnTransformer on TRAIN ONLY, transform both splits.
    """
    df = df.copy()

    # 1) Basic cleaning that is safe to do before the split.
    df = df.dropna(axis=1, how="all")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows", before - len(df))

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not present in dataframe.")

    # Rows with a missing label are unusable.
    df = df[df[target].notna()].reset_index(drop=True)

    # Remove participant/identifier columns (e.g. subject_id) to prevent identity leakage.
    identifier_cols = [
        c for c in df.columns
        if c != target
        and (
            c.lower() in ("id", "subject", "participant", "subject_id", "participant_id")
            or c.lower().endswith("_id")
        )
    ]
    feature_cols = [c for c in df.columns if c != target and c not in identifier_cols]
    X = df[feature_cols]
    y_raw = df[target].astype(str)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_labels = [str(c) for c in label_encoder.classes_]

    # 3) Reproducible split.
    stratify = y if _can_stratify(y_raw) else None
    idx = np.arange(len(df))
    X_train_df, X_test_df, y_train, y_test, _, test_idx = train_test_split(
        X,
        y,
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    # 4) Fit transformers on training data only (no leakage).
    preprocessor = _build_column_transformer(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    X_train = _to_dense(X_train)
    X_test = _to_dense(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:  # noqa: BLE001
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    split_meta = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features_out": int(X_train.shape[1]),
        "stratified": stratify is not None,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
    }
    logger.info("Prepared data: %s", split_meta)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        class_labels=class_labels,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        raw_feature_columns=feature_cols,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        test_index=[int(i) for i in test_idx],
        split_meta=split_meta,
    )


def _to_dense(arr: Any) -> np.ndarray:
    """Return a dense float array whether or not the transformer emitted sparse."""
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return np.asarray(arr, dtype=float)
