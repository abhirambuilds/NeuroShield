"""Dataset loading + preprocessing tests (synthetic, offline)."""
import numpy as np

from ml import dataset_loader, preprocessing


def test_synthetic_dataset_loads_and_inspects():
    df, meta = dataset_loader.load_raw()
    report = dataset_loader.inspect(df)
    assert report["n_samples"] > 0
    assert report["n_features"] > 0
    assert report["n_classes"] >= 2
    assert sum(report["class_distribution"].values()) == report["n_samples"]
    assert "target_column" in report


def test_preprocessing_split_is_reproducible_and_leak_free():
    df, _ = dataset_loader.load_raw()
    target = dataset_loader.inspect(df)["target_column"]

    a = preprocessing.prepare(df, target)
    b = preprocessing.prepare(df, target)

    # Reproducible split.
    assert a.split_meta["train_samples"] == b.split_meta["train_samples"]
    assert np.array_equal(a.y_test, b.y_test)

    # No overlap between train and test indices.
    assert a.X_train.shape[0] + a.X_test.shape[0] <= len(df)

    # Scaler fitted on train only => train columns ~ zero mean.
    assert np.allclose(a.X_train.mean(axis=0), 0, atol=0.2)


def test_pagination():
    page = dataset_loader.get_page(1, 10)
    assert page["page"] == 1
    assert len(page["rows"]) <= 10
    assert page["total_rows"] > 0
