"""Generic CSV loaders for regression and classification tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, standardize

FIXTURE_DIR = Path(__file__).resolve().parent / "_fixtures"


def _load_csv(path: Path, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in CSV")
    y = df.pop(target_col).to_numpy()
    X = df.to_numpy(dtype=np.float32)
    return X.astype(np.float32), y


def _default_path(name: str) -> Path:
    return FIXTURE_DIR / name


@register_dataset("csv_regression")
def load_csv_regression(
    *,
    csv_path: str | Path | None = None,
    target_col: str = "target",
    offline: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    standardize_inputs: bool = True,
    standardize_targets: bool = True,
    cache_dir: str | Path | None = None,
) -> DatasetSpec:
    """Load a regression dataset from a CSV file."""

    path = Path(csv_path) if csv_path else _default_path("csv_regression_fixture.csv")
    X, y_raw = _load_csv(path, target_col)
    y = np.asarray(y_raw, dtype=np.float32).reshape(-1, 1)

    normalization: dict[str, dict[str, list[float]]] = {}
    if standardize_inputs:
        X, mean, std = standardize(X)
        normalization["inputs"] = {
            "mean": mean.flatten().tolist(),
            "std": std.flatten().tolist(),
        }
    if standardize_targets:
        y, t_mean, t_std = standardize(y)
        normalization["targets"] = {
            "mean": t_mean.flatten().tolist(),
            "std": t_std.flatten().tolist(),
        }

    splits = deterministic_split(
        X.shape[0], val_split=val_split, test_split=test_split, seed=seed
    )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(X, y, indices, batch_size=batch_size, seed=split_seed)

    data_spec = DataSpec(
        d_in=int(X.shape[1]),
        d_out=int(y.shape[1]),
        task_type="regression",
        normalization=normalization,
    )

    provenance = {
        "path": str(path),
        "val_split": val_split,
        "test_split": test_split,
        "seed": seed,
        "target_col": target_col,
        "standardize_inputs": standardize_inputs,
        "standardize_targets": standardize_targets,
    }

    return DatasetSpec(
        name="csv_regression",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


@register_dataset("csv_classification")
def load_csv_classification(
    *,
    csv_path: str | Path | None = None,
    target_col: str = "target",
    offline: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
    cache_dir: str | Path | None = None,
) -> DatasetSpec:
    """Load a classification dataset from a CSV file."""

    path = (
        Path(csv_path) if csv_path else _default_path("csv_classification_fixture.csv")
    )
    X, y_raw = _load_csv(path, target_col)
    X = X.astype(np.float32)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = int(np.max(y_encoded)) + 1
    if one_hot:
        y = np.eye(num_classes, dtype=np.float32)[y_encoded]
    else:
        y = y_encoded.astype(np.float32).reshape(-1, 1)

    splits = deterministic_split(
        X.shape[0], val_split=val_split, test_split=test_split, seed=seed
    )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(X, y, indices, batch_size=batch_size, seed=split_seed)

    data_spec = DataSpec(
        d_in=int(X.shape[1]),
        d_out=num_classes if one_hot else 1,
        task_type="multiclass",
        num_classes=num_classes,
        normalization={},
    )

    provenance = {
        "path": str(path),
        "val_split": val_split,
        "test_split": test_split,
        "seed": seed,
        "target_col": target_col,
        "one_hot": one_hot,
        "classes": encoder.classes_.tolist(),
    }

    return DatasetSpec(
        name="csv_classification",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["load_csv_regression", "load_csv_classification"]
