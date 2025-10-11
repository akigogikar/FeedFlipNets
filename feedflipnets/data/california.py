"""California Housing regression dataset with offline fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.datasets import fetch_california_housing

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir, standardize


def _offline_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic synthetic regression dataset."""

    rng = np.random.default_rng(2718)
    n_samples = 256
    n_features = 8
    features = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(
        np.float32
    )
    weights = rng.normal(loc=0.0, scale=1.0, size=(n_features, 1)).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=0.1, size=(n_samples, 1)).astype(np.float32)
    targets = features @ weights + noise
    return features, targets.astype(np.float32)


@register_dataset("california_housing")
def build_california_dataset(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    standardize_inputs: bool = True,
    standardize_targets: bool = False,
) -> DatasetSpec:
    """Return the California Housing dataset specification."""

    cache_root = resolve_cache_dir(cache_dir)

    if offline:
        X, y = _offline_dataset()
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        dataset = fetch_california_housing(data_home=str(cache_root), return_X_y=False)
        X = dataset.data.astype(np.float32)
        y = dataset.target.astype(np.float32).reshape(-1, 1)
        provenance = {
            "mode": "download",
            "sklearn_version": getattr(
                fetch_california_housing, "__module__", "sklearn"
            ),
        }

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
        d_out=1,
        task_type="regression",
        num_classes=None,
        normalization=normalization,
    )

    provenance.update(
        {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "standardize_inputs": standardize_inputs,
            "standardize_targets": standardize_targets,
        }
    )

    return DatasetSpec(
        name="california_housing",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["build_california_dataset"]
