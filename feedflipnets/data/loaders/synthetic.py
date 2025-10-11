"""Pure in-memory synthetic datasets."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..registry import DataSpec, DatasetSpec, register_dataset
from ..utils import batch_iterator, deterministic_split


def _make_dataset(freq: int, n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_points, dtype=np.float32).reshape(-1, 1)
    y_true = np.sin(freq * np.pi * x)
    noise = 0.05 * rng.standard_normal(size=y_true.shape)
    y = (y_true + noise).astype(np.float32)
    return x.astype(np.float32), y.astype(np.float32)


def _factory(
    freq: int = 3,
    n_points: int = 512,
    seed: int = 0,
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    **_: object,
) -> DatasetSpec:
    x, y = _make_dataset(freq=freq, n_points=n_points, seed=seed)
    splits = deterministic_split(
        x.shape[0], val_split=val_split, test_split=test_split, seed=seed
    )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(x, y, indices, batch_size=batch_size, seed=split_seed)

    provenance = {
        "type": "synthetic",
        "freq": freq,
        "n_points": n_points,
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
    }

    data_spec = DataSpec(
        d_in=int(x.shape[1]),
        d_out=int(y.shape[1]),
        task_type="regression",
        normalization={},
    )

    return DatasetSpec(
        name="synthetic",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


register_dataset("synthetic", _factory)
