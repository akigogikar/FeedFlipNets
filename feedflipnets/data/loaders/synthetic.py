"""Pure in-memory synthetic datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..registry import DatasetSpec, register_dataset


@dataclass
class _SyntheticDataset:
    x: np.ndarray
    y: np.ndarray

    def batches(self, batch_size: int, rng: np.random.Generator) -> Iterator[Batch]:
        n = self.x.shape[0]
        while True:
            idx = rng.integers(0, n, size=batch_size)
            yield Batch(inputs=self.x[idx], targets=self.y[idx])


def _make_dataset(freq: int, n_points: int, seed: int) -> _SyntheticDataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_points, dtype=np.float32)
    x = x.reshape(-1, 1)
    y_true = np.sin(freq * np.pi * x)
    noise = 0.05 * rng.standard_normal(size=y_true.shape)
    y = y_true + noise
    return _SyntheticDataset(x=x.astype(np.float32), y=y.astype(np.float32))


def _factory(freq: int = 3, n_points: int = 512, seed: int = 0, **_: object) -> DatasetSpec:
    dataset = _make_dataset(freq=freq, n_points=n_points, seed=seed)
    provenance = {
        "type": "synthetic",
        "freq": freq,
        "n_points": n_points,
        "seed": seed,
    }

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        rng = np.random.default_rng(seed if split == "train" else seed + 1)
        return dataset.batches(batch_size, rng)

    return DatasetSpec(name="synthetic", provenance=provenance, loader=loader)


register_dataset("synthetic", _factory)

