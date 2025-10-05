"""UCR/UEA time-series fixtures."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..cache_manager import fetch
from ..registry import DatasetSpec, register_dataset

_URL = "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip"
_CHECKSUM = "94b8ac5753e76d7052252e380e11564d268758813a8fd0a6d5632fe94feb676e"


def _build_offline_fixture(path: Path, dataset: str = "sample") -> None:
    rng = np.random.default_rng(abs(hash(dataset)) % 2**32)
    length = 128
    timesteps = 60
    X = rng.standard_normal((length, timesteps)).astype(np.float32)
    y = rng.integers(0, 3, size=(length,), dtype=np.int32)
    np.savez(path, X=X, y=y)


def _factory(name: str = "sample", seed: int = 0, **options: object) -> DatasetSpec:
    del options
    path, provenance = fetch(
        _URL,
        checksum=_CHECKSUM,
        filename=f"ucr_{name}.npz",
        offline_builder=partial(_build_offline_fixture, dataset=name),
    )
    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
    classes = int(np.max(y)) + 1
    targets = np.eye(classes, dtype=np.float32)[y]

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        rng = np.random.default_rng(seed if split == "train" else seed + 1)
        n = X.shape[0]
        while True:
            idx = rng.integers(0, n, size=batch_size)
            yield Batch(inputs=X[idx], targets=targets[idx])

    provenance = dict(provenance)
    provenance.update({"dataset": name, "samples": X.shape[0]})
    return DatasetSpec(name="ucr_uea", provenance=provenance, loader=loader)


register_dataset("ucr_uea", _factory)

