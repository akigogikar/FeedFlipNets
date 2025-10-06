"""UCR/UEA time-series fixtures."""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..cache import fetch
from ..registry import DatasetSpec, register_dataset

_URL = "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip"
_CHECKSUM = "2476234343311b7f97a83bc26b1378fac835e12d8d327ce2e9bb4ed0a5926c7c"


def _build_offline_fixture(path: Path, dataset: str = "sample") -> None:
    seed = int(hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    length = 128
    timesteps = 60
    X = rng.standard_normal((length, timesteps)).astype(np.float32)
    y = rng.integers(0, 3, size=(length,), dtype=np.int32)
    np.savez(path, X=X, y=y)


def _factory(
    name: str = "sample",
    seed: int = 0,
    *,
    dataset_name: str | None = None,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    **_: object,
) -> DatasetSpec:
    dataset_id = dataset_name or name
    base_cache = Path(cache_dir) if cache_dir is not None else Path(".cache/feedflip")
    offline_path = base_cache / "offline" / f"ucr_{dataset_id}.npz"
    checksum = _CHECKSUM if dataset_id == "sample" else None
    path, provenance = fetch(
        name=f"ucr_{dataset_id}",
        url=_URL,
        checksum=checksum,
        filename=f"ucr_{dataset_id}.npz",
        offline_path=offline_path,
        offline_builder=partial(_build_offline_fixture, dataset=dataset_id),
        offline=offline,
        cache_dir=cache_dir,
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
    provenance.update({"dataset": dataset_id, "samples": X.shape[0]})
    return DatasetSpec(
        name="ucr_uea",
        provenance=provenance,
        loader=loader,
        checksum=provenance.get("checksum"),
    )


register_dataset("ucr_uea", _factory)
