"""UCR/UEA time-series datasets with offline fixtures."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.types import Batch
from .cache import CacheError, fetch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir

BASE_URL = "https://www.timeseriesclassification.com/Downloads/{name}.zip"


def _parse_ts(content: str) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    data_started = False
    series: list[list[float]] = []
    labels: list[str] = []
    for line in lines:
        if line.startswith("@data"):
            data_started = True
            continue
        if not data_started or line.startswith("@"):
            continue
        if ":" not in line:
            continue
        values_str, label_str = line.rsplit(":", 1)
        values = [float(v) for v in values_str.split(",") if v]
        series.append(values)
        labels.append(label_str.strip())
    if not series:
        raise ValueError("No series parsed from TS file")
    X = np.asarray(series, dtype=np.float32)
    y = np.asarray(labels)
    uniques, encoded = np.unique(y, return_inverse=True)
    return X, encoded.astype(np.int64)


def _load_from_zip(path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(path, "r") as archive:
        train_name = next(
            n for n in archive.namelist() if n.lower().endswith("_train.ts")
        )
        test_name = next(
            n for n in archive.namelist() if n.lower().endswith("_test.ts")
        )
        train_bytes = archive.read(train_name).decode("utf-8", errors="ignore")
        test_bytes = archive.read(test_name).decode("utf-8", errors="ignore")
    X_train, y_train = _parse_ts(train_bytes)
    X_test, y_test = _parse_ts(test_bytes)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return X, y


def _offline_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, class-structured UCR-style dataset."""

    encoded = dataset.lower().encode("utf-8")
    seed = int.from_bytes(encoded, "little", signed=False) % (2**32 - 1)
    seed = max(seed, 1)
    rng = np.random.default_rng(seed)

    sequence_length = int(80 + (seed % 40))
    num_classes = int(max(2, (seed % 4) + 2))
    samples_per_class = 48

    time_axis = np.linspace(0.0, 2.0 * np.pi, sequence_length, dtype=np.float32)
    X: list[np.ndarray] = []
    y: list[int] = []

    base_freqs = np.linspace(0.5, 2.0, num_classes)
    base_phases = np.linspace(0.0, np.pi / 2.0, num_classes)

    for cls, (freq, phase) in enumerate(zip(base_freqs, base_phases)):
        prototype = np.sin(freq * time_axis + phase).astype(np.float32)
        for _ in range(samples_per_class):
            scale = rng.uniform(0.8, 1.2)
            drift = rng.normal(0.0, 0.1, size=sequence_length).astype(np.float32)
            sample = scale * prototype + drift
            X.append(sample)
            y.append(cls)

    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64)


@register_dataset("ucr")
def build_ucr_dataset(
    *,
    ucr_name: str = "GunPoint",
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for a UCR/UEA dataset."""

    dataset = ucr_name
    cache_root = resolve_cache_dir(cache_dir)

    if offline:
        X, y = _offline_dataset(dataset)
        provenance: dict[str, object] = {
            "mode": "offline",
            "dataset": dataset,
            "source": "synthetic",
        }
    else:
        url = BASE_URL.format(name=dataset)
        filename = f"{dataset}.zip"
        try:
            path, provenance = fetch(
                name=f"ucr_{dataset}",
                url=url,
                checksum=None,
                filename=filename,
                offline_path=None,
                offline_builder=None,
                offline=False,
                cache_dir=cache_root,
            )
        except CacheError:
            X, y = _offline_dataset(dataset)
            provenance = {
                "mode": "offline-fallback",
                "dataset": dataset,
                "source": "synthetic",
            }
        else:
            if path.suffix == ".npz":
                with np.load(path) as data:
                    X = data["X"].astype(np.float32)
                    y = data["y"].astype(np.int64)
            else:
                X, y = _load_from_zip(path, dataset)

    sequence_length = X.shape[1]
    features = X.reshape(X.shape[0], -1).astype(np.float32)
    num_classes = int(np.max(y)) + 1 if y.size else 0
    if num_classes == 0:
        raise ValueError("UCR dataset must contain at least one class")
    targets = np.eye(num_classes, dtype=np.float32)[y]

    splits = deterministic_split(
        features.shape[0], val_split=val_split, test_split=test_split, seed=seed
    )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(
            features, targets, indices, batch_size=batch_size, seed=split_seed
        )

    data_spec = DataSpec(
        d_in=int(features.shape[1]),
        d_out=num_classes,
        task_type="multiclass",
        num_classes=num_classes,
        normalization={},
        extra={"sequence_length": int(sequence_length)},
    )

    provenance = dict(provenance)
    provenance.update(
        {
            "dataset": dataset,
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "sequence_length": sequence_length,
        }
    )

    return DatasetSpec(
        name="ucr",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["build_ucr_dataset"]
