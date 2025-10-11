"""Utility helpers for dataset loaders."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from ..core.types import Batch

DEFAULT_CACHE_SUBDIR = Path.home() / ".cache" / "feedflipnets"


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the effective cache directory for datasets."""

    env_dir = os.environ.get("FFN_CACHE_DIR") or os.environ.get("FEEDFLIP_CACHE_DIR")
    base = Path(cache_dir or env_dir or DEFAULT_CACHE_SUBDIR)
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_file(cache_dir: Path, *parts: str, checksum: str | None = None) -> Path:
    """Return a deterministic cache file path under ``cache_dir``."""

    filename = "-".join(parts)
    if checksum:
        filename = f"{filename}-{checksum[:8]}"
    return cache_dir / filename


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs and return a generator."""

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class SplitIndices:
    """Indices for train/validation/test partitions."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def sizes(self) -> Mapping[str, int]:
        return {
            "train": int(self.train.size),
            "val": int(self.val.size),
            "test": int(self.test.size),
        }


def deterministic_split(
    n_samples: int,
    *,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
) -> SplitIndices:
    """Return deterministic indices for the requested split ratios."""

    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1)")
    if not 0 <= test_split < 1:
        raise ValueError("test_split must be in [0, 1)")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be < 1")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    test_size = int(round(n_samples * test_split))
    val_size = int(round(n_samples * val_split))
    # Ensure at least one sample per split when possible
    test_size = min(max(test_size, 1 if test_split > 0 else 0), n_samples)
    remaining = n_samples - test_size
    val_size = min(max(val_size, 1 if val_split > 0 else 0), remaining)
    train_size = n_samples - val_size - test_size
    if train_size <= 0:
        raise ValueError("Not enough samples for the requested splits")

    test_idx = indices[:test_size]
    val_idx = indices[test_size : test_size + val_size]
    train_idx = indices[test_size + val_size :]

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def batch_iterator(
    features: np.ndarray,
    targets: np.ndarray,
    indices: Sequence[int],
    *,
    batch_size: int,
    seed: int,
) -> Iterator[Batch]:
    """Yield deterministic batches by sampling with replacement."""

    rng = np.random.default_rng(seed)
    index_array = np.asarray(indices)
    n = index_array.size
    while True:
        sampled = rng.integers(0, n, size=batch_size)
        subset = index_array[sampled]
        yield Batch(inputs=features[subset], targets=targets[subset])


def ensure_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def checksum_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def standardize(
    array: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply standard scaling returning the scaled array and parameters."""

    if mean is None or std is None:
        mean = array.mean(axis=0, keepdims=True)
        std = array.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
    scaled = (array - mean) / std
    return scaled.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def identity_batches(
    features: np.ndarray, targets: np.ndarray, split: str, batch_size: int
) -> Iterator[Batch]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    # Deterministic sequential batches (no sampling) used for fixtures/tests.
    start = 0
    while True:
        end = start + batch_size
        idx = np.arange(start, min(end, features.shape[0]))
        if idx.size == 0:
            idx = np.arange(0, min(batch_size, features.shape[0]))
        yield Batch(inputs=features[idx], targets=targets[idx])
        start = end % features.shape[0]
