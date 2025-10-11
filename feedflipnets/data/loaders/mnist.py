"""MNIST loader backed by the cache manager."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..cache import fetch
from ..registry import DatasetSpec, register_dataset

_URL = "https://storage.googleapis.com/tf-keras-datasets/mnist.npz"
_CHECKSUM = "8ecf920312e1afce37bc2c6c96142e1698af7837f2ca82bb28d5f633cb3517a2"


def _build_offline_fixture(path: Path) -> None:
    """Build a deterministic MNIST-like fixture for offline use."""

    # The previous implementation relied on ``np.random.default_rng`` which
    # produces different streams across NumPy releases (notably 1.26 vs 2.0).
    # That meant the serialized ``.npz`` archive changed depending on the
    # Python/NumPy combo that built it, leading to checksum mismatches in CI.
    #
    # To make the fixture stable we generate the data procedurally using plain
    # ``np.arange`` and modular arithmetic.  Everything is derived from simple
    # integer sequences, so the resulting arrays – and therefore the archive –
    # are bit-for-bit identical across platforms and NumPy versions.
    train_x = np.arange(256 * 784, dtype=np.uint32).reshape(256, 784) % 256
    train_y = np.arange(256, dtype=np.uint8) % 10
    test_x = np.arange(64 * 784, dtype=np.uint32).reshape(64, 784)[::-1] % 256
    test_y = np.arange(64, dtype=np.uint8)[::-1] % 10

    np.savez(
        path,
        X_train=train_x.astype(np.uint8),
        y_train=train_y,
        X_test=test_x.astype(np.uint8),
        y_test=test_y,
    )


def _prepare(
    one_hot: bool, X: np.ndarray, y: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    X = X.astype(np.float32)
    y = y.astype(int)
    X /= 255.0 if X.max() > 1 else 1.0
    if not one_hot:
        return X, y.astype(np.float32).reshape(-1, 1)
    targets = np.eye(num_classes, dtype=np.float32)[y]
    return X, targets


def _factory(
    subset: str = "train",
    max_items: int | None = None,
    one_hot: bool = True,
    num_classes: int = 10,
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    **_: object,
) -> DatasetSpec:
    base_cache = Path(cache_dir) if cache_dir is not None else Path(".cache/feedflip")
    offline_root = base_cache / "offline"
    offline_path = offline_root / "mnist_fixture.npz"
    path, provenance = fetch(
        name="mnist",
        url=_URL,
        checksum=_CHECKSUM,
        filename="mnist_fixture.npz",
        offline_path=offline_path,
        offline_builder=_build_offline_fixture,
        offline=offline,
        cache_dir=cache_dir,
    )
    with np.load(path) as data:
        train_x, train_y, test_x, test_y = (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
        )
    train_x, train_y = _prepare(one_hot, train_x, train_y, num_classes)
    test_x, test_y = _prepare(one_hot, test_x, test_y, num_classes)

    if max_items is not None:
        train_x, train_y = train_x[:max_items], train_y[:max_items]
        test_x, test_y = (
            test_x[: max(1, min(max_items, len(test_x)))],
            test_y[: max(1, min(max_items, len(test_y)))],
        )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if split == "train":
            if subset == "test":
                data_x, data_y = test_x, test_y
            else:
                data_x, data_y = train_x, train_y
        else:
            data_x, data_y = test_x, test_y
        rng = np.random.default_rng(0 if split == "train" else 1)
        n = data_x.shape[0]
        while True:
            indices = rng.integers(0, n, size=batch_size)
            yield Batch(inputs=data_x[indices], targets=data_y[indices])

    provenance = dict(provenance)
    provenance.update(
        {
            "subset": subset,
            "max_items": max_items,
            "one_hot": one_hot,
            "num_classes": num_classes,
        }
    )
    return DatasetSpec(
        name="mnist",
        provenance=provenance,
        loader=loader,
        checksum=provenance.get("checksum"),
    )


register_dataset("mnist", _factory)
