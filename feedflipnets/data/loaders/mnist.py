"""MNIST loader backed by the cache manager."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..cache_manager import fetch
from ..registry import DatasetSpec, register_dataset

_URL = "https://storage.googleapis.com/tf-keras-datasets/mnist.npz"
_CHECKSUM = "cdd4c7da39a7ad7989ea0613f1ab75b842272c24e5bdccce94d4ce7fee8be855"


def _build_offline_fixture(path: Path) -> None:
    rng = np.random.default_rng(123)
    train_x = rng.integers(0, 256, size=(256, 784), dtype=np.uint8)
    train_y = rng.integers(0, 10, size=(256,), dtype=np.uint8)
    test_x = rng.integers(0, 256, size=(64, 784), dtype=np.uint8)
    test_y = rng.integers(0, 10, size=(64,), dtype=np.uint8)
    np.savez(path, X_train=train_x, y_train=train_y, X_test=test_x, y_test=test_y)


def _prepare(one_hot: bool, X: np.ndarray, y: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
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
    **options: object,
) -> DatasetSpec:
    del options
    path, provenance = fetch(
        _URL,
        checksum=_CHECKSUM,
        filename="mnist_fixture.npz",
        offline_builder=_build_offline_fixture,
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
        test_x, test_y = test_x[:max(1, min(max_items, len(test_x)))], test_y[:max(1, min(max_items, len(test_y)))]

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
    provenance.update({"subset": subset, "max_items": max_items, "one_hot": one_hot, "num_classes": num_classes})
    return DatasetSpec(name="mnist", provenance=provenance, loader=loader)


register_dataset("mnist", _factory)

