"""MNIST dataset with deterministic splits and offline fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.types import Batch
from .cache import fetch
from .registry import DataSpec, DatasetSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir

MNIST_URL = "https://storage.googleapis.com/tf-keras-datasets/mnist.npz"
MNIST_CHECKSUM = "8ecf920312e1afce37bc2c6c96142e1698af7837f2ca82bb28d5f633cb3517a2"


def _load_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    x_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    x_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y


def _prepare_inputs(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32)
    if images.max() > 1:
        images /= 255.0
    images = images.reshape(images.shape[0], -1)
    return images.astype(np.float32)


def _prepare_targets(labels: np.ndarray, *, one_hot: bool, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64)
    if one_hot:
        eye = np.eye(num_classes, dtype=np.float32)
        return eye[labels]
    return labels.astype(np.float32).reshape(-1, 1)


def _offline_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic synthetic MNIST-like dataset."""

    rng = np.random.default_rng(12345)
    num_samples = 256
    images = rng.integers(0, 256, size=(num_samples, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=(num_samples,), dtype=np.int64)
    return images.astype(np.float32), labels


@register_dataset("mnist")
def build_mnist(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for MNIST."""

    cache_root = resolve_cache_dir(cache_dir)
    if offline:
        inputs_raw, labels_raw = _offline_dataset()
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        path, provenance = fetch(
            name="mnist",
            url=MNIST_URL,
            checksum=MNIST_CHECKSUM,
            filename="mnist.npz",
            offline_path=None,
            offline_builder=None,
            offline=False,
            cache_dir=cache_root,
        )

        inputs_raw, labels_raw = _load_archive(path)
    inputs = _prepare_inputs(inputs_raw)
    targets = _prepare_targets(labels_raw, one_hot=one_hot, num_classes=10)

    splits = deterministic_split(
        inputs.shape[0], val_split=val_split, test_split=test_split, seed=seed
    )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(
            inputs, targets, indices, batch_size=batch_size, seed=split_seed
        )

    target_dim = 10 if one_hot else 1
    data_spec = DataSpec(
        d_in=int(inputs.shape[1]),
        d_out=target_dim,
        task_type="multiclass",
        num_classes=10,
        normalization={"inputs": {"method": "minmax", "range": [0.0, 1.0]}},
        extra={"input_shape": (28, 28)},
    )

    provenance = dict(provenance)
    provenance.update({
        "val_split": val_split,
        "test_split": test_split,
        "seed": seed,
        "one_hot": one_hot,
    })

    split_sizes = splits.sizes

    return DatasetSpec(
        name="mnist",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in split_sizes.items()},
    )


__all__ = ["build_mnist"]
