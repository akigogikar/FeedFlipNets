from __future__ import annotations

from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..registry import DatasetSpec, DataSpec, register_dataset
from ..utils import batch_iterator, deterministic_split


def _make_dataset(
    length: int, seed: int, freq: int, amplitude: float, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, length, dtype=np.float32).reshape(-1, 1)
    y_true = amplitude * np.sin(freq * np.pi * x)
    if noise > 0:
        y_true = y_true + noise * rng.standard_normal(size=y_true.shape)
    y = y_true.astype(np.float32)
    return x.astype(np.float32), y


def _factory(
    *,
    length: int = 64,
    seed: int = 0,
    freq: int = 3,
    amplitude: float = 1.0,
    noise: float = 0.0,
    offline: bool = True,  # parity with other datasets
    cache_dir: str | None = None,
    **_: object,
) -> DatasetSpec:
    inputs, targets = _make_dataset(
        length=length, seed=seed, freq=freq, amplitude=amplitude, noise=noise
    )
    splits = deterministic_split(inputs.shape[0], val_split=0.1, test_split=0.2, seed=seed)
    provenance = {
        "type": "synth_fixture",
        "length": length,
        "seed": seed,
        "freq": freq,
        "amplitude": amplitude,
        "noise": noise,
        "offline": bool(offline),
    }

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(inputs, targets, indices, batch_size=batch_size, seed=split_seed)

    data_spec = DataSpec(
        d_in=int(inputs.shape[1]),
        d_out=int(targets.shape[1]),
        task_type="regression",
        normalization={},
    )

    return DatasetSpec(
        name="synth_fixture",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


register_dataset("synth_fixture", _factory)

__all__ = ["_factory"]
