from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..registry import DatasetSpec, register_dataset


@dataclass
class _FixtureDataset:
    inputs: np.ndarray
    targets: np.ndarray

    def batches(self, batch_size: int) -> Iterator[Batch]:
        n = self.inputs.shape[0]
        if n == 0:
            raise ValueError("Fixture dataset cannot be empty")
        offset = 0
        while True:
            idx = (np.arange(batch_size) + offset) % n
            offset = (offset + batch_size) % n
            yield Batch(inputs=self.inputs[idx], targets=self.targets[idx])


def _make_dataset(
    length: int, seed: int, freq: int, amplitude: float, noise: float
) -> _FixtureDataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, length, dtype=np.float32).reshape(-1, 1)
    y_true = amplitude * np.sin(freq * np.pi * x)
    if noise > 0:
        y_true = y_true + noise * rng.standard_normal(size=y_true.shape)
    y = y_true.astype(np.float32)
    return _FixtureDataset(inputs=x.astype(np.float32), targets=y)


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
    dataset = _make_dataset(
        length=length, seed=seed, freq=freq, amplitude=amplitude, noise=noise
    )
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
        if split not in {"train", "test", "eval"}:
            raise ValueError(f"Unsupported split: {split}")
        return dataset.batches(batch_size)

    return DatasetSpec(name="synth_fixture", provenance=provenance, loader=loader)


register_dataset("synth_fixture", _factory)

__all__ = ["_factory"]
