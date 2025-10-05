"""TinyStories text shards for offline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..cache_manager import fetch
from ..registry import DatasetSpec, register_dataset

_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
_CHECKSUM = "fb7e21f9a172717ecc3f4664ef3a0bcb0ed20a7daab8471e394239c1414584ed"


def _build_offline_fixture(path: Path) -> None:
    tokens = np.array(
        [
            "<bos>",
            "the",
            "tiny",
            "robot",
            "saw",
            "a",
            "star",
            "and",
            "waved",
            ".",
            "it",
            "dreamed",
            "of",
            "friends",
            "and",
            "shared",
            "a",
            "story",
            "before",
            "sleep",
            "<eos>",
        ],
        dtype=object,
    )
    np.save(path, tokens)


def _factory(window: int = 4, seed: int = 0, **options: object) -> DatasetSpec:
    del options
    path, provenance = fetch(
        _URL,
        checksum=_CHECKSUM,
        filename="tinystories_fixture.npy",
        offline_builder=_build_offline_fixture,
    )
    tokens = np.load(path, allow_pickle=True)
    vocab = {tok: idx for idx, tok in enumerate(sorted(set(tokens)))}
    encoded = np.array([vocab[t] for t in tokens], dtype=np.float32)
    encoded /= max(len(vocab), 1)

    sequences = []
    targets = []
    for start in range(len(encoded) - window):
        sequences.append(encoded[start : start + window])
        targets.append(encoded[start + window])
    X = np.stack(sequences, axis=0)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        rng = np.random.default_rng(seed if split == "train" else seed + 1)
        n = X.shape[0]
        while True:
            idx = rng.integers(0, n, size=batch_size)
            yield Batch(inputs=X[idx], targets=y[idx])

    provenance = dict(provenance)
    provenance.update({"window": window, "vocab_size": len(vocab)})
    return DatasetSpec(name="tinystories", provenance=provenance, loader=loader)


register_dataset("tinystories", _factory)

