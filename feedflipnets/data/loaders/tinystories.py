"""TinyStories text shards for offline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.types import Batch
from ..cache import fetch
from ..registry import DatasetSpec, DataSpec, register_dataset
from ..utils import batch_iterator, deterministic_split, resolve_cache_dir

_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
_CHECKSUM = "a78da77ff36f30c5e6a4467348b5f683afde3787e243be00b9340c306c8ad3fc"


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
    )
    np.save(path, tokens)


def _factory(
    window: int = 4,
    seed: int = 0,
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    **_: object,
) -> DatasetSpec:
    base_cache = resolve_cache_dir(cache_dir)
    offline_path = base_cache / "offline" / "tinystories_fixture.npy"
    path, provenance = fetch(
        name="tinystories",
        url=_URL,
        checksum=_CHECKSUM,
        filename="tinystories_fixture.npy",
        offline_path=offline_path,
        offline_builder=_build_offline_fixture,
        offline=offline,
        cache_dir=cache_dir,
    )
    tokens = np.load(path, allow_pickle=False)
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

    splits = deterministic_split(X.shape[0], val_split=0.1, test_split=0.2, seed=seed)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(X, y, indices, batch_size=batch_size, seed=split_seed)

    provenance = dict(provenance)
    provenance.update({"window": window, "vocab_size": len(vocab)})

    data_spec = DataSpec(
        d_in=int(X.shape[1]),
        d_out=int(y.shape[1]),
        task_type="regression",
        normalization={},
    )

    return DatasetSpec(
        name="tinystories",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


register_dataset("tinystories", _factory)
