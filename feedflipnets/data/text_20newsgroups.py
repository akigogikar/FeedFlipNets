"""20 Newsgroups dataset with hashing vectorizer and offline fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir


def _offline_dataset(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic sparse bag-of-words style dataset."""

    rng = np.random.default_rng(4242)
    num_classes = 8
    samples_per_class = 30
    tokens_per_doc = max(6, n_features // 48)

    anchors: list[np.ndarray] = []
    for cls in range(num_classes):
        anchor = np.zeros(n_features, dtype=np.float32)
        active = rng.choice(n_features, size=min(tokens_per_doc * 2, n_features), replace=False)
        anchor[active] = rng.uniform(0.6, 1.0, size=active.size).astype(np.float32)
        anchors.append(anchor)

    documents: list[np.ndarray] = []
    labels: list[int] = []
    for cls, anchor in enumerate(anchors):
        for _ in range(samples_per_class):
            doc = anchor.copy()
            mask = rng.random(size=n_features) < 0.15
            doc *= mask.astype(np.float32)
            noise_idx = rng.choice(
                n_features, size=min(tokens_per_doc // 2 + 1, n_features), replace=False
            )
            doc[noise_idx] += rng.uniform(0.0, 0.3, size=noise_idx.size).astype(np.float32)
            documents.append(doc)
            labels.append(cls)

    X = np.stack(documents, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


@register_dataset("20newsgroups")
def build_20newsgroups(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    subset: str = "all",
    n_features: int = 4096,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for 20 Newsgroups."""

    if offline:
        X, y = _offline_dataset(n_features)
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        cache_root = resolve_cache_dir(cache_dir)
        raw = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers"), data_home=str(cache_root)
        )
        vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )
        X_sparse = vectorizer.transform(raw.data)
        X = X_sparse.toarray().astype(np.float32)
        y = raw.target.astype(np.int64)
        provenance = {
            "mode": "download",
            "subset": subset,
            "n_features": n_features,
            "target_names": list(raw.target_names),
        }

    num_classes = int(np.max(y)) + 1 if y.size else 0
    y_one_hot = np.eye(num_classes, dtype=np.float32)[y] if num_classes else y.reshape(-1, 1)

    splits = deterministic_split(X.shape[0], val_split=val_split, test_split=test_split, seed=seed)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(X, y_one_hot, indices, batch_size=batch_size, seed=split_seed)

    data_spec = DataSpec(
        d_in=int(X.shape[1]),
        d_out=num_classes,
        task_type="multiclass",
        num_classes=num_classes,
        normalization={"inputs": {"method": "l2"}},
    )

    provenance.update(
        {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
        }
    )

    return DatasetSpec(
        name="20newsgroups",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["build_20newsgroups"]
