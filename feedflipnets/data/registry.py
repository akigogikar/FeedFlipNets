"""Dataset registry and metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, MutableMapping

import numpy as np

from ..core.types import Batch


TaskType = "regression", "multiclass", "binary", "multilabel"


@dataclass(frozen=True)
class DataSpec:
    """Structural information about a dataset.

    Attributes
    ----------
    d_in:
        Flattened dimensionality of the model inputs.
    d_out:
        Dimensionality of the targets as consumed by the model.
    task_type:
        One of ``{"regression", "multiclass", "binary", "multilabel"}``.
    num_classes:
        Optional number of discrete classes when ``task_type`` is
        ``"multiclass"``.
    normalization:
        Arbitrary metadata describing normalization that has been applied to
        either inputs or targets.  The registry does not interpret these
        values but preserving them allows experiments to remain reproducible.
    extra:
        Free-form metadata – for example the original sequence length of a
        time-series sample – that future models might find useful.
    """

    d_in: int
    d_out: int
    task_type: str
    num_classes: int | None = None
    normalization: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a dataset registered in the system."""

    name: str
    loader: Callable[[str, int], Iterator[Batch]]
    data_spec: DataSpec
    provenance: Dict[str, Any]
    splits: Dict[str, int]

    def iter_split(self, split: str, batch_size: int) -> Iterator[Batch]:
        """Return an iterator over ``split`` with ``batch_size`` batches."""

        return self.loader(split, batch_size)


DatasetFactory = Callable[..., DatasetSpec]


_REGISTRY: MutableMapping[str, DatasetFactory] = {}


def register_dataset(
    name: str | None = None,
    factory: DatasetFactory | None = None,
) -> Callable[[DatasetFactory], DatasetFactory] | None:
    """Register a dataset factory.

    ``register_dataset`` can be used both as a decorator::

        @register_dataset("mnist")
        def make_mnist(**kwargs):
            ...

    or directly::

        def make_mnist(**kwargs):
            ...
        register_dataset("mnist", make_mnist)
    """

    def _decorator(func: DatasetFactory) -> DatasetFactory:
        _REGISTRY[str(name or func.__name__)] = func
        return func

    if factory is not None:
        return _decorator(factory)
    if name is None:
        raise TypeError("register_dataset requires a name when used without a decorator")
    return _decorator


def get_dataset(
    dataset: str | None = None,
    /,
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    **options: Any,
) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``dataset``."""

    if dataset is None:
        if "dataset" in options:
            dataset = str(options.pop("dataset"))
        elif "name" in options:
            dataset = str(options.pop("name"))
        else:
            raise TypeError("Dataset name must be provided")

    if dataset not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset}")

    factory = _REGISTRY[dataset]
    spec = factory(offline=offline, cache_dir=cache_dir, **options)
    _validate_spec(spec)
    return spec


def available_datasets() -> Iterable[str]:
    """Return the sorted list of available dataset identifiers."""

    return sorted(_REGISTRY)


def _validate_spec(spec: DatasetSpec) -> None:
    if spec.data_spec.task_type not in {"regression", "multiclass", "binary", "multilabel"}:
        raise ValueError(f"Invalid task type: {spec.data_spec.task_type}")
    if spec.data_spec.task_type == "multiclass" and spec.data_spec.num_classes is None:
        raise ValueError("Multiclass datasets must define num_classes")
    if not isinstance(spec.splits, dict):
        raise TypeError("DatasetSpec.splits must be a mapping")
    for split, count in spec.splits.items():
        if count < 0:
            raise ValueError(f"Split {split!r} has negative sample count {count}")


# Backwards compatibility -----------------------------------------------------------------


def get(
    dataset: str | None = None,
    /,
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    **options: Any,
) -> DatasetSpec:
    """Alias for :func:`get_dataset` to preserve existing imports."""

    return get_dataset(dataset, offline=offline, cache_dir=cache_dir, **options)


def iter_batches(
    spec: DatasetSpec,
    split: str,
    batch_size: int,
    *,
    seed: int | None = None,
) -> Iterator[Batch]:
    """Utility helper that returns a deterministic iterator for ``split``."""

    rng = np.random.default_rng(seed)
    for batch in spec.iter_split(split, batch_size):
        if seed is not None:
            # ``loader`` implementations are already deterministic but to avoid
            # double-randomisation we simply reseed the RNG per batch.
            np.random.seed(rng.integers(0, 2**32 - 1, dtype=np.uint32))
        yield batch


__all__ = [
    "Batch",
    "DataSpec",
    "DatasetSpec",
    "available_datasets",
    "get",
    "get_dataset",
    "iter_batches",
    "register_dataset",
]
