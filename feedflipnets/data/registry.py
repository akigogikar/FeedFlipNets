"""Dataset registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping

from ..core.types import Batch


@dataclass
class DatasetSpec:
    """Metadata about an available dataset."""

    name: str
    provenance: Mapping[str, object]
    loader: Callable[[str, int], Iterator[Batch]]


_REGISTRY: MutableMapping[str, Callable[..., DatasetSpec]] = {}


def register_dataset(name: str, factory: Callable[..., DatasetSpec]) -> None:
    """Register a dataset factory under ``name``."""

    _REGISTRY[name] = factory


def get_dataset(name: str, **options) -> DatasetSpec:
    """Return a dataset specification."""

    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return _REGISTRY[name](**options)


def available_datasets() -> Iterable[str]:
    return sorted(_REGISTRY)

