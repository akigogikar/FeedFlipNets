"""Dataset registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, MutableMapping

from ..core.types import Batch


@dataclass
class DatasetSpec:
    """Metadata about an available dataset."""

    name: str
    provenance: Mapping[str, object]
    loader: Callable[[str, int], Iterator[Batch]]
    checksum: str | None = None


_REGISTRY: MutableMapping[str, Callable[..., DatasetSpec]] = {}


def register_dataset(name: str, factory: Callable[..., DatasetSpec]) -> None:
    """Register a dataset factory under ``name``."""

    _REGISTRY[name] = factory


def get(
    dataset: str | None = None,
    /,
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    **options,
) -> DatasetSpec:
    """Return a dataset specification."""

    if dataset is None:
        if "dataset" in options:
            dataset = str(options.pop("dataset"))
        elif "name" in options:
            dataset = str(options.pop("name"))
        else:
            raise TypeError("Dataset name must be provided")
    if dataset not in _REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset}")

    if (
        "dataset_name" not in options
        and "dataset_id" not in options
        and "name" in options
    ):
        options["dataset_name"] = str(options.pop("name"))

    factory = _REGISTRY[dataset]
    return factory(offline=offline, cache_dir=cache_dir, **options)


def available_datasets() -> Iterable[str]:
    return sorted(_REGISTRY)
