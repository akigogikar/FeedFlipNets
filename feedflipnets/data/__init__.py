"""Dataset registry and loader helpers."""

# Ensure built-in datasets register themselves when the package is imported.
from . import california as _california  # noqa: F401
from . import csv_generic as _csv_generic  # noqa: F401
from . import mnist as _mnist  # noqa: F401
from . import text_20newsgroups as _text_20newsgroups  # noqa: F401
from . import ucr as _ucr  # noqa: F401
from .loaders import mnist as _legacy_mnist  # noqa: F401
from .loaders import synth_fixture as _legacy_synth_fixture  # noqa: F401
from .loaders import synthetic as _legacy_synthetic  # noqa: F401
from .loaders import tinystories as _legacy_tinystories  # noqa: F401
from .registry import (
    DatasetSpec,
    DataSpec,
    available_datasets,
    get,
    get_dataset,
    iter_batches,
    register_dataset,
)

__all__ = [
    "DataSpec",
    "DatasetSpec",
    "available_datasets",
    "get",
    "get_dataset",
    "iter_batches",
    "register_dataset",
]
