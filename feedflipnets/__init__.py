"""FeedFlipNets public API."""

from .core import activations  # noqa: F401
from .core import quant  # Backwards compatible alias
from .core import strategies  # Backwards compatible alias
from .core import types  # noqa: F401
from .train import sweep_and_log, train_single
from .training.pipelines import load_preset, presets, run_pipeline
from .training.trainer import Trainer
from .utils import make_dataset

__all__ = [
    "Trainer",
    "activations",
    "strategies",
    "quant",
    "feedback",
    "quantization",
    "types",
    "load_preset",
    "presets",
    "run_pipeline",
    "train_single",
    "sweep_and_log",
    "make_dataset",
]
