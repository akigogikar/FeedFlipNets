"""FeedFlipNets public API."""

from .core import activations, quant, strategies, types  # noqa: F401
from .core import quant as quantization  # Backwards compatible alias
from .core import strategies as feedback  # Backwards compatible alias
from .training.pipelines import load_preset, presets, run_pipeline
from .training.trainer import Trainer
from .train import sweep_and_log, train_single
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
