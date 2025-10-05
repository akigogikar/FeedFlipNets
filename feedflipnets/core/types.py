"""Core typing contracts for FeedFlipNets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Batch:
    """A single mini-batch of data."""

    inputs: Array
    targets: Array


@dataclass(frozen=True)
class RunResult:
    """Summary returned by :func:`feedflipnets.training.trainer.Trainer.run`."""

    steps: int
    metrics_path: str
    manifest_path: str


class FeedbackStrategy(Protocol):
    """Protocol implemented by feedback alignment strategies."""

    def compute_updates(
        self,
        activations: Dict[str, Array],
        error: Array,
    ) -> Dict[str, Array]:
        """Return parameter updates indexed by parameter name."""

