"""Core typing contracts for FeedFlipNets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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
    summary_path: str = ""


@dataclass
class ActivationState:
    """Intermediate activations captured during the forward pass."""

    layer_inputs: List[Array]
    layer_derivs: List[Array]
    weights: List[Array]


Gradients = Dict[str, Array]


@dataclass(frozen=True)
class ModelDescription:
    """Description of the feed-forward network architecture."""

    layer_dims: List[int]


@dataclass
class StrategyState:
    """State persisted by a feedback strategy between iterations."""

    feedback: List[Array] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
