"""Loss registry used by the upgraded training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np

from ..core.types import Array

LossFn = Callable[[Array, Array], tuple[float, Array]]


@dataclass(frozen=True)
class Loss:
    """Loss wrapper returning both the scalar loss and dL/dy."""

    name: str
    fn: LossFn

    def __call__(self, predictions: Array, targets: Array) -> tuple[float, Array]:
        return self.fn(predictions, targets)


class LossRegistry:
    """Central registry for loss functions."""

    def __init__(self) -> None:
        self._registry: Dict[str, Loss] = {}

    def register(self, name: str, fn: LossFn) -> None:
        self._registry[name] = Loss(name, fn)

    def get(self, name: str) -> Loss:
        try:
            return self._registry[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown loss: {name}") from exc

    def names(self) -> Iterable[str]:
        return sorted(self._registry)

    def resolve(self, name: str, *, task_type: str) -> Loss:
        if name == "auto":
            if task_type == "regression":
                name = "mse"
            elif task_type == "multiclass":
                name = "ce"
            elif task_type in {"binary", "multilabel"}:
                name = "bce"
            else:  # pragma: no cover - safeguard
                raise ValueError(f"Unknown task type: {task_type}")
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Unknown loss {name!r}. Available losses: {available}")
        return self._registry[name]


REGISTRY = LossRegistry()


def _mse(pred: Array, target: Array) -> tuple[float, Array]:
    diff = pred - target
    loss = float(np.mean(np.square(diff)))
    return loss, diff


def _mae(pred: Array, target: Array) -> tuple[float, Array]:
    diff = pred - target
    loss = float(np.mean(np.abs(diff)))
    grad = np.sign(diff)
    return loss, grad


def _huber(pred: Array, target: Array, delta: float = 1.0) -> tuple[float, Array]:
    diff = pred - target
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    loss = float(np.mean(0.5 * quadratic**2 + delta * linear))
    grad = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
    return loss, grad


def _softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _ensure_one_hot(target: Array, num_classes: int) -> Array:
    if target.ndim == 2 and target.shape[1] == num_classes:
        return target.astype(np.float32)
    indices = target.reshape(-1).astype(int)
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[indices]


def _cross_entropy(logits: Array, target: Array) -> tuple[float, Array]:
    num_classes = logits.shape[1]
    one_hot = _ensure_one_hot(target, num_classes)
    probs = _softmax(logits)
    eps = 1e-9
    loss = float(-np.mean(np.sum(one_hot * np.log(probs + eps), axis=1)))
    grad = probs - one_hot
    return loss, grad


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def _bce_with_logits(logits: Array, target: Array) -> tuple[float, Array]:
    target = target.astype(np.float32)
    probs = _sigmoid(logits)
    eps = 1e-9
    loss = float(-np.mean(target * np.log(probs + eps) + (1 - target) * np.log(1 - probs + eps)))
    grad = probs - target
    return loss, grad


REGISTRY.register("mse", _mse)
REGISTRY.register("mae", _mae)
REGISTRY.register("huber", _huber)
REGISTRY.register("ce", _cross_entropy)
REGISTRY.register("bce", _bce_with_logits)
# Alias for parity with research code naming
REGISTRY.register("bcewithlogits", _bce_with_logits)

__all__ = ["Loss", "LossRegistry", "REGISTRY"]
