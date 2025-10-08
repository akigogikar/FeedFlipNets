"""Deterministic training loops for FeedFlipNets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableSequence, Sequence

import numpy as np

from ..core.activations import relu
from ..core.quant import quantize_ternary_det, quantize_ternary_stoch
from ..core.strategies import FeedbackStrategy
from ..core.types import (
    ActivationState,
    Array,
    Batch,
    Gradients,
    ModelDescription,
    RunResult,
    StrategyState,
)


def _relu_deriv(z: Array) -> Array:
    return (z > 0).astype(np.float32)


@dataclass
class FeedForwardModel:
    """Lightweight feed-forward network with ternary quantisation."""

    layer_dims: Sequence[int]
    tau: float
    quant: str = "det"
    seed: int = 0
    weights: MutableSequence[Array] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def describe(self) -> ModelDescription:
        return ModelDescription(layer_dims=list(self.layer_dims))

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._quant_rng = np.random.default_rng(seed + 1)
        weights: list[Array] = []
        dims = list(self.layer_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            W = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.05
            weights.append(W)
        self.weights = weights

    def forward(self, inputs: Array) -> tuple[Array, ActivationState]:
        layer_inputs: list[Array] = [inputs]
        layer_derivs: list[Array] = []
        x = inputs
        for idx, W in enumerate(self.weights):
            z = x @ W
            if idx < len(self.weights) - 1:
                layer_derivs.append(_relu_deriv(z))
                x = relu(z)
                layer_inputs.append(x)
            else:
                x = z
        activations = ActivationState(
            layer_inputs=layer_inputs,
            layer_derivs=layer_derivs,
            weights=[w.copy() for w in self.weights],
        )
        return x, activations

    def apply_gradients(self, grads: Gradients) -> None:
        for idx, W in enumerate(self.weights):
            grad = grads.get(f"W{idx}")
            if grad is None:
                continue
            self.weights[idx] = W + grad

    def quantise(self) -> None:
        for idx, W in enumerate(self.weights):
            if self.quant == "det":
                self.weights[idx] = quantize_ternary_det(W, self.tau)
            elif self.quant == "stoch":
                self.weights[idx] = quantize_ternary_stoch(W, self.tau, self._quant_rng)
            else:
                raise ValueError(f"Unknown quantisation mode: {self.quant}")


@dataclass
class SGDOptimizer:
    """Vanilla SGD with optional momentum (unused but extendable)."""

    lr: float

    def step(self, model: FeedForwardModel, grads: Gradients) -> None:
        scaled = {name: -self.lr * grad for name, grad in grads.items()}
        model.apply_gradients(scaled)


class Trainer:
    """Run deterministic training loops with pluggable feedback strategies."""

    def __init__(
        self,
        model: FeedForwardModel,
        strategy: FeedbackStrategy,
        optimizer: SGDOptimizer,
        callbacks: Sequence[object] | None = None,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer
        self.callbacks = list(callbacks or [])
        self._state: StrategyState | None = None

    def run(
        self,
        dataloader: Iterable[Batch],
        epochs: int,
        seed: int,
        device: str = "cpu",
        *,
        determinism: bool = True,
        steps_per_epoch: int | None = None,
    ) -> RunResult:
        if device != "cpu":  # pragma: no cover - guardrail
            raise ValueError("Only CPU execution is supported in the reference trainer")

        self._set_seed(seed, determinism)
        self.model.reset(seed)
        state = self.strategy.init(self.model.describe())
        iterator = iter(dataloader)
        total_steps = 0
        last_metrics: Mapping[str, float] = {}

        for epoch in range(epochs):
            steps_this_epoch = steps_per_epoch or self._infer_steps(dataloader)
            for _ in range(steps_this_epoch):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dataloader)
                    batch = next(iterator)
                predictions, activations = self.model.forward(batch.inputs)
                error = predictions - batch.targets
                loss = float(np.mean(np.square(error)))
                grads, state = self.strategy.backward(activations, error, state)
                self.optimizer.step(self.model, grads)
                self.model.quantise()

                metrics = {"loss": loss}
                self._emit_step(total_steps, metrics)
                last_metrics = metrics
                total_steps += 1

            state.metadata["pending_refresh"] = True
            self._emit_epoch(epoch, last_metrics)

        self._state = state
        return RunResult(
            steps=total_steps, metrics_path="", manifest_path="", summary_path=""
        )

    # ------------------------------------------------------------------
    # Internal helpers

    def _emit_step(self, step: int, metrics: Mapping[str, float]) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_step"):
                callback.on_step(step, metrics)  # type: ignore[attr-defined]
            elif callable(callback):
                callback(step, metrics)

    def _emit_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch"):
                callback.on_epoch(epoch, metrics)  # type: ignore[attr-defined]

    @staticmethod
    def _set_seed(seed: int, determinism: bool) -> None:
        if determinism:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _infer_steps(dataloader: Iterable[Batch]) -> int:
        if hasattr(dataloader, "__len__"):
            return len(dataloader)  # type: ignore[arg-type]
        return 1
