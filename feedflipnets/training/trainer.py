"""Deterministic, modality-aware training loops for FeedFlipNets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
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
from .losses import REGISTRY as LOSS_REGISTRY
from .metrics import compute_metrics, default_metrics


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
            else:  # pragma: no cover - guardrail
                raise ValueError(f"Unknown quantisation mode: {self.quant}")

    def state_dict(self) -> Mapping[str, Array]:
        return {f"W{idx}": W.copy() for idx, W in enumerate(self.weights)}

    def load_state_dict(self, state: Mapping[str, Array]) -> None:
        for idx in range(len(self.weights)):
            key = f"W{idx}"
            if key not in state:
                raise KeyError(f"Missing weight {key} in state dict")
            self.weights[idx] = state[key].copy()

    def parameter_count(self) -> int:
        return int(sum(int(w.size) for w in self.weights))


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
        dataloader: Iterable[Batch] | Mapping[str, Iterable[Batch]],
        epochs: int,
        seed: int,
        device: str = "cpu",
        *,
        determinism: bool = True,
        steps_per_epoch: int | None = None,
        val_loader: Iterable[Batch] | None = None,
        test_loader: Iterable[Batch] | None = None,
        val_steps: int | None = None,
        test_steps: int | None = None,
        task_type: str = "regression",
        num_classes: int | None = None,
        loss: str = "auto",
        metric_names: Sequence[str] | str = (),
        eval_every: int = 1,
        split_loggers: Mapping[str, Sequence[object]] | None = None,
        ternary_mode: str = "per_step",
        early_stopping_patience: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> RunResult:
        if device != "cpu":  # pragma: no cover - guardrail
            raise ValueError("Only CPU execution is supported in the reference trainer")

        if isinstance(dataloader, Mapping):
            train_loader = dataloader.get("train")
            if train_loader is None:
                raise ValueError("train loader missing from dataloader mapping")
            val_loader = val_loader or dataloader.get("val")
            test_loader = test_loader or dataloader.get("test")
        else:
            train_loader = dataloader

        if isinstance(metric_names, str):
            if metric_names == "default" or metric_names.strip() == "":
                metric_names = default_metrics(task_type, num_classes=num_classes)
            else:
                metric_names = [m.strip() for m in metric_names.split(",") if m.strip()]
        if not metric_names:
            metric_names = default_metrics(task_type, num_classes=num_classes)

        loss_fn = LOSS_REGISTRY.resolve(loss, task_type=task_type)
        self._set_seed(seed, determinism)
        self.model.reset(seed)
        state = self.strategy.init(self.model.describe())
        train_steps = steps_per_epoch or self._infer_steps(train_loader)
        ternary_mode = ternary_mode or "per_step"
        if ternary_mode not in {"off", "per_step", "per_epoch"}:
            raise ValueError(
                "ternary_mode must be one of {'off','per_step','per_epoch'}"
            )

        best_loss = float("inf")
        best_state: Mapping[str, Array] | None = None
        epochs_no_improve = 0
        total_steps = 0
        split_loggers = split_loggers or {}
        checkpoint_dir = Path(checkpoint_dir or ".")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_metrics, state = self._run_phase(
                train_loader,
                train_steps,
                state,
                loss_fn,
                metric_names,
                task_type,
                num_classes,
                training=True,
                ternary_mode=ternary_mode,
            )
            total_steps += train_steps
            self._emit_epoch("train", epoch, train_metrics, split_loggers)

            if ternary_mode == "per_epoch":
                self.model.quantise()

            should_eval = epoch % max(1, eval_every) == 0
            val_metrics = None
            if should_eval and val_loader is not None and (val_steps or 0) > 0:
                val_metrics, _ = self._run_phase(
                    val_loader,
                    val_steps or self._infer_steps(val_loader),
                    state,
                    loss_fn,
                    metric_names,
                    task_type,
                    num_classes,
                    training=False,
                    ternary_mode="off",
                )
                self._emit_epoch("val", epoch, val_metrics, split_loggers)

            if should_eval and test_loader is not None and (test_steps or 0) > 0:
                test_metrics, _ = self._run_phase(
                    test_loader,
                    test_steps or self._infer_steps(test_loader),
                    state,
                    loss_fn,
                    metric_names,
                    task_type,
                    num_classes,
                    training=False,
                    ternary_mode="off",
                )
                self._emit_epoch("test", epoch, test_metrics, split_loggers)

            target_metrics = val_metrics or train_metrics
            current_loss = float(target_metrics.get("loss", 0.0))
            if current_loss < best_loss - 1e-9:
                best_loss = current_loss
                epochs_no_improve = 0
                best_state = self.model.state_dict()
                self._save_checkpoint(checkpoint_dir / "best.ckpt", best_state)
            else:
                epochs_no_improve += 1
                if (
                    early_stopping_patience
                    and epochs_no_improve >= early_stopping_patience
                ):
                    break

        last_state = self.model.state_dict()
        self._save_checkpoint(checkpoint_dir / "last.ckpt", last_state)
        self._state = state
        return RunResult(
            steps=total_steps,
            metrics_path="",
            manifest_path="",
            summary_path="",
        )

    # ------------------------------------------------------------------
    # Internal helpers

    def _run_phase(
        self,
        loader: Iterable[Batch],
        steps: int,
        state: StrategyState,
        loss_fn,
        metric_names: Sequence[str],
        task_type: str,
        num_classes: int | None,
        *,
        training: bool,
        ternary_mode: str,
    ) -> tuple[Mapping[str, float], StrategyState]:
        iterator = iter(loader)
        losses: list[float] = []
        preds_all: list[Array] = []
        targets_all: list[Array] = []
        current_state = state
        for _ in range(max(1, steps)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            predictions, activations = self.model.forward(batch.inputs)
            loss_value, delta = loss_fn(predictions, batch.targets)
            losses.append(loss_value)
            preds_all.append(predictions.copy())
            targets_all.append(batch.targets.copy())
            if training:
                grads, current_state = self.strategy.backward(
                    activations, delta, current_state
                )
                self.optimizer.step(self.model, grads)
                if ternary_mode == "per_step":
                    self.model.quantise()
        metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
        if preds_all:
            predictions = np.concatenate(preds_all, axis=0)
            targets = np.concatenate(targets_all, axis=0)
            metrics.update(
                compute_metrics(
                    metric_names,
                    predictions,
                    targets,
                    task_type=task_type,
                    num_classes=num_classes,
                )
            )
        return metrics, current_state

    def _emit_epoch(
        self,
        split: str,
        epoch: int,
        metrics: Mapping[str, float],
        loggers: Mapping[str, Sequence[object]],
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch"):
                callback.on_epoch(epoch, metrics)  # type: ignore[attr-defined]
        for callback in loggers.get(split, []):
            if hasattr(callback, "on_epoch"):
                callback.on_epoch(epoch, metrics)  # type: ignore[attr-defined]
            elif callable(callback):
                callback(epoch, metrics)

    @staticmethod
    def _set_seed(seed: int, determinism: bool) -> None:
        if determinism:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _infer_steps(dataloader: Iterable[Batch]) -> int:
        if hasattr(dataloader, "__len__"):
            try:
                return len(dataloader)  # type: ignore[arg-type]
            except TypeError:  # pragma: no cover - defensive
                pass
        return 1

    @staticmethod
    def _save_checkpoint(path: Path, state: Mapping[str, Array]) -> None:
        payload = {name: value for name, value in state.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **payload)


__all__ = ["FeedForwardModel", "SGDOptimizer", "Trainer"]
