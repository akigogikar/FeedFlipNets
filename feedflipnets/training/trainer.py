"""Training loops for FeedFlipNets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from ..core.activations import relu
from ..core.quantization import quantize_ternary_det, quantize_ternary_stoch
from ..core.types import Array, Batch, FeedbackStrategy, RunResult

Callback = Callable[[int, Mapping[str, float]], None]


def _relu_deriv(z: Array) -> Array:
    return (z > 0).astype(np.float32)


@dataclass
class Trainer:
    """Orchestrates optimisation using feedback alignment strategies."""

    @staticmethod
    def run(
        config: Mapping[str, object],
        data_iter: Iterator[Batch],
        callbacks: Sequence[Callback] | None = None,
    ) -> RunResult:
        callbacks = list(callbacks or [])
        model_cfg = config.get("model", {})  # type: ignore[assignment]
        train_cfg = config.get("train", {})  # type: ignore[assignment]
        strategy: FeedbackStrategy = config["strategy"]  # type: ignore[index]

        d_in = int(model_cfg.get("d_in", 1))
        d_out = int(model_cfg.get("d_out", 1))
        hidden = list(model_cfg.get("hidden", [32]))  # type: ignore[call-arg]
        dims = [d_in, *hidden, d_out]

        seed = int(train_cfg.get("seed", 0))
        steps = int(train_cfg.get("steps", 0))
        lr = float(train_cfg.get("lr", 0.01))
        tau = float(model_cfg.get("tau", 0.05))
        quant_kind = model_cfg.get("quant", "det")
        eval_interval = int(train_cfg.get("eval_interval", steps or 1))
        if eval_interval <= 0:
            eval_interval = 1

        metrics_path = config.get("metrics_path", "metrics.jsonl")  # type: ignore[assignment]
        manifest_path = config.get("manifest_path", "manifest.json")  # type: ignore[assignment]

        rng = np.random.default_rng(seed)
        quant_rng = np.random.default_rng(seed + 1)

        weights: list[Array] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            W = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.05
            weights.append(W)

        # Prepare data iterator
        iterator = data_iter

        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(data_iter)
                batch = next(iterator)

            activations: Dict[str, Array | list[Array]] = {}
            layer_inputs: list[Array] = []
            layer_derivs: list[Array] = []
            x = batch.inputs
            layer_inputs.append(x)
            for idx, W in enumerate(weights):
                z = x @ W
                if idx < len(weights) - 1:
                    layer_derivs.append(_relu_deriv(z))
                    x = relu(z)
                else:
                    x = z
                if idx < len(weights) - 1:
                    layer_inputs.append(x)
            activations["weights"] = [w.copy() for w in weights]
            activations["layer_inputs"] = layer_inputs
            activations["layer_derivs"] = layer_derivs

            predictions = x
            error = predictions - batch.targets
            loss = float(np.mean(error ** 2))

            grads = strategy.compute_updates(activations, error)
            for idx, W in enumerate(weights):
                grad = grads.get(f"W{idx}")
                if grad is None:
                    continue
                W = W - lr * grad
                if quant_kind == "det":
                    W = quantize_ternary_det(W, tau)
                elif quant_kind == "stoch":
                    W = quantize_ternary_stoch(W, tau, quant_rng)
                weights[idx] = W

            metrics = {"loss": loss}
            for callback in callbacks:
                if hasattr(callback, "on_step"):
                    callback.on_step(step, metrics)  # type: ignore[attr-defined]
                else:
                    callback(step, metrics)

            if hasattr(strategy, "on_epoch_end") and (step + 1) % eval_interval == 0:
                strategy.on_epoch_end()  # type: ignore[attr-defined]

        return RunResult(steps=steps, metrics_path=str(metrics_path), manifest_path=str(manifest_path))

