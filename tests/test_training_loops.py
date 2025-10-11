from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

import numpy as np
import pytest

from feedflipnets.core.strategies import Backprop
from feedflipnets.core.types import Batch
from feedflipnets.training.losses import REGISTRY as LOSS_REGISTRY
from feedflipnets.training.trainer import FeedForwardModel, SGDOptimizer, Trainer


class _LoopLoader:
    """Simple iterable that loops over a fixed batch sequence."""

    def __init__(self, batches: List[Batch]):
        self._batches = batches

    def __iter__(self) -> Iterable[Batch]:
        while True:
            for batch in self._batches:
                yield batch

    def __len__(self) -> int:
        return len(self._batches)


@dataclass
class _Capture:
    history: list[tuple[int, Mapping[str, float]]]

    def __init__(self) -> None:
        self.history = []

    def on_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        self.history.append((epoch, {k: float(v) for k, v in metrics.items()}))


def _make_batches(inputs: np.ndarray, targets: np.ndarray, batch_size: int) -> List[Batch]:
    batches: List[Batch] = []
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        batches.append(Batch(inputs=inputs[start:end], targets=targets[start:end]))
    return batches


def test_regression_training_improves_r2(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(-1, 1)
    y = np.sin(np.pi * x).astype(np.float32) + 0.05 * rng.standard_normal(x.shape, dtype=np.float32)
    batches = _make_batches(x, y, batch_size=16)
    loader = _LoopLoader(batches)

    model = FeedForwardModel(layer_dims=[1, 32, 1], tau=0.05, quant="det", seed=0)
    optimizer = SGDOptimizer(lr=0.05)
    trainer = Trainer(model=model, strategy=Backprop(), optimizer=optimizer)
    capture = _Capture()

    trainer.run(
        loader,
        epochs=30,
        seed=0,
        steps_per_epoch=len(loader),
        task_type="regression",
        num_classes=None,
        loss="auto",
        metric_names=["r2", "mae"],
        split_loggers={"train": [capture]},
        flip="off",
        checkpoint_dir=tmp_path,
    )

    assert capture.history, "expected metrics to be recorded"
    first = capture.history[0][1]
    last = capture.history[-1][1]
    assert last["r2"] > 0.3
    assert last["loss"] < first["loss"]


def test_multiclass_accuracy_with_auto_loss() -> None:
    rng = np.random.default_rng(1)
    centers = np.array([[2, 2], [-2, -2], [2, -2]], dtype=np.float32)
    samples_per_class = 60
    inputs = []
    labels = []
    for idx, center in enumerate(centers):
        noise = 0.4 * rng.standard_normal((samples_per_class, 2), dtype=np.float32)
        inputs.append(center + noise)
        labels.append(np.full((samples_per_class,), idx, dtype=np.int32))
    X = np.vstack(inputs).astype(np.float32)
    y_idx = np.concatenate(labels)
    eye = np.eye(3, dtype=np.float32)
    y = eye[y_idx]

    batches = _make_batches(X, y, batch_size=30)
    loader = _LoopLoader(batches)
    model = FeedForwardModel(layer_dims=[2, 32, 3], tau=0.05, quant="det", seed=1)
    optimizer = SGDOptimizer(lr=0.1)
    trainer = Trainer(model=model, strategy=Backprop(), optimizer=optimizer)
    capture = _Capture()

    trainer.run(
        loader,
        epochs=40,
        seed=1,
        steps_per_epoch=len(loader),
        task_type="multiclass",
        num_classes=3,
        loss="auto",
        metric_names="default",
        split_loggers={"train": [capture]},
        flip="off",
    )

    last = capture.history[-1][1]
    assert last["accuracy"] > 0.7
    assert np.isfinite(last["loss"])


@pytest.mark.parametrize("flip_schedule", ["off", "per_step", "per_epoch"])
def test_binary_metrics_and_checkpoints(tmp_path, flip_schedule: str) -> None:
    rng = np.random.default_rng(2)
    n = 120
    X = rng.standard_normal((n, 2), dtype=np.float32)
    true_w = np.array([[1.5], [-2.0]], dtype=np.float32)
    logits = X @ true_w
    y = (logits > 0).astype(np.float32)

    batches = _make_batches(X, y, batch_size=24)
    loader = _LoopLoader(batches)

    model = FeedForwardModel(layer_dims=[2, 16, 1], tau=0.05, quant="det", seed=2)
    optimizer = SGDOptimizer(lr=0.05)
    trainer = Trainer(model=model, strategy=Backprop(), optimizer=optimizer)
    capture = _Capture()

    ckpt_dir = tmp_path / flip_schedule
    ckpt_dir.mkdir()
    run_kwargs = (
        {"flip": "off"}
        if flip_schedule == "off"
        else {"flip": "ternary", "flip_schedule": flip_schedule}
    )
    trainer.run(
        loader,
        epochs=25,
        seed=2,
        steps_per_epoch=len(loader),
        task_type="binary",
        num_classes=None,
        loss="auto",
        metric_names="default",
        split_loggers={"train": [capture]},
        **run_kwargs,
        checkpoint_dir=ckpt_dir,
    )

    last = capture.history[-1][1]
    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert metric in last
        assert np.isfinite(last[metric])
    assert (ckpt_dir / "best.ckpt").exists()
    assert (ckpt_dir / "last.ckpt").exists()


def test_auto_loss_registry() -> None:
    assert LOSS_REGISTRY.resolve("auto", task_type="regression").name == "mse"
    assert LOSS_REGISTRY.resolve("auto", task_type="multiclass").name == "ce"
    assert LOSS_REGISTRY.resolve("auto", task_type="binary").name == "bce"
