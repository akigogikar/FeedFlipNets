"""Legacy utility shims for backwards compatibility."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np

from .core.quant import quantize_ternary_det, quantize_ternary_stoch, ternary
from .data import registry
from .data.loaders import mnist as _mnist  # noqa: F401
from .data.loaders import synth_fixture as _synth_fixture  # noqa: F401
from .data.loaders import synthetic as _synthetic  # noqa: F401
from .data.loaders import tinystories as _tinystories  # noqa: F401
from .data.loaders import ucr_uea as _ucr  # noqa: F401


def make_dataset(
    freq: int,
    n: int = 200,
    seed: int = 42,
    dataset: str | None = None,
    max_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compat shim that mirrors the historic ``make_dataset`` behaviour."""

    if dataset is None or dataset == "synthetic":
        rng = np.random.default_rng(seed)
        n_points = min(n, max_points) if max_points is not None else n
        x = np.linspace(-1, 1, n_points, dtype=np.float32).reshape(1, -1)
        y_true = np.sin(freq * np.pi * x)
        return x, y_true + 0.05 * rng.standard_normal(size=y_true.shape)

    if dataset == "mnist":
        spec = registry.get("mnist", subset="train", max_items=max_points or 1, one_hot=False)
        batch = next(spec.loader("train", 1))
        x = batch.inputs[0:1]
        if max_points is not None:
            x = x[:, :max_points]
        label = float(batch.targets[0, 0])
        y = np.full((1, x.shape[1] if max_points else 1), label, dtype=np.float32)
        return x, y

    if dataset and dataset.startswith("ucr:"):
        name = dataset.split(":", 1)[1]
        spec = registry.get("ucr_uea", name=name)
        batch = next(spec.loader("train", 1))
        x = batch.inputs[0:1]
        if max_points is not None:
            x = x[:, :max_points]
        target = float(np.argmax(batch.targets[0]))
        y = np.full((1, x.shape[1]), target, dtype=np.float32)
        return x, y

    if dataset == "tinystories":
        spec = registry.get("tinystories")
        batch = next(spec.loader("train", 1))
        x = batch.inputs[0:1]
        if max_points is not None:
            x = x[:, :max_points]
        y = batch.targets[0:1].T
        if max_points is not None:
            y = np.repeat(y[:, :1], x.shape[1], axis=1)
        return x, y.astype(np.float32)

    raise ValueError(f"Unsupported dataset: {dataset}")


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def quantize_stoch(W: np.ndarray, thr: float) -> np.ndarray:
    warnings.warn("Use feedflipnets.core.quant.quantize_ternary_stoch instead", DeprecationWarning)
    rng = np.random.default_rng()
    return quantize_ternary_stoch(W, thr, rng)


def quantize_fixed(W: np.ndarray, thr: float = 0.0) -> np.ndarray:
    warnings.warn("Use feedflipnets.core.quant.quantize_ternary_det instead", DeprecationWarning)
    return quantize_ternary_det(W, thr)


def quantize_sign(W: np.ndarray) -> np.ndarray:
    warnings.warn("Use feedflipnets.core.quant.ternary instead", DeprecationWarning)
    return ternary(W)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
