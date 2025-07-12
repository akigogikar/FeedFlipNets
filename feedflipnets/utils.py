from __future__ import annotations
import os
from typing import Tuple
import numpy as np


def make_dataset(freq: int, n: int = 200, seed: int = 42, dataset: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return a toy sinusoid or a small slice from one of the datasets.

    Parameters
    ----------
    freq : int
        Frequency of the synthetic sinusoid when ``dataset`` is ``None`` or
        ``"synthetic"``.
    n : int, optional
        Number of points for the synthetic dataset, by default 200.
    seed : int, optional
        Random seed for noise generation, by default 42.
    dataset : str | None, optional
        Name of a dataset to load. Supported options are ``"mnist"``,
        ``"tinystories"`` and strings of the form ``"ucr:<name>"``.  If ``None``
        or ``"synthetic"`` the toy sinusoid is returned.
    """

    if dataset is None or dataset == "synthetic":
        rng = np.random.default_rng(seed)
        x = np.linspace(-1, 1, n).reshape(1, -1)
        y_true = np.sin(freq * np.pi * x)
        return x, y_true + 0.1 * rng.standard_normal(size=y_true.shape)

    if dataset == "mnist":
        from datasets import load_mnist

        X_train, y_train, _, _ = load_mnist()
        # use a single digit as a sequence of pixel values to keep the toy
        # network's one-dimensional input interface
        return X_train[0].reshape(1, -1), y_train[0:1].astype(float).reshape(1, -1)

    if dataset.startswith("ucr:"):
        from datasets import load_ucr

        name = dataset.split(":", 1)[1]
        X_train, y_train, _, _ = load_ucr(name)
        # treat the first time-series as a one-dimensional signal
        return X_train[0].reshape(1, -1), y_train[0:1].astype(float).reshape(1, -1)

    if dataset == "tinystories":
        from datasets import load_tinystories

        tokens = load_tinystories()
        x = np.arange(len(tokens)).reshape(1, -1) / len(tokens)
        y = np.array(tokens == tokens).astype(float).reshape(1, -1)
        return x, y

    raise ValueError(f"Unsupported dataset: {dataset}")


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def quantize_stoch(W: np.ndarray, thr: float) -> np.ndarray:
    noise = np.random.uniform(-thr, thr, W.shape)
    Wn = W + noise
    out = np.zeros_like(W, dtype=float)
    out[Wn > thr] = 1.0
    out[Wn < -thr] = -1.0
    return out


def quantize_fixed(W: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """Deterministic ternary quantization with a fixed threshold."""
    out = np.zeros_like(W, dtype=float)
    out[W > thr] = 1.0
    out[W < -thr] = -1.0
    return out


def quantize_sign(W: np.ndarray) -> np.ndarray:
    """Return the sign of ``W`` as {-1, 0, 1} floats."""
    return np.sign(W).astype(float)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
