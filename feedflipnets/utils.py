from __future__ import annotations
import os
from typing import Tuple
import numpy as np


def make_dataset(freq: int, n: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n).reshape(1, -1)
    y_true = np.sin(freq * np.pi * x)
    return x, y_true + 0.1 * rng.standard_normal(size=y_true.shape)


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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
