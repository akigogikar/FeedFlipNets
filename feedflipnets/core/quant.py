"""Ternary quantisation helpers."""

from __future__ import annotations

import numpy as np

from .types import Array


def ternary(x: Array) -> Array:
    """Return the sign of ``x`` as ternary floats."""

    out = np.zeros_like(x, dtype=float)
    out[x > 0.0] = 1.0
    out[x < 0.0] = -1.0
    return out


def quantize_ternary_det(weights: Array, tau: float) -> Array:
    """Deterministic ternary quantisation with threshold ``tau``."""

    out = np.zeros_like(weights, dtype=float)
    out[weights > tau] = 1.0
    out[weights < -tau] = -1.0
    return out


def quantize_ternary_stoch(
    weights: Array, tau: float, rng: np.random.Generator
) -> Array:
    """Stochastic ternary quantisation matching the legacy behaviour."""

    noise = rng.uniform(-tau, tau, size=weights.shape)
    jittered = weights + noise
    out = np.zeros_like(weights, dtype=float)
    out[jittered > tau] = 1.0
    out[jittered < -tau] = -1.0
    return out


def pack_ternary(weights: Array) -> np.ndarray:
    """Pack ternary weights into ``int8`` vectors for logging or storage."""

    return ternary(weights).astype(np.int8)
