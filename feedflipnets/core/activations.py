"""Activation utilities for FeedFlipNets."""

from __future__ import annotations

import numpy as np

from .types import Array


def relu(x: Array) -> Array:
    """Return the ReLU activation."""

    return np.maximum(x, 0.0)


def hadamard_pre(x: Array) -> Array:
    """Pad ``x`` to the next power of two for Hadamard transforms."""

    if x.ndim == 0:
        x = x.reshape(1)
    size = x.shape[-1]
    if size == 0:
        return x
    next_pow = 1 << (size - 1).bit_length()
    if next_pow == size:
        return x
    pad_width = next_pow - size
    pad_shape = [(0, 0)] * x.ndim
    pad_shape[-1] = (0, pad_width)
    return np.pad(x, pad_shape, mode="constant") / np.sqrt(next_pow / size)
