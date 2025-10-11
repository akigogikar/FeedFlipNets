"""Utility helpers for dataset loading."""

from __future__ import annotations

import os
import urllib.request
from typing import Iterable, Tuple

import numpy as np


def download_file(url: str, dest: str) -> str:
    """Download a file if it does not exist."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def normalize(data: np.ndarray, axis=None) -> np.ndarray:
    """Return zero mean / unit variance normalisation."""
    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True) + 1e-8
    return (data - mean) / std


def batch_iter(
    data: np.ndarray, labels: np.ndarray, batch_size: int
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Simple batch iterator."""
    n = data.shape[0]
    for i in range(0, n, batch_size):
        yield data[i : i + batch_size], labels[i : i + batch_size]
