"""MNIST dataset loader using sklearn."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .utils import normalize, download_file


_DEF_PATH = os.path.join("datasets_cache", "mnist.npz")


def load_mnist(path: str = _DEF_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (if needed) and return MNIST as numpy arrays."""
    if os.path.exists(path):
        with np.load(path) as f:
            return f["X_train"], f["y_train"], f["X_test"], f["y_test"]
    from sklearn.datasets import fetch_openml  # local import to avoid hard dependency

    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.to_numpy().astype(np.float32) / 255.0
    y = mnist.target.astype(int).to_numpy()
    X = normalize(X, axis=0)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test
