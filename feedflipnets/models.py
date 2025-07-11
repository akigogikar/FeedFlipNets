from __future__ import annotations
from typing import List
import numpy as np
from .utils import tanh, tanh_deriv


def forward_pass(weights: List[np.ndarray], x: np.ndarray) -> List[np.ndarray]:
    activs = [x]
    for W in weights:
        activs.append(tanh(W @ activs[-1]))
    return activs


def backprop_deltas(weights: List[np.ndarray], activs: List[np.ndarray], err: np.ndarray) -> List[np.ndarray]:
    L = len(weights) - 1
    deltas: List[np.ndarray] = [None] * (L + 1)
    deltas[L] = err
    for l in reversed(range(L)):
        deltas[l] = (weights[l + 1].T @ deltas[l + 1]) * tanh_deriv(weights[l] @ activs[l])
    return deltas
