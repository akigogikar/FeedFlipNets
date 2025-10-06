from __future__ import annotations
from typing import List
import numpy as np
from .utils import tanh, tanh_deriv


def forward_pass(weights: List[np.ndarray], x: np.ndarray) -> List[np.ndarray]:
    activs = [x]
    for W in weights:
        activs.append(tanh(W @ activs[-1]))
    return activs


def backprop_deltas(
    weights: List[np.ndarray], activs: List[np.ndarray], err: np.ndarray
) -> List[np.ndarray]:
    last_layer = len(weights) - 1
    deltas: List[np.ndarray] = [None] * (last_layer + 1)
    deltas[last_layer] = err
    for layer_idx in reversed(range(last_layer)):
        deltas[layer_idx] = (
            weights[layer_idx + 1].T @ deltas[layer_idx + 1]
        ) * tanh_deriv(weights[layer_idx] @ activs[layer_idx])
    return deltas
