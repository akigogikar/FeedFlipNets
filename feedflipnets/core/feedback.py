"""Feedback alignment strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .types import Array, FeedbackStrategy


@dataclass
class DFA(FeedbackStrategy):
    """Direct Feedback Alignment with fixed random matrices."""

    rng: np.random.Generator
    layer_dims: List[int]

    def __post_init__(self) -> None:
        output_dim = self.layer_dims[-1]
        self._feedback: List[Array] = []
        for hidden_dim in self.layer_dims[:-1]:
            scale = 1.0 / np.sqrt(output_dim)
            self._feedback.append(
                self.rng.standard_normal((output_dim, hidden_dim)) * scale
            )

    def compute_updates(self, activations: Dict[str, Array], error: Array) -> Dict[str, Array]:
        grads: Dict[str, Array] = {}
        batch = error.shape[0]
        delta = error
        weights = activations["weights"]
        layer_inputs: List[Array] = activations["layer_inputs"]
        layer_derivs: List[Array] = activations["layer_derivs"]

        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            feedback = self._feedback[idx]
            delta = (delta @ feedback) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads


@dataclass
class FlipFeedback(FeedbackStrategy):
    """Use sign-flipped forward weights for the feedback signal."""

    refresh: str = "per_step"

    def compute_updates(self, activations: Dict[str, Array], error: Array) -> Dict[str, Array]:
        grads: Dict[str, Array] = {}
        weights: List[Array] = activations["weights"]
        layer_inputs: List[Array] = activations["layer_inputs"]
        layer_derivs: List[Array] = activations["layer_derivs"]
        batch = error.shape[0]

        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            forward = weights[idx + 1]
            feedback = -np.sign(forward)
            delta = (delta @ feedback.T) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads


@dataclass
class BackpropLite(FeedbackStrategy):
    """A lightweight backprop variant used for smoke comparisons."""

    def compute_updates(self, activations: Dict[str, Array], error: Array) -> Dict[str, Array]:
        grads: Dict[str, Array] = {}
        weights: List[Array] = activations["weights"]
        layer_inputs: List[Array] = activations["layer_inputs"]
        layer_derivs: List[Array] = activations["layer_derivs"]
        batch = error.shape[0]

        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            forward = weights[idx + 1]
            delta = (delta @ forward.T) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads


class StructuredFeedback(FeedbackStrategy):
    """Feedback alignment using structured random matrices."""

    def __init__(
        self,
        rng: np.random.Generator,
        dim_in: int,
        dim_out: int,
        *,
        structure_type: str = "orthogonal",
        refresh: str = "fixed",
        rank: int | None = None,
        blocks: int | None = None,
        hidden_dims: Sequence[int] | None = None,
        input_dim: int | None = None,
    ) -> None:
        self.rng = rng
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.structure_type = structure_type
        self.refresh = refresh
        self.rank = rank
        self.blocks = blocks
        self.hidden_dims = list(hidden_dims or [])
        self.input_dim = int(input_dim) if input_dim is not None else self.dim_in

        self.B: Array = self._make_matrix(self.dim_out, self.dim_in)
        self._stack: List[Array] = []
        self._signature: List[Tuple[int, int]] = []
        self._pending_refresh = True

    def _make_matrix(self, out_dim: int, in_dim: int) -> Array:
        if self.structure_type == "orthogonal":
            return _orthogonal(self.rng, out_dim, in_dim)
        if self.structure_type == "hadamard":
            return _hadamard_matrix(self.rng, out_dim, in_dim)
        if self.structure_type == "blockdiag":
            return _blockdiag_orthogonal(
                self.rng,
                in_dim,
                out_dim,
                self.blocks,
            )
        if self.structure_type == "lowrank":
            return _lowrank(self.rng, out_dim, in_dim, self.rank)
        raise ValueError(f"Unknown structure_type: {self.structure_type}")

    def _refresh_required(self) -> bool:
        return self._pending_refresh or self.refresh == "per_step"

    def _ensure_stack(self, weights: Sequence[Array]) -> None:
        dims: List[Tuple[int, int]] = []
        for idx in range(len(weights) - 1):
            next_dim = int(weights[idx + 1].shape[1])
            curr_dim = int(weights[idx].shape[1])
            dims.append((next_dim, curr_dim))

        if not self._refresh_required() and dims == self._signature:
            return

        self._stack = []
        for idx, (out_dim, in_dim) in enumerate(dims):
            matrix = self._make_matrix(out_dim, in_dim)
            self._stack.append(matrix)
            if idx == 0:
                self.B = matrix

        if not dims:
            # No hidden layers; still refresh base matrix if requested.
            if self._refresh_required() or not self._signature:
                self.B = self._make_matrix(self.dim_out, self.dim_in)

        self._signature = dims
        self._pending_refresh = False

    def on_epoch_end(self) -> None:
        if self.refresh == "per_epoch":
            self._pending_refresh = True

    def compute_updates(self, activations: Dict[str, Array], error: Array) -> Dict[str, Array]:
        weights: Sequence[Array] = activations["weights"]  # type: ignore[index]
        layer_inputs: Sequence[Array] = activations["layer_inputs"]  # type: ignore[index]
        layer_derivs: Sequence[Array] = activations["layer_derivs"]  # type: ignore[index]

        self._ensure_stack(weights)

        grads: Dict[str, Array] = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            feedback = self._stack[idx]
            delta = (delta @ feedback) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads


def _orthogonal(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    A = rng.standard_normal((out_dim, in_dim))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:out_dim, :in_dim].astype(np.float32)


def _hadamard_matrix(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    size = max(out_dim, in_dim)
    H = _hadamard(size)
    if H.shape[0] < size:
        # Fallback to orthogonal if size constraint violated.
        return _orthogonal(rng, out_dim, in_dim)
    return H[:out_dim, :in_dim].astype(np.float32)


def _lowrank(
    rng: np.random.Generator,
    out_dim: int,
    in_dim: int,
    rank: int | None,
) -> Array:
    r = rank or max(1, min(out_dim, in_dim) // 16)
    U = rng.standard_normal((out_dim, r))
    V = rng.standard_normal((in_dim, r))
    B = U @ V.T
    norms = np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    return (B / norms).astype(np.float32)


def _hadamard(n: int) -> Array:
    m = 1 << (max(1, n) - 1).bit_length()
    H = np.array([[1]], dtype=np.float32)
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    scale = np.sqrt(np.float32(H.shape[0]))
    return H / scale


def _blockdiag_orthogonal(
    rng: np.random.Generator,
    dim_in: int,
    dim_out: int,
    blocks: int | None,
) -> Array:
    if blocks is None:
        if dim_in % 4 == 0 and dim_out % 4 == 0 and min(dim_in, dim_out) >= 4:
            blocks = 4
        else:
            blocks = np.gcd(dim_in, dim_out)
            if blocks <= 0:
                blocks = 1
    rows = np.array_split(np.arange(dim_out), blocks)
    cols = np.array_split(np.arange(dim_in), blocks)
    B = np.zeros((dim_out, dim_in), dtype=np.float32)
    for r_idx, c_idx in zip(rows, cols):
        if len(r_idx) == 0 or len(c_idx) == 0:
            continue
        A = rng.standard_normal((len(r_idx), len(c_idx)))
        Q, _ = np.linalg.qr(A.T)
        blk = Q.T[: len(r_idx), : len(c_idx)].astype(np.float32)
        B[np.ix_(r_idx, c_idx)] = blk
    return B

