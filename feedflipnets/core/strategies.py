"""Feedback strategies for FeedFlipNets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np

from .types import ActivationState, Array, Gradients, ModelDescription, StrategyState


class FeedbackStrategy(Protocol):
    """Protocol implemented by feedback-alignment strategies."""

    def init(self, model: ModelDescription) -> StrategyState:
        """Initialise internal state for ``model``."""

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        """Return parameter gradients and the (possibly updated) state."""


@dataclass
class _SimpleState(StrategyState):
    """State container for simple strategies."""

    feedback: List[Array] = field(default_factory=list)


@dataclass
class DFA:
    """Direct Feedback Alignment with fixed random matrices."""

    rng: np.random.Generator

    def init(self, model: ModelDescription) -> StrategyState:
        dims = model.layer_dims
        output_dim = dims[-1]
        matrices: List[Array] = []
        for hidden_dim in dims[1:-1]:
            scale = 1.0 / np.sqrt(output_dim)
            matrices.append(
                self.rng.standard_normal((output_dim, hidden_dim)).astype(np.float32)
                * scale
            )
        return _SimpleState(feedback=matrices)

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs
        feedback_mats = list(getattr(state, "feedback", []))

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            feedback = feedback_mats[idx]
            delta = (delta @ feedback) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads, state


@dataclass
class FlipTernary:
    """Use sign-flipped forward weights for the feedback signal."""

    refresh: str = "per_step"

    def init(self, model: ModelDescription) -> StrategyState:  # noqa: D401
        return StrategyState(metadata={"refresh": self.refresh})

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            forward = weights[idx + 1]
            feedback = -np.sign(forward).T
            delta = (delta @ feedback) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch
        return grads, state


@dataclass
class StructuredFeedback:
    """Feedback alignment using structured random matrices."""

    rng: np.random.Generator
    structure_type: str = "orthogonal"
    refresh: str = "fixed"
    rank: int | None = None
    blocks: int | None = None

    def init(self, model: ModelDescription) -> StrategyState:
        feedback = self._build_stack(model.layer_dims)
        return StrategyState(
            metadata={
                "signature": self._signature(model.layer_dims),
                "pending_refresh": False,
                "layer_dims": list(model.layer_dims),
            },
            feedback=feedback,
        )

    def backward(
        self,
        activations: ActivationState,
        error: Array,
        state: StrategyState,
    ) -> tuple[Gradients, StrategyState]:
        feedback = list(state.feedback)
        metadata = dict(state.metadata)
        dims = metadata.get("layer_dims", [])

        if self._needs_refresh(feedback, dims, metadata):
            feedback = self._build_stack(dims)
            metadata["pending_refresh"] = False

        weights = activations.weights
        layer_inputs = activations.layer_inputs
        layer_derivs = activations.layer_derivs

        grads: Gradients = {}
        batch = error.shape[0]
        delta = error
        last_idx = len(weights) - 1
        grads[f"W{last_idx}"] = layer_inputs[last_idx].T @ delta / batch
        for idx in reversed(range(last_idx)):
            matrix = feedback[idx]
            delta = (delta @ matrix) * layer_derivs[idx]
            grads[f"W{idx}"] = layer_inputs[idx].T @ delta / batch

        new_state = StrategyState(feedback=feedback, metadata=metadata)
        return grads, new_state

    # ------------------------------------------------------------------
    # Helpers

    def _needs_refresh(
        self,
        feedback: Sequence[Array],
        dims: Sequence[int],
        metadata: Dict[str, object],
    ) -> bool:
        if not feedback:
            return True
        if self.refresh == "per_step":
            return True
        if self.refresh == "per_epoch":
            return bool(metadata.get("pending_refresh", False))
        return False

    def _signature(self, dims: Sequence[int]) -> List[Tuple[int, int]]:
        output_dim = dims[-1]
        return [(output_dim, dims[idx + 1]) for idx in range(len(dims) - 2)]

    def _build_stack(self, dims: Sequence[int]) -> List[Array]:
        matrices: List[Array] = []
        for out_dim, in_dim in self._signature(dims):
            matrices.append(self._make_matrix(out_dim, in_dim))
        return matrices

    def _make_matrix(self, out_dim: int, in_dim: int) -> Array:
        if self.structure_type == "orthogonal":
            return _orthogonal(self.rng, out_dim, in_dim)
        if self.structure_type == "hadamard":
            return _hadamard_matrix(self.rng, out_dim, in_dim)
        if self.structure_type == "blockdiag":
            return _blockdiag_orthogonal(self.rng, in_dim, out_dim, self.blocks)
        if self.structure_type == "lowrank":
            return _lowrank(self.rng, out_dim, in_dim, self.rank)
        raise ValueError(f"Unknown structure_type: {self.structure_type}")


def _orthogonal(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    A = rng.standard_normal((out_dim, in_dim))
    Q, _ = np.linalg.qr(A.T)
    return Q.T[:out_dim, :in_dim].astype(np.float32)


def _hadamard_matrix(rng: np.random.Generator, out_dim: int, in_dim: int) -> Array:
    size = max(out_dim, in_dim)
    H = _hadamard(size)
    if H.shape[0] < size:
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
    blocks = blocks or max(1, np.gcd(dim_in, dim_out))
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


# Backwards compatibility aliases -------------------------------------------------

FlipFeedback = FlipTernary

__all__ = [
    "FeedbackStrategy",
    "DFA",
    "FlipTernary",
    "FlipFeedback",
    "StructuredFeedback",
]
