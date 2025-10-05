import numpy as np

from feedflipnets.core.feedback import StructuredFeedback


def _make_activations(input_dim: int, hidden_dims: list[int], output_dim: int, batch: int = 4):
    dims = [input_dim, *hidden_dims, output_dim]
    weights = [
        np.zeros((dims[idx], dims[idx + 1]), dtype=np.float32)
        for idx in range(len(dims) - 1)
    ]
    layer_inputs = []
    layer_derivs = []
    layer_inputs.append(np.zeros((batch, dims[0]), dtype=np.float32))
    for idx in range(len(dims) - 2):
        hidden_dim = dims[idx + 1]
        layer_derivs.append(np.ones((batch, hidden_dim), dtype=np.float32))
        layer_inputs.append(np.zeros((batch, hidden_dim), dtype=np.float32))
    return {"weights": weights, "layer_inputs": layer_inputs, "layer_derivs": layer_derivs}


def _mk(kind, din=16, dout=8, refresh="fixed", seed=0):
    rng = np.random.default_rng(seed)
    return StructuredFeedback(rng, din, dout, structure_type=kind, refresh=refresh)


def test_shapes_and_determinism():
    s1 = _mk("orthogonal", 16, 8, "fixed", 0)
    s2 = _mk("orthogonal", 16, 8, "fixed", 0)
    assert np.allclose(s1.B, s2.B)


def test_hadamard_exists_and_norms():
    s = _mk("hadamard", 16, 8)
    assert s.B.shape == (8, 16)
    row_norms = np.linalg.norm(s.B, axis=1)
    assert np.all(row_norms > 0)


def test_blockdiag_partition():
    s = StructuredFeedback(
        np.random.default_rng(0),
        32,
        16,
        structure_type="blockdiag",
        refresh="fixed",
        blocks=4,
    )
    assert s.B.shape == (16, 32)


def test_lowrank_rank_behavior():
    s = StructuredFeedback(
        np.random.default_rng(0),
        64,
        32,
        structure_type="lowrank",
        refresh="fixed",
        rank=4,
    )
    singular_values = np.linalg.svd(s.B, compute_uv=False)
    assert (singular_values > 1e-6).sum() >= 4


def test_per_step_refresh_changes_B():
    s = _mk("orthogonal", 16, 8, "per_step", 0)
    old = s.B.copy()
    activations = _make_activations(12, [16], 8)
    error = np.zeros((4, 8), dtype=np.float32)
    s.compute_updates(activations, error)
    assert not np.allclose(old, s.B)
