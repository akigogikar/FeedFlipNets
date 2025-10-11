import numpy as np

from feedflipnets.core.strategies import DFA, StructuredFeedback, TernaryDFA
from feedflipnets.core.types import ModelDescription
from feedflipnets.data.cache import CacheManifest, fetch
from feedflipnets.reporting.plots import PlotAdapter
from feedflipnets.training.trainer import FeedForwardModel


def _make_activations(layer_dims):
    model = FeedForwardModel(layer_dims=layer_dims, tau=0.05, quant="det", seed=123)
    inputs = np.ones((4, layer_dims[0]), dtype=np.float32)
    outputs, activations = model.forward(inputs)
    error = outputs - np.ones_like(outputs)
    return activations, error


def test_dfa_gradient_shapes():
    dims = [3, 5, 2]
    activations, error = _make_activations(dims)
    strategy = DFA(np.random.default_rng(0))
    state = strategy.init(ModelDescription(layer_dims=dims))
    grads, _ = strategy.backward(activations, error, state)
    assert grads["W0"].shape == (dims[0], dims[1])
    assert grads["W1"].shape == (dims[1], dims[2])


def test_ternary_dfa_uses_quantised_feedback():
    dims = [2, 4, 1]
    activations, error = _make_activations(dims)
    strategy = TernaryDFA(np.random.default_rng(0), threshold=0.05)
    state = strategy.init(ModelDescription(layer_dims=dims))
    grads, _ = strategy.backward(activations, error, state)
    assert set(grads) == {"W0", "W1"}


def test_structured_strategy_refresh(tmp_path):
    dims = [2, 3, 2]
    activations, error = _make_activations(dims)
    strategy = StructuredFeedback(
        np.random.default_rng(1), structure_type="orthogonal", refresh="per_epoch"
    )
    state = strategy.init(ModelDescription(layer_dims=dims))
    grads, state = strategy.backward(activations, error, state)
    assert grads["W0"].shape == (dims[0], dims[1])
    state.metadata["pending_refresh"] = True
    _, state2 = strategy.backward(activations, error, state)
    assert state2.feedback[0].shape == (dims[-1], dims[1])


def test_cache_manifest_records(tmp_path):
    manifest = CacheManifest(tmp_path)
    offline_path = tmp_path / "fixture.bin"

    def _builder(path):
        path.write_bytes(b"data")

    path, record = fetch(
        name="unit-fixture",
        url="http://example.com/unit",
        offline_path=offline_path,
        offline_builder=_builder,
        offline=True,
        cache_dir=tmp_path,
        filename="unit.bin",
        manifest=manifest,
    )
    assert path.exists()
    stored = manifest.get("unit-fixture")
    assert stored["mode"].startswith("offline")
    assert stored["checksum"] == record["checksum"]
    assert (tmp_path / "manifest.json").exists()


def test_plot_adapter_headless(tmp_path):
    adapter = PlotAdapter(tmp_path, enable_plots=True)
    adapter.on_step(0, {"loss": 1.0})
    adapter.on_step(1, {"loss": 0.5})
    adapter.close()
    assert (tmp_path / "loss.png").exists()
