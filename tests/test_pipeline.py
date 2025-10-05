from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from feedflipnets import train
from feedflipnets.core import feedback
from feedflipnets.training import pipelines


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")


def test_feedback_strategy_contract():
    rng = np.random.default_rng(0)
    weights = [rng.standard_normal((4, 3)), rng.standard_normal((3, 2))]
    layer_inputs = [rng.standard_normal((5, 4)), rng.standard_normal((5, 3))]
    layer_derivs = [np.ones((5, 3))]
    activations = {
        "weights": [w.copy() for w in weights],
        "layer_inputs": layer_inputs,
        "layer_derivs": layer_derivs,
    }
    error = rng.standard_normal((5, 2))

    dfa = feedback.DFA(np.random.default_rng(1), [3, 2])
    grads_dfa = dfa.compute_updates(activations, error)
    assert set(grads_dfa) == {"W0", "W1"}
    assert grads_dfa["W0"].shape == (4, 3)
    assert grads_dfa["W1"].shape == (3, 2)

    flip = feedback.FlipFeedback()
    grads_flip = flip.compute_updates(activations, error)
    assert set(grads_flip) == {"W0", "W1"}
    assert grads_flip["W0"].shape == (4, 3)
    assert grads_flip["W1"].shape == (3, 2)


def test_dataset_registry_offline_provenance():
    from feedflipnets.data import registry
    from feedflipnets.data.loaders import mnist  # noqa: F401

    spec = registry.get_dataset("mnist", subset="train", max_items=2, one_hot=False)
    provenance = spec.provenance
    assert provenance["mode"] == "offline"
    assert Path(provenance["local_path"]).exists()
    batch = next(spec.loader("train", 2))
    assert batch.inputs.shape[0] == 2


def test_pipeline_smoke_synthetic(tmp_path):
    config = pipelines.load_preset("synthetic-min")
    config = json.loads(json.dumps(config))
    config["train"]["run_dir"] = str(tmp_path / "run")
    result = pipelines.run_pipeline(config)
    assert not isinstance(result, list)
    assert result.steps == config["train"]["steps"]
    assert Path(result.metrics_path).exists()


def test_metrics_determinism(tmp_path):
    base = pipelines.load_preset("synthetic-min")
    base = json.loads(json.dumps(base))
    base["train"]["steps"] = 40
    base["train"]["run_dir"] = str(tmp_path / "det")

    first = pipelines.run_pipeline(base)
    second = pipelines.run_pipeline(base)

    path = Path(first.metrics_path)
    with path.open() as handle:
        lines = handle.readlines()
    assert len(lines) >= 20
    values_first = [json.loads(line)["loss"] for line in lines[:20]]

    with open(second.metrics_path) as handle:
        lines_second = handle.readlines()
    values_second = [json.loads(line)["loss"] for line in lines_second[:20]]

    assert np.allclose(values_first, values_second, atol=1e-7)


def test_legacy_train_shim_warns(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        losses, auc, t01 = train.train_single(
            method="Backprop",
            depth=1,
            freq=1,
            seed=0,
            epochs=5,
            dataset="synthetic",
            max_points=16,
        )
    assert any(isinstance(item.message, DeprecationWarning) for item in caught)
    assert len(losses) == 5
    assert isinstance(auc, float)
    assert isinstance(t01, int)

