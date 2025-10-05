"""Assembly helpers that wire together data, models and reporting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np

from ..core.feedback import DFA, FlipFeedback, StructuredFeedback
from ..core.types import RunResult
from ..data import registry
from ..data.loaders import mnist as _mnist_loader  # noqa: F401
from ..data.loaders import synthetic as _synthetic_loader  # noqa: F401
from ..data.loaders import tinystories as _tinystories_loader  # noqa: F401
from ..data.loaders import ucr_uea as _ucr_loader  # noqa: F401
from ..reporting.artifacts import write_manifest
from ..reporting.metrics import JsonlSink
from ..reporting.plots import PlotAdapter
from .trainer import Trainer

_PRESETS: Dict[str, Mapping[str, object]] = {
    "synthetic-min": {
        "data": {"name": "synthetic", "options": {"freq": 3, "n_points": 128, "seed": 0}},
        "model": {"d_in": 1, "d_out": 1, "hidden": [8], "quant": "det", "tau": 0.05, "strategy": "flip"},
        "train": {
            "steps": 200,
            "batch_size": 16,
            "seed": 42,
            "lr": 0.02,
            "run_dir": "runs/synthetic-min",
            "enable_plots": False,
        },
    },
    "mnist-flip-det": {
        "data": {"name": "mnist", "options": {"subset": "train", "max_items": 16, "one_hot": True}},
        "model": {"d_in": 784, "d_out": 10, "hidden": [32], "quant": "det", "tau": 0.05, "strategy": "flip"},
        "train": {
            "steps": 50,
            "batch_size": 8,
            "seed": 1,
            "lr": 0.05,
            "run_dir": "runs/mnist-flip-det",
            "enable_plots": False,
        },
    },
    "tinystories-dfa-stoch": {
        "data": {"name": "tinystories", "options": {"window": 3, "seed": 0}},
        "model": {"d_in": 3, "d_out": 1, "hidden": [16], "quant": "stoch", "tau": 0.1, "strategy": "dfa"},
        "train": {
            "steps": 80,
            "batch_size": 4,
            "seed": 7,
            "lr": 0.03,
            "run_dir": "runs/tinystories-dfa-stoch",
            "enable_plots": False,
        },
    },
    "depth-frequency-sweep": {
        "sweep": {
            "methods": ["flip", "dfa"],
            "depths": [1, 2],
            "freqs": [2, 4],
            "seeds": [0, 1],
            "epochs": 20,
        },
        "data": {"name": "synthetic", "options": {"freq": 3, "n_points": 128, "seed": 0}},
        "model": {"d_in": 1, "d_out": 1, "tau": 0.05, "quant": "det"},
        "train": {"batch_size": 16, "lr": 0.02, "enable_plots": False, "run_dir": "runs/sweep"},
    },
    "synthetic-structured-orthogonal-fixed": {
        "data": {"name": "synthetic", "options": {"freq": 3, "n_points": 512, "seed": 0}},
        "model": {
            "d_in": 1,
            "d_out": 8,
            "hidden": [32],
            "quant": "det",
            "tau": 0.05,
            "strategy": "structured",
            "structure_type": "orthogonal",
            "feedback_refresh": "fixed",
        },
        "train": {
            "steps": 300,
            "batch_size": 64,
            "seed": 123,
            "lr": 0.05,
            "run_dir": "runs/syn-orth-fixed",
            "enable_plots": False,
        },
    },
    "synthetic-structured-hadamard-perstep": {
        "data": {"name": "synthetic", "options": {"freq": 3, "n_points": 512, "seed": 0}},
        "model": {
            "d_in": 1,
            "d_out": 8,
            "hidden": [32],
            "quant": "det",
            "tau": 0.05,
            "strategy": "structured",
            "structure_type": "hadamard",
            "feedback_refresh": "per_step",
        },
        "train": {
            "steps": 300,
            "batch_size": 64,
            "seed": 321,
            "lr": 0.05,
            "run_dir": "runs/syn-hadamard-step",
            "enable_plots": False,
        },
    },
}


def presets() -> Mapping[str, Mapping[str, object]]:
    return deepcopy(_PRESETS)


def _build_hidden(config: Mapping[str, object], depth: int | None = None) -> List[int]:
    if "hidden" in config:
        return list(config["hidden"])  # type: ignore[list-item]
    hidden_dim = int(config.get("hidden_dim", 32))
    depth = depth or int(config.get("depth", 1))
    return [hidden_dim for _ in range(depth)]


def _build_strategy(model_cfg: Mapping[str, object], hidden_dims: List[int], d_out: int, seed: int) -> object:
    name = str(model_cfg.get("strategy", "flip"))
    if name == "flip":
        refresh = str(model_cfg.get("feedback_refresh", "per_step"))
        return FlipFeedback(refresh=refresh)
    if name == "dfa":
        rng = np.random.default_rng(seed)
        return DFA(rng, hidden_dims + [d_out])
    if name == "structured":
        structure_type = model_cfg.get("structure_type")
        if structure_type is None:
            raise KeyError("Structured strategy requires `structure_type` in the model config")
        rng = np.random.default_rng(seed)
        refresh = str(model_cfg.get("feedback_refresh", "fixed"))
        rank = model_cfg.get("rank")
        blocks = model_cfg.get("blocks")
        base_dim = hidden_dims[-1] if hidden_dims else int(model_cfg.get("d_in", 1))
        strategy = StructuredFeedback(
            rng,
            base_dim,
            d_out,
            structure_type=str(structure_type),
            refresh=refresh,
            rank=int(rank) if rank is not None else None,
            blocks=int(blocks) if blocks is not None else None,
            hidden_dims=hidden_dims,
            input_dim=int(model_cfg.get("d_in", base_dim)),
        )
        return strategy
    raise ValueError(f"Unknown strategy: {name}")


def _train_single(config: Mapping[str, object]) -> RunResult:
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    train_cfg = dict(config["train"])

    dataset = registry.get_dataset(data_cfg["name"], **data_cfg.get("options", {}))
    batch_size = int(train_cfg.get("batch_size", 1))
    data_iter = dataset.loader("train", batch_size)

    hidden_dims = _build_hidden(model_cfg)
    model_cfg["hidden"] = hidden_dims

    strategy = _build_strategy(model_cfg, hidden_dims, int(model_cfg["d_out"]), int(train_cfg.get("seed", 0)))

    run_dir = Path(train_cfg.get("run_dir", "runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    manifest_path = run_dir / "manifest.json"

    sink = JsonlSink(metrics_path)
    plot = PlotAdapter(run_dir, enable_plots=bool(train_cfg.get("enable_plots", False)))

    trainer_config: Dict[str, object] = {
        "data": data_cfg,
        "model": model_cfg,
        "train": train_cfg,
        "strategy": strategy,
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
    }

    result = Trainer.run(trainer_config, data_iter, callbacks=[sink, plot])
    plot.close()

    safe_config = deepcopy(config)
    safe_config.setdefault("model", {})["hidden"] = hidden_dims
    manifest_real = write_manifest(manifest_path, config=safe_config, dataset_provenance=dataset.provenance)
    return RunResult(steps=result.steps, metrics_path=result.metrics_path, manifest_path=manifest_real)


def run_pipeline(config: Mapping[str, object]) -> RunResult | List[RunResult]:
    if "sweep" in config:
        sweep_cfg = config["sweep"]
        results: List[RunResult] = []
        methods = sweep_cfg.get("methods", [config["model"].get("strategy", "flip")])
        depths = sweep_cfg.get("depths", [len(_build_hidden(config["model"]))])
        freqs = sweep_cfg.get("freqs", [config["data"].get("options", {}).get("freq", 1)])
        seeds = sweep_cfg.get("seeds", [config["train"].get("seed", 0)])
        epochs = int(sweep_cfg.get("epochs", config["train"].get("steps", 100)))

        for method in methods:
            for depth in depths:
                for freq in freqs:
                    for seed in seeds:
                        cfg = deepcopy(config)
                        cfg.pop("sweep", None)
                        cfg["data"] = deepcopy(config["data"])
                        cfg["data"].setdefault("options", {})["freq"] = freq
                        cfg["model"] = deepcopy(config["model"])
                        cfg["model"]["strategy"] = method
                        cfg["model"]["depth"] = depth
                        cfg["train"] = deepcopy(config["train"])
                        cfg["train"]["seed"] = seed
                        cfg["train"]["steps"] = epochs
                        run_name = f"{method}-d{depth}-f{freq}-s{seed}"
                        cfg["train"]["run_dir"] = str(Path(config["train"]["run_dir"]) / run_name)
                        results.append(_train_single(cfg))
        return results
    return _train_single(config)


def load_preset(name: str) -> Mapping[str, object]:
    presets_map = presets()
    if name not in presets_map:
        raise KeyError(f"Unknown preset: {name}")
    return presets_map[name]

