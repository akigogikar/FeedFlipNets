"""Pipeline assembly for FeedFlipNets."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ..core.strategies import DFA, FlipTernary, StructuredFeedback
from ..core.types import RunResult
from ..data import registry
from ..reporting.artifacts import write_manifest
from ..reporting.metrics import CsvSink, JsonlSink
from ..reporting.plots import PlotAdapter
from ..reporting.summary import write_summary
from .trainer import FeedForwardModel, SGDOptimizer, Trainer

_PRESETS: Dict[str, Mapping[str, object]] = {
    "basic_dfa_cpu": {
        "data": {
            "name": "synthetic",
            "options": {"freq": 3, "n_points": 64, "seed": 0},
        },
        "model": {
            "d_in": 1,
            "d_out": 1,
            "hidden": [4],
            "quant": "det",
            "tau": 0.05,
            "strategy": "dfa",
        },
        "train": {
            "epochs": 1,
            "steps_per_epoch": 90,
            "batch_size": 8,
            "seed": 7,
            "lr": 0.03,
            "run_dir": "runs/basic-dfa-cpu",
            "enable_plots": False,
        },
    },
    "synthetic-min": {
        "data": {
            "name": "synthetic",
            "options": {"freq": 3, "n_points": 128, "seed": 0},
        },
        "model": {
            "d_in": 1,
            "d_out": 1,
            "hidden": [8],
            "quant": "det",
            "tau": 0.05,
            "strategy": "flip",
        },
        "train": {
            "epochs": 1,
            "steps_per_epoch": 200,
            "batch_size": 16,
            "seed": 42,
            "lr": 0.02,
            "run_dir": "runs/synthetic-min",
            "enable_plots": False,
        },
    },
    "mnist-flip-det": {
        "data": {
            "name": "mnist",
            "options": {"subset": "train", "max_items": 16, "one_hot": True},
        },
        "model": {
            "d_in": 784,
            "d_out": 10,
            "hidden": [32],
            "quant": "det",
            "tau": 0.05,
            "strategy": "flip",
        },
        "train": {
            "epochs": 1,
            "steps_per_epoch": 50,
            "batch_size": 8,
            "seed": 1,
            "lr": 0.05,
            "run_dir": "runs/mnist-flip-det",
            "enable_plots": False,
        },
    },
    "tinystories-dfa-stoch": {
        "data": {"name": "tinystories", "options": {"window": 3, "seed": 0}},
        "model": {
            "d_in": 3,
            "d_out": 1,
            "hidden": [16],
            "quant": "stoch",
            "tau": 0.1,
            "strategy": "dfa",
        },
        "train": {
            "epochs": 1,
            "steps_per_epoch": 80,
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
            "epochs": 1,
            "steps_per_epoch": 20,
        },
        "data": {
            "name": "synthetic",
            "options": {"freq": 3, "n_points": 128, "seed": 0},
        },
        "model": {"d_in": 1, "d_out": 1, "tau": 0.05, "quant": "det"},
        "train": {
            "batch_size": 16,
            "lr": 0.02,
            "enable_plots": False,
            "run_dir": "runs/sweep",
        },
    },
    "synthetic-structured-orthogonal-fixed": {
        "data": {
            "name": "synthetic",
            "options": {"freq": 3, "n_points": 512, "seed": 0},
        },
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
            "epochs": 1,
            "steps_per_epoch": 300,
            "batch_size": 64,
            "seed": 123,
            "lr": 0.05,
            "run_dir": "runs/syn-orth-fixed",
            "enable_plots": False,
        },
    },
    "synthetic-structured-hadamard-perstep": {
        "data": {
            "name": "synthetic",
            "options": {"freq": 3, "n_points": 512, "seed": 0},
        },
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
            "epochs": 1,
            "steps_per_epoch": 300,
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


def load_preset(name: str) -> Mapping[str, object]:
    try:
        return deepcopy(_PRESETS[name])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown preset: {name}") from exc


def run_pipeline(config: Mapping[str, object]) -> RunResult | List[RunResult]:
    if "sweep" in config:
        return _run_sweep(config)
    return _train_single(config)


def _run_sweep(config: Mapping[str, object]) -> List[RunResult]:
    sweep_cfg = config["sweep"]
    results: List[RunResult] = []
    for method in sweep_cfg["methods"]:
        for depth in sweep_cfg["depths"]:
            for freq in sweep_cfg["freqs"]:
                for seed in sweep_cfg["seeds"]:
                    cfg = deepcopy(config)
                    cfg.pop("sweep", None)
                    cfg.setdefault("model", {})
                    cfg["model"].update({"strategy": method, "depth": depth})
                    cfg.setdefault("data", {})
                    cfg["data"].setdefault("options", {})["freq"] = freq
                    cfg.setdefault("train", {})["seed"] = seed
                    cfg["offline"] = config.get("offline", True)
                    result = _train_single(cfg)
                    results.append(result)
    return results


def _train_single(config: Mapping[str, object]) -> RunResult:
    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    train_cfg = dict(config["train"])

    offline = bool(config.get("offline", True))
    cache_dir = train_cfg.get("cache_dir")
    dataset = registry.get(
        data_cfg["name"],
        offline=offline,
        cache_dir=cache_dir,
        **data_cfg.get("options", {}),
    )
    batch_size = int(train_cfg.get("batch_size", 1))
    data_iter = _IterableLoader(lambda: dataset.loader("train", batch_size))

    hidden_dims = _build_hidden(model_cfg)
    model_cfg["hidden"] = hidden_dims
    dims = _build_dims(model_cfg)

    seed = int(train_cfg.get("seed", 0))
    epochs, steps_per_epoch = _resolve_schedule(train_cfg)
    strategy = _build_strategy(model_cfg, dims, seed)

    run_dir = Path(train_cfg.get("run_dir", "runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    manifest_path = run_dir / "manifest.json"

    sink = JsonlSink(metrics_path, seed=seed)
    csv_sink = CsvSink(run_dir / "metrics.csv")
    plot = PlotAdapter(run_dir, enable_plots=bool(train_cfg.get("enable_plots", False)))

    model = FeedForwardModel(
        layer_dims=dims,
        tau=float(model_cfg.get("tau", 0.05)),
        quant=str(model_cfg.get("quant", "det")),
        seed=seed,
    )
    optimizer = SGDOptimizer(lr=float(train_cfg.get("lr", 0.01)))
    trainer = Trainer(
        model=model,
        strategy=strategy,
        optimizer=optimizer,
        callbacks=[sink, csv_sink, plot],
    )

    result = trainer.run(
        data_iter,
        epochs=epochs,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
    )
    plot.close()

    manifest = write_manifest(
        manifest_path,
        config=_safe_config(config, hidden_dims),
        dataset_provenance=dataset.provenance,
    )
    summary_tail = int(train_cfg.get("summary_tail", 32))
    summary_path = write_summary(
        metrics_path, run_dir / "summary.json", tail=summary_tail
    )
    return RunResult(
        steps=result.steps,
        metrics_path=str(metrics_path),
        manifest_path=manifest,
        summary_path=str(summary_path),
    )


def _build_hidden(config: Mapping[str, object], depth: int | None = None) -> List[int]:
    if "hidden" in config:
        return list(config["hidden"])  # type: ignore[list-item]
    hidden_dim = int(config.get("hidden_dim", 32))
    depth = depth or int(config.get("depth", 1))
    return [hidden_dim for _ in range(depth)]


def _build_dims(model_cfg: Mapping[str, object]) -> List[int]:
    dims = [int(model_cfg.get("d_in", 1))]
    dims.extend(int(h) for h in model_cfg.get("hidden", []))
    dims.append(int(model_cfg.get("d_out", 1)))
    return dims


def _build_strategy(model_cfg: Mapping[str, object], dims: Sequence[int], seed: int):
    name = str(model_cfg.get("strategy", "flip"))
    rng = np.random.default_rng(seed)
    if name == "flip":
        refresh = str(model_cfg.get("feedback_refresh", "per_step"))
        return FlipTernary(refresh=refresh)
    if name == "dfa":
        return DFA(rng)
    if name == "structured":
        structure_type = model_cfg.get("structure_type")
        if structure_type is None:
            raise KeyError(
                "Structured strategy requires `structure_type` in the model config"
            )
        refresh = str(model_cfg.get("feedback_refresh", "fixed"))
        rank = model_cfg.get("rank")
        blocks = model_cfg.get("blocks")
        return StructuredFeedback(
            rng,
            structure_type=str(structure_type),
            refresh=refresh,
            rank=int(rank) if rank is not None else None,
            blocks=int(blocks) if blocks is not None else None,
        )
    raise ValueError(f"Unknown strategy: {name}")


def _resolve_schedule(train_cfg: Mapping[str, object]) -> tuple[int, int]:
    if "epochs" in train_cfg:
        epochs = int(train_cfg["epochs"])
        steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1))
        return epochs, steps_per_epoch
    if "steps" in train_cfg:
        total_steps = int(train_cfg["steps"])
        steps_per_epoch = int(train_cfg.get("steps_per_epoch", total_steps))
        epochs = math.ceil(total_steps / max(1, steps_per_epoch))
        return epochs, steps_per_epoch
    return 1, int(train_cfg.get("steps_per_epoch", 1))


def _safe_config(
    config: Mapping[str, object], hidden_dims: Iterable[int]
) -> Mapping[str, object]:
    copied = json.loads(json.dumps(config))
    copied.setdefault("model", {})["hidden"] = list(hidden_dims)
    return copied


class _IterableLoader:
    """Wrap a factory into a re-iterable loader."""

    def __init__(self, factory: Callable[[], Iterable]):
        self._factory = factory

    def __iter__(self):
        return iter(self._factory())
