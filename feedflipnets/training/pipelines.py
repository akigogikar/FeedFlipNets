"""Pipeline assembly for FeedFlipNets with modality-aware training."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from ..core.strategies import DFA, Backprop, FlipTernary, StructuredFeedback, TernaryDFA
from ..core.types import Batch, RunResult
from ..data import registry
from ..reporting.artifacts import write_manifest
from ..reporting.metrics import CsvSink, JsonlSink
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

_PRESET_DIR = Path(__file__).resolve().parents[2] / "configs" / "presets"
_FILE_PRESETS_CACHE: Dict[str, Mapping[str, object]] | None = None


def _read_preset_file(path: Path) -> Mapping[str, object]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to load preset files in YAML format") from exc
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text or "{}")
    else:
        raise ValueError(f"Unsupported preset file type: {path.suffix}")

    if not isinstance(data, Mapping):
        raise TypeError(f"Preset {path.name} must decode to a mapping")
    return data


def _file_presets() -> Dict[str, Mapping[str, object]]:
    global _FILE_PRESETS_CACHE
    if _FILE_PRESETS_CACHE is None:
        presets: Dict[str, Mapping[str, object]] = {}
        if _PRESET_DIR.exists():
            for file in sorted(_PRESET_DIR.iterdir()):
                if file.suffix.lower() not in {".yaml", ".yml", ".json"}:
                    continue
                data = _read_preset_file(file)
                required = {"data", "model", "train"}
                missing = required - set(data)
                if missing:
                    missing_str = ", ".join(sorted(missing))
                    message = f"Preset {file.name} is missing required sections: " f"{missing_str}"
                    raise KeyError(message)
                presets[file.stem] = json.loads(json.dumps(data))
        _FILE_PRESETS_CACHE = presets
    cache = _FILE_PRESETS_CACHE or {}
    return {name: deepcopy(cfg) for name, cfg in cache.items()}


def presets() -> Mapping[str, Mapping[str, object]]:
    combined: Dict[str, Mapping[str, object]] = {}
    combined.update({name: deepcopy(cfg) for name, cfg in _PRESETS.items()})
    combined.update(_file_presets())
    return combined


def load_preset(name: str) -> Mapping[str, object]:
    file_overrides = _file_presets()
    if name in file_overrides:
        return file_overrides[name]
    try:
        return deepcopy(_PRESETS[name])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown preset: {name}") from exc


def run_pipeline(config: Mapping[str, object]) -> RunResult | List[RunResult]:
    if "sweep" in config:
        return _run_sweep(config)
    return _train_single(config)


class _SplitLoader:
    """Re-iterable loader with deterministic reseeding per epoch."""

    def __init__(
        self,
        spec: registry.DatasetSpec,
        split: str,
        batch_size: int,
        seed: int,
        steps: int,
    ) -> None:
        self.spec = spec
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.steps = steps
        self._epoch = 0

    def __iter__(self) -> Iterable[Batch]:
        rng_seed = self.seed + self._epoch
        self._epoch += 1
        return registry.iter_batches(self.spec, self.split, self.batch_size, seed=rng_seed)

    def __len__(self) -> int:
        return max(1, self.steps)


class _MetricsCapture:
    def __init__(self) -> None:
        self.history: list[tuple[int, Mapping[str, float]]] = []
        self.last: Mapping[str, float] = {}

    def on_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        payload = {k: float(v) for k, v in metrics.items()}
        self.history.append((int(epoch), payload))
        self.last = payload


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
    data_spec = dataset.data_spec

    batch_size = int(train_cfg.get("batch_size", 1))
    seed = int(train_cfg.get("seed", 0))
    eval_every = int(train_cfg.get("eval_every", 1))
    early_stopping = train_cfg.get("early_stopping_patience")
    early_stopping = int(early_stopping) if early_stopping is not None else None
    loss_name = str(train_cfg.get("loss", "auto"))
    metrics_cfg = train_cfg.get("metrics", "default")
    if isinstance(metrics_cfg, str):
        metrics_list = metrics_cfg
    else:
        metrics_list = ",".join(str(item) for item in metrics_cfg)

    ternary_mode = str(train_cfg.get("ternary", "per_step"))
    ternary_threshold = train_cfg.get("ternary_threshold")
    if ternary_threshold is not None:
        model_cfg["tau"] = float(ternary_threshold)

    split_sizes = dict(dataset.splits)
    if split_sizes.get("test", 0) == 0:
        train_size = int(split_sizes.get("train", 0))
        fallback = max(1, int(train_size * 0.1)) if train_size else batch_size
        split_sizes["test"] = fallback

    steps_per_epoch, val_steps, test_steps = _resolve_steps(split_sizes, batch_size, train_cfg)

    train_loader = _SplitLoader(dataset, "train", batch_size, seed, steps_per_epoch)
    val_loader = None
    if split_sizes.get("val", 0) > 0:
        val_loader = _SplitLoader(dataset, "val", batch_size, seed + 1, val_steps)
    test_loader = None
    if split_sizes.get("test", 0) > 0:
        test_loader = _SplitLoader(dataset, "test", batch_size, seed + 2, test_steps)

    sample_batch = next(iter(dataset.loader("train", 1)))
    inferred_in = sample_batch.inputs.reshape(sample_batch.inputs.shape[0], -1).shape[1]
    inferred_out = sample_batch.targets.reshape(sample_batch.targets.shape[0], -1).shape[1]
    d_in = int(model_cfg.get("d_in", inferred_in))
    d_out = int(model_cfg.get("d_out", inferred_out))
    model_cfg.setdefault("d_in", d_in)
    model_cfg.setdefault("d_out", d_out)

    if inferred_in != d_in:
        raise ValueError(f"Configured d_in={d_in} but observed batch has {inferred_in}")
    if inferred_out != d_out:
        raise ValueError(f"Configured d_out={d_out} but observed batch has {inferred_out}")

    hidden_dims = _build_hidden(model_cfg)
    model_cfg["hidden"] = hidden_dims
    dims = _build_dims(model_cfg)

    strategy = _build_strategy(model_cfg, dims, seed)

    run_dir = _resolve_run_dir(train_cfg, dataset.name, model_cfg.get("strategy", "flip"))
    run_dir.mkdir(parents=True, exist_ok=True)

    _print_startup_summary(
        dataset_name=dataset.name,
        dims=dims,
        loss=loss_name,
        metrics=metrics_list,
        strategy=str(model_cfg.get("strategy", "flip")),
        ternary_mode=ternary_mode,
        param_count=sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1)),
    )

    train_jsonl = JsonlSink(run_dir / "metrics_train.jsonl", split="train", seed=seed)
    train_csv = CsvSink(run_dir / "metrics_train.csv", split="train")
    val_jsonl = JsonlSink(run_dir / "metrics_val.jsonl", split="val", seed=seed)
    val_csv = CsvSink(run_dir / "metrics_val.csv", split="val")
    test_jsonl = JsonlSink(run_dir / "metrics_test.jsonl", split="test", seed=seed)
    test_csv = CsvSink(run_dir / "metrics_test.csv", split="test")

    capture_train = _MetricsCapture()
    capture_val = _MetricsCapture()
    capture_test = _MetricsCapture()

    model = FeedForwardModel(
        layer_dims=dims,
        tau=float(model_cfg.get("tau", 0.05)),
        quant=str(model_cfg.get("quant", "det")),
        seed=seed,
    )
    optimizer_name = str(train_cfg.get("optimizer", "sgd")).lower()
    if optimizer_name != "sgd":
        raise ValueError("Only 'sgd' optimizer is currently supported")
    optimizer = SGDOptimizer(lr=float(train_cfg.get("lr", 0.01)))
    trainer = Trainer(
        model=model,
        strategy=strategy,
        optimizer=optimizer,
        callbacks=[],
    )

    split_loggers = {
        "train": [train_jsonl, train_csv, capture_train],
        "val": [val_jsonl, val_csv, capture_val],
        "test": [test_jsonl, test_csv, capture_test],
    }

    result = trainer.run(
        train_loader,
        epochs=int(train_cfg.get("epochs", 1)),
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        val_loader=val_loader,
        test_loader=test_loader,
        val_steps=val_steps,
        test_steps=test_steps,
        task_type=data_spec.task_type,
        num_classes=data_spec.num_classes,
        loss=loss_name,
        metric_names=metrics_list,
        eval_every=eval_every,
        split_loggers=split_loggers,
        ternary_mode=ternary_mode,
        early_stopping_patience=early_stopping,
        checkpoint_dir=run_dir,
    )

    test_metrics_final = capture_test.last or {}
    (run_dir / "metrics_test.json").write_text(json.dumps(test_metrics_final, indent=2))

    manifest = write_manifest(
        run_dir / "manifest.json",
        config=_safe_config(config, hidden_dims),
        dataset_provenance=dataset.provenance,
    )
    summary_tail = int(train_cfg.get("summary_tail", 32))
    summary_path = write_summary(train_jsonl.path, run_dir / "summary.json", tail=summary_tail)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(_safe_config(config, hidden_dims), indent=2))

    metrics_alias = run_dir / "metrics.jsonl"
    if train_jsonl.path.exists():
        metrics_alias.write_text(train_jsonl.path.read_text())
    csv_alias = run_dir / "metrics.csv"
    train_csv_path = run_dir / "metrics_train.csv"
    if train_csv_path.exists():
        csv_alias.write_text(train_csv_path.read_text())

    return RunResult(
        steps=result.steps,
        metrics_path=str(train_jsonl.path),
        manifest_path=manifest,
        summary_path=str(summary_path),
    )


def _resolve_run_dir(train_cfg: Mapping[str, object], dataset: str, strategy: str) -> Path:
    if "run_dir" in train_cfg:
        return Path(train_cfg["run_dir"])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("runs") / timestamp / dataset / strategy


def _resolve_steps(
    split_sizes: Mapping[str, int],
    batch_size: int,
    train_cfg: Mapping[str, object],
) -> tuple[int, int, int]:
    if "steps_per_epoch" in train_cfg:
        train_steps = int(train_cfg["steps_per_epoch"])
    else:
        train_steps = max(1, math.ceil(split_sizes.get("train", 1) / batch_size))
    val_steps = (
        max(1, math.ceil(split_sizes.get("val", 0) / batch_size))
        if split_sizes.get("val", 0)
        else 0
    )
    test_steps = (
        max(1, math.ceil(split_sizes.get("test", 0) / batch_size))
        if split_sizes.get("test", 0)
        else 0
    )
    return train_steps, val_steps, test_steps


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
    if name == "ternary_dfa":
        threshold = float(model_cfg.get("feedback_threshold", model_cfg.get("tau", 0.05)))
        return TernaryDFA(rng, threshold=threshold)
    if name == "backprop":
        return Backprop()
    if name == "structured":
        structure_type = model_cfg.get("structure_type")
        if structure_type is None:
            raise KeyError("Structured strategy requires `structure_type` in the model config")
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


def _safe_config(config: Mapping[str, object], hidden_dims: Iterable[int]) -> Mapping[str, object]:
    copied = json.loads(json.dumps(config))
    copied.setdefault("model", {})["hidden"] = list(hidden_dims)
    return copied


def _print_startup_summary(
    *,
    dataset_name: str,
    dims: Sequence[int],
    loss: str,
    metrics: str,
    strategy: str,
    ternary_mode: str,
    param_count: int,
) -> None:
    print("=== FeedFlipNets run ===")
    print(f"Dataset       : {dataset_name}")
    print(f"Dimensions    : {dims}")
    print(f"Loss          : {loss}")
    print(f"Metrics       : {metrics}")
    print(f"Feedback      : {strategy}")
    print(f"Ternary mode  : {ternary_mode}")
    print(f"Parameters    : {param_count}")
    print("========================")


class _IterableLoader:
    """Legacy compatibility wrapper retained for completeness."""

    def __init__(self, factory: Callable[[], Iterable]):
        self._factory = factory

    def __iter__(self):
        return iter(self._factory())


__all__ = ["run_pipeline", "load_preset", "presets"]
