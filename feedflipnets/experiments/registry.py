"""Experiment registry loader and helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REGISTRY = _DEFAULT_ROOT / "experiments" / "registry.json"
_DEFAULT_SCHEMA = _DEFAULT_ROOT / "experiments" / "schema.json"


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def _normalise(value):  # type: ignore[override]
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_hash(config: Mapping[str, object]) -> str:
    """Return a stable 12-character hash for ``config``."""

    canonical = json.dumps(_normalise(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved experiment configuration."""

    name: str
    parameters: Mapping[str, object]
    config: Mapping[str, object]
    version: int

    def to_pipeline_config(self) -> Dict[str, object]:
        return json.loads(json.dumps(self.config))

    @property
    def run_id(self) -> str:
        return config_hash(self.config)


def _validate(raw: Mapping[str, object], schema: Mapping[str, object]) -> None:
    required = schema.get("required_keys", [])
    for key in required:
        if key not in raw:
            raise ValueError(f"Registry missing required key: {key}")

    experiments = raw.get("experiments")
    if not isinstance(experiments, Mapping):
        raise TypeError("Registry 'experiments' must be a mapping")

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise TypeError("Registry 'defaults' must be a mapping")

    experiment_required = schema.get("experiment_required", [])
    allowed_strategies = set(schema.get("strategy_enum", []))

    for name, entry in experiments.items():
        if not isinstance(entry, Mapping):
            raise TypeError(f"Experiment '{name}' must be a mapping")
        merged: MutableMapping[str, object] = dict(defaults)
        merged.update(entry)
        for field in experiment_required:
            if field not in merged:
                raise ValueError(f"Experiment '{name}' missing field '{field}'")
        strategy = merged.get("strategy")
        if allowed_strategies and strategy not in allowed_strategies:
            raise ValueError(f"Experiment '{name}' has unsupported strategy '{strategy}'")


def _build_pipeline_config(params: Mapping[str, object]) -> Dict[str, object]:
    dataset = str(params["dataset"])
    strategy = str(params["strategy"])
    steps = int(params["steps"])
    seed = int(params["seed"])

    data_options = dict(params.get("data_options", {}))
    data_options.setdefault("seed", seed)
    if dataset == "synth_fixture":
        data_options.setdefault("length", steps)
    if dataset == "synthetic":
        data_options.setdefault("n_points", max(steps * 2, 32))
        data_options.setdefault("freq", 3)

    model_cfg = dict(params.get("model", {}))
    model_cfg.setdefault("d_in", int(params.get("d_in", 1)))
    model_cfg.setdefault("d_out", int(params.get("d_out", 1)))
    hidden = params.get("hidden")
    if hidden is not None:
        model_cfg["hidden"] = list(hidden)  # type: ignore[list-item]
    else:
        model_cfg.setdefault("hidden", [8])
    model_cfg.setdefault("quant", str(params.get("quant", "det")))
    model_cfg.setdefault("tau", float(params.get("tau", 0.05)))
    model_cfg["strategy"] = strategy
    for key in ("structure_type", "feedback_refresh", "rank", "blocks"):
        if key in params and key not in model_cfg:
            model_cfg[key] = params[key]

    train_cfg = dict(params.get("train", {}))
    train_cfg.setdefault("epochs", int(params.get("epochs", 1)))
    train_cfg.setdefault("steps_per_epoch", int(params.get("steps_per_epoch", steps)))
    train_cfg.setdefault("batch_size", int(params.get("batch_size", 8)))
    train_cfg.setdefault("seed", seed)
    train_cfg.setdefault("lr", float(params.get("lr", 0.05)))
    train_cfg.setdefault("enable_plots", bool(params.get("enable_plots", False)))

    config = {
        "data": {"name": dataset, "options": data_options},
        "model": model_cfg,
        "train": train_cfg,
        "offline": bool(params.get("offline", True)),
    }
    return config


def load_registry(
    path: str | Path | None = None, schema_path: str | Path | None = None
) -> Mapping[str, object]:
    """Load and validate the experiment registry."""

    registry_path = Path(path or _DEFAULT_REGISTRY)
    schema_path = Path(schema_path or _DEFAULT_SCHEMA)
    raw = _load_json(registry_path)
    schema = _load_json(schema_path)
    _validate(raw, schema)

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, Mapping):
        defaults = {}

    experiments = {}
    version = int(raw.get("version", 1))
    for name, entry in raw["experiments"].items():  # type: ignore[index]
        params: MutableMapping[str, object] = dict(defaults)
        params.update(entry)
        config = _build_pipeline_config(params)
        experiments[name] = ExperimentConfig(
            name=name,
            parameters=json.loads(json.dumps(params)),
            config=json.loads(json.dumps(config)),
            version=version,
        )

    return {
        "version": version,
        "defaults": json.loads(json.dumps(defaults)),
        "experiments": experiments,
    }


def get_experiment(name: str, *, path: str | Path | None = None) -> ExperimentConfig:
    """Return the resolved configuration for ``name``."""

    registry = load_registry(path)
    experiments = registry["experiments"]
    if name not in experiments:
        raise KeyError(f"Experiment '{name}' not found in registry")
    return experiments[name]


__all__ = ["ExperimentConfig", "config_hash", "get_experiment", "load_registry"]
