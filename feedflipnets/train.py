"""Legacy training entry points (deprecated)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .data import registry
from .training import pipelines

_trapezoid = getattr(np, "trapezoid", np.trapz)

_DEPRECATION_EMITTED = False


def _warn_once() -> None:
    global _DEPRECATION_EMITTED
    if not _DEPRECATION_EMITTED:
        warnings.warn(
            "feedflipnets.train is deprecated; use `python -m cli.main --preset <name>`.",
            DeprecationWarning,
            stacklevel=3,
        )
        _DEPRECATION_EMITTED = True


def _dataset_options(
    dataset: str | None, freq: int, max_points: int | None, seed: int
) -> Dict[str, object]:
    if dataset is None or dataset == "synthetic":
        opts: Dict[str, object] = {
            "freq": freq,
            "n_points": max(max_points or 128, 32),
            "seed": seed,
        }
        if max_points is not None:
            opts["n_points"] = max_points
        return {"name": "synthetic", "options": opts}
    if dataset == "mnist":
        options = {"subset": "train", "max_items": max_points or 32, "one_hot": True}
        return {"name": "mnist", "options": options}
    if dataset == "tinystories":
        return {"name": "tinystories", "options": {}}
    if dataset and dataset.startswith("ucr:"):
        name = dataset.split(":", 1)[1]
        return {"name": "ucr", "options": {"ucr_name": name}}
    raise ValueError(f"Unsupported dataset: {dataset}")


def _infer_dims(data_cfg: Dict[str, object]) -> Tuple[int, int]:
    spec = registry.get(data_cfg["name"], **data_cfg.get("options", {}))  # type: ignore[arg-type]
    batch = next(spec.loader("train", 1))
    return batch.inputs.shape[1], batch.targets.shape[1]


_METHOD_MAP = {
    "Backprop": "dfa",
    "Vanilla DFA": "dfa",
    "Structured DFA": "dfa",
    "Ternary static Δ": "flip",
    "Ternary + adaptive + ortho B": "flip",
    "Ternary + adaptive + ortho B + cal": "flip",
    "+Shadow": "flip",
    "+Momentum": "flip",
    "Ternary DFA on Transformer/LLM": "flip",
}


def _legacy_run_dir(method: str, depth: int, freq: int, seed: int) -> str:
    return f"runs/legacy/{method.replace(' ', '_')}_d{depth}_f{freq}_s{seed}"


def train_single(
    method: str,
    depth: int,
    freq: int,
    seed: int,
    epochs: int = 500,
    dataset: str | None = None,
    max_points: int | None = None,
) -> Tuple[List[float], float, int]:
    """Shim that proxies the legacy API onto :mod:`cli.main`."""

    _warn_once()
    data_cfg = _dataset_options(dataset, freq, max_points, seed)
    d_in, d_out = _infer_dims(data_cfg)
    hidden = [16] * max(depth, 1)
    strategy = _METHOD_MAP.get(method, "flip")

    config = {
        "data": data_cfg,
        "model": {
            "d_in": d_in,
            "d_out": d_out,
            "hidden": hidden,
            "quant": "det",
            "tau": 0.05,
            "strategy": strategy,
        },
        "train": {
            "steps": epochs,
            "batch_size": 8,
            "seed": seed,
            "lr": 0.02,
            "run_dir": _legacy_run_dir(method, depth, freq, seed),
            "enable_plots": False,
        },
    }

    result = pipelines.run_pipeline(config)
    run_result = result if not isinstance(result, list) else result[-1]

    metrics_path = Path(run_result.metrics_path)
    losses: List[float] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            record = json.loads(line)
            losses.append(float(record.get("loss", 0.0)))
    auc = float(_trapezoid(losses)) if losses else 0.0
    t01 = next((idx for idx, loss in enumerate(losses) if loss < 0.01), epochs + 1)
    return losses, auc, t01


def sweep_and_log(
    methods: Sequence[str],
    depths: Sequence[int],
    freqs: Sequence[int],
    seeds: Iterable[int],
    epochs: int,
    outdir: str,
    dataset: str | None = None,
    max_points: int | None = None,
) -> Dict[str, np.ndarray]:
    _warn_once()
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    seeds_list = list(seeds)
    summary = {
        "methods": list(methods),
        "depths": list(depths),
        "freqs": list(freqs),
        "seeds": seeds_list,
        "epochs": epochs,
        "dataset": dataset or "synthetic",
        "max_points": max_points,
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))

    final_tables: Dict[str, np.ndarray] = {m: np.zeros((len(depths), len(freqs))) for m in methods}

    for method in methods:
        for i, depth in enumerate(depths):
            for j, freq in enumerate(freqs):
                curves: List[List[float]] = []
                for seed in seeds_list:
                    curve, _, _ = train_single(
                        method,
                        depth,
                        freq,
                        seed,
                        epochs=epochs,
                        dataset=dataset,
                        max_points=max_points,
                    )
                    curves.append(curve)
                    np.save(
                        out_path
                        / f"curve_{method.replace(' ', '_')}_d{depth}_k{freq}_seed{seed}.npy",
                        np.array(curve),
                    )
                if curves:
                    final_tables[method][i, j] = curves[-1][-1] if curves[-1] else 0.0
    return final_tables


__all__ = ["train_single", "sweep_and_log"]
