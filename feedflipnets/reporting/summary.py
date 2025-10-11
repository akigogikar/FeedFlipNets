"""Deterministic experiment summarisation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _area(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if callable(trapezoid):
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_auc(points: Sequence[float]) -> float:
    """Return the area-under-curve of ``points`` along an implicit step axis."""

    if not points:
        return 0.0
    y = np.asarray(points, dtype=np.float64)
    x = np.arange(len(points), dtype=np.float64)
    return _area(y, x)


def _extract_numeric(
    records: Iterable[Mapping[str, object]]
) -> Mapping[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            if key in {"step", "epoch"}:
                continue
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(float(value))
    return metrics


def _build_summary(
    metrics_path: Path, records: list[Mapping[str, object]], tail: int
) -> Mapping[str, object]:
    metrics = _extract_numeric(records)
    tail_window = min(tail, len(records)) if records else 0
    summary_metrics: dict[str, Mapping[str, float]] = {}
    for name, values in metrics.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        tail_arr = arr[-tail_window:] if tail_window else arr[:0]
        summary_metrics[name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "last": float(arr[-1]),
            "tail_auc": compute_auc(tail_arr.tolist()) if tail_window else 0.0,
        }

    return {
        "version": 1,
        "records": len(records),
        "tail_window": tail_window,
        "metrics": summary_metrics,
    }


def write_summary(
    metrics_jsonl: str | Path, out_summary_json: str | Path, *, tail: int = 32
) -> str:
    """Write a deterministic summary for ``metrics_jsonl``."""

    metrics_path = Path(metrics_jsonl)
    out_path = Path(out_summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[Mapping[str, object]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    summary = _build_summary(metrics_path, records, tail)
    out_path.write_text(json.dumps(summary, sort_keys=True, indent=2))
    return str(out_path)


__all__ = ["compute_auc", "write_summary"]
