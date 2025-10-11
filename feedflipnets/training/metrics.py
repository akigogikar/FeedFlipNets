"""Metric helpers for the upgraded trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np

from ..core.types import Array


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float


def default_metrics(task_type: str, *, num_classes: int | None = None) -> List[str]:
    if task_type == "regression":
        return ["mae", "rmse", "r2"]
    if task_type == "multiclass":
        metrics = ["accuracy"]
        if num_classes and num_classes <= 20:
            metrics.append("macro_f1")
        return metrics
    if task_type in {"binary", "multilabel"}:
        return ["accuracy", "precision", "recall", "f1"]
    raise ValueError(f"Unknown task type: {task_type}")


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metric(
    name: str,
    predictions: Array,
    targets: Array,
    *,
    task_type: str,
    num_classes: int | None = None,
) -> MetricResult:
    key = name.lower()
    preds = predictions
    targs = targets
    if key == "mae":
        value = float(np.mean(np.abs(preds - targs)))
    elif key == "rmse":
        value = float(np.sqrt(np.mean((preds - targs) ** 2)))
    elif key == "r2":
        mean = np.mean(targs, axis=0, keepdims=True)
        ss_res = float(np.sum((targs - preds) ** 2))
        ss_tot = float(np.sum((targs - mean) ** 2))
        value = 1.0 if ss_tot == 0 else float(1 - ss_res / (ss_tot + 1e-9))
    elif key == "accuracy":
        if task_type == "multiclass":
            pred_idx = np.argmax(preds, axis=1)
            if targs.ndim == 2 and targs.shape[1] > 1:
                targ_idx = np.argmax(targs, axis=1)
            else:
                targ_idx = targs.reshape(-1).astype(int)
        else:
            prob = _sigmoid(preds)
            pred_idx = (prob >= 0.5).astype(int)
            targ_idx = targs.astype(int)
        value = float(np.mean(pred_idx == targ_idx))
    elif key == "macro_f1":
        if num_classes is None:
            raise ValueError("macro_f1 requires num_classes")
        pred_idx = np.argmax(preds, axis=1)
        targ_idx = (
            np.argmax(targs, axis=1)
            if targs.ndim == 2 and targs.shape[1] == num_classes
            else targs.reshape(-1).astype(int)
        )
        f1_scores = []
        for cls in range(num_classes):
            tp = np.sum((pred_idx == cls) & (targ_idx == cls))
            fp = np.sum((pred_idx == cls) & (targ_idx != cls))
            fn = np.sum((pred_idx != cls) & (targ_idx == cls))
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            f1_scores.append(f1)
        value = float(np.mean(f1_scores))
    elif key in {"precision", "recall", "f1"}:
        prob = _sigmoid(preds)
        pred_idx = (prob >= 0.5).astype(int)
        targ_idx = targs.astype(int)
        tp = float(np.sum((pred_idx == 1) & (targ_idx == 1)))
        fp = float(np.sum((pred_idx == 1) & (targ_idx == 0)))
        fn = float(np.sum((pred_idx == 0) & (targ_idx == 1)))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        if key == "precision":
            value = float(precision)
        elif key == "recall":
            value = float(recall)
        else:
            value = float(2 * precision * recall / (precision + recall + 1e-9))
    else:
        raise KeyError(f"Unknown metric: {name}")
    return MetricResult(name=key, value=value)


def compute_metrics(
    names: Iterable[str],
    predictions: Array,
    targets: Array,
    *,
    task_type: str,
    num_classes: int | None = None,
) -> Mapping[str, float]:
    results: Dict[str, float] = {}
    for name in names:
        metric = compute_metric(
            name, predictions, targets, task_type=task_type, num_classes=num_classes
        )
        results[metric.name] = metric.value
    return results


__all__ = ["MetricResult", "default_metrics", "compute_metrics"]
