"""Metrics sinks for experiment tracking."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Mapping


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:  # pragma: no cover - git may be unavailable in tests
        return "unknown"


class JsonlSink:
    """Append-only JSONL writer for metrics."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str = "train",
        seed: int | None = None,
        sha: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.split = split
        self.seed = seed
        self.sha = sha or _git_sha()

    def on_step(self, step: int, metrics: Mapping[str, float]) -> None:
        record = {
            "step": int(step),
            "split": self.split,
            "loss": float(metrics.get("loss", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "seed": self.seed,
            "sha": self.sha,
        }
        record.update(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    __call__ = on_step


class CsvSink:
    """Write metrics to CSV with a stable schema."""

    def __init__(self, path: str | Path, *, split: str = "train") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.split = split

    def on_step(self, step: int, metrics: Mapping[str, float]) -> None:
        row = {
            "step": int(step),
            "split": self.split,
            "loss": float(metrics.get("loss", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
        }
        row.update(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        )
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            fieldnames = sorted(row.keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
