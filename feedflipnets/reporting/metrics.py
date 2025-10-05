"""Metrics sinks for experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class JsonlSink:
    """Append-only JSONL writer for metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def on_step(self, step: int, metrics: Mapping[str, float]) -> None:
        record = {"step": int(step), "ts": float(step)}
        record.update({k: float(v) for k, v in metrics.items()})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    __call__ = on_step

