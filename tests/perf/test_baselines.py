import json
import time
from pathlib import Path

import pytest

from feedflipnets.training import pipelines


@pytest.mark.perf
def test_registry_baseline_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    config = {
        "data": {"name": "synth_fixture", "options": {"length": 64, "seed": 123}},
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
            "steps_per_epoch": 32,
            "batch_size": 4,
            "seed": 123,
            "lr": 0.05,
            "run_dir": str(tmp_path / "run"),
            "enable_plots": False,
        },
        "offline": True,
    }

    start = time.perf_counter()
    result = pipelines.run_pipeline(config)
    duration = time.perf_counter() - start

    assert duration <= 2.0
    assert result.steps <= 64

    metrics = Path(result.metrics_path)
    summary = Path(result.summary_path)
    assert metrics.exists()
    assert summary.exists()

    metrics_data = [
        json.loads(line) for line in metrics.read_text().splitlines() if line
    ]
    assert metrics_data, "metrics should not be empty"
