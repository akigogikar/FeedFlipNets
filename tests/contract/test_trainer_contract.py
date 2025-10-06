import json
from pathlib import Path

from feedflipnets.training import pipelines


def test_trainer_pipeline_produces_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    config = {
        "data": {
            "name": "synthetic",
            "options": {"freq": 2, "n_points": 64, "seed": 0},
        },
        "model": {
            "d_in": 1,
            "d_out": 1,
            "hidden": [4],
            "quant": "det",
            "tau": 0.05,
            "strategy": "flip",
        },
        "train": {
            "epochs": 1,
            "steps_per_epoch": 20,
            "batch_size": 4,
            "seed": 11,
            "lr": 0.05,
            "run_dir": str(tmp_path / "run"),
            "enable_plots": False,
        },
    }

    result = pipelines.run_pipeline(config)
    assert Path(result.metrics_path).exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["config"]["train"]["seed"] == 11
    dataset_meta = manifest["dataset"]
    mode = dataset_meta.get("mode")
    if mode is not None:
        assert mode in {"offline", "offline-fallback", "cache"}
    else:
        fallback = dataset_meta.get("name") or dataset_meta.get("type")
        assert fallback == "synthetic"

    metrics = [
        json.loads(line)
        for line in Path(result.metrics_path).read_text().splitlines()
        if line
    ]
    assert metrics, "metrics should not be empty"
    first = metrics[0]
    assert "split" in first and first["split"] == "train"
    assert "sha" in first
    assert "seed" in first
    assert all("loss" in entry for entry in metrics)

    csv_path = Path(config["train"]["run_dir"]) / "metrics.csv"
    assert csv_path.exists()


def test_pipeline_determinism(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    base_config = {
        "data": {
            "name": "synthetic",
            "options": {"freq": 2, "n_points": 64, "seed": 0},
        },
        "model": {
            "d_in": 1,
            "d_out": 1,
            "hidden": [4],
            "quant": "det",
            "tau": 0.05,
            "strategy": "flip",
        },
        "train": {
            "epochs": 1,
            "steps_per_epoch": 15,
            "batch_size": 4,
            "seed": 99,
            "lr": 0.05,
            "run_dir": str(tmp_path / "run1"),
            "enable_plots": False,
        },
    }

    first = pipelines.run_pipeline(base_config)
    metrics_1 = Path(first.metrics_path).read_text()

    base_config["train"]["run_dir"] = str(tmp_path / "run2")
    second = pipelines.run_pipeline(base_config)
    metrics_2 = Path(second.metrics_path).read_text()

    assert metrics_1 == metrics_2
