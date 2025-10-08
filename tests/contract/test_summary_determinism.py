from pathlib import Path

from feedflipnets.training import pipelines


def test_summary_outputs_are_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    config = {
        "data": {"name": "synth_fixture", "options": {"length": 64, "seed": 123}},
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
            "steps_per_epoch": 32,
            "batch_size": 4,
            "seed": 55,
            "lr": 0.05,
            "run_dir": str(tmp_path / "run_a"),
            "enable_plots": False,
        },
        "offline": True,
    }

    first = pipelines.run_pipeline(config)
    summary_a = Path(first.summary_path).read_bytes()
    metrics_a = Path(first.metrics_path).read_bytes()

    config["train"]["run_dir"] = str(tmp_path / "run_b")
    second = pipelines.run_pipeline(config)
    summary_b = Path(second.summary_path).read_bytes()
    metrics_b = Path(second.metrics_path).read_bytes()

    assert metrics_a == metrics_b
    assert summary_a == summary_b
