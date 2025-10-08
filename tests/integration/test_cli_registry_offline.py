import json
from pathlib import Path

from cli import main as cli_main
from feedflipnets.experiments import registry as exp_registry


def test_cli_experiment_runs_are_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")

    cli_main.main(["--experiment", "dfa_baseline"])
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "CLI should emit at least one line"
    first_payload = json.loads(captured[-1])

    metrics_path = Path(first_payload["metrics"])
    summary_path = Path(first_payload["summary"])
    assert metrics_path.exists()
    assert summary_path.exists()
    first_metrics = metrics_path.read_bytes()
    first_summary = summary_path.read_bytes()

    cli_main.main(["--experiment", "dfa_baseline"])
    captured = capsys.readouterr().out.strip().splitlines()
    second_payload = json.loads(captured[-1])

    assert first_payload["run_id"] == second_payload["run_id"]
    assert Path(second_payload["metrics"]).read_bytes() == first_metrics
    assert Path(second_payload["summary"]).read_bytes() == first_summary

    experiment = exp_registry.get_experiment("dfa_baseline")
    expected_run_id = exp_registry.config_hash(experiment.to_pipeline_config())
    assert first_payload["run_id"] == expected_run_id
    assert metrics_path.parent == Path(".artifacts") / expected_run_id
