from pathlib import Path

from cli.main import main


def test_cli_basic_preset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FEEDFLIP_DATA_OFFLINE", "1")
    main(["--preset", "basic_dfa_cpu"])
    run_dir = Path("runs/basic-dfa-cpu")
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
