import subprocess, sys
from pathlib import Path

def test_bench_micro_runs_quickly(tmp_path):
    out = tmp_path / "bench"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "scripts/bench_micro.py", "--seeds","123","--steps","8","--out", str(out)])
    md = (out / "bench_micro.md").read_text(encoding="utf-8")
    assert "| BP |" in md and "| DFA |" in md and "| FLIP |" in md
