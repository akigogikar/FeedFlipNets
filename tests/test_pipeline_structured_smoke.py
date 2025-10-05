import json
import subprocess
import sys


def test_cli_preset_structured_runs():
    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "--preset", "synthetic-structured-orthogonal-fixed"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["steps"] >= 300
