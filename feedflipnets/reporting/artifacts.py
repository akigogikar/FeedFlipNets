"""Experiment artifact helpers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Mapping


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # pragma: no cover - git may be unavailable in tests
        return "unknown"


def write_manifest(
    path: str | Path,
    *,
    config: Mapping[str, object],
    dataset_provenance: Mapping[str, object],
) -> str:
    """Write a manifest JSON file capturing reproducibility metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "dataset": dict(dataset_provenance),
        "environment": {
            "python": os.environ.get("PYTHON_VERSION", "unknown"),
            "offline": os.environ.get("FEEDFLIP_DATA_OFFLINE", "0"),
        },
    }
    path.write_text(json.dumps(manifest, indent=2))
    return str(path)

