#!/usr/bin/env bash
set -euo pipefail

export FFN_DATA_OFFLINE="1"

python - <<'PY'
from pathlib import Path

import numpy as np

from feedflipnets.data.registry import get_dataset

cache = Path(".cache/smoke-datasets")
cache.mkdir(parents=True, exist_ok=True)

fixtures = Path("feedflipnets/data/_fixtures")

datasets = [
    ("mnist", {}),
    ("ucr", {"ucr_name": "GunPoint"}),
    ("california_housing", {}),
    ("20newsgroups", {}),
    (
        "csv_regression",
        {"csv_path": str(fixtures / "csv_regression_fixture.csv"), "target_col": "target"},
    ),
    (
        "csv_classification",
        {"csv_path": str(fixtures / "csv_classification_fixture.csv"), "target_col": "target"},
    ),
]

for name, options in datasets:
    spec = get_dataset(name, offline=True, cache_dir=cache, **options)
    batch = next(spec.loader("train", 4))
    print(f"{name}: inputs={batch.inputs.shape} targets={batch.targets.shape} d_in={spec.data_spec.d_in} d_out={spec.data_spec.d_out}")
    assert batch.inputs.dtype == np.float32
    assert batch.targets.dtype == np.float32
PY
