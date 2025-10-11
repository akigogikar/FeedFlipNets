"""UCR/UEA time-series dataset utilities."""

from __future__ import annotations

import os
import zipfile
import numpy as np
from typing import Tuple

from .utils import download_file, normalize

_BASE_URL = "http://www.timeseriesclassification.com/Downloads"


def _extract_dataset(name: str, root: str) -> str:
    zip_path = os.path.join(root, f"{name}.zip")
    download_file(f"{_BASE_URL}/{name}.zip", zip_path)
    extract_dir = os.path.join(root, name)
    if not os.path.isdir(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
    return extract_dir


def load_ucr(
    name: str, root: str = "datasets_cache"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from the UCR/UEA archive.

    Returns (X_train, y_train, X_test, y_test) arrays normalised per time-series.
    """
    path = _extract_dataset(name, root)
    train_path = os.path.join(path, f"{name}_TRAIN.txt")
    test_path = os.path.join(path, f"{name}_TEST.txt")

    def _load(p: str):
        data = np.loadtxt(p, delimiter=",")
        labels = data[:, 0].astype(int)
        series = data[:, 1:]
        series = normalize(series, axis=1)
        return series, labels

    X_train, y_train = _load(train_path)
    X_test, y_test = _load(test_path)
    return X_train, y_train, X_test, y_test
