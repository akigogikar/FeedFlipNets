"""TinyStories dataset loader (HuggingFace)."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .utils import download_file


_DEF_PATH = os.path.join("datasets_cache", "tinystories.txt")
_HF_URL = "https://raw.githubusercontent.com/karpathy/tinygrad/master/extra/TinyStories-short.txt"


def load_tinystories(path: str = _DEF_PATH) -> Tuple[np.ndarray]:
    """Download TinyStories and return array of tokenized lines."""
    download_file(_HF_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # simple whitespace tokenisation
    tokens = text.split()
    arr = np.array(tokens)
    return arr
