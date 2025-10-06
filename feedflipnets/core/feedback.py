"""Compatibility shim for legacy imports."""

from __future__ import annotations

import warnings

from . import strategies as _strategies
from .strategies import *  # noqa: F401,F403

warnings.warn(
    "feedflipnets.core.feedback is deprecated; use feedflipnets.core.strategies instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _strategies.__all__
