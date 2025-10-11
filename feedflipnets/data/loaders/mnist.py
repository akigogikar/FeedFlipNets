"""Compatibility wrapper for the new MNIST loader."""

from __future__ import annotations

from ..mnist import build_mnist  # noqa: F401

__all__ = ["build_mnist"]
