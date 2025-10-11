"""Built-in dataset loaders.

This module exists for backwards compatibility. Importing it ensures that legacy
dataset registrations remain available.
"""

from . import mnist, synth_fixture, synthetic, tinystories, ucr_uea  # noqa: F401

__all__ = ["mnist", "synth_fixture", "synthetic", "tinystories", "ucr_uea"]
