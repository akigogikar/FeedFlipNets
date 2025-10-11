"""Built-in dataset loaders.

This module exists for backwards compatibility. Importing it ensures that legacy
dataset registrations remain available.
"""

from . import (  # noqa: F401
    mnist,
    synth_fixture,
    synthetic,
    tinystories,
    ucr_uea,
)

__all__ = ["mnist", "synth_fixture", "synthetic", "tinystories", "ucr_uea"]
