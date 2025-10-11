"""Compatibility wrapper for the experiment entry point.

Historically :mod:`ternary_dfa_experiment` exposed helper functions such as
``make_dataset`` and ``sweep_and_log``.  The implementation now lives under
``experiments/ternary_dfa_experiment.py`` but tests (and user code) may still
import these utilities from this top level module.  To keep backwards
compatibility we re-export the relevant functions and ``main``.
"""

from experiments.ternary_dfa_experiment import main, parse_args, sweep_and_log
from feedflipnets.utils import make_dataset

__all__ = ["main", "parse_args", "sweep_and_log", "make_dataset"]

if __name__ == "__main__":
    main()
