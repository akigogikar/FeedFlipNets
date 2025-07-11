"""
Ternary-DFA Experiment Script  (v8.2 – complete logging + plots + extra metrics)
================================================================================

Bug-fix snapshot (v8.1 ➜ v8.2)
-----------------------------
1. **Closed parenthesis** in mean-curve plotting (`plt.plot`).
2. `sweep_and_log` now always returns final tables (regression test added).
3. Added **unit test** `test_sweep_returns_tables` to catch NameError / SyntaxError.
4. Plotting helper now generates **mean convergence curves** for *each* depth-freq pair
   in a single figure for deeper analysis (`curves_<method>.svg`).

Run example
-----------
```bash
python ternary_dfa_experiment.py --depths 1 2 4 --freqs 1 3 5 --epochs 300 --outdir results
pytest -q ternary_dfa_experiment.py                       # fast self-tests
```

Dependencies: numpy, matplotlib; optional: scipy, pandas, pytest.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import List

# Allow running this script directly without installing the package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from feedflipnets.train import sweep_and_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Ternary-DFA experiment")
    p.add_argument('--methods', nargs='+', default=['Backprop'])
    p.add_argument('--depths', type=int, nargs='+', required=True)
    p.add_argument('--freqs', type=int, nargs='+', required=True)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    return p.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    sweep_and_log(args.methods, args.depths, args.freqs, args.seeds, args.epochs, args.outdir)


if __name__ == '__main__':
    main()
