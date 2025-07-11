# FeedFlipNets


This repository provides minimal implementations of FeedFlipNets utilities.
Core modules live under `feedflipnets/` and experiment scripts under
`experiments/`.

Run the example experiment (package layout):

```bash
python experiments/ternary_dfa_experiment.py --depths 1 2 --freqs 1 3
```

The legacy entry point `python ternary_dfa_experiment.py` continues to work and
forwards to the script under `experiments/`.

