# FeedFlipNets


FeedFlipNets is a minimal research code base exploring "flip" style feedback in neural networks. The project accompanies the FeedFlipNets paper and provides reference implementations for running small‐scale experiments with direct feedback alignment (DFA) and quantized weights.

The repository currently contains a single script `ternary_dfa_experiment.py` which sweeps over network depth and training frequency on a toy time‑series regression problem. Additional scripts (not included here) were used in the paper to benchmark the approach on MNIST and the TinyStories language dataset.

## Dependencies

The experiments require Python 3.8+ and a few scientific Python packages:

```
pip install -r requirements.txt
```

The `requirements.txt` file lists the minimal packages: `numpy`, `matplotlib`, `pandas`, `scipy` and `pytest` (used for quick sanity tests).

## Running experiments

The general pattern for an experiment is

```
python ternary_dfa_experiment.py --depths <d1 d2 ...> --freqs <f1 f2 ...> --epochs <E> --outdir <results_dir>
```

This will train networks for the specified depths and frequencies, store curves and summary tables under `<results_dir>` and produce plots in `<results_dir>/plots`.

### Time‑series example

```
python ternary_dfa_experiment.py --depths 1 2 4 --freqs 1 3 5 --epochs 300 --outdir results/timeseries
```

### MNIST sweep

```
python ternary_dfa_experiment.py --dataset mnist --depths 2 4 --epochs 20 --outdir results/mnist
```

### TinyStories sweep

```
python ternary_dfa_experiment.py --dataset tinystories --depths 2 4 --epochs 50 --outdir results/tinystories
```

## Reproducing paper benchmarks

To reproduce the main benchmarks from the paper:

1. Install the dependencies as above.
2. Run the time‑series sweep command to match the regression results.
3. Repeat for MNIST and TinyStories using the respective dataset options or scripts.
4. Compare the final tables and plots in the `results/*/` folders with the ones reported in the paper.


