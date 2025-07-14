# FeedFlipNets

## Project Overview
FeedFlipNets is a small research code base exploring **flip-style feedback** in neural networks. The package contains reference implementations of ternary direct feedback alignment (DFA) experiments along with data loaders and utilities for running small benchmarks. It accompanies the FeedFlipNets paper which describes the approach in detail.

### Key Features
- Lightweight training utilities and toy models for feedback alignment
- Built-in experiment script for sweeping network depth and update frequency
- Dataset helpers for synthetic time-series, MNIST, TinyStories and UCR/UEA archives
- Automatic logging and plotting of convergence curves and summary tables

## Installation
FeedFlipNets requires Python 3.8 or newer. Install the minimal dependencies with

```bash
pip install -r requirements.txt
```

Additional development tools such as `pytest` can be installed via the optional extras defined in `pyproject.toml`:

```bash
pip install -e .[dev]
```

## Getting Started
Experiments are launched via `ternary_dfa_experiment.py`. The general pattern is

```bash
python ternary_dfa_experiment.py --depths <d1 d2 ...> --freqs <f1 f2 ...> \
    --epochs <E> --outdir <results_dir>
```

### Examples
Run a small synthetic time-series sweep:

```bash
python ternary_dfa_experiment.py --depths 1 2 4 --freqs 1 3 5 --epochs 300 \
    --outdir results/timeseries
```

Use a dataset from the UCR/UEA archive:

```bash
python ternary_dfa_experiment.py --dataset ucr:GunPoint --depths 1 --freqs 1 \
    --epochs 10 --max-points 50 --outdir results/gunpoint
```

MNIST benchmark sweep:

```bash
python ternary_dfa_experiment.py --dataset mnist --depths 2 4 --epochs 20 \
    --outdir results/mnist
```

Short MNIST run used in tests:

```bash
python ternary_dfa_experiment.py --dataset mnist --depths 1 --freqs 1 \
    --epochs 1 --outdir results/mnist-mini --methods Backprop "Vanilla DFA" Momentum
```

TinyStories experiment:

```bash
python ternary_dfa_experiment.py --dataset tinystories --depths 2 4 --epochs 50 \
    --outdir results/tinystories
```

Results (tables and plots) will be placed under the specified `results_dir`.

## Module Guide
- `feedflipnets/` contains the core implementation:
  - `models.py` – simple feed-forward models and backprop utilities
  - `train.py` – training loop and experiment orchestration
  - `utils.py` – activation functions and helper routines
- `datasets/` provides loaders for several datasets:
  - `timeseries.py` – interface to the UCR/UEA archive
  - `mnist.py` – MNIST download and preprocessing
  - `tinystories.py` – TinyStories text dataset
  - `utils.py` – shared dataset utilities

## Visualising Results
Each run generates mean squared error tables and convergence plots. Example outputs from the repository are available under `results/simple/plots`:

![Heatmap](results/simple/plots/heat_Backprop.svg)

![Convergence curves](results/simple/plots/curves_Backprop.svg)

## Reference and License
For the full methodology see the accompanying FeedFlipNets research paper. This repository is released under the terms of the MIT License; see the `LICENSE` file for details.
