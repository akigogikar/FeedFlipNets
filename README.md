# FeedFlipNets

![CI](https://github.com/akigogikar/FeedFlipNets/actions/workflows/ci.yml/badge.svg)

FeedFlipNets is a compact research rig for experimenting with feedback-alignment
(DFA), ternary quantisation, and classic backpropagation across multiple data
modalities. The project bundles deterministic offline fixtures, manifest-driven
artifacts, and a CLI so that classification, regression, and text workloads can
run repeatably on a laptop or in CI without network access.

## Quickstart

```bash
# 1. Install pinned dependencies into a virtual environment
make setup

# 2. Run formatting, linting, tests, and the offline smoke suite
make format lint test smoke

# 3. Launch a preset end-to-end run (defaults to offline fixtures)
make run PRESET=mnist_mlp_dfa
```

Each modality ships with an end-to-end preset under `configs/presets/`:

| Modality            | Command                                                      | Notes |
| ------------------- | ------------------------------------------------------------- | ----- |
| Vision (MNIST)      | `make run PRESET=mnist_mlp_dfa`                               | Two hidden layers, DFA, ternary per-step |
| Time series (UCR)   | `make run PRESET=ucr_gunpoint_mlp_dfa`                        | GunPoint offline fixture, DFA |
| Tabular regression  | `make run PRESET=california_housing_mlp_dfa`                  | California Housing synthetic regression, ternary off |
| Text (20 Newsgroups)| `make run PRESET=20newsgroups_bow_mlp_dfa`                    | HashingVectorizer features, DFA |

Presets declare dataset parameters, model topology, feedback strategy, ternary
mode, optimiser, learning rate, epochs, batch size, and evaluation cadence. All
runs write manifests, JSONL metrics, CSV summaries, and checkpoints inside the
configured `train.run_dir`.

### Sweeps in one command

Grid sweeps over feedback strategy, ternary schedule, learning rate, and hidden
widths can be launched with:

```bash
python -m scripts.preset_sweep --preset mnist_mlp_dfa \
  --feedback backprop dfa ternary_dfa \
  --ternary off per_step \
  --lr 0.1 0.01 \
  --hidden 128 256
```

The script clones the preset configuration, adjusts the requested knobs, and
stores each run under `runs/sweeps/<preset>/feedback-.../`.

## Dataset modes

FeedFlipNets datasets default to deterministic offline fixtures so smoke tests
never touch the network. Switching to the real datasets is as simple as passing
`--no-offline` or setting `FEEDFLIP_DATA_OFFLINE=0`.

| Dataset              | Offline fixture characteristics                               | Online toggle |
| -------------------- | ------------------------------------------------------------- | ------------- |
| `mnist`              | Linearly separable synthetic digits (10×64 samples)           | `--no-offline` downloads the classic MNIST archive |
| `ucr`                | Class-specific sinusoid time series with noise                | `--no-offline --ucr-name <dataset>` pulls from UCR/UEA |
| `california_housing` | Linear regression with Gaussian noise, standardised features  | `--no-offline` fetches via scikit-learn |
| `20newsgroups`       | Sparse hashing-vectorised pseudo-documents per class          | `--no-offline --text-subset train` streams from scikit-learn |

Additional CSV helpers and custom datasets are documented in
[`docs/how_to_add_dataset.md`](docs/how_to_add_dataset.md).

## Losses, metrics, and ternary support

| Task type      | Default loss (`--loss auto`) | Default metrics            | Ternary scheduling (`--ternary`) | Feedback strategies (`--feedback`) |
| -------------- | ---------------------------- | -------------------------- | -------------------------------- | ---------------------------------- |
| Regression     | MSE                          | `mae`, `mse`               | `off`, `per_epoch`, `per_step`  | `backprop`, `dfa`, `ternary_dfa`   |
| Multiclass     | Cross-entropy                | `accuracy`, `macro_f1`     | `off`, `per_step`               | `backprop`, `dfa`, `ternary_dfa`   |
| Binary         | BCE                          | `accuracy`, `auc`          | `off`, `per_step`               | `backprop`, `dfa`, `ternary_dfa`   |

The trainer chooses defaults based on the dataset metadata but every preset can
override them in its YAML/JSON. CLI flags mirror the config keys for interactive
experimentation.

## Reproducibility checklist

- Every preset specifies `train.seed`; the trainer derives deterministic child
  seeds for validation/test loaders and quantisation.
- `manifest.json` captures the resolved configuration, dataset provenance, and
  package versions so runs can be replayed later.
- Metrics are written as JSONL (`metrics_<split>.jsonl`) and CSV files alongside
  a snapshot of the last test metrics (`metrics_test.json`).
- `make smoke` replays all presets offline and is wired into CI, guaranteeing
  end-to-end coverage on clean clones.

## Tooling

- `make setup` — create a virtual environment, install pinned dependencies, and
  register pre-commit hooks.
- `make format` / `make lint` — run `isort`, `black`, `ruff`, and `flake8` over
  `cli/`, `feedflipnets/`, `scripts/`, and `tests/`.
- `make test` — execute the offline pytest suite.
- `make smoke` — execute all presets offline, writing metrics to `runs/`.
- `.pre-commit-config.yaml` — standardises formatting, linting, and whitespace
  hygiene before committing.

Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding standards and release
policies.
