# FeedFlipNets

![CI](https://github.com/akigogikar/FeedFlipNets/actions/workflows/ci.yml/badge.svg)

**FeedFlipNets – DFA-trained Ternary Networks for Edge-Ready, Low-Precision Deep Learning.**

## What is FeedFlipNets?

FeedFlipNets is a lightweight research library for DFA-trained ternary neural
networks. It combines Direct Feedback Alignment—which replaces symmetric weight
transport with fixed random feedback matrices—with ternary weight quantisation
{−1, 0, +1} to deliver low-precision, hardware-friendly training without full
backprop’s constraints. Use it to prototype edge-ready models, study
biologically plausible learning, and compare DFA vs. BP under aggressive
quantisation. “Flip” refers strictly to ternary weight states in the forward
path; feedback matrices stay fixed and random as dictated by DFA.

## Why it matters

- **Edge ML efficiency.** Ternary weights slash memory and multiply-accumulate
  cost, while DFA avoids weight symmetry in the backward pass—ideal for
  neuromorphic or on-device training explorations.
- **Research credibility.** DFA is a published alternative to backprop capable
  of learning in deep networks, and ternary networks are a well studied
  compression method. FeedFlipNets unifies both for practical experimentation.

## How “Flip” works

- Maintain float “shadow” weights `V` for optimisation.
- Quantise forward weights `W = Qτ(V)` to {−1, 0, +1} on a schedule
  (`per_step` or `per_epoch`).
- Backward signals rely on DFA via fixed random matrices `B_l`.
- Gradients update the float weights; the flip schedule refreshes the ternary
  forward path.

## Not what it is

- ❌ Not “sign-flipped feedback.”
- ✅ “Flip” refers to ternary weight states; feedback remains random fixed
  matrices (DFA).

The library bundles deterministic offline fixtures, manifest-driven artefacts,
and a CLI so that classification, regression, and text workloads can run
repeatably on a laptop or in CI without network access.

**Keywords:** Direct Feedback Alignment, DFA, feedback alignment, ternary weight
networks, binary/ternary neural networks, quantised neural networks,
low-precision training, edge AI, neuromorphic learning, hardware-friendly deep
learning.

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
| Vision (MNIST)      | `make run PRESET=mnist_mlp_dfa`                               | Two hidden layers, DFA, ternary flip per-step |
| Time series (UCR)   | `make run PRESET=ucr_gunpoint_mlp_dfa`                        | GunPoint offline fixture, DFA |
| Tabular regression  | `make run PRESET=california_housing_mlp_dfa`                  | California Housing synthetic regression, flip off |
| Text (20 Newsgroups)| `make run PRESET=20newsgroups_bow_mlp_dfa`                    | HashingVectorizer features, DFA |

Presets declare dataset parameters, model topology, feedback strategy, flip
mode, optimiser, learning rate, epochs, batch size, and evaluation cadence. All
runs write manifests, JSONL metrics, CSV summaries, and checkpoints inside the
configured `train.run_dir`.

### CLI flag summary

- `--feedback {backprop, dfa, ternary_dfa, structured}`
- `--flip {off, ternary}` with `--flip-schedule {per_step, per_epoch}`
- `--flip-threshold τ` to control the ternary quantisation boundary

### Sweeps in one command

Grid sweeps over feedback strategy, flip options, learning rate, and hidden
widths can be launched with:

```bash
python -m scripts.preset_sweep --preset mnist_mlp_dfa \
  --feedback backprop dfa ternary_dfa \
  --flip off ternary \
  --flip-schedule per_step per_epoch \
  --lr 0.1 0.01 \
  --hidden 128 256
```

The script clones the preset configuration, adjusts the requested knobs, and
stores each run under `runs/sweeps/<preset>/feedback-.../`.

## FAQ

### Does FeedFlip flip the feedback weights?

No. “Flip” is forward-path ternarisation only. The feedback path is DFA with
fixed random matrices.

### Why ternary instead of binary?

Ternary weights strike a better accuracy/efficiency trade-off than pure binary,
while keeping most of the compute and memory savings.

### Is this credible beyond toy tasks?

DFA has been shown to learn in deep architectures and has modern variants;
ternary and binary networks have extensive literature and real hardware appeal.
FeedFlipNets unifies both for practical experimentation.

## Straight talk on trade-offs

You’re optimising for training simplicity and hardware realism, not SOTA
accuracy. Expect more epochs vs. BP in some settings—the known cost of DFA and
aggressive quantisation. The win is a clean experimental surface to study
low-precision learning and error-transport alternatives with minimal moving
parts.

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

## Losses, metrics, and flip support

| Task type      | Default loss (`--loss auto`) | Default metrics        | Flip mode (`--flip`) | Flip schedule (`--flip-schedule`) | Feedback strategies (`--feedback`) |
| -------------- | ---------------------------- | ---------------------- | -------------------- | --------------------------------- | ---------------------------------- |
| Regression     | MSE                          | `mae`, `mse`           | `off`, `ternary`     | `off`, `per_epoch`, `per_step`    | `backprop`, `dfa`, `ternary_dfa`   |
| Multiclass     | Cross-entropy                | `accuracy`, `macro_f1` | `off`, `ternary`     | `off`, `per_step`                 | `backprop`, `dfa`, `ternary_dfa`   |
| Binary         | BCE                          | `accuracy`, `auc`      | `off`, `ternary`     | `off`, `per_step`                 | `backprop`, `dfa`, `ternary_dfa`   |

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
