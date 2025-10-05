# FeedFlipNets

## Project Overview
FeedFlipNets implements "flip" feedback alignment: networks update ternary weights
in {-1, 0, 1} while projecting output errors to hidden layers with fixed feedback
matrices. The vNext architecture splits the project into well-defined packages so
that training strategies, datasets and reporting utilities can be combined
modularly. Deterministic pipelines and offline fixtures make the experiments
reproducible without network access.

### Key Capabilities
- **Core primitives** – pure NumPy activations, ternary quantisation helpers and
  feedback-alignment strategies.
- **Data layer** – cache-first dataset registry with offline shards for MNIST,
  TinyStories and UCR/UEA samples.
- **Training pipelines** – configurable trainer that applies post-update
  quantisation and pluggable feedback strategies.
- **Reporting** – JSONL metrics sink, optional headless-safe plotting and run
  manifests with git/version metadata.

## Installation
FeedFlipNets targets Python 3.10+. The dependency lock file is the single source
of truth:

```bash
pip install -r requirements-lock.txt
```

For editable installs:

```bash
pip install -e .
```

## Run Modes
FeedFlipNets is designed for automation-first, reproducible runs. The following
modes are supported:

- **Offline datasets** – set `FEEDFLIP_DATA_OFFLINE=1` to force loaders to use
  bundled fixtures. No network access is attempted in this mode.
- **Headless plotting** – plotting is disabled by default. Enable it per run with
  `"enable_plots": true` in the training config or `--enable-plots` once exposed.
- **Deterministic metrics** – fixed seeds plus deterministic JSONL timestamps
  ensure repeated runs produce identical first 20 metric entries (within 1e-7).
- **Structured feedback** – orthogonal, Hadamard, block-diagonal and low-rank
  feedback matrices are reproducible under fixed seeds; refresh policies
  (`fixed`, `per_step`, `per_epoch`) control when matrices are regenerated.

## Quickstart
Experiments are executed through the unified CLI:

```bash
python -m cli.main --preset synthetic-min
```

Available presets:

- `synthetic-min` – 200 deterministic steps on an in-memory sinusoid.
- `mnist-flip-det` – flip-strategy ternary run using the MNIST fixture.
- `tinystories-dfa-stoch` – DFA strategy with stochastic ternary quantisation.
- `synthetic-structured-orthogonal-fixed` – structured feedback with fixed
  orthogonal refresh on a medium synthetic dataset.
- `synthetic-structured-hadamard-perstep` – Hadamard-structured feedback that
  regenerates per step for a stochastic baseline.
- `depth-frequency-sweep` – automation wrapper that reuses the new pipeline for
  depth/update-frequency sweeps.

Run details are written to `runs/<preset>/` (or a custom `run_dir`). The JSONL
metrics file contains `{step, ts, metric_name: value}` entries; the manifest in
the same directory records the git SHA, dataset provenance and seeds.

## Module Map
The repository is organised as follows:

- `feedflipnets/core/`
  - `types.py` – shared type aliases and `FeedbackStrategy` protocol.
  - `activations.py` – pure NumPy activations (e.g. ReLU, Hadamard padding).
  - `quantization.py` – ternary helpers including deterministic and stochastic
    quantisers.
  - `feedback.py` – DFA, flip-feedback, structured matrices and backprop-lite
    implementations.
- `feedflipnets/data/`
  - `cache_manager.py` – cache.fetch with offline fixtures, checksum validation
    and retries.
  - `registry.py` – dataset registry returning `DatasetSpec` objects with
    provenance metadata.
  - `loaders/` – synthetic, MNIST, TinyStories and UCR/UEA loaders that operate
    offline-first.
- `feedflipnets/training/`
  - `trainer.py` – core training loop applying feedback strategies and ternary
    quantisation.
  - `pipelines.py` – assembles data, model configs, reporting adapters and
    exposes presets.
- `feedflipnets/reporting/`
  - `metrics.py` – JSONL sink with deterministic timestamps.
  - `plots.py` – optional matplotlib plotting gated behind `enable_plots`.
  - `artifacts.py` – manifest writer capturing git SHA, seeds and dataset
    fingerprints.
- `cli/main.py` – single entrypoint CLI orchestrating presets and overrides.
- `feedflipnets/train.py` – deprecated shim that forwards legacy calls to the
  new pipeline and emits a deprecation warning.

## Development & Testing
Convenience targets are provided via `make`:

```bash
make setup      # install dependencies
make test       # run unit tests
make lint       # run import-linter contracts
make smoke      # smoke test synthetic preset
```

Manual commands:

```bash
pip install -r requirements-lock.txt
pytest
lint-imports
python -m cli.main --preset synthetic-min
```

Set `FEEDFLIP_DATA_OFFLINE=1` in CI or local shells to ensure offline fixtures
are used. The CLI defaults to headless mode, so matplotlib is only imported when
plots are explicitly enabled.

## Reference & License
FeedFlipNets is released under the MIT License. See `LICENSE` for details.
