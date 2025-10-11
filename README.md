# FeedFlipNets

![CI](https://github.com/akigogikar/FeedFlipNets/actions/workflows/ci.yml/badge.svg)

FeedFlipNets implements deterministic feedback-alignment experiments with ternary
weights. The repository is structured into four focused packages—`core`, `data`,
`training`, and `reporting`—with a thin CLI that wires them together. All
pipelines are offline-first and produce reproducible metrics/artefacts suitable
for publication workflows.

## 90-second first run

```bash
# 1. Create the virtualenv and install pinned dependencies
make bootstrap

# 2. Run the lint + test suite (offline)
make lint
make test

# 3. Execute the offline smoke preset and inspect outputs
make smoke
ls runs/basic-dfa-cpu
```

`make smoke` writes `metrics_train.jsonl`, `metrics_val.jsonl`,
`metrics_test.json`, split CSV files, checkpoints, and `manifest.json` under
`runs/basic-dfa-cpu/`. Running the command twice yields identical metrics hashes
thanks to deterministic seeds.

## CLI usage

The CLI defaults to offline mode and the `basic_dfa_cpu` preset:

```bash
python -m cli.main --preset basic_dfa_cpu --dump-config tmp/config.json
python -m cli.main --preset synthetic-min --enable-plots --offline
```

Optional `--config` overrides accept JSON or YAML files. When the override
contains complete `data`/`model`/`train` sections it is treated as a standalone
configuration; otherwise it patches the selected preset. The CLI exports the
resolved configuration, including the offline flag, to the pipeline. When
`FEEDFLIP_DATA_OFFLINE=1` (default) no network calls are attempted; fixtures are
generated locally via the cache manifest.

### Training & evaluation controls

The upgraded trainer is modality-aware and automatically selects sensible
defaults for each dataset type. You can override the behaviour from the CLI:

- `--loss {auto,mse,mae,huber,ce,bce}` selects the loss function (auto uses the
  dataset task type).
- `--metrics default` or a comma-separated list (e.g. `accuracy,macro_f1`) drives
  the per-split evaluation.
- `--feedback {flip,dfa,structured,backprop,ternary_dfa}` swaps feedback
  strategies while keeping the legacy names available.
- `--ternary {off,per_step,per_epoch}` and `--ternary-threshold` control the
  quantisation schedule.
- `--eval-every` and `--early-stopping-patience` enable validation-driven early
  stopping and periodic evaluation.

Each run prints a concise startup summary with dataset, inferred dimensions,
loss/metric selections, feedback strategy, ternary mode, and parameter count.

### Datasets & offline fixtures

FeedFlipNets exposes a unified dataset registry via `feedflipnets.data`. The
built-in loaders cover MNIST, UCR/UEA time-series (defaulting to GunPoint), the
California Housing regression task, 20 Newsgroups text classification, and
generic CSV regression/classification adapters. Every loader supports
deterministic `train`/`val`/`test` splits, `seed` overrides, and an offline mode
powered by deterministic synthetic fixtures (CSV helpers ship text fixtures in
`feedflipnets/data/_fixtures/`). To explore a dataset without touching presets:

```bash
python -m cli.main --dataset mnist --offline --val-split 0.05 --test-split 0.1
python -m cli.main --dataset ucr --ucr-name GunPoint --seed 123
python -m cli.main --dataset csv_regression --csv-path feedflipnets/data/_fixtures/csv_regression_fixture.csv
```

Set `FFN_CACHE_DIR=...` to control the download cache location and
`FFN_DATA_OFFLINE=0` to allow network fetches for the real datasets. The helper
script `scripts/smoke_datasets.sh` runs a fast offline sanity check across all
registered datasets.

### Run an experiment from the registry

```bash
python -m cli.main --experiment dfa_baseline
python -m cli.main --experiment dfa_baseline  # identical metrics + summary bytes
```

Registry-backed runs derive a deterministic `run_id` from the configuration
hash. Artefacts are written to `.artifacts/<run_id>/` with
`metrics.jsonl`, `summary.json`, and the manifest. Re-running the same
experiment reuses the directory and produces byte-identical outputs.

### Build a paper bundle

After an experiment completes, generate a reproducible archive:

```bash
python scripts/build_paper_bundle.py --run-dir .artifacts/<run_id> --include-plots
```

The script copies metrics, recomputes a deterministic summary, materialises
CSV tables, renders optional plots using the Agg backend, and writes a
`paper_bundle.zip` alongside the `paper_bundle/` directory for upload.

## Module map

```text
feedflipnets/
  core/        -> numerical primitives (activations, quant, strategies, types)
  data/        -> dataset registry, cache manifest, offline loaders
  training/    -> trainer abstraction, pipelines, schedulers
  reporting/   -> metrics sinks (JSONL/CSV), headless plotting, manifests
  cli/         -> argparse entrypoint with presets and config overrides
```

Key architecture rules:

- `core` has no dependencies on other packages.
- `data` depends on `core` only; loaders rely on the cache manifest for offline
  provenance.
- `training` orchestrates `core`, `data`, and `reporting` components.
- `reporting` consumes `core` utilities only, ensuring headless execution.

See [`_design/ARCHITECTURE_TARGET.md`](./_design/ARCHITECTURE_TARGET.md) for the
full dependency matrix and Mermaid diagram.

## Development workflow

- **Install** – `make bootstrap`
- **Lint** – `make lint` (Ruff + Black + Import Linter contracts)
- **Test** – `make test` (offline, coverage ≥ 75%)
- **Smoke** – `make smoke` (offline deterministic preset)

The dependency lock file (`requirements-lock.txt`) is the single source of truth
for tooling. All tests and smoke runs set `FEEDFLIP_DATA_OFFLINE=1` to prevent
network calls.

## Reporting artefacts

`feedflipnets.reporting.metrics.JsonlSink` records per-epoch metrics for each
split (`metrics_train.jsonl`, `metrics_val.jsonl`, `metrics_test.jsonl`). The
`CsvSink` mirrors the schema for spreadsheet workflows. A compact
`metrics_test.json` captures the last test metrics, while `best.ckpt` and
`last.ckpt` store model weights for reproducibility. `PlotAdapter` writes loss
curves using the Agg backend so that CI remains headless. Manifests include Git
SHA, dataset provenance (with checksums), and runtime environment details.

## Legacy shims

`feedflipnets.train` exposes `train_single` and `sweep_and_log` for backwards
compatibility. The functions forward to the new pipeline and emit
`DeprecationWarning`s with upgrade guidance. Modern consumers should rely on the
CLI or call `feedflipnets.training.pipelines.run_pipeline` directly.

## Further reading

- [`_design/MIGRATION_RUNBOOK.md`](./_design/MIGRATION_RUNBOOK.md) – phased
  rollout plan with rollback steps.
- [`UPGRADING.md`](./UPGRADING.md) – guidance for migrating legacy scripts.
- [`CHANGELOG.md`](./CHANGELOG.md) – highlights for the v1.0.0 release cycle.
