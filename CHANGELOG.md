# Changelog

## [v1.0.0-rc1] - 2024-03-20
### Added
- Modularised the codebase into `core`, `data`, `training`, and `reporting`
  packages with a deterministic `Trainer` abstraction.【F:feedflipnets/training/trainer.py†L1-L134】
- Offline-first dataset registry with cache manifest, checksum validation, and
  deterministic offline builders.【F:feedflipnets/data/cache.py†L1-L132】
- Reporting sinks (JSONL/CSV) and headless plotting adapters decoupled from the
  training loop.【F:feedflipnets/reporting/metrics.py†L1-L63】【F:feedflipnets/reporting/plots.py†L1-L33】
- CLI presets with offline defaults, config overrides, and smoke artifact
  generation.【F:cli/main.py†L1-L89】
- GitHub Actions workflow running lint, tests (coverage gate), smoke test, and
  uploading artefacts for Python 3.10/3.11.【F:.github/workflows/ci.yml†L1-L33】

### Changed
- Deprecated legacy `train_single`/`sweep_and_log` in favour of pipeline-backed
  implementations emitting `DeprecationWarning`s.【F:feedflipnets/train.py†L1-L156】
- Replaced `np.trapz` usage with `np.trapezoid` to silence NumPy deprecation
  warnings.【F:feedflipnets/train.py†L122-L156】

### Removed
- Monolithic trainer logic and in-loop plotting; responsibilities are now split
  across the new packages and callbacks.

## [Unreleased]
- No unreleased changes.
