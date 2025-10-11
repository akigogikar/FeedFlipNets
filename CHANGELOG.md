# Changelog

## [Unreleased]
- No unreleased changes.

## [1.0.0-rc1] - 2024-06-12
### Added
- Offline-first presets for MNIST, UCR GunPoint, California Housing, and
  20 Newsgroups under `configs/presets/`, plus a configurable sweep helper in
  `scripts/preset_sweep.py` for grid-searching feedback, flip modes and schedules,
  learning rates, and hidden sizes.
- GitHub Actions workflow covering Python 3.10/3.11 linting, formatting, tests,
  and preset smoke runs that upload metrics artifacts.
- Pre-commit configuration (`.pre-commit-config.yaml`), extended Makefile
  targets (`setup`, `format`, `lint`, `test`, `smoke`, `run`), and repository
  documentation (`README.md`, `CONTRIBUTING.md`, `docs/how_to_add_dataset.md`).
- Deterministic offline dataset fixtures for MNIST, UCR time series, and
  20 Newsgroups that achieve above-random accuracy during smoke runs.

### Changed
- `feedflipnets/training/pipelines.py` now discovers presets from
  `configs/presets/`, validates optimiser selections, and exposes them via the
  CLI.
- Dependency metadata updated with PyYAML, linting, and tooling packages to
  support the new presets and developer workflow.

### Fixed
- Offline dataset generators produce structured signals so loss decreases and
  accuracy exceeds random chance in regression and classification presets.
