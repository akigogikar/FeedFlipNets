# Contributing to FeedFlipNets

Thank you for improving FeedFlipNets! This document summarises the engineering
expectations so experiments remain reproducible and easy to collaborate on.

## Environment setup

1. Create a virtual environment and install pinned dependencies:
   ```bash
   make setup
   ```
2. Activate the environment (`source .venv/bin/activate`) when developing.
3. Install the Git hooks registered by `make setup`; they mirror the checks run
   in CI and help keep diffs clean.

## Coding standards

- Format imports with `isort` and code with `black` (see `make format`).
- Lint with `ruff` and `flake8` (see `make lint`). Warnings introduced by a
  change must be fixed or explicitly justified in code comments.
- Type hints are required for new public functions. Prefer `from __future__ import
  annotations` to keep annotations lightweight.
- Avoid adding heavy dependencies. The core stack deliberately sticks to NumPy,
  SciPy, pandas, scikit-learn, and PyYAML.
- Do not wrap imports in `try`/`except` blocks unless guarding optional
  third-party packages.

## Tests and quality gates

- Run `make test` before submitting a pull request. The pytest suite must pass
  with `FEEDFLIP_DATA_OFFLINE=1`.
- Execute `make smoke` to ensure all presets complete end-to-end and write
  metrics under `runs/`.
- New features should include automated tests where practical. Regression tests
  belong under `tests/` and should avoid network calls.

## Documentation

- Update `README.md`, `CHANGELOG.md`, and relevant docs when behaviour changes.
- Reference new presets or datasets in `configs/presets/` and `docs/` so other
  contributors understand how to reproduce results.

## Commit and PR guidelines

- Keep commits focused; include context in the message body when touching
  multiple areas.
- All pull requests must describe how to reproduce the change (commands run,
  seeds used, and where outputs are written). The CI template enforces linting,
  tests, and smoke runs on Python 3.10 and 3.11.
- Target branches should remain rebased on the latest `main` (or `work` in this
  training environment) before requesting review.

Following these conventions keeps FeedFlipNets reproducible and friendly for new
experiments. Thanks again for contributing!
