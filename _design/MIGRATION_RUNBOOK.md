# Migration Runbook – FeedFlipNets v1.0

## Objective
Deliver FeedFlipNets v1.0 with modular packages, offline-first datasets, and CI
coverage while keeping legacy entrypoints functional.

## Phase 0 – Preparation
- Branch from the default branch (`work`) into `restructure/v1`.
- Freeze dependencies in `requirements-lock.txt`; ensure all tooling (pytest,
  pytest-cov, ruff, black, import-linter) is pinned.
- Stand up `.github/workflows/ci.yml` with the matrix (Python 3.10/3.11) calling
  `make bootstrap`, `make lint`, `make test`, and `make smoke`.
- Owners: Platform Engineer (CI), Tech Lead (lockfile).
- Rollback: Revert branch; retain previous workflow if bootstrapping fails.

## Phase 1 – Package extraction
- Create `feedflipnets.core.quant`, `.strategies`, `.types` with the
  `FeedbackStrategy` protocol (`init`/`backward`).
- Move dataset registry + loaders under `feedflipnets.data`, introducing the
  cache manifest with checksum validation and offline builders.
- Refactor `feedflipnets.training.trainer` into the new `Trainer` abstraction
  (constructor + `run`) and expose pipelines wiring model/strategy/data.
- Add reporting sinks (`JsonlSink`, `CsvSink`, `PlotAdapter`) gated for headless
  execution.
- Owners: Research Engineer (core/training), Data Engineer (registry/cache).
- Rollback: Restore monolithic trainer, keep new modules behind feature flag.

## Phase 2 – Legacy shims & determinism
- Update `feedflipnets.train` to delegate to pipelines while emitting
  `DeprecationWarning`s; replace `np.trapz` with `np.trapezoid` and cover with a
  regression test.
- Implement contract tests: strategy gradient shapes, cache manifest integrity,
  headless plotting, deterministic pipeline results, CLI smoke test.
- Ensure `make smoke` artefacts (metrics JSONL/CSV, manifest, plots) are stable
  across runs; document determinism guarantees.
- Owners: QA (tests), Research Engineer (shims).
- Rollback: keep legacy implementations callable, disable warnings.

## Phase 3 – Documentation & release readiness
- Refresh README, CHANGELOG, UPGRADING, architecture docs (Mermaid dependency
  graph) and quality audit snapshot.
- Provide migration guidance for preset/CLI changes and offline-first behaviour.
- Verify CI uploads smoke artefacts; confirm coverage threshold (≥75%).
- Owners: Tech Lead (docs), Platform Engineer (QA sign-off).
- Rollback: Revert docs if blockers discovered; maintain pre-migration docs in
  `_design/RESTRUCTURE_PLAN.md` for reference.

## Rollback strategy
- If a phase fails validation, revert commits in `restructure/v1` and restore the
  previous lockfile/CI workflow.
- Legacy shim remains operational, providing a safe fallback for scripts.
- Cached fixtures are pure local assets; purge `.cache/feedflip` when rolling
  back to avoid stale manifests.

## Communication
- Weekly status update in #feedflipnets with checklist progress and blockers.
- Demo deterministic smoke artefact to stakeholders before merge.
- Final sign-off requires CI green + QA acknowledgement on the contract tests.
