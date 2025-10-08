# Changelog

## [Unreleased]
- No unreleased changes.

## [1.0.0-rc1] - 2024-03-20
### Added
- JSON experiment registry with schema validation, config hashing, and CLI
  integration that writes artefacts to `.artifacts/<run_id>/`.【F:experiments/registry.json†L1-L27】【F:feedflipnets/experiments/registry.py†L1-L157】【F:cli/main.py†L1-L132】
- Deterministic summary writer computing min/max/mean/last and tail AUC for
  metrics streams, plus contract tests verifying byte-identical outputs.【F:feedflipnets/reporting/summary.py†L1-L87】【F:tests/contract/test_summary_determinism.py†L1-L34】
- Paper bundle generator with optional plots, CSV tables, methods stub, and
  reproducible zip archives for publication workflows.【F:scripts/build_paper_bundle.py†L1-L140】
- Performance baseline test suite and Makefile targets (`perf`, `bundle`,
  `release-rc`) to exercise micro benchmarks locally.【F:tests/perf/test_baselines.py†L1-L43】【F:Makefile†L1-L28】
- Reproducibility documentation and formal citation metadata for the release
  candidate.【F:docs/reproducibility.md†L1-L52】【F:CITATION.cff†L1-L7】

### Changed
- CLI now supports registry experiments, full-config execution, and deterministic
  run directories while still honouring preset overrides.【F:cli/main.py†L1-L132】
- Training pipelines emit deterministic `summary.json` files and expose the
  summary path on `RunResult` for downstream tooling.【F:feedflipnets/training/pipelines.py†L1-L287】【F:feedflipnets/core/types.py†L1-L38】
- CI adds a determinism workflow that reruns registry experiments, checks
  SHA256 digests, and uploads the generated paper bundle as an artefact.【F:.github/workflows/ci.yml†L1-L64】

### Fixed
- Config hashing now normalises nested structures, ensuring run IDs stay stable
  across equivalent dictionary orderings.【F:feedflipnets/experiments/registry.py†L1-L157】
