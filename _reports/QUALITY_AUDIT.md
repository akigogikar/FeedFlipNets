# Quality & Risk Audit – FeedFlipNets v1.0 Snapshot

## Tooling & Coverage
- **Automation:** `.github/workflows/ci.yml` runs linting, tests with coverage
  ≥75%, and the deterministic smoke preset on Python 3.10/3.11, uploading the
  generated artefacts.【F:.github/workflows/ci.yml†L1-L33】
- **Commands:** `Makefile` exposes `bootstrap`, `lint`, `test`, and `smoke`
  targets; lint bundles Ruff, Black, and import-linter contracts while tests run
  offline with `pytest --cov` and a fail-under gate.【F:Makefile†L1-L23】
- **Determinism:** Contract tests assert that repeated pipeline executions yield
  identical JSONL metrics, and the CLI smoke test ensures offline completion.
  【F:tests/contract/test_trainer_contract.py†L1-L52】【F:tests/integration/test_cli.py†L1-L11】

## Residual Risks & Mitigations
- **Offline cache integrity** – `feedflipnets.data.cache.fetch` records
  provenance (mode, checksum, URL) via the `CacheManifest`. Risk of stale
  fixtures mitigated by deterministic builders and manifest writes before
  returning.【F:feedflipnets/data/cache.py†L1-L132】
- **Strategy refresh semantics** – `StructuredFeedback` relies on the trainer to
  flag `pending_refresh` metadata each epoch; contract tests cover gradient
  shapes but additional stress tests on large hidden stacks remain TODO.【F:feedflipnets/core/strategies.py†L1-L207】
- **Legacy usage** – `feedflipnets/train.py` now delegates to pipelines and emits
  deprecation warnings, but downstream notebooks should migrate to the CLI to
  benefit from deterministic artefacts.【F:feedflipnets/train.py†L1-L156】

## Observability & Artefacts
- Metrics sinks emit JSONL (`{step, split, loss, accuracy, seed, sha}`) and CSV
  records, while `PlotAdapter` forces Matplotlib's Agg backend to remain
  headless.【F:feedflipnets/reporting/metrics.py†L1-L63】【F:feedflipnets/reporting/plots.py†L1-L33】
- `write_manifest` captures git SHA, config, dataset provenance (including
  checksums), and environment flags; smoke artefacts are archived in CI for
  regression triage.【F:feedflipnets/reporting/artifacts.py†L1-L33】【F:.github/workflows/ci.yml†L24-L33】

## Follow-up Actions
- Expand structured-feedback regression tests to cover `hadamard`/`lowrank`
  refresh policies under deeper networks.
- Evaluate provenance signing for cached datasets if external distribution is
  required.
- Monitor CI wall-clock time; smoke artefact compression is currently best
  effort (`tar` stage tolerates missing outputs).
