# Migration Runbook – FeedFlipNets Restructure

## Overview
- **Objective:** Transition from monolithic trainer to modular architecture without breaking existing experiment scripts.  
- **Owners:**
  - Tech Lead – overall coordination.  
  - Platform Engineer – CI/dataset caching.  
  - Research Engineer – strategy extraction.  
  - QA – contract & regression testing.

## Phase 1 Checklist
1. Create feature branch `restructure/phase1-guardrails`.  
2. Add GitHub Actions workflow (`.github/workflows/ci.yml`) running `pip install -e .[dev]` and `pytest -m "not network"`.  
3. Introduce `requirements-lock.txt` via `pip-compile` or Poetry lock; document in README.  
4. Implement dataset cache manifest (`feedflipnets/data/cache_manifest.py`) storing URL, checksum, timestamp.  
5. Swap `np.trapz` ➜ `np.trapezoid`; add unit test verifying identical results on sample curve.  
6. Update README with offline instructions.  
**Checkpoint:** CI green on synthetic-only tests; lockfile committed.  
**Rollback:** Revert branch; ensure dataset manifest optional and disabled by default until validated.

## Phase 2 Checklist
1. Branch `restructure/phase2-modules` from updated main.  
2. Create new packages:
   - `feedflipnets/core` (activations, quantisation, feedback strategies).  
   - `feedflipnets/data` (registry, loaders, cache manager).  
   - `feedflipnets/training` (Trainer, RunResult).  
   - `feedflipnets/reporting` (metrics sinks, plotting adapters).  
3. Move existing functionality into new modules while keeping shim functions in `feedflipnets/train.py` forwarding to new classes.  
4. Add contract tests:
   - Feedback strategy interface compliance.  
   - Dataset registry returns deterministic metadata + iterables.  
   - Trainer-run integration with synthetic dataset verifying metrics output.  
5. Update CLI (`experiments/ternary_dfa_experiment.py`) to consume new Trainer and reporting callbacks.  
6. Document migration path in `CHANGELOG.md` with deprecation notes.  
**Checkpoint:** All tests (unit + integration) pass; shims log deprecation warnings.  
**Rollback:** Keep shims pointing to old monolithic code (retain copy under `legacy/`) and toggle via env var.

## Phase 3 Checklist
1. Branch `restructure/phase3-optimise`.  
2. Implement experiment registry with YAML/JSON configs and validation.  
3. Add performance benchmarks (e.g., `pytest -m perf`) capturing baseline metrics.  
4. Enhance CI: coverage report, cache dataset artifacts, publish run metadata as artifacts.  
5. Remove legacy `train_single`/`sweep_and_log` implementations after confirming no external dependencies remain; update wrapper to raise informative errors if legacy path invoked.  
6. Publish documentation updates and tag release `v1.0.0`.  
**Checkpoint:** CI includes coverage + artifact upload; changelog + release notes published.  
**Rollback:** Retain `legacy` module for one release cycle; if regressions appear, re-enable via feature flag.

## Communication Plan
- Weekly sync with research + platform leads to review progress and unblock issues.  
- Publish status updates in project channel (Monday/Thursday).  
- After each phase, circulate summary + retro notes; update roadmap accordingly.  
- Notify stakeholders before Phase 3 deprecations; provide migration snippet examples.

## Owner Matrix
| Activity | Primary | Backup |
| --- | --- | --- |
| CI setup & lockfiles | Platform Engineer | Tech Lead |
| Dataset registry | Data Engineer | Research Engineer |
| Trainer extraction | Research Engineer | Tech Lead |
| Reporting adapter | Platform Engineer | Research Engineer |
| Documentation & comms | Tech Lead | QA |
