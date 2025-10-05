# TL;DR
- Alignment ≈55%: Core ternary feedback alignment loop exists, but architecture and automation lag behind the README promises of modular datasets, reproducible sweeps, and polished tooling.【F:README.md†L16-L104】【F:feedflipnets/train.py†L20-L240】
- Top Risks: (1) Monolithic training routine; (2) Network-tethered dataset loaders; (3) No CI/CD; (4) Plotting/IO tightly coupled to training; (5) Lack of dependency/version controls.
- Recommended Option: **Restructure** – staged extraction of data/model/training boundaries, CI scaffolding, and reproducible experiment surfaces.
- 2-week Quick Wins: carve out data adapter interface, add smoke-test CI, pin runtime deps, and swap deprecated NumPy API.
- 90-day Outlook: land modular core package, experiment registry, cached datasets, and release-quality automation for research reproducibility.

## Scores (0–5)
| Category | Score |
| --- | --- |
| Plan Alignment | 2 |
| Modularity | 1 |
| Test Rigor | 3 |
| CI/CD Maturity | 0 |
| Security Hygiene | 1 |
| DX Onboarding | 2 |
| Observability | 0 |
| Docs Quality | 3 |
| Release Discipline | 1 |

## Alignment Snapshot
**Finding:** README advertises modular datasets, sweep tooling, and plotting, establishing the intended scope.【F:README.md†L16-L104】  
**Evidence:** README outlines dataset helpers and experiment orchestration.  
**Why it matters:** Sets expectations for separable layers and reproducible research.  
**Recommendation:** Use the restructure plan to harden module boundaries and automation.

**Finding:** Actual implementation concentrates dataset loading, quantisation, training, logging, and plotting inside `train_single`/`sweep_and_log` without seams for reuse.【F:feedflipnets/train.py†L20-L240】  
**Evidence:** Single module houses model math, IO, plotting, and metrics.  
**Why it matters:** Hinders extensibility, reproducibility, and testing; increases coupling risk.  
**Recommendation:** Execute Phase 2 extraction (see `_design/RESTRUCTURE_PLAN.md`) to separate trainers, reporters, and IO.

**Finding:** Dataset utilities reach out to remote sources synchronously every run, lacking caching controls or offline fallbacks.【F:datasets/timeseries.py†L15-L43】【F:datasets/tinystories.py†L17-L25】  
**Evidence:** Loaders call `download_file` on each invocation.  
**Why it matters:** Breaks deterministic runs and slows CI; risk for offline environments.  
**Recommendation:** Phase 1 guardrails should introduce cache manifest + retry logic.

**Finding:** Repository ships tests but no CI/CD workflow or release automation is defined.【F:_reports/raw/tree.txt†L1-L61】  
**Evidence:** Tree shows `.github` contains no workflows; automation absent.  
**Why it matters:** Manual validation only; high regression risk.  
**Recommendation:** Add GitHub Actions smoke suite in Phase 1.

**Finding:** Test suite passes locally yet raises deprecated NumPy API warnings, highlighting impending breakage.【F:_reports/raw/pytest.txt†L1-L20】【F:feedflipnets/train.py†L143-L144】  
**Why it matters:** Future NumPy releases may remove `np.trapz`, undermining metrics.  
**Recommendation:** Quick win: migrate to `np.trapezoid` and pin dependencies.

## Top Risks
1. **God Function Trainer** – `train_single` coordinates data prep, model evolution, logging, and quantisation with implicit state.【F:feedflipnets/train.py†L20-L144】  Impact: fragile to extend new methods or devices.  Mitigation: extract trainer class + strategy interfaces (Phase 2).
2. **Unbounded External Downloads** – Dataset loaders fetch remote assets inline without retries/auth caching.【F:datasets/utils.py†L12-L31】【F:datasets/tinystories.py†L17-L25】  Impact: flaky runs/CI; potential ToS violations.  Mitigation: centralise download manager with cache manifest.
3. **Missing Automation** – No CI workflows or release cadence.【F:_reports/raw/tree.txt†L1-L61】  Impact: regressions land unnoticed; research unreproducible.  Mitigation: Add GitHub Actions with lint/test matrix in Phase 1.
4. **Tightly Coupled Plotting** – Matplotlib usage embedded in training loop prevents headless usage and unit testing.【F:feedflipnets/train.py†L203-L238】  Impact: evaluation-only runs still incur plotting cost; CLI can't run on servers without display.  Mitigation: move plotting to reporter module with optional enable flag.
5. **Dependency Drift** – Dependencies are unpinned and split between `pyproject` and `requirements.txt`, risking environment skew.【F:pyproject.toml†L6-L24】【F:requirements.txt†L1-L6】  Impact: reproduction fails across machines.  Mitigation: consolidate dependency management and lock versions.

## Recommended Path
Adopt the restructure option outlined in `_design/RESTRUCTURE_PLAN.md`: 
- Phase 1 guardrails (2 weeks): CI, dependency pinning, dataset cache policy, documentation of run modes. 
- Phase 2 modular extraction (4–6 weeks): separate `core`, `data`, `training`, and `reporting` packages with adapters. 
- Phase 3 optimisation (4 weeks): performance profiling, API hardening, and deletion of deprecated pathways.

## Quick Wins (Weeks 1–2)
- Replace `np.trapz` with `np.trapezoid` and add regression test for metric calculation.【F:feedflipnets/train.py†L143-L144】
- Introduce GitHub Actions workflow running lint + pytest smoke on synthetic dataset.【F:_reports/raw/tree.txt†L1-L61】
- Create dataset cache manifest + offline mode to bypass downloads in tests.【F:datasets/utils.py†L12-L31】
- Add pinned `requirements-lock.txt` to document tested versions.【F:requirements.txt†L1-L6】

## 90-Day Outlook
- Modular package boundaries enforced via code owners and PR template (Phase 2 deliverable). 
- Artifacted experiment runs with reproducibility metadata (Phase 3). 
- Release cadence with tagged datasets and published changelog.
