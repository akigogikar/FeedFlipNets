# Quality & Risk Audit

## Test & Tooling Snapshot
- **Pytest**: 16 passed, 2 skipped in ~34s; deprecation warnings for `np.trapz` remain.【F:_reports/raw/pytest.txt†L1-L20】
- **Code Size**: ~708 LOC of Python tracked by quick LOC scan; significant CSV/SVG artifacts also present.【F:_reports/raw/loc_summary.txt†L1-L12】
- **Hotspots**: Training loop and README dominate history churn, signalling unstable requirements vs. implementation.【F:_reports/raw/hotspots.txt†L1-L10】

## Risk Register
### Justification Block – Monolithic Trainer
- **Finding:** `train_single` handles data prep, trainer state, logging, and quantisation within one function.【F:feedflipnets/train.py†L20-L144】  
- **Why it matters:** Hard to test and extend new training strategies; invites bugs when experimenting.  
- **Recommendation:** Refactor into trainer class + strategy interfaces (Phase 2).  
- **Owner Suggestion:** Research engineering lead.

### Justification Block – Plotting Coupling
- **Finding:** Matplotlib figure generation executed inside sweep loop with no headless guard.【F:feedflipnets/train.py†L203-L238】  
- **Why it matters:** Headless servers and CI may fail or slow down runs; plotting can't be skipped.  
- **Recommendation:** Move plotting into optional reporter service with CLI flag.  
- **Owner Suggestion:** Tooling engineer.

### Justification Block – Network-Dependent Datasets
- **Finding:** Dataset loaders call `download_file` on every request without retry/offline logic.【F:datasets/utils.py†L12-L31】【F:datasets/tinystories.py†L17-L25】  
- **Why it matters:** CI flakes and ToS risks; no cache manifests for provenance.  
- **Recommendation:** Build dataset registry with manifest + offline test fixtures (Phase 1).

### Justification Block – Missing CI/CD
- **Finding:** `.github` lacks workflows; no automated gates.【F:_reports/raw/tree.txt†L1-L61】  
- **Why it matters:** Regression risk and inconsistent environments.  
- **Recommendation:** Add GitHub Actions running lint/pytest + caching dataset fixtures.

### Justification Block – Dependency Drift
- **Finding:** Requirements split between `pyproject.toml` and `requirements.txt` with unpinned versions.【F:pyproject.toml†L6-L24】【F:requirements.txt†L1-L6】  
- **Why it matters:** Reproducibility suffers; environment mismatches likely.  
- **Recommendation:** Consolidate under Poetry/pip-tools lockfile and document supported versions (Phase 1).

### Justification Block – Observability Gap
- **Finding:** Metrics limited to print statements and saved plots; no structured logs or metadata beyond JSON summary.【F:feedflipnets/train.py†L163-L238】  
- **Why it matters:** Difficult to compare runs programmatically or integrate with experiment trackers.  
- **Recommendation:** Introduce metrics emitter interface (e.g., CSV/JSONL) in Phase 2.

## Security & Compliance Notes
- Remote dataset downloads use plain HTTP for UCR archive.【F:datasets/timeseries.py†L12-L41】 – risk of MITM; prefer HTTPS mirror or checksums.
- No secret management issues detected (no `.env` or API keys present). Unknown CVE status due to lack of dependency locking.

## Developer Experience
- README provides install/run instructions but lacks "first run" script or Makefile.【F:README.md†L22-L118】
- Tests cover primary flows yet rely on live network; add fixtures/mocks for deterministic onboarding.
