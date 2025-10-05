# Restructure Plan

## Decision Matrix
| Option | Scope | Effort | Timeline | Risks | Opportunity | Est. ROI |
| --- | --- | --- | --- | --- | --- | --- |
| Keep-as-is | Bug fixes only | S | 1-2 weeks | Monolithic trainer and flaky datasets remain | Minimal disruption | Low (10-15 engineer hours saved short-term, but tech debt grows) |
| Adjust | Extract plotting + add CI | M | 4 weeks | Core coupling persists; dataset downloads still fragile | Faster CI onboarding | Medium (30-40 hours, partial debt reduction) |
| **Restructure (Chosen)** | Full modular split, data registry, CI/CD, reproducibility tooling | L | 10-12 weeks | Requires coordinated refactor; temporary double-write paths | Sustainable research velocity, reproducible experiments, easier method additions | High (≈200 engineer hours, avoids repeated rework per experiment) |

## Phase Plan
### Phase 1 – Stabilise (Weeks 1-2)
- Add GitHub Actions workflow (lint + pytest on synthetic fixture).  
- Introduce dependency lockfile and document supported Python/NumPy combo.  
- Create dataset cache manifest & offline fixtures; add retries + checksum validation for HTTP downloads.  
- Replace deprecated `np.trapz` usage; add regression tests for metrics.【F:feedflipnets/train.py†L143-L144】  
- Draft developer quick-start script (Makefile/Taskfile) to run smoke sweep.

### Phase 2 – Extract & Re-compose (Weeks 3-8)
- Split `feedflipnets` into `core`, `data`, `training`, `reporting` namespaces (see `_design/ARCHITECTURE_TARGET.md`).  
- Implement `FeedbackStrategy` interface; migrate existing variants (`Backprop`, `DFA`, `Ternary`, etc.) into strategy classes.  
- Build trainer orchestration class that accepts dataset iterators and emits structured metrics.  
- Move plotting + artifact generation into reporting layer with toggles.  
- Establish dataset registry with manifest metadata and offline-friendly caching.  
- Write contract tests covering dataset registry, trainer, and reporting seams.

### Phase 3 – Optimise & De-risk (Weeks 9-12)
- Introduce experiment registry & configuration schema (e.g., pydantic).  
- Add performance profiling hooks and baseline benchmarks for critical configs.  
- Publish documentation for reproducible runs (config + dataset manifest).  
- Deprecate legacy monolithic functions; provide shims with warnings.  
- Tighten CI (coverage threshold, artifact upload, reproducibility checks).

## ROI & Risk Analysis
| Risk | Severity | Likelihood | Mitigation (Phase) | Residual Risk |
| --- | --- | --- | --- | --- |
| Regression during module split | High | Medium | Phase 2 dual-runner tests + feature flags | Medium |
| Dataset downtime/offline env | High | High | Phase 1 cache manifest + mirrors | Low |
| Developer learning curve | Medium | Medium | Phase 1 quick-start docs & templates | Low |
| CI flakiness due to large downloads | Medium | High | Phase 1 offline fixtures, Phase 3 artifact caching | Low |
| Plotting headless failures | Medium | Medium | Phase 2 reporting adapter with CLI flag | Low |

## Staffing Assumptions
- 1 Senior Research Engineer (lead, 50% allocation).  
- 1 Tooling/Platform Engineer (Phase 1-2 focus on CI/data infra).  
- 1 Data Engineer (Phase 2 dataset registry + caching).  
- QA support (0.5 FTE) for contract/coverage tests during Phase 2-3.
