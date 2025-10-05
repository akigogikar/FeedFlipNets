## Summary
- 

## Testing
- [ ] `pytest -m "not network"`
- [ ] Other (specify): 

## Boundaries & Quality Gates
- [ ] No cross-package imports outside allowed dependency graph.
- [ ] Dataset loaders respect cache/offline policy.
- [ ] Metrics/reporting outputs validated (JSONL/CSV as applicable).
- [ ] Updated docs/changelog.
- [ ] Added/updated issue in `_reports/QUALITY_AUDIT.md` if new risk introduced.

## Deployment / Rollback
- Describe rollout plan and rollback strategy.

## Reviewer Checklist
- [ ] Architecture alignment (see `_design/ARCHITECTURE_TARGET.md`).
- [ ] Tests cover new behaviour.
- [ ] CI status is green.
- [ ] Risks acknowledged and mitigations documented.
