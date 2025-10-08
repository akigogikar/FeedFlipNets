# Reproducibility Playbook

FeedFlipNets v1.0.0-rc1 ships with an experiment registry, deterministic
summaries, and an automated paper bundle workflow. This document captures the
practical steps for running repeatable experiments and packaging the results.

## Experiment registry

- The canonical registry lives in `experiments/registry.json` and is validated by
  `experiments/schema.json`.
- Each experiment merges registry defaults with per-entry overrides before being
  converted into a full pipeline configuration.
- The registry loader exposes:
  - `load_registry()` – load and validate the JSON registry.
  - `get_experiment(name)` – return an `ExperimentConfig` with helper methods.
  - `config_hash(config)` – compute a 12-character SHA256 run identifier over a
    canonicalised configuration.

```bash
python -m cli.main --experiment dfa_baseline
# artefacts -> .artifacts/<run_id>/metrics.jsonl, summary.json, manifest.json
```

## Deterministic outputs

- The run identifier is derived from the canonical config (sorted JSON) before
  injecting paths, so repeated runs reuse the same directory.
- Metrics are written via `JsonlSink`; summaries are recomputed with sorted-key
  JSON and canonical statistics (`min`, `max`, `mean`, `last`, `tail_auc`).
- CI runs the same experiment twice and asserts identical SHA256 hashes for both
  `metrics.jsonl` and `summary.json` to guard against regression.

```bash
python -m cli.main --experiment dfa_baseline
python -m cli.main --experiment dfa_baseline  # metrics + summary hashes match
```

## Paper bundle workflow

1. Run an experiment via the registry or a full config file.
2. Generate a bundle from the resulting `.artifacts/<run_id>` directory:

   ```bash
   python scripts/build_paper_bundle.py --run-dir .artifacts/<run_id> --include-plots
   ```

3. Inspect the generated `paper_bundle/` tree:
   - `metrics.jsonl` – copy of the run metrics.
   - `summary.json` – regenerated deterministic summary (sorted keys).
   - `manifest.json` – registry/manuscript metadata for the run.
   - `figures/` – optional deterministic plots rendered via the Agg backend.
   - `tables/metrics_summary.csv` – tail statistics for rapid reporting.
   - `methods.md` – stub describing dataset, seed, strategy, and run directory.
4. `paper_bundle.zip` captures the entire directory with fixed timestamps for
   byte-identical archives.

## Determinism checklist

- Always run with `FEEDFLIP_DATA_OFFLINE=1` (the CLI sets this by default).
- Avoid injecting timestamps into artefacts; manifests are the only timestamped
  files and are excluded from determinism checks.
- Prefer the experiment registry for publishable runs—the hash-based run IDs are
  portable and easy to cite.
