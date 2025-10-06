# Upgrading to FeedFlipNets v1.0

## Summary of breaking changes
- The legacy `feedflipnets.train.train_single` and `sweep_and_log` functions now
  emit `DeprecationWarning`s and internally call the new pipeline; existing code
  continues to work but should migrate to the CLI or `pipelines.run_pipeline`.
- Feedback strategies implement the new `FeedbackStrategy` protocol with
  `init(model)` and `backward(activations, error, state)` methods. Custom
  strategies must be updated accordingly.【F:feedflipnets/core/strategies.py†L1-L207】
- Datasets are resolved via `feedflipnets.data.registry.get` with explicit
  offline/cache controls; the old `get_dataset` helper has been removed.

## Migration recipes

### From direct trainer calls to pipelines
```python
from feedflipnets.training import pipelines

config = pipelines.load_preset("synthetic-min")
config["train"]["run_dir"] = "runs/my-run"
result = pipelines.run_pipeline(config)
print(result.metrics_path)
```

### CLI replacement for scripts
```bash
python -m cli.main --preset basic_dfa_cpu --dump-config runs/run.json
```
This reproduces the functionality of the old `ternary_dfa_experiment.py` entry
point while honouring offline caching and deterministic seeds.【F:cli/main.py†L1-L89】

### Custom dataset registration
```python
from feedflipnets.data.registry import register_dataset, DatasetSpec
from feedflipnets.core.types import Batch


def my_dataset_factory(*, offline=True, cache_dir=None, **kwargs) -> DatasetSpec:
    # Build or load fixtures respecting offline/cache_dir arguments
    ...
    return DatasetSpec(name="my_dataset", provenance=meta, loader=loader)
```
Factories should support the `offline`/`cache_dir` keyword arguments to remain
compatible with the CLI and test suite.【F:feedflipnets/data/registry.py†L1-L53】

## Validation checklist
- Run `make lint && make test` to ensure imports, formatting, and coverage gates
  pass before committing.
- Execute `make smoke` twice and verify `metrics.jsonl` digests match to confirm
  deterministic behaviour.
