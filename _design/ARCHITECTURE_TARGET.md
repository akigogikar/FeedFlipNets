# Target Architecture – FeedFlipNets vNext

## Guiding Principles
1. **Deterministic experiments** – isolate data acquisition, training, and reporting to enable reproducible runs.  
2. **Pluggable strategies** – decouple feedback alignment variants from trainer orchestration.  
3. **Automation first** – every surface callable from CLI, API, or CI with the same entrypoints.

## Module Map
```mermaid
graph TD
    subgraph feedflipnets.core
        activations[activations]
        quant[quant]
        strategies[strategies]
        types[types]
    end
    subgraph feedflipnets.data
        registry[data.registry]
        cache[data.cache]
        loaders[data.loaders]
    end
    subgraph feedflipnets.training
        trainer[training.trainer]
        pipelines[training.pipelines]
    end
    subgraph feedflipnets.reporting
        metrics[reporting.metrics]
        plots[reporting.plots]
        artifacts[reporting.artifacts]
    end
    subgraph cli
        entry[cli.main]
    end

    registry --> loaders
    loaders --> cache
    trainer --> strategies
    trainer --> quant
    trainer --> types
    pipelines --> trainer
    pipelines --> registry
    pipelines --> metrics
    pipelines --> plots
    pipelines --> artifacts
    entry --> pipelines
```

## Dependency Rules
- `feedflipnets.core` contains pure numpy operations with **no IO** or plotting.  
- `feedflipnets.data` may depend on `core` for normalization utilities but **never** on training/reporting.  
- `feedflipnets.training` orchestrates batches using interfaces from `core` and `data`; it can emit events to `reporting` via abstract ports.  
- `feedflipnets.reporting` depends on core types only; plotting is optional and headless-safe.  
- CLI/experiments import only `training` and `reporting` interfaces.

## Public APIs
- `feedflipnets.core.strategies.FeedbackStrategy` defines `init(model)` and `backward(activations, error, state)`.
- `feedflipnets.training.trainer.Trainer.run(dataloader, epochs, seed, *, determinism=True)` returns a `RunResult`.
- `feedflipnets.data.registry.get(name, *, offline=True, cache_dir=None, **options)` returns dataset metadata + loader callable.
- `feedflipnets.reporting.metrics.emit(event)` writes to JSONL/CSV sinks; CLI chooses sinks via config.

## Data Flow
1. CLI parses config ➜ resolves dataset via registry (ensures cached assets, offline checks).  
2. Trainer pulls batches via iterator, applies quantisation/feedback strategy, records metrics.  
3. Reporting callbacks serialize metrics and optionally plot via adapter (skippable in CI).  
4. Pipelines capture metadata (git SHA, config, dataset manifest) for reproducibility.

## Non-Functional Guardrails
- All modules pure-Python/NumPy; GPU/accelerator hooks implemented via optional adapters.  
- Deterministic random seeds enforced by trainer; dataset loader returns provenance manifest.  
- Logging uses structured JSON to integrate with experiment trackers.
