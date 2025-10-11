# How to add a new dataset

This guide explains the minimal steps required to register a dataset with the
FeedFlipNets pipeline.

## 1. Implement the loader

1. Create a module under `feedflipnets/data/` (or extend an existing one).
2. Write a factory function that returns a `DatasetSpec` and decorate it with
   `@register_dataset("<name>")`.
3. Implement an offline path that produces deterministic fixtures so CI and
   smoke tests remain network-free.
4. Return a `DataSpec` describing `d_in`, `d_out`, `task_type`, and optional
   `num_classes`. Record provenance metadata for reproducibility.

Refer to existing loaders such as `mnist.py` or `ucr.py` for patterns covering
splits, caching, and batching utilities.

## 2. Wire the preset

1. Create a preset under `configs/presets/` that references the new dataset.
2. Specify model `hidden` sizes, `strategy`, optimiser, learning rate, epochs,
   batch size, flip schedule, and evaluation cadence.
3. Choose a deterministic `train.seed` and assign a unique `train.run_dir`.

Presets appear automatically in `python -m cli.main --list-presets` once added.

## 3. Document the change

- Update `README.md` with the new preset or dataset options.
- Mention the addition in `CHANGELOG.md`.
- If the dataset requires external assets, add instructions to this document.

## 4. Validate

1. Run `make format lint test` to ensure style and unit tests pass.
2. Execute `make run PRESET=<new_preset>` followed by `make smoke` to verify the
   preset integrates cleanly with the CI smoke workflow.

Following these steps keeps datasets reproducible and easy to consume from the
CLI and sweep utilities.
