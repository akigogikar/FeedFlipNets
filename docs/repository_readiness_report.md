# FeedFlipNets Repository Readiness Report

## Overview
FeedFlipNets provides a modular research environment for studying ternary neural networks trained with feedback-alignment variants. The CLI surfaces presets and registry-based experiment definitions, letting users launch runs with a single command while controlling offline data usage and artifact logging.【F:cli/main.py†L1-L155】 The pipeline assembles datasets, models, and training schedules into reproducible runs, optionally sweeping over configurations while writing manifest, metrics, and summary artifacts for each experiment.【F:feedflipnets/training/pipelines.py†L22-L292】

The reference trainer implements a deterministic CPU-only loop over NumPy-based feed-forward models that quantise weights to ternary values after each optimisation step. Loss tracking is limited to mean squared error (MSE) computed on each batch, which is the only metric emitted by default callbacks.【F:feedflipnets/training/trainer.py†L29-L191】 Feedback strategies are pluggable, covering Direct Feedback Alignment, ternary flip alignment, and structured random feedback matrices for orthogonal, Hadamard, block-diagonal, or low-rank variants.【F:feedflipnets/core/strategies.py†L1-L160】

## Modality Readiness

| Modality | Status | Evidence |
| --- | --- | --- |
| Regression | 🟢 Ready | Synthetic sine regression is available out-of-the-box via the `synthetic` loader and multiple presets, providing continuous targets suitable for MSE optimisation.【F:feedflipnets/data/loaders/synthetic.py†L1-L52】【F:feedflipnets/training/pipelines.py†L22-L182】|
| Classification | 🟢 Ready (with caveats) | The MNIST loader supports offline fixtures, one-hot outputs, and numeric labels, and presets configure 10-way classification. Tests validate shapes for offline batches.【F:feedflipnets/data/loaders/mnist.py†L18-L125】【F:feedflipnets/training/pipelines.py†L69-L110】【F:tests/unit/test_dataset_loaders.py†L4-L16】|
| Time Series | 🟡 Partial | The UCR/UEA loader prepares synthetic fixtures and one-hot targets but lacks real archive parsing; tests confirm offline integration. Additional work is required to use real datasets or structured temporal models.【F:feedflipnets/data/loaders/ucr_uea.py†L1-L78】【F:tests/unit/test_dataset_loaders.py†L26-L31】|
| Text / Sequential | 🟡 Partial | TinyStories loader encodes a cached token list into sliding windows for next-token regression, and presets demonstrate training, but coverage is limited to small fixtures without rich language modelling support.【F:feedflipnets/data/loaders/tinystories.py†L1-L99】【F:feedflipnets/training/pipelines.py†L92-L110】|

## Strengths
* **Offline-first datasets.** The cache manager consistently produces deterministic fixtures and records provenance, ensuring experiments run without network access.【F:feedflipnets/data/cache.py†L1-L162】【F:feedflipnets/data/loaders/mnist.py†L18-L125】【F:feedflipnets/data/loaders/tinystories.py†L18-L95】【F:feedflipnets/data/loaders/ucr_uea.py†L20-L75】
* **Configurable pipelines.** Presets and sweeps simplify experimentation across strategies, depths, and frequencies while keeping manifest metadata for reproducibility.【F:feedflipnets/training/pipelines.py†L22-L220】【F:feedflipnets/training/pipelines.py†L278-L292】
* **Deterministic training core.** The trainer enforces CPU determinism, seeds RNGs, and emits per-step loss metrics through callbacks that can be extended for logging or plotting.【F:feedflipnets/training/trainer.py†L118-L191】

## Gaps and Recommended Improvements
1. **Real dataset ingestion.** Extend loaders such as UCR/UEA and TinyStories to parse real archives or document preprocessing steps so experiments can move beyond synthetic fixtures.【F:feedflipnets/data/loaders/ucr_uea.py†L39-L66】【F:feedflipnets/data/loaders/tinystories.py†L55-L95】
2. **Evaluation metrics.** Introduce accuracy or modality-appropriate validation metrics, as the trainer currently emits only MSE on training batches.【F:feedflipnets/training/trainer.py†L138-L159】
3. **Loss function flexibility.** Provide cross-entropy or other task-specific losses to complement the existing MSE workflow, which may hinder classification performance.【F:feedflipnets/training/trainer.py†L146-L154】
4. **Architectural variety.** Add optional convolutional or recurrent modules to better exploit modality structure; the current feed-forward network flattens inputs for every task.【F:feedflipnets/training/trainer.py†L29-L88】
5. **Dimension inference helpers.** Reintroduce automated input/output shape inference in the pipeline to reduce configuration errors when plugging in new datasets.【F:feedflipnets/training/pipelines.py†L223-L307】

## Suggested Next Steps
1. Harden dataset loaders for real-world archives, starting with at least one UCR dataset and a small open-text corpus, while updating documentation on supplying custom assets.【F:feedflipnets/data/loaders/ucr_uea.py†L39-L75】【F:feedflipnets/data/loaders/tinystories.py†L55-L95】
2. Implement optional validation phases in the trainer to compute metrics such as classification accuracy or regression RMSE at epoch boundaries.【F:feedflipnets/training/trainer.py†L138-L163】
3. Offer configurable loss functions and output activations in the trainer/pipeline configuration to support classification norms alongside DFA research goals.【F:feedflipnets/training/trainer.py†L146-L154】【F:feedflipnets/training/pipelines.py†L223-L334】
4. Prototype additional model classes (e.g., convolutional or sequential layers) and corresponding feedback strategy hooks to broaden modality coverage.【F:feedflipnets/training/trainer.py†L29-L88】【F:feedflipnets/core/strategies.py†L1-L160】
5. Expand usage guides or tutorials describing preset usage, registry-based experiments, and dataset expectations to make onboarding easier for new users.【F:cli/main.py†L28-L155】【F:feedflipnets/training/pipelines.py†L22-L292】

