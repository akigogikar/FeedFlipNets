FeedFlipNets
===========

DFA-trained Ternary Networks for Edge-Ready, Low-Precision Deep Learning

One-liner: Feedback (Direct Feedback Alignment) + Flip (ternary forward weights) = FeedFlip — a pragmatic toolkit for training accurate, hardware-friendly neural nets without full backprop’s symmetry constraints.


⸻

Why FeedFlipNets
----------------

Modern teams need efficient deep learning that actually ships: low power, small memory, predictable behavior. FeedFlipNets does exactly that by combining:

* **Direct Feedback Alignment (DFA)** — replace backprop’s weight-transpose path with fixed random feedback matrices. This removes the weight transport constraint and simplifies the backward graph.
* **Ternary weight “flip”** — constrain forward weights to {−1, 0, +1} on a configurable schedule while updating “shadow” float weights under the hood. You keep learning capacity, you gain hardware efficiency.

Net outcome: simpler training signals, binary/ternary-friendly models, and a clean surface for edge AI and neuromorphic experiments.


⸻

What you get
------------

* **Drop-in training strategies:** backprop, dfa, structured feedback, and FeedFlip (dfa + ternary forward).
* **Low-precision knobs:** flip thresholds, deterministic/stochastic ternarization, and scheduling (per-step or per-epoch).
* **Determinism by design:** seeds, fixtures, and smoke tests for CI.
* **Small, readable codepaths:** great for experimentation, pedagogy, and fast iteration.
* **No GPU required:** CPU-friendly NumPy-first approach (works anywhere you can run Python).

Straight talk: FeedFlip optimizes practical efficiency and deployability, not leaderboard SOTA. Expect excellent footprint and stability; do not expect it to beat full-precision backprop on very large benchmarks out of the box.


⸻

How it works (at a glance)
--------------------------

Forward: use flipped (ternary) weights $W_l = Q_\tau(V_l)$ derived from float “shadow” weights $V_l$.

Backward (DFA): project output error with fixed matrices $B_l$:

$$
\delta_l = (B_l e) \odot f'(h_l)
$$

Update: compute $\nabla V_l$ with the usual local rule and step the optimizer. Refresh $W_l$ per your flip schedule.

This separation lets you train with simple, fixed feedback while keeping the forward path quantized and deployable.


⸻

Installation
------------

```bash
# From source (recommended)
git clone https://github.com/akigogikar/FeedFlipNets.git
cd FeedFlipNets
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

Requirements: Python 3.8+ plus NumPy, matplotlib, pandas, scikit-learn, and PyYAML (see `pyproject.toml` / `requirements-lock.txt` for exact pins).


⸻

Quickstart (use the smoke tests)
--------------------------------

The fastest path to value is to run the smoke tests. They validate the end-to-end training loop with small, deterministic datasets.

```bash
# Option A: run the whole smoke suite (datasets, trainer, CLI)
pytest -q tests/test_datasets_smoke.py tests/test_training_loops.py tests/integration/test_cli.py

# Option B: run a single scenario
pytest -q tests/integration/test_cli.py::test_cli_basic_preset
```

What to expect:

* A tiny model trains for a few epochs.
* Metrics (loss/accuracy) write into `runs/` alongside manifests.
* Runs are deterministic with the shipped seeds and offline fixtures.


⸻

Usage patterns
--------------

1. **CLI presets (recommended for teams)**

   Presets let you snap together dataset + model + feedback + flip schedule without touching code.

   ```bash
   # Example: DFA + ternary forward on the MNIST offline fixture
   python -m cli.main \
     --preset mnist_mlp_dfa \
     --feedback dfa \
     --flip ternary \
     --flip-schedule per_step \
     --flip-threshold 0.05
   ```

   Prefer `make run PRESET=<name>` if you want an even shorter command (`make run PRESET=mnist_mlp_dfa`).

2. **Python API (for experiments)**

   ```python
   from pathlib import Path

   import numpy as np

   from feedflipnets.core.strategies import DFA
   from feedflipnets.data.registry import get_dataset
   from feedflipnets.training.trainer import FeedForwardModel, SGDOptimizer, Trainer

   rng = np.random.default_rng(0)
   spec = get_dataset("mnist", offline=True, cache_dir=Path(".cache"))
   train_loader = spec.loader("train", batch_size=32)

   model = FeedForwardModel([784, 256, 10], tau=0.05, quant="det", seed=0)
   strategy = DFA(rng)
   optimizer = SGDOptimizer(lr=0.05)

   trainer = Trainer(model=model, strategy=strategy, optimizer=optimizer)
   result = trainer.run(
       train_loader,
       epochs=10,
       seed=0,
       steps_per_epoch=spec.splits["train"] // 32,
       task_type="multiclass",
       num_classes=10,
       flip="ternary",
       flip_schedule="per_step",
       checkpoint_dir=Path("runs/example"),
   )

   print(result.metrics_path)
   ```

   Swap in `feedflipnets.core.strategies.Backprop()` or `feedflipnets.core.strategies.TernaryDFA(...)` to explore other feedback paths.


⸻

Configuration surface
---------------------

Key toggles you’ll expose in code/CLI:

* `--feedback {backprop, dfa, ternary_dfa, structured}`
* `--flip {off, ternary}`
* `--flip-schedule {per_step, per_epoch}`
* `--flip-threshold <float>` (ternary threshold)
* `--seed <int>`
* Optimizer hyperparameters via config files (`train.lr`, `train.batch_size`, etc.)
* Dataset and model size flags (hidden units, depth, activation) via presets/config overrides.


⸻

Datasets & tasks
----------------

* **Vision:** MNIST-scale offline fixture (`mnist_mlp_dfa` preset).
* **Time series:** UCR GunPoint offline fixture (`ucr_gunpoint_mlp_dfa`).
* **Tabular regression:** California Housing offline fixture (`california_housing_mlp_dfa`).
* **Text:** 20 Newsgroups hashing-vectorized offline fixture (`20newsgroups_bow_mlp_dfa`).
* **CSV helpers:** `csv_regression` / `csv_classification` loaders for custom experiments.

Each preset lives under `configs/presets/` and works fully offline unless you pass `--offline false` or clear the `FEEDFLIP_DATA_OFFLINE` environment variable.


⸻

Performance posture (managing expectations)
-------------------------------------------

* **Convergence:** DFA + ternary typically requires more epochs vs. full-precision backprop.
* **Footprint:** Ternary forward weights and simple feedback paths make the approach edge-deployable and hardware-amenable.
* **Determinism:** Reproducible by default (seeded in data shuffles, init, and feedback matrices).

Add your own smoke-test metrics table once you have results to share.


⸻

FAQ
----

**Is “Flip” about the feedback path?**

No. “Flip” refers to ternary forward weights only. Feedback remains DFA with fixed random matrices.

**Why ternary and not binary?**

Ternary often hits a better accuracy/efficiency frontier while preserving the low-precision benefits.

**Can I turn off quantization?**

Yes—set `--flip off` (or the equivalent API flag) to run pure DFA or backprop.

**Do I need a GPU?**

No. CPU is fine for the supported experiments and smoke tests.


⸻

Roadmap
-------

* Optional binary feedback matrices (±1) with variance scaling.
* Structured feedback (e.g., orthogonal/Hadamard) for stability.
* Plug-in quantizers (LSQ, DoReFa) and per-layer thresholds.
* Exporters for on-device inference formats.

Update these bullets as plans evolve.


⸻

SEO keywords (for GitHub/Google)
--------------------------------

Direct Feedback Alignment, DFA, Feedback Alignment, ternary weight networks, binary neural networks, quantized neural networks, low-precision training, efficient deep learning, edge AI, on-device learning, neuromorphic learning, random feedback matrices, hardware-friendly deep learning.


⸻

Contributing
------------

PRs welcome. Please run the smoke suite locally before submitting.

```bash
# Lint + tests
make format lint test smoke
```

Add `ruff`, `black`, `mypy`, or pre-commit hooks here if your workflow uses them.


⸻

Citation
--------

If this work helps your research or product, please cite:

```bibtex
@software{FeedFlipNets,
  author = {Gogikar, A.},
  title  = {FeedFlipNets: DFA-trained Ternary Networks for Efficient Deep Learning},
  year   = {2025},
  url    = {https://github.com/akigogikar/FeedFlipNets}
}
```

Swap in a paper or arXiv entry once available.


⸻

License
-------

Apache-2.0 © Aki Gogikar


⸻

Drop-in checklist for maintainers
---------------------------------

* Replace placeholders with fresh metrics or CLI shortcuts as the project matures.
* Confirm Python version and dependency pins.
* Add one tiny metrics table from a smoke run (before/after flip) to make the README “pop.”
* If you have CI, wire the badge to your workflow URL.
