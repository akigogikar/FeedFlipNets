"""Headless-safe plotting adapters."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple


class PlotAdapter:
    """Collect metrics and optionally emit matplotlib figures."""

    def __init__(self, run_dir: str | Path, enable_plots: bool = False):
        self.enable_plots = enable_plots
        self.run_dir = Path(run_dir)
        self._history: List[Tuple[int, float]] = []
        if self.enable_plots:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def on_step(self, step: int, metrics):
        if not self.enable_plots:
            return
        loss = float(metrics.get("loss", 0.0))
        self._history.append((step, loss))

    def close(self) -> None:
        if not self.enable_plots or not self._history:
            return
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # imported lazily for headless safety

        steps, losses = zip(*self._history)
        fig, ax = plt.subplots()
        ax.plot(steps, losses)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curve")
        plot_path = self.run_dir / "loss.png"
        fig.savefig(plot_path)
        plt.close(fig)

    __call__ = on_step
