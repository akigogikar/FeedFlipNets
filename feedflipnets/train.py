from __future__ import annotations
import json
import os
import datetime
from typing import List, Dict, Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt

from .utils import make_dataset, quantize_stoch, ensure_dir, tanh_deriv
from .models import forward_pass, backprop_deltas


def train_single(
    method: str,
    depth: int,
    freq: int,
    seed: int,
    epochs: int = 500,
    dataset: str | None = None,
) -> Tuple[List[float], float, int]:
    """Train a single network instance and return metrics."""

    # second argument of ``make_dataset`` is the number of points; the previous
    # implementation mistakenly passed ``seed`` here which resulted in empty
    # datasets when ``seed`` was 0.  Explicitly bind the keyword to avoid such
    # mixups.
    X, Y = make_dataset(freq, seed=seed, dataset=dataset)
    N = X.shape[1]
    np.random.seed(seed)

    hidden_dim, lr, alpha = 8, 0.01, 0.7
    beta, calib_k = 0.9, 50

    # ---- weight init ----
    Ws: List[np.ndarray] = []
    for d in range(depth):
        in_dim = 1 if d == 0 else hidden_dim
        init = (np.random.randn(hidden_dim, in_dim) * 0.5 if method == "Backprop" else
                 np.random.choice([-1.0, 0.0, 1.0], (hidden_dim, in_dim)))
        Ws.append(init.astype(float))
    Ws.append((np.random.randn(1, hidden_dim) * 0.5 if method == "Backprop" else
               np.random.choice([-1.0, 0.0, 1.0], (1, hidden_dim))).astype(float))

    B = [np.random.randn(hidden_dim, 1) for _ in range(depth)]
    B = [b / np.linalg.norm(b) for b in B]
    M = [np.zeros_like(w) for w in Ws]

    curve: List[float] = []
    t01_reached = epochs + 1
    for epoch in range(epochs):
        acts = forward_pass(Ws, X)
        err = acts[-1] - Y
        mse = float(np.mean(err ** 2))
        curve.append(mse)
        if mse < 0.01 and t01_reached == epochs + 1:
            t01_reached = epoch

        if method == "Backprop":
            deltas = backprop_deltas(Ws, acts, err)
            for l in range(len(Ws) - 1, -1, -1):
                Ws[l] -= lr * (deltas[l] @ acts[l].T) / N
            continue

        G_out = (err @ acts[-2].T) / N
        W_new = Ws[-1] - lr * G_out
        Ws[-1] = W_new if method in {"Vanilla DFA", "Structured DFA"} else \
                 quantize_stoch(W_new, alpha * np.mean(np.abs(W_new)))

        delta_dfa = err
        for l in reversed(range(depth)):
            pseudo = B[l] @ delta_dfa
            pseudo *= tanh_deriv(Ws[l] @ acts[l])
            grad = (pseudo @ acts[l].T) / N
            if method in {"Vanilla DFA", "Structured DFA"}:
                Ws[l] -= lr * grad
            else:
                Ws[l] = quantize_stoch(Ws[l] - lr * grad, alpha * np.mean(np.abs(Ws[l])))
                if method == "Momentum":
                    M[l] = beta * M[l] + (1 - beta) * grad
                    Ws[l] = quantize_stoch(Ws[l] - lr * M[l], alpha * np.mean(np.abs(Ws[l])))
            if method == "Ternary adaptive + cal" and epoch % calib_k == 0:
                true_d = backprop_deltas(Ws, acts, err)[l]
                Bl = (true_d @ delta_dfa.T) / (np.sum(delta_dfa ** 2) + 1e-8)
                B[l] = Bl / (np.linalg.norm(Bl) + 1e-8)

    auc = float(np.trapz(curve))
    return curve, auc, t01_reached


def sweep_and_log(
    methods: List[str],
    depths: List[int],
    freqs: List[int],
    seeds: Iterable[int],
    epochs: int,
    outdir: str,
    dataset: str | None = None,
) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)
    plots_dir = os.path.join(outdir, 'plots')
    ensure_dir(plots_dir)

    meta = {
        'timestamp': datetime.datetime.now().isoformat(),
        'methods': methods,
        'depths': depths,
        'freqs': freqs,
        'seeds': list(seeds),
        'epochs': epochs,
        'dataset': dataset or 'synthetic',
    }
    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Running sweep: dataset={meta['dataset']} methods={methods}")

    final_tbls = {m: np.zeros((len(depths), len(freqs))) for m in methods}

    for m in methods:
        # store mean curves for every depth-freq pair
        mean_curves: Dict[Tuple[int, int], np.ndarray] = {}

        for d in depths:
            for k in freqs:
                curves = []
                for s in seeds:
                    curve, _, _ = train_single(m, d, k, s, epochs, dataset)
                    curves.append(curve)
                    np.save(os.path.join(outdir, f"curve_{m.replace(' ','_')}_d{d}_k{k}_seed{s}.npy"), np.array(curve))
                mean_curve = np.mean(curves, axis=0)
                mean_curves[(d, k)] = mean_curve
                final_tbls[m][depths.index(d), freqs.index(k)] = float(np.mean([c[-1] for c in curves]))

        # save CSV of final MSE
        try:
            import pandas as pd
            pd.DataFrame(final_tbls[m], index=depths, columns=freqs).to_csv(
                os.path.join(outdir, f"final_table_{m.replace(' ','_')}.csv"))
        except ImportError:
            np.savetxt(
                os.path.join(outdir, f"final_table_{m.replace(' ','_')}.csv"),
                final_tbls[m], delimiter=',')

        print(f"{m} results:\n{final_tbls[m]}")

        # heat-map
        plt.figure(figsize=(5,4))
        plt.imshow(final_tbls[m], origin='lower', cmap='viridis')
        plt.colorbar(label='MSE')
        plt.xticks(range(len(freqs)), freqs)
        plt.yticks(range(len(depths)), depths)
        plt.xlabel('Frequency k')
        plt.ylabel('Depth')
        plt.title(f'Final MSE — {m}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"heat_{m.replace(' ','_')}.svg"))
        plt.close()

        # convergence curves grid per method
        plt.figure(figsize=(6,4))
        for (d,k), mc in mean_curves.items():
            plt.plot(mc, label=f'd{d}-k{k}')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title(f'Mean curves — {m}')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"curves_{m.replace(' ','_')}.svg"))
        plt.close()

    return final_tbls
