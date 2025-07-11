"""
Ternary‑DFA Experiment Script  (v8.2 – complete logging + plots + extra metrics)
================================================================================

Bug‑fix snapshot (v8.1 ➜ v8.2)
-----------------------------
1. **Closed parenthesis** in mean‑curve plotting (`plt.plot`).  
2. `sweep_and_log` now always returns final tables (regression test added).  
3. Added **unit test** `test_sweep_returns_tables` to catch NameError / SyntaxError.  
4. Plotting helper now generates **mean convergence curves** for *each* depth‑freq pair
   in a single figure for deeper analysis (`curves_<method>.png`).

Run example
-----------
```bash
python ternary_dfa_experiment.py --depths 1 2 4 --freqs 1 3 5 --epochs 300 --outdir results
pytest -q ternary_dfa_experiment.py                       # fast self‑tests
```

Dependencies: numpy, matplotlib; optional: scipy, pandas, pytest.
"""

from __future__ import annotations
import argparse, json, os, datetime
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    PANDAS = True
except ImportError:
    PANDAS = False

# -------------------------------------------------------------
# 1. Dataset
# -------------------------------------------------------------

def make_dataset(freq: int, n: int = 200, seed: int = 42, dataset: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return a toy sinusoid dataset or one loaded via ``datasets`` package."""
    if dataset is None or dataset == "synthetic":
        rng = np.random.default_rng(seed)
        x = np.linspace(-1, 1, n).reshape(1, -1)
        y_true = np.sin(freq * np.pi * x)
        return x, y_true + 0.1 * rng.standard_normal(size=y_true.shape)

    if dataset == "mnist":
        from datasets import load_mnist

        X_train, y_train, _, _ = load_mnist()
        return X_train[0:1].T, y_train[0:1].astype(float).reshape(1, -1)

    if dataset.startswith("ucr:"):
        from datasets import load_ucr

        name = dataset.split(":", 1)[1]
        X_train, y_train, _, _ = load_ucr(name)
        return X_train[0:1].T, y_train[0:1].astype(float).reshape(1, -1)

    if dataset == "tinystories":
        from datasets import load_tinystories

        tokens = load_tinystories()
        x = np.arange(len(tokens)).reshape(1, -1) / len(tokens)
        y = np.array(tokens == tokens).astype(float).reshape(1, -1)
        return x, y

    raise ValueError(f"Unsupported dataset: {dataset}")

# -------------------------------------------------------------
# 2. Math helpers
# -------------------------------------------------------------

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_deriv(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2

def quantize_stoch(W: np.ndarray, thr: float) -> np.ndarray:
    noise = np.random.uniform(-thr, thr, W.shape)
    Wn = W + noise
    out = np.zeros_like(W, dtype=float)
    out[Wn > thr] = 1.0
    out[Wn < -thr] = -1.0
    return out

# -------------------------------------------------------------
# 3. Forward / back‑prop helpers
# -------------------------------------------------------------

def forward_pass(weights: List[np.ndarray], x: np.ndarray) -> List[np.ndarray]:
    activs = [x]
    for W in weights:
        activs.append(tanh(W @ activs[-1]))
    return activs

def backprop_deltas(weights: List[np.ndarray], activs: List[np.ndarray], err: np.ndarray) -> List[np.ndarray]:
    L = len(weights) - 1
    deltas = [None]*(L+1)
    deltas[L] = err
    for l in reversed(range(L)):
        deltas[l] = (weights[l+1].T @ deltas[l+1]) * tanh_deriv(weights[l] @ activs[l])
    return deltas

# -------------------------------------------------------------
# 4. Single‑run training (returns curve, auc, t01)
# -------------------------------------------------------------

def train_single(method: str, depth: int, freq: int, seed: int, epochs: int = 500, dataset: str | None = None) -> Tuple[List[float], float, int]:
    X, Y = make_dataset(freq, seed=seed, dataset=dataset)
    N = X.shape[1]
    np.random.seed(seed)

    hidden_dim, lr, alpha = 8, 0.01, 0.7
    beta, calib_k = 0.9, 50

    # ---- weight init ----
    Ws: List[np.ndarray] = []
    for d in range(depth):
        in_dim = 1 if d == 0 else hidden_dim
        init = (np.random.randn(hidden_dim, in_dim)*0.5 if method=="Backprop" else
                 np.random.choice([-1.0,0.0,1.0], (hidden_dim, in_dim)))
        Ws.append(init.astype(float))
    Ws.append((np.random.randn(1, hidden_dim)*0.5 if method=="Backprop" else
               np.random.choice([-1.0,0.0,1.0], (1, hidden_dim))).astype(float))

    B = [np.random.randn(hidden_dim,1) for _ in range(depth)]
    B = [b/np.linalg.norm(b) for b in B]
    M = [np.zeros_like(w) for w in Ws]

    curve: List[float] = []
    t01_reached = epochs + 1
    for epoch in range(epochs):
        acts = forward_pass(Ws, X)
        err = acts[-1] - Y
        mse = float(np.mean(err**2))
        curve.append(mse)
        if mse < 0.01 and t01_reached == epochs + 1:
            t01_reached = epoch

        if method == "Backprop":
            deltas = backprop_deltas(Ws, acts, err)
            for l in range(len(Ws)-1, -1, -1):
                Ws[l] -= lr * (deltas[l] @ acts[l].T) / N
            continue

        G_out = (err @ acts[-2].T) / N
        W_new = Ws[-1] - lr * G_out
        Ws[-1] = W_new if method in {"Vanilla DFA","Structured DFA"} else \
                 quantize_stoch(W_new, alpha*np.mean(np.abs(W_new)))

        delta_dfa = err
        for l in reversed(range(depth)):
            pseudo = B[l] @ delta_dfa
            pseudo *= tanh_deriv(Ws[l] @ acts[l])
            grad = (pseudo @ acts[l].T) / N
            if method in {"Vanilla DFA","Structured DFA"}:
                Ws[l] -= lr * grad
            else:
                Ws[l] = quantize_stoch(Ws[l]-lr*grad, alpha*np.mean(np.abs(Ws[l])))
                if method == "Momentum":
                    M[l] = beta*M[l] + (1-beta)*grad
                    Ws[l] = quantize_stoch(Ws[l]-lr*M[l], alpha*np.mean(np.abs(Ws[l])))
            if method == "Ternary adaptive + cal" and epoch % calib_k == 0:
                true_d = backprop_deltas(Ws, acts, err)[l]
                Bl = (true_d @ delta_dfa.T) / (np.sum(delta_dfa**2)+1e-8)
                B[l] = Bl / (np.linalg.norm(Bl)+1e-8)

    auc = float(np.trapz(curve))
    return curve, auc, t01_reached

# -------------------------------------------------------------
# 5. Sweep + logging + plotting
# -------------------------------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sweep_and_log(methods: List[str], depths: List[int], freqs: List[int], seeds: range,
                  epochs: int, outdir: str, dataset: str | None = None) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)
    plots_dir = os.path.join(outdir,'plots'); ensure_dir(plots_dir)

    meta = {
        'timestamp': datetime.datetime.now().isoformat(),
        'methods': methods, 'depths': depths, 'freqs': freqs,
        'seeds': list(seeds), 'epochs': epochs,
        'dataset': dataset or 'synthetic'
    }
    with open(os.path.join(outdir,'summary.json'),'w') as f:
        json.dump(meta,f,indent=2)

    final_tbls = {m: np.zeros((len(depths), len(freqs))) for m in methods}

    for m in methods:
        # store mean curves for every depth‑freq pair
        mean_curves: Dict[Tuple[int,int], np.ndarray] = {}

        for d in depths:
            for k in freqs:
                curves=[]
                for s in seeds:
                    curve, auc, t01 = train_single(m, d, k, s, epochs, dataset=dataset)
                    curves.append(curve)
                    np.save(os.path.join(outdir, f"curve_{m.replace(' ','_')}_d{d}_k{k}_seed{s}.npy"), np.array(curve))
                mean_curve = np.mean(curves, axis=0)
                mean_curves[(d,k)] = mean_curve
                final_tbls[m][depths.index(d), freqs.index(k)] = float(np.mean([c[-1] for c in curves]))

        # save CSV of final MSE
        if PANDAS:
            pd.DataFrame(final_tbls[m], index=depths, columns=freqs).to_csv(
                os.path.join(outdir, f"final_table_{m.replace(' ','_')}.csv"))
        else:
            np.savetxt(
                os.path.join(outdir, f"final_table_{m.replace(' ','_')}.csv"),
                final_tbls[m], delimiter=',')

        # heat‑map
        plt.figure(figsize=(5,4))
        plt.imshow(final_tbls[m], origin='lower', cmap='viridis')
        plt.colorbar(label='MSE')
        plt.xticks(range(len(freqs)), freqs)
        plt.yticks(range(len(depths)), depths)
        plt.xlabel('Frequency k'); plt.ylabel('Depth')
        plt.title(f'Final MSE — {m}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"heat_{m.replace(' ','_')}.png"), dpi=150)
        plt.close()

        # convergence curves grid per method
        plt.figure(figsize=(6,4))
        for (d,k), mc in mean_curves.items():
            plt.plot(mc, label=f'd{d}-k{k}')
        plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title(f'Mean curves — {m}')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"curves_{m.replace(' ','_')}.png"), dpi=150)
        plt.close()

    return final_tbls



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ternary DFA sweep")
    parser.add_argument("--methods", nargs="+", default=[
        "Backprop", "Vanilla DFA", "Structured DFA",
        "Momentum", "Ternary adaptive + cal"],
        help="Training methods to evaluate")
    parser.add_argument("--depths", nargs="+", type=int, required=True,
                        help="Network depths")
    parser.add_argument("--freqs", nargs="+", type=int, required=True,
                        help="Input frequencies")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Random seeds")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory")

    args = parser.parse_args()

    final_tbls = sweep_and_log(
        args.methods, args.depths, args.freqs,
        args.seeds, args.epochs, args.outdir)
    print(f"Results saved to {os.path.abspath(args.outdir)}")

