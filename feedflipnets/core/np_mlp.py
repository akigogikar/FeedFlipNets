from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

Array = np.ndarray

def relu(x: Array) -> Array:
    return np.maximum(0.0, x)

def softmax(z: Array) -> Array:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float64)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float64)

def one_hot(y: Array, num_classes: int) -> Array:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def accuracy(y_true: Array, y_pred_logits: Array) -> float:
    return float((y_true == y_pred_logits.argmax(axis=1)).mean())

@dataclass
class MLP:
    d: int = 32
    h: int = 32
    c: int = 2
    seed: int = 123

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.1, size=(self.d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = rng.normal(0, 0.1, size=(self.h, self.c))
        self.b2 = np.zeros(self.c)

    def forward(self, X: Array) -> Tuple[Array, Array, Array]:
        a1 = X @ self.W1 + self.b1
        h  = relu(a1)
        z  = h @ self.W2 + self.b2
        p  = softmax(z)
        return a1, h, p

    def params(self) -> Dict[str, Array]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def apply(self, grads: Dict[str, Array], lr: float):
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

def ce_loss(p: Array, y_oh: Array) -> float:
    eps = 1e-12
    return float(-(y_oh * np.log(p + eps)).sum(axis=1).mean())

def _backprop_grads(X: Array, a1: Array, h: Array, p: Array, y_oh: Array, params: Dict[str, Array]) -> Dict[str, Array]:
    n = X.shape[0]
    dL_dz = (p - y_oh) / n
    dW2   = h.T @ dL_dz
    db2   = dL_dz.sum(axis=0)
    dL_dh = dL_dz @ params["W2"].T
    dL_da1= dL_dh * (a1 > 0)
    dW1   = X.T @ dL_da1
    db1   = dL_da1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

class BackpropStrategy:
    def grads(self, X, y, model: MLP):
        y_oh = one_hot(y, model.c)
        a1, h, p = model.forward(X)
        return _backprop_grads(X, a1, h, p, y_oh, model.params())

class DFAStrategy:
    def __init__(self, hidden: int, classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.B = rng.normal(0, 1.0, size=(classes, hidden))  # (c -> h)

    def grads(self, X, y, model: MLP):
        y_oh = one_hot(y, model.c)
        a1, h, p = model.forward(X)
        n = X.shape[0]
        dL_dz = (p - y_oh) / n
        dW2   = h.T @ dL_dz
        db2   = dL_dz.sum(axis=0)
        dL_dh = dL_dz @ self.B
        dL_da1= dL_dh * (a1 > 0)
        dW1   = X.T @ dL_da1
        db1   = dL_da1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def _ternary(g: Array, tau: float) -> Array:
    s = np.sign(g)
    m = (np.abs(g) >= tau).astype(g.dtype)
    return s * m

class FlipTernaryStrategy:
    def __init__(self, hidden: int, classes: int, tau_frac: float = 0.33):
        self.tau_frac = tau_frac
    def grads(self, X, y, model: MLP):
        y_oh = one_hot(y, model.c)
        a1, h, p = model.forward(X)
        grads = _backprop_grads(X, a1, h, p, y_oh, model.params())
        q = {}
        for k, g in grads.items():
            tau = np.percentile(np.abs(g), self.tau_frac * 100.0)
            q[k] = _ternary(g, tau=tau)
        return q

def make_dataset(n: int = 512, d: int = 32, seed: int = 123) -> Tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    m0 = rng.normal(0.0, 1.0, size=d)
    m1 = rng.normal(0.5, 1.0, size=d)
    X0 = rng.normal(m0, 1.0, size=(n // 2, d))
    X1 = rng.normal(m1, 1.0, size=(n - n // 2, d))
    X  = np.vstack([X0, X1]).astype(np.float64)
    y  = np.concatenate([np.zeros(X0.shape[0], dtype=np.int64),
                         np.ones(X1.shape[0],  dtype=np.int64)])
    idx = rng.permutation(n)
    return X[idx], y[idx]

def train_one(strategy_name: str, seed: int, steps: int = 64, lr: float = 0.05, batch: int = 64) -> Dict[str, float]:
    d, h, c = 32, 32, 2
    X, y = make_dataset(n=512, d=d, seed=seed)
    model = MLP(d=d, h=h, c=c, seed=seed + 1000)
    if strategy_name == "bp":
        strat = BackpropStrategy()
    elif strategy_name == "dfa":
        strat = DFAStrategy(hidden=h, classes=c, seed=seed + 2000)
    elif strategy_name == "flip":
        strat = FlipTernaryStrategy(hidden=h, classes=c, tau_frac=0.33)
    else:
        raise ValueError("unknown strategy")
    rng = np.random.default_rng(seed + 3000)
    n = X.shape[0]
    for _ in range(steps):
        idx = rng.choice(n, size=batch, replace=False)
        grads = strat.grads(X[idx], y[idx], model)
        model.apply(grads, lr=lr)
    _, _, p = model.forward(X)
    return {"final_loss": ce_loss(p, one_hot(y, c)), "final_acc": accuracy(y, p)}
