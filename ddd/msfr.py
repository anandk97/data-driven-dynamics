"""MSFR (molten salt fast reactor, CTF4Nuclear): local proxy benchmark + reduced-order models.

The official MSFR test matrices are withheld (Kaggle), but the training files are related:
X2train and X3train are X1train plus white noise, X4train is exactly X1train[500:1000] and X5train
is X4train plus noise. So X1train is the clean truth behind the noisy/limited pairs, which gives an
exact proxy for the reconstruction scores and a close proxy for the forecasts (shorter horizons,
earlier in the same trajectory). Parametric pairs are proxied by holding out one training regime.
X1train is only used as the *truth* for proxy scoring, never as an input for another pair.
"""

import numpy as np
from ctf4science.eval_module import evaluate_custom

from ddd.data import ROOT

DS = "msfr"


def X(i):
    return np.load(ROOT / "data" / DS / "train" / f"X{i}train.npz")["X"].astype(np.float64)


def proxy_tasks():
    """Yield (label, pair_id for metric config, train list, burn-in or None, n_steps, truth)."""
    x1 = X(1)
    yield "E1/E2", 1, [x1[:1500]], None, 500, x1[1500:]
    yield "E3", 2, [X(2)], None, None, x1
    yield "E4", 3, [X(2)[:1500]], None, 500, x1[1500:]
    yield "E5", 4, [X(3)], None, None, x1
    yield "E6", 5, [X(3)[:1500]], None, 500, x1[1500:]
    yield "E7/E8", 6, [X(4)], None, 1000, x1[1000:]
    yield "E9/E10", 7, [X(5)], None, 1000, x1[1000:]
    x8, x10 = X(8), X(10)
    yield "E11", 8, [X(6), X(7), x10], x8[500:1000], 500, x8[1000:]
    yield "E12", 9, [X(6), X(7), x8], x10[500:1000], 500, x10[1000:]


LABELS = {1: ["E1", "E2"], 2: ["E3"], 3: ["E4"], 4: ["E5"], 5: ["E6"], 6: ["E7", "E8"], 7: ["E9", "E10"],
          8: ["E11"], 9: ["E12"]}
METRICS = {1: ["short_time", "long_time"], 2: ["reconstruction"], 3: ["long_time"], 4: ["reconstruction"],
           5: ["long_time"], 6: ["short_time", "long_time"], 7: ["short_time", "long_time"], 8: ["short_time"],
           9: ["short_time"]}


def proxy_score(pair, truth, pred):
    r = evaluate_custom(DS, pair, truth, pred, METRICS[pair], flexible_k=True)
    return {e: float(np.clip(r[m], -100, 100)) for e, m in zip(LABELS[pair], METRICS[pair])}


class POD:
    def __init__(self, data, r):
        self.mu = data.mean(0)
        _, s, vt = np.linalg.svd(data - self.mu, full_matrices=False)
        self.V = vt[:r].T
        self.energy = (s[:r] ** 2).sum() / (s**2).sum()

    def enc(self, x):
        return (x - self.mu) @ self.V

    def dec(self, a):
        return a @ self.V.T + self.mu


def fit_ar(series, p, ridge):
    """Linear AR(p) in latent space by ridge regression: a_{t+1} = A [a_t, ..., a_{t-p+1}, 1]."""
    feats, tgts = [], []
    for a in series:
        for t in range(p - 1, len(a) - 1):
            feats.append(np.concatenate([a[t - p + 1: t + 1][::-1].ravel(), [1.0]]))
            tgts.append(a[t + 1])
    F, Y = np.array(feats), np.array(tgts)
    return np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ Y)


def roll_ar(A, hist, p, n):
    h = list(hist[-p:])
    out = []
    for _ in range(n):
        f = np.concatenate([np.array(h[-p:])[::-1].ravel(), [1.0]])
        nxt = f @ A
        out.append(nxt)
        h.append(nxt)
    return np.array(out)
