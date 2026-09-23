"""Sparse ODE discovery (SINDy-style polynomial library) fitted through a differentiable RK4 integrator.

x' = Theta(x) @ W, Theta = all monomials of the state up to degree 2 (10 terms for 3 states).
W is fitted by matching one-step (or multi-step) rollouts of the integrator to the data, which is
much more accurate than regressing on finite-difference derivatives at dt = 0.05. Small terms
are then pruned and the rest refitted (sequential thresholding).
"""

import itertools

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def library_names(d, degree=2, var="xyz"):
    names = ["1"]
    for deg in range(1, degree + 1):
        names += ["*".join(var[i] for i in c) for c in itertools.combinations_with_replacement(range(d), deg)]
    return names


class PolyODE(torch.nn.Module):
    def __init__(self, d=3, degree=2, dt=0.05, substeps=10, W=None, mask=None, shift=None, scale=None):
        super().__init__()
        self.d, self.degree, self.dt, self.substeps = d, degree, dt, substeps
        self.combos = [c for deg in range(1, degree + 1)
                       for c in itertools.combinations_with_replacement(range(d), deg)]
        p = 1 + len(self.combos)
        self.W = torch.nn.Parameter(torch.tensor(W if W is not None else np.zeros((p, d)), dtype=torch.float64))
        self.register_buffer("mask", torch.tensor(mask if mask is not None else np.ones((p, d)), dtype=torch.float64))

    def theta(self, x):
        cols = [torch.ones_like(x[..., :1])]
        for c in self.combos:
            t = x[..., c[0]:c[0] + 1]
            for i in c[1:]:
                t = t * x[..., i:i + 1]
            cols.append(t)
        return torch.cat(cols, -1)

    def f(self, x, W):
        return self.theta(x) @ (W * self.mask)

    def rollout(self, x0, steps, W=None):
        W = self.W if W is None else W
        h = self.dt / self.substeps
        x, out = x0, []
        for _ in range(steps):
            for _ in range(self.substeps):
                k1 = self.f(x, W)
                k2 = self.f(x + h / 2 * k1, W)
                k3 = self.f(x + h / 2 * k2, W)
                k4 = self.f(x + h * k3, W)
                x = (x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).clamp(-1e4, 1e4)  # keep bad trial params finite
            out.append(x)
        return torch.stack(out, 1)


def _pairs(trajs, horizon):
    x0 = np.concatenate([x[: len(x) - horizon] for x in trajs])
    tgt = np.concatenate([np.stack([x[i + 1: i + 1 + horizon] for i in range(len(x) - horizon)]) for x in trajs])
    return torch.tensor(x0, device=DEV), torch.tensor(tgt, device=DEV)


def least_squares_init(trajs, dt, degree=2):
    """Classic SINDy: regress central-difference derivatives on the library (used only as a starting point)."""
    m = PolyODE(d=trajs[0].shape[1], degree=degree, dt=dt)
    X = np.concatenate([x[1:-1] for x in trajs])
    dX = np.concatenate([(x[2:] - x[:-2]) / (2 * dt) for x in trajs])
    with torch.no_grad():
        T = m.theta(torch.tensor(X)).numpy()
    return np.linalg.lstsq(T, dX, rcond=None)[0]


def weak_init(trajs, dt, degree=2, width=15, stride=3, threshold=0.05, rounds=5):
    """Weak-form SINDy (Messenger & Bortz 2021): integrate against compactly supported test functions,
    so derivatives fall on the smooth test function instead of the noisy data. Sparse via STLSQ."""
    m = PolyODE(d=trajs[0].shape[1], degree=degree, dt=dt)
    s = np.linspace(-1, 1, width)
    phi, dphi = (1 - s**2) ** 4, -8 * s * (1 - s**2) ** 3 / ((width - 1) / 2 * dt)
    G, b = [], []
    for x in trajs:
        with torch.no_grad():
            T = m.theta(torch.tensor(x)).numpy()
        for i in range(0, len(x) - width + 1, stride):
            G.append(dt * phi @ T[i: i + width])
            b.append(-dt * dphi @ x[i: i + width])
    G, b = np.array(G), np.array(b)
    norm = np.linalg.norm(G, axis=0)
    W = np.linalg.lstsq(G / norm, b, rcond=None)[0]
    for _ in range(rounds):  # sequential thresholding on normalised coefficients, per equation
        keep = np.abs(W) > threshold * np.abs(W).max(0, keepdims=True)
        for j in range(W.shape[1]):
            W[:, j] = 0
            W[keep[:, j], j] = np.linalg.lstsq(G[:, keep[:, j]] / norm[keep[:, j]], b[:, j], rcond=None)[0]
    return W / norm[:, None], (W != 0).astype(float)


def fit(trajs, dt, W0, mask=None, horizon=1, degree=2, substeps=10, iters=200, verbose=False):
    model = PolyODE(d=trajs[0].shape[1], degree=degree, dt=dt, substeps=substeps, W=W0, mask=mask).to(DEV)
    x0, tgt = _pairs(trajs, horizon)
    ref = (tgt - x0[:, None]).pow(2).mean()
    opt = torch.optim.LBFGS([model.W], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-16, history_size=50)

    def closure():
        opt.zero_grad()
        loss = ((model.rollout(x0, horizon) - tgt) ** 2).mean() / ref
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    if verbose:
        print(f"  fit: loss/persistence {closure().item():.3e}")
    return model


def discover(trajs, dt, threshold=1e-2, degree=2, verbose=False, **kw):
    """Fit the full quadratic library, then prune terms whose contribution is tiny, then refit."""
    W0 = least_squares_init(trajs, dt, degree)
    model = fit(trajs, dt, W0, degree=degree, verbose=verbose, **kw)
    W = model.W.detach().cpu().numpy()
    with torch.no_grad():
        T = model.theta(torch.tensor(np.concatenate(trajs), device=DEV)).cpu().numpy()
    contrib = np.abs(W) * T.std(0)[:, None]
    contrib[0] = np.abs(W[0])  # constant term
    mask = (contrib > threshold * contrib.max(0, keepdims=True)).astype(float)
    model = fit(trajs, dt, W * mask, mask=mask, degree=degree, verbose=verbose, **kw)
    if verbose:
        names = library_names(trajs[0].shape[1], degree)
        Wm = model.W.detach().cpu().numpy() * mask
        for j in range(Wm.shape[1]):
            print(f"  d{'xyz'[j]}/dt = " + " + ".join(f"{Wm[i, j]:.5g}*{names[i]}" for i in range(len(names)) if mask[i, j]))
    return model


def lorenz_form(sigma=10.0, rho=28.0, beta=8 / 3):
    """Known Lorenz structure from the dataset README (x'=s(y-x), y'=x(r-z)-y, z'=xy-bz) as (W, mask)."""
    names = library_names(3)
    W, mask = np.zeros((10, 3)), np.zeros((10, 3))
    for (term, eq), val in {("x", 0): -sigma, ("y", 0): sigma, ("x", 1): rho, ("y", 1): -1.0, ("x*z", 1): -1.0,
                            ("z", 2): -beta, ("x*y", 2): 1.0}.items():
        W[names.index(term), eq], mask[names.index(term), eq] = val, 1.0
    return W, mask
