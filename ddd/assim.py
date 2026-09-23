"""Model-based denoising by weak-constraint 4D-Var, parallel in time.

Unknowns: the whole clean trajectory X [T, D] and the model's physical parameters p. Minimise

    sum_t |X_t - Y_t|^2 / r  +  sum_t |X_{t+1} - step(X_t; p)|^2 / q

where Y is the noisy data, r the observation-noise variance (estimated from the data) and q a small
model-error variance. Every step(X_t) is evaluated for all t at once, so one L-BFGS iteration is a
single batched model step on the GPU instead of a sequential rollout. q is lowered in stages
(continuation) so the problem stays well conditioned while the model constraint tightens.
"""

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def weak_4dvar(step, Y, X0, p0, r, q_schedule, free=None, iters=200, chunk=None, fit_from=0, verbose=False):
    """step(X [B, D], p) -> X one sample later. p0: parameter tensor; `free` boolean mask of params to fit.
    For each q in the schedule: optimise the trajectory with parameters fixed, then (from stage `fit_from`
    on) refit the free parameters to the current trajectory, alternating once more.

    Returns (X numpy [T, D], fitted params tensor)."""
    Y = torch.as_tensor(Y, device=DEV)
    X = torch.nn.Parameter(torch.as_tensor(X0, device=DEV).clone())
    p0 = torch.as_tensor(p0, device=DEV, dtype=torch.float64)
    free = torch.ones_like(p0, dtype=torch.bool) if free is None else torch.as_tensor(free, device=DEV)
    scale = torch.where(p0.abs() > 1e-6, p0, torch.full_like(p0, 1e-6))  # signed, so theta starts at 1
    theta = torch.nn.Parameter(torch.ones_like(p0))  # relative parameter units
    T = len(Y)
    chunk = chunk or T

    def params():
        return torch.where(free, theta * scale, p0)

    def model_term(q):
        total = 0.0
        for i in range(0, T - 1, chunk):  # in chunks to bound memory; gradients accumulate
            j = min(i + chunk, T - 1)
            mod = ((X[i + 1: j + 1] - step(X[i:j], params())) ** 2).sum() / q
            mod.backward()
            total += mod.item()
        return total

    def run(variables, q, with_obs):
        opt = torch.optim.LBFGS(variables, lr=1, max_iter=iters, line_search_fn="strong_wolfe", history_size=20,
                                tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            total = 0.0
            if with_obs:
                obs = ((X - Y) ** 2).sum() / r
                obs.backward()
                total += obs.item()
            total += model_term(q)
            return torch.tensor(total / T)

        opt.step(closure)

    for stage, q in enumerate(q_schedule):
        for _ in range(2 if stage >= fit_from else 1):
            run([X], q, True)
            if stage >= fit_from:
                run([theta], q, False)
        if verbose:
            with torch.no_grad():
                p = params()
                mis = ((X - Y) ** 2).mean().item()
                mod = sum(((X[i + 1: min(i + chunk, T - 1) + 1] - step(X[i: min(i + chunk, T - 1)], p)) ** 2).sum().item()
                          for i in range(0, T - 1, chunk)) / (T - 1)
            print(f"    q={q:.1e}: obs misfit/entry {mis:.4f}, model misfit/step {mod:.2e}, "
                  f"params {np.round(p.cpu().numpy()[free.cpu().numpy()], 5)}")
    return X.detach().cpu().numpy(), params().detach()
