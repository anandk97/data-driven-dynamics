"""Sparse PDE discovery for 1D periodic data, fitted through a differentiable ETDRK4 solver.

Model (all coefficients learned, time unit = one sample):
    u_t = c0*u + c1*u_x + c2*u_xx + c3*u_xxx + c4*u_xxxx + c5*u*u_x + c6*u^2 + c7*u^3

The linear terms go into the exponential integrator, the rest is the nonlinear part.
Coefficients are fitted by matching solver rollouts to the data (not finite differences, which
are useless at this sampling rate), then small terms are pruned and the rest refitted
(SINDy-style sequential thresholding).
"""

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TERMS = ["u", "u_x", "u_xx", "u_xxx", "u_xxxx", "u*u_x", "u^2", "u^3"]
KS_GUESS = np.array([0, 0, -1, 0, -1, -1, 0, 0.0])


class KS(torch.nn.Module):
    def __init__(self, n=1024, length=32 * np.pi, substeps=4, c=None, mask=None):
        super().__init__()
        self.n, self.substeps = n, substeps
        k = 2 * np.pi * np.fft.rfftfreq(n, d=length / n)
        k[-1] = 0.0  # the Nyquist mode is a static artifact in the data: keep it out of the dynamics
        self.register_buffer("ik", torch.tensor(1j * k, dtype=torch.complex128))
        self.register_buffer("dealias", torch.tensor(np.arange(len(k)) < (2 / 3) * (n // 2)))
        self.c = torch.nn.Parameter(torch.tensor(np.zeros(8) if c is None else c, dtype=torch.float64))
        self.register_buffer("mask", torch.tensor(np.ones(8) if mask is None else mask, dtype=torch.float64))

    def nonlinear(self, v, c):
        u = torch.fft.irfft(v, n=self.n)
        ux = torch.fft.irfft(self.ik * v, n=self.n)
        return torch.fft.rfft(c[5] * u * ux + c[6] * u**2 + c[7] * u**3) * self.dealias

    @staticmethod
    def etd_coeffs(lin, h, m=64):
        # Kassam & Trefethen (2005) ETDRK4 coefficients; full contour circle because lin may be complex
        r = torch.exp(2j * np.pi * (torch.arange(1, m + 1, device=lin.device, dtype=torch.float64) - 0.5) / m)
        lr = h * lin[:, None] + r[None, :]
        q = h * torch.mean((torch.exp(lr / 2) - 1) / lr, dim=1)
        f1 = h * torch.mean((-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3, dim=1)
        f2 = h * torch.mean((2 + lr + torch.exp(lr) * (-2 + lr)) / lr**3, dim=1)
        f3 = h * torch.mean((-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / lr**3, dim=1)
        return torch.exp(h * lin), torch.exp(h * lin / 2), q, f1, f2, f3

    def rollout(self, u0, steps, c=None):
        """u0: [B, n] real. Returns [B, steps, n]: the states 1..steps samples later."""
        c = (self.c if c is None else c) * self.mask
        ik = self.ik
        lin = c[0] + c[1] * ik + c[2] * ik**2 + c[3] * ik**3 + c[4] * ik**4
        e, e2, q, f1, f2, f3 = self.etd_coeffs(lin, 1.0 / self.substeps)
        v = torch.fft.rfft(u0)
        nyq = v[:, -1:]
        out = []
        for _ in range(steps):
            for _ in range(self.substeps):
                nv = self.nonlinear(v, c)
                a = e2 * v + q * nv
                na = self.nonlinear(a, c)
                b = e2 * v + q * na
                nb = self.nonlinear(b, c)
                cc = e2 * a + q * (2 * nb - nv)
                nc = self.nonlinear(cc, c)
                v = e * v + nv * f1 + 2 * (na + nb) * f2 + nc * f3
                v = torch.cat([v[:, :-1], nyq], dim=1)
            out.append(torch.fft.irfft(v, n=self.n))
        return torch.stack(out, 1)

    def features(self, u):
        """Library columns evaluated on states u [T, n], used to judge term sizes."""
        v = torch.fft.rfft(u)
        v[:, -1] = 0
        d = [torch.fft.irfft(self.ik**p * v, n=self.n) for p in range(5)]
        return [d[0], d[1], d[2], d[3], d[4], d[0] * d[1], d[0] ** 2, d[0] ** 3]


def _pairs(trajs, horizon, batch, seed=0):
    idx = [(i, t) for i, x in enumerate(trajs) for t in range(len(x) - horizon)]
    sel = np.random.default_rng(seed).choice(len(idx), size=min(batch, len(idx)), replace=False)
    u0 = np.stack([trajs[idx[s][0]][idx[s][1]] for s in sel])
    tgt = np.stack([trajs[idx[s][0]][idx[s][1] + 1: idx[s][1] + 1 + horizon] for s in sel])
    return torch.tensor(u0, device=DEV), torch.tensor(tgt, device=DEV)


def fit(trajs, c0, mask=None, horizon=1, batch=256, iters=100, verbose=False):
    """Fit coefficients so that `horizon`-step rollouts match clean trajectories (full-batch L-BFGS)."""
    model = KS(n=trajs[0].shape[1], c=c0, mask=mask).to(DEV)
    u0, tgt = _pairs(trajs, horizon, batch)
    ref = (tgt - u0[:, None]).pow(2).mean()  # loss relative to a persistence forecast
    scale = model.c.detach().abs().clamp(min=1e-3)
    theta = torch.nn.Parameter(model.c.detach() / scale)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=iters, line_search_fn="strong_wolfe", tolerance_change=1e-15)

    def closure():
        opt.zero_grad()
        loss = ((model.rollout(u0, horizon, theta * scale) - tgt) ** 2).mean() / ref
        loss.backward()
        return loss

    opt.step(closure)
    model.c.data = (theta * scale).detach()
    if verbose:
        print(f"    fit: loss/persistence {closure().item():.2e}  coeffs {np.round(coeffs(model), 5)}")
    return model


def coeffs(model):
    return (model.c * model.mask).detach().cpu().numpy()


def discover(trajs, threshold=0.01, verbose=False):
    """Scan the time scale of a generic KS guess, fit all 8 library terms, prune small ones, refit."""
    u0, tgt = _pairs(trajs, 1, 64, seed=1)
    errs = []
    with torch.no_grad():
        for s in np.logspace(-3, 0, 31):
            p = KS(n=trajs[0].shape[1], c=s * KS_GUESS).to(DEV).rollout(u0, 1)
            errs.append(((p - tgt).norm() / (tgt - u0[:, None]).norm()).item())
    s = np.logspace(-3, 0, 31)[int(np.argmin(errs))]
    c0 = s * KS_GUESS + 1e-4 * (KS_GUESS == 0)
    model = fit(trajs, c0, verbose=verbose)
    mask = np.ones(8)
    for _ in range(3):  # sequential thresholding
        with torch.no_grad():
            x = torch.tensor(np.concatenate(trajs)[:: max(1, sum(map(len, trajs)) // 300)], device=DEV)
            size = np.array([f.std().item() for f in model.features(x)])
        contrib = np.abs(coeffs(model)) * size
        new = mask * (contrib > threshold * contrib.max())
        if verbose:
            print("    kept terms:", [t for t, m in zip(TERMS, new) if m])
        if (new == mask).all():
            break
        mask = new
        model = fit(trajs, coeffs(model) * mask, mask=mask, verbose=verbose)
    return model


def forecast(model, u_last, steps):
    with torch.no_grad():
        return model.rollout(torch.tensor(u_last[None], device=DEV), steps)[0].cpu().numpy()
