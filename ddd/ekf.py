"""Extended Kalman filter + Rauch-Tung-Striebel smoother for the Lorenz family, with maximum-likelihood
estimation of (sigma, rho, beta) and the observation noise from the noisy series itself.

Pure numpy: the state is only 3-dimensional, so a sequential smoother is fast and, unlike window
4D-Var on a chaotic system, has no local-minimum problem.
"""

import numpy as np
import scipy.optimize


def f(x, p):
    s, r, b = p
    return np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])


def jac(x, p):
    s, r, b = p
    return np.array([[-s, s, 0.0], [r - x[2], -1.0, -x[0]], [x[1], x[0], -b]])


def step(x, p, dt, sub=10):
    """RK4 over one sample with `sub` substeps, propagating the tangent-linear map M alongside."""
    h = dt / sub
    M = np.eye(3)
    for _ in range(sub):
        k1 = f(x, p)
        K1 = jac(x, p) @ M
        x2 = x + h / 2 * k1
        k2 = f(x2, p)
        K2 = jac(x2, p) @ (M + h / 2 * K1)
        x3 = x + h / 2 * k2
        k3 = f(x3, p)
        K3 = jac(x3, p) @ (M + h / 2 * K2)
        x4 = x + h * k3
        k4 = f(x4, p)
        K4 = jac(x4, p) @ (M + h * K3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        M = M + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return x, M


def rollout(x, p, dt, n, sub=10):
    out = []
    for _ in range(n):
        x, _ = step(x, p, dt, sub)
        out.append(x)
    return np.array(out)


def filter_smooth(Y, p, r, q, dt, smooth=True, sub=10):
    """Returns (smoothed states [T,3], negative log-likelihood of the innovations)."""
    T = len(Y)
    I = np.eye(3)
    R, Q = r * I, q * I
    xf, Pf, xp, Pp, Ms = np.zeros((T, 3)), np.zeros((T, 3, 3)), np.zeros((T, 3)), np.zeros((T, 3, 3)), np.zeros((T, 3, 3))
    x, P = Y[0].copy(), 4 * R
    nll = 0.0
    for t in range(T):
        if t > 0:
            x, M = step(x, p, dt, sub)
            P = M @ P @ M.T + Q
            Ms[t] = M
        xp[t], Pp[t] = x, P
        S = P + R
        innov = Y[t] - x
        Sinv = np.linalg.inv(S)
        nll += 0.5 * (innov @ Sinv @ innov + np.log(np.linalg.det(S)))
        K = P @ Sinv
        x = x + K @ innov
        P = (I - K) @ P
        xf[t], Pf[t] = x, P
        if not np.isfinite(x).all() or np.abs(x).max() > 1e4:
            return None, np.inf
    if not smooth:
        return xf, nll
    xs = xf.copy()
    for t in range(T - 2, -1, -1):  # RTS backward pass
        G = Pf[t] @ Ms[t + 1].T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xf[t] + G @ (xs[t + 1] - xp[t + 1])
    return xs, nll


def estimate(Y, dt, p0=(10.0, 28.0, 8 / 3), n_fit=2000, sub=5):
    """ML fit of (sigma, rho, beta, noise var r, model-error var q) on the first n_fit frames (Nelder-Mead)."""
    Yf = Y[:n_fit]
    d2 = Yf[2:] - 2 * Yf[1:-1] + Yf[:-2]
    r0 = max(np.var(d2) / 6, 1e-3)  # rough: second differences are dominated by white noise (var 6r)

    def nll(v):
        _, val = filter_smooth(Yf, (v[0], v[1], v[2]), np.exp(v[3]), np.exp(v[4]), dt, smooth=False, sub=sub)
        return val

    # the likelihood surface is multimodal in rho at high noise: coarse scan, then refine the best start.
    # q (model-error variance) is fitted jointly because it trades off against r and the EKF linearisation.
    starts = [[p0[0], rho, p0[2], np.log(r), np.log(q)] for rho in (20, 24, 28, 32, 36, 40)
              for r in (r0 / 2, r0, 2 * r0) for q in (1e-4, 1e-2)]
    best = min(starts, key=nll)
    res = scipy.optimize.minimize(nll, best, method="Nelder-Mead",
                                  options={"maxfev": 400, "xatol": 1e-4, "fatol": 1e-3})
    p, r, q = tuple(res.x[:3]), float(np.exp(res.x[3])), float(max(np.exp(res.x[4]), 1e-4))
    return p, r, q
