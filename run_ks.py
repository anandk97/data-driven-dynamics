"""PDE_KS: all 9 CTF pairs (E1-E12) with sparse PDE identification + model-based denoising/forecasting.

Clean training data: blind sparse discovery over an 8-term library (ddd.ks.discover), fitted through a
differentiable ETDRK4 solver; it recovers exactly u_xx, u_xxxx, u*u_x. Parametric pairs: the structure
found on the three training regimes, coefficients refitted on the 100-step burn-in.
Noisy training data: spectral Wiener filter as a starting point, then weak-constraint 4D-Var
(ddd.assim) that jointly estimates the clean trajectory and the KS coefficients (KS form from the README).
Forecasts roll the identified PDE forward from the last (denoised) training state.
"""

import json
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ddd import assim, ks
from ddd.data import ROOT, load, score

DS = "PDE_KS"
KS_MASK = (ks.KS_GUESS != 0).astype(float)


def prefilter(Y):
    """Per-frame Wiener filter in Fourier space (noise level from the flat high-wavenumber floor) plus a
    light temporal Gaussian. Returns (filtered, noise variance per grid point)."""
    U = np.fft.rfft(Y, axis=1)
    N = Y.shape[1]
    r = (np.abs(U[:, 200:500]) ** 2).mean() / N
    P = (np.abs(U) ** 2).mean(0)
    w = np.clip((P - N * r) / P, 0, 1)
    w[-1] = 1  # static Nyquist artifact
    V = U * w
    V = gaussian_filter1d(V.real, 1, axis=0) + 1j * gaussian_filter1d(V.imag, 1, axis=0)
    return np.fft.irfft(V, n=N, axis=1), r


def denoise(Y, iters=100):
    X0, r = prefilter(Y)
    # time scale of the README's KS form from a coarse scan, then a 1-step fit on the prefiltered data
    u0 = torch.tensor(X0[:-1][:: max(1, len(X0) // 256)], device=ks.DEV)
    tg = torch.tensor(X0[1:][:: max(1, len(X0) // 256)], device=ks.DEV)
    with torch.no_grad():
        scales = np.logspace(-2, 0.5, 26)
        errs = [(ks.KS(c=s * ks.KS_GUESS).to(ks.DEV).rollout(u0, 1)[:, 0] - tg).norm().item() for s in scales]
    c0 = ks.coeffs(ks.fit([X0], scales[int(np.argmin(errs))] * ks.KS_GUESS, mask=KS_MASK))
    model = ks.KS(c=c0, mask=KS_MASK).to(ks.DEV)
    X, c = assim.weak_4dvar(lambda x, c: model.rollout(x, 1, c)[:, 0], Y, X0, torch.tensor(c0), r,
                            [1e-1, 1e-2, 1e-3], free=torch.tensor(KS_MASK > 0), iters=iters,
                            chunk=1000, fit_from=1, verbose=True)
    model.c.data = c
    return X, model, {"noise_var": round(float(r), 4), "coeffs": np.round(c.cpu().numpy(), 5).tolist()}


def main():
    t0 = time.time()
    scores, notes = {}, {}

    # Pair 1 (E1, E2) and pair 6 (E7, E8): clean data, blind discovery
    for pair in (1, 6):
        tr, _, n = load(DS, pair)
        m = ks.discover(tr, verbose=True)
        scores.update(score(DS, pair, ks.forecast(m, tr[0][-1], n)))
        notes[f"pair{pair}"] = {"discovered": dict(zip(ks.TERMS, np.round(ks.coeffs(m), 5).tolist()))}
        print(f"  pair {pair} done {time.time() - t0:.0f}s", flush=True)

    # Pairs 8, 9 (E11, E12): structure from the training regimes, coefficients from the burn-in
    tr, _, _ = load(DS, 8)
    structure = ks.discover(tr, verbose=True)
    mask = structure.mask.cpu().numpy()
    for pair in (8, 9):
        _, init, n = load(DS, pair)
        m = ks.fit([init], ks.coeffs(structure), mask=mask)
        scores.update(score(DS, pair, ks.forecast(m, init[-1], n)))
        notes[f"pair{pair}"] = {"burn_in_coeffs": dict(zip(ks.TERMS, np.round(ks.coeffs(m), 5).tolist()))}
    print(f"  pairs 8/9 done {time.time() - t0:.0f}s", flush=True)

    # Pairs 2-5 (E3-E6): noisy reconstruction + long-time forecast; pair 7 (E9, E10): 100 noisy samples
    for rec_pair, fc_pair in [(2, 3), (4, 5), (None, 7)]:
        Y = load(DS, rec_pair or fc_pair)[0][0]
        X, m, info = denoise(Y)
        if rec_pair:
            scores.update(score(DS, rec_pair, X))
        scores.update(score(DS, fc_pair, ks.forecast(m, X[-1], load(DS, fc_pair)[2])))
        notes[f"pair{rec_pair or fc_pair}"] = info
        print(f"  pairs {rec_pair}/{fc_pair} done {time.time() - t0:.0f}s {scores}", flush=True)

    order = [f"E{i}" for i in range(1, 13)]
    scores = {k: round(scores[k], 2) for k in order}
    scores["Avg"] = round(float(np.mean([scores[k] for k in order])), 2)
    out = ROOT / "results" / "ks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"scores": scores, "notes": notes}, indent=2))
    print(json.dumps(scores), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
