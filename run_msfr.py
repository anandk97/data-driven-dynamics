"""MSFR (CTF4Nuclear): proxy benchmark of learned reduced-order models vs denoise + persistence.

The official test set is withheld, so everything here is scored on the local proxy in ddd/msfr.py.
Candidates:
  - DMD / linear AR in a 30-mode POD space (the family of the paper's winner, PyKoopman)
  - damped AR (the DMD forecast relaxed toward persistence)
  - ours: Gavish-Donoho optimal singular-value hard thresholding for the noisy pairs, persistence of the
    (denoised) last frame for every forecast
Because our forecasts equal the paper's "Baseline Last" predictions (up to denoising the last frame), the
paper's hidden-test scores for that baseline give an estimate of our hidden-test forecast scores.
"""

import json

import numpy as np

from ddd import msfr
from ddd.data import ROOT

# Paper (CTF4Nuclear, Table 1): hidden-test scores
PYKOOPMAN = [87.46, 31.12, 90.91, 71.81, 86.92, 85.00, 92.72, 61.21, 92.59, 65.95, 43.36, 42.56]
BASELINE_LAST = [81.20, 71.79, 25.45, 71.79, -25.46, 71.79, 85.90, 49.52, 85.88, 49.53, 94.03, 90.65]
E = [f"E{i}" for i in range(1, 13)]


def svht(Y):
    """Gavish & Donoho (2014) optimal hard threshold for singular values, noise level unknown."""
    mu = Y.mean(0)
    U, s, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    m, n = sorted(Y.shape)
    beta = m / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    r = int((s > omega * np.median(s)).sum())
    return (U[:, :r] * s[:r]) @ Vt[:r] + mu


def main():
    rows = {"Ours: SVHT denoise + persistence": {}, "DMD / AR(1), 30 POD modes": {},
            "Damped DMD (tau = 20 steps)": {}, "Persistence of raw last frame": {}}
    for label, pair, train, init, n, truth in msfr.proxy_tasks():
        noisy = pair in (2, 3, 4, 5, 7)
        if n is None:  # reconstruction
            rows["Ours: SVHT denoise + persistence"].update(msfr.proxy_score(pair, truth, svht(train[0])))
            rows["Persistence of raw last frame"].update(
                msfr.proxy_score(pair, truth, np.tile(train[0][-1], (len(truth), 1))))
            rows["DMD / AR(1), 30 POD modes"].update(msfr.proxy_score(pair, truth, train[0]))
            rows["Damped DMD (tau = 20 steps)"].update(msfr.proxy_score(pair, truth, train[0]))
            continue
        clean = [svht(x) for x in train] if noisy else train
        hist = init if init is not None else clean[0]
        last = np.tile(hist[-1], (n, 1))
        pod = msfr.POD(np.concatenate(clean), 30)
        A = msfr.fit_ar([pod.enc(x) for x in clean], 1, 1e-3)
        ar = pod.dec(msfr.roll_ar(A, pod.enc(hist), 1, n))
        w = np.exp(-np.arange(1, n + 1) / 20)[:, None]
        rows["Ours: SVHT denoise + persistence"].update(msfr.proxy_score(pair, truth, last))
        rows["DMD / AR(1), 30 POD modes"].update(msfr.proxy_score(pair, truth, ar))
        rows["Damped DMD (tau = 20 steps)"].update(msfr.proxy_score(pair, truth, last + w * (ar - last)))
        raw_hist = init if init is not None else train[0]
        rows["Persistence of raw last frame"].update(msfr.proxy_score(pair, truth, np.tile(raw_hist[-1], (n, 1))))
        print(label, "done", flush=True)
    proxy = {k: {e: round(v[e], 2) for e in E} | {"Avg": round(float(np.mean([v[e] for e in E])), 2)}
             for k, v in rows.items()}
    ours = proxy["Ours: SVHT denoise + persistence"]
    # hidden-test estimate: forecasts = paper's Baseline Last; reconstructions = our proxy (exact truth)
    est = [ours[e] if e in ("E3", "E5") else BASELINE_LAST[i] for i, e in enumerate(E)]
    out = {"proxy": proxy, "hidden_estimate": {e: est[i] for i, e in enumerate(E)} | {"Avg": round(float(np.mean(est)), 2)},
           "paper": {"PyKoopman": PYKOOPMAN, "Baseline Last": BASELINE_LAST}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "msfr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
