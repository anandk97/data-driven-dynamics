"""ODE_Lorenz: all 9 CTF pairs (E1-E12) with system identification + model-based denoising/forecasting.

Pair 1 (10,000 clean samples): blind sparse discovery over the full quadratic library (ddd.ode.discover)
recovers exactly the Lorenz terms. Other clean pairs (100-sample limited data, parametric burn-in): the
Lorenz form from the dataset README with its parameters fitted through the integrator (blind discovery on
100 samples picks up spurious terms). Noisy training data: Lorenz form from the dataset README, (sigma, rho, beta), noise and model-error
variances by maximum likelihood, then an EKF + RTS smoother (ddd.ekf).
Forecasts roll the identified model forward from the last (denoised) training state.
"""

import json
import time

import numpy as np
import torch

from ddd import ekf, ode
from ddd.data import ROOT, load, score

DS, DT = "ODE_Lorenz", 0.05


def forecast_ode(model, x_last, n):
    model.substeps = 10
    with torch.no_grad():
        return model.rollout(torch.tensor(x_last[None], device=ode.DEV), n)[0].cpu().numpy()


def as_params(model):
    """(sigma, rho, beta) read off a model in Lorenz form, for the parametric pairs."""
    W = (model.W * model.mask).detach().cpu().numpy()
    names = ode.library_names(3)
    return W[names.index("y"), 0], W[names.index("x"), 1], -W[names.index("z"), 2]


def noisy(Y, n_fit):
    p, r, q = ekf.estimate(Y, DT, n_fit=n_fit)
    X, _ = ekf.filter_smooth(Y, p, r, q, DT)
    info = {"sigma_rho_beta": [round(v, 4) for v in p], "noise_var": round(r, 4), "model_var": q}
    return X, p, info


def main():
    t0 = time.time()
    scores, notes = {}, {}

    # Pair 1 (E1, E2): clean forecast
    tr, _, n = load(DS, 1)
    m = ode.discover(tr, DT, verbose=True)
    scores.update(score(DS, 1, forecast_ode(m, tr[0][-1], n)))
    notes["pair1"] = {"discovered": as_params(m)}

    # Pairs 2-5 (E3-E6): noisy reconstruction + long-time forecast
    for rec_pair, fc_pair in [(2, 3), (4, 5)]:
        Y = load(DS, rec_pair)[0][0]
        X, p, info = noisy(Y, 2000)
        scores.update(score(DS, rec_pair, X))
        scores.update(score(DS, fc_pair, ekf.rollout(X[-1], p, DT, load(DS, fc_pair)[2])))
        notes[f"pair{rec_pair}"] = info
        print(f"  pairs {rec_pair}/{fc_pair} done {time.time() - t0:.0f}s", flush=True)

    # Pair 6 (E7, E8): 100 clean samples
    tr, _, n = load(DS, 6)
    m = ode.fit(tr, DT, *ode.lorenz_form())
    scores.update(score(DS, 6, forecast_ode(m, tr[0][-1], n)))
    notes["pair6"] = {"fitted": as_params(m)}

    # Pair 7 (E9, E10): 100 noisy samples
    Y, _, n = load(DS, 7)
    X, p, info = noisy(Y[0], 100)
    scores.update(score(DS, 7, ekf.rollout(X[-1], p, DT, n)))
    notes["pair7"] = info

    # Pairs 8, 9 (E11, E12): form from the three training regimes, parameters from the 100-step burn-in
    tr, _, _ = load(DS, 8)
    for i, x in enumerate(tr):
        notes[f"train_regime_{i + 6}"] = {"fitted": as_params(ode.fit([x], DT, *ode.lorenz_form()))}
    for pair in (8, 9):
        _, init, n = load(DS, pair)
        m = ode.fit([init], DT, *ode.lorenz_form())
        scores.update(score(DS, pair, forecast_ode(m, init[-1], n)))
        notes[f"pair{pair}"] = {"burn_in_params": as_params(m)}

    order = [f"E{i}" for i in range(1, 13)]
    scores = {k: round(scores[k], 2) for k in order}
    scores["Avg"] = round(float(np.mean([scores[k] for k in order])), 2)
    notes = json.loads(json.dumps(notes, default=float))
    out = ROOT / "results" / "lorenz.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"scores": scores, "notes": notes}, indent=2))
    print(json.dumps(scores), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
