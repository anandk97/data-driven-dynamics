# data-driven-dynamics

Trying to beat the best models from the **Common Task Framework (CTF) for scientific ML**
([Wyder et al., NeurIPS 2025 Datasets & Benchmarks](https://arxiv.org/abs/2510.23166),
[code](https://github.com/CTF-for-Science/ctf4science), [data on OSF](https://osf.io/6rzhm/)).

**Result:** on both datasets with a public test set, the approach here beats every model in the
paper's leaderboard by a wide margin: Lorenz **78.85** vs 64.54 (LSTM), Kuramoto-Sivashinsky
**83.41** vs 18.88 (reservoir computing). The approach identifies the governing equation from the
training data and then forecasts or denoises with it, rather than training a black-box forecaster.

On the nuclear reactor dataset MSFR ([CTF4Nuclear](https://arxiv.org/abs/2605.15549)) the test set is
withheld. There, an *estimated* **79.33** vs 70.97 (PyKoopman) comes from denoising plus persistence;
no learned model beat persistence on a local proxy (details below).

**Code:** [`run_lorenz.py`](run_lorenz.py) · [`run_ks.py`](run_ks.py) · [`run_msfr.py`](run_msfr.py) · [`ddd/`](ddd/) (model
identification, solvers, denoising) · [`leaderboard.py`](leaderboard.py) (rebuilds the tables below) ·
[`download_data.py`](download_data.py)

## Leaderboards

Scores are the official CTF metrics, computed with the `ctf4science` package on the public test
sets. Each score runs from -100 to 100, where 100 is perfect. **Avg** is the mean of E1–E12 and is
the paper's ranking metric.

The paper rows are its Table 1: the top 5 models, plus SINDy and the best naive baseline. **Bold**
per-metric scores beat every paper model on that metric.

| Metrics | What they test |
|---|---|
| E1, E2 | Short- and long-term forecast |
| E3–E6 | Reconstruction and forecast from medium and high-noise data |
| E7–E10 | Forecast from only 100 samples, clean and noisy |
| E11, E12 | Interpolation and extrapolation to unseen parameters |

### Lorenz (`ODE_Lorenz`, 3-D chaotic ODE)

<!-- LEADERBOARD:lorenz:START -->
| Rank | Model | **Avg** | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Ours: identified model + 4D-Var/EKF** | **78.85** | **100.0** | **58.5** | **99.6** | **72.5** | **99.1** | **72.3** | **81.3** | **70.8** | 45.1 | **47.1** | **100.0** | **100.0** |
| 2 | LSTM | 64.54 | 99.3 | 52.4 | 97.4 | 50.8 | 96.4 | 72.1 | 66.6 | 32.8 | 36.4 | 29.6 | 80.6 | 60.1 |
| 3 | DeepONet | 57.80 | 99.1 | 45.3 | 96.2 | 62.3 | 91.8 | 14.3 | 21.8 | 40.8 | 34.5 | 25.1 | 85.5 | 76.7 |
| 4 | Reservoir | 54.87 | 99.9 | 56.7 | 97.4 | -33.5 | 95.1 | 19.0 | 65.2 | 61.9 | 45.4 | -48.5 | 99.8 | 100.0 |
| 5 | KAN | 47.28 | 82.9 | 20.5 | 96.2 | 55.2 | 93.0 | 66.7 | 50.5 | -2.5 | 33.7 | -31.5 | 41.7 | 60.9 |
| 6 | ODE-LSTM | 41.67 | 97.8 | 17.1 | 97.9 | 69.9 | 96.5 | -82.4 | 55.8 | 15.2 | 36.8 | 14.6 | 39.9 | 41.0 |
| 7 | SINDy | 19.19 | 81.8 | 36.0 | 36.9 | -96.8 | -29.2 | -18.3 | 55.8 | 15.2 | 36.8 | 14.6 | 82.4 | 15.1 |
| 8 | Baseline: average | -4.73 | 51.7 | -91.2 | 54.9 | -91.9 | 56.5 | -91.3 | 66.0 | -91.1 | 51.9 | -90.3 | 57.1 | 60.9 |
<!-- LEADERBOARD:lorenz:END -->

### Kuramoto-Sivashinsky (`PDE_KS`, 1024-point chaotic PDE)

<!-- LEADERBOARD:ks:START -->
| Rank | Model | **Avg** | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Ours: identified model + 4D-Var/EKF** | **83.41** | **100.0** | **100.0** | **95.1** | **64.3** | **87.2** | **61.0** | **100.0** | **83.0** | **26.2** | **84.2** | **100.0** | **100.0** |
| 2 | Reservoir | 18.88 | 100.0 | 88.8 | 88.6 | 23.5 | 80.7 | -2.6 | -12.4 | -12.6 | -100.0 | -100.0 | 32.4 | 40.1 |
| 3 | LSTM | 15.61 | 95.2 | -1.9 | 90.1 | -43.4 | 79.8 | -27.5 | 7.3 | 48.7 | 4.5 | 28.8 | -54.1 | -40.3 |
| 4 | ODE-LSTM | 11.85 | 80.1 | 0.5 | 88.7 | -31.5 | 52.2 | -47.0 | 1.7 | 49.5 | 6.4 | 8.5 | -54.1 | -12.8 |
| 5 | DeepONet | 6.99 | 36.5 | 17.4 | -1.4 | 6.5 | 6.3 | 24.5 | -9.5 | 1.5 | -1.9 | -0.1 | -4.6 | 8.8 |
| 6 | KAN | 6.54 | -4.4 | 4.9 | 50.4 | 5.3 | 36.9 | 24.7 | -22.5 | 26.5 | -43.1 | 1.8 | 0.8 | -2.8 |
| 7 | Baseline: zeros | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | SINDy | -3.00 | 84.4 | -13.8 | -2.9 | -100.0 | -1.2 | -88.5 | -0.2 | 45.4 | -14.0 | 34.4 | 10.0 | 10.5 |
<!-- LEADERBOARD:ks:END -->

### Molten Salt Fast Reactor (`msfr`, CTF4Nuclear, 5 coupled fields on 3,880 mesh nodes)

The MSFR test set is withheld, so this row is an **estimate, not a verified score**. The paper rows
are CTF4Nuclear's Table 1.

How the estimate is built:
- **Forecasts (E1, E2, E4, E6–E12):** these predictions repeat the last (denoised) training frame,
  which is what the paper's "last frame" baseline predicts. So these cells are that baseline's
  published hidden-test scores.
- **Reconstructions (E3, E5):** scored locally against `X1train`.
  - `X2train` and `X3train` are exactly `X1train` plus white noise, so `X1train` is the clean truth.
  - The local proxy reproduces the paper's hidden-test baseline scores on E5, E7 and E9 to 0.01, which
    confirms this.

<!-- LEADERBOARD:msfr:START -->
| Rank | Model | **Avg** | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Ours: SVHT denoise + persistence (estimated)** | **79.33** | 81.2 | 71.8 | **100.0** | 71.8 | **99.9** | 71.8 | 85.9 | 49.5 | 85.9 | 49.5 | 94.0 | 90.7 |
| 2 | PyKoopman | 70.97 | 87.5 | 31.1 | 90.9 | 71.8 | 86.9 | 85.0 | 92.7 | 61.2 | 92.6 | 66.0 | 43.4 | 42.6 |
| 3 | Baseline: last frame | 62.67 | 81.2 | 71.8 | 25.4 | 71.8 | -25.5 | 71.8 | 85.9 | 49.5 | 85.9 | 49.5 | 94.0 | 90.7 |
| 4 | Reservoir | 57.22 | 92.4 | 87.4 | 98.8 | 88.4 | 98.8 | 66.0 | 92.1 | 92.8 | -100.0 | -100.0 | 86.1 | 83.8 |
| 5 | SINDy | 55.52 | 75.3 | 42.9 | 41.4 | 42.9 | 41.5 | 42.8 | 73.8 | 52.9 | 73.8 | 52.9 | 70.1 | 55.9 |
| 6 | Opt DMD | 53.14 | 72.1 | 41.2 | 90.7 | 41.2 | 91.3 | 41.1 | 60.9 | 50.1 | 61.9 | 52.3 | 36.6 | -1.8 |
| 7 | ODE-LSTM | 52.71 | 80.7 | 61.4 | 5.7 | 30.1 | 93.0 | 52.5 | 78.3 | 42.3 | 34.4 | 66.8 | 56.5 | 30.7 |
| 8 | LSTM | 50.06 | 49.3 | 30.4 | 97.0 | 14.2 | 96.4 | 12.2 | 81.4 | 49.2 | 58.0 | 23.6 | 59.1 | 30.0 |
<!-- LEADERBOARD:msfr:END -->

Local proxy results (see [`ddd/msfr.py`](ddd/msfr.py)):
- **Proxy averages:** DMD in 30 POD modes 48.9, damped DMD 57.8. Denoise plus persistence scores
  higher, because persistence won every short-horizon forecast.
- **Why:** MSFR is driven by external forcing (pump velocity, heat sink) that is not in the data, so
  there is no closed dynamics to learn from 500–2,000 frames.
- **Damped-trend extrapolation** helped E1 (+6) but hurt E7, E11 and E12, so I did not use it.
- **Reconstruction:** the metric uses the matrix 2-norm, which barely registers white noise, so even
  the raw noisy input scores 99.98 and 99.88. Optimal singular-value thresholding (Gavish–Donoho)
  keeps those scores. The paper's best is 98.85.

## Datasets

I downloaded 6 CTF datasets from OSF. Only two have public test sets, so only those two can be scored
exactly here. MSFR is scored through the proxy and estimate described above.

| Dataset | Size | Test set | Best in the literature |
|---|---|---|---|
| `ODE_Lorenz`: Lorenz system | 2 MB | public | LSTM 64.54 (NeurIPS paper) |
| `PDE_KS`: Kuramoto-Sivashinsky | 0.7 GB | public | Reservoir 18.88 (NeurIPS paper) |
| `seismo`: global seismic wavefields | 0.2 GB | withheld (Kaggle) | LSTM 13.18 ([Seismic CTF](https://arxiv.org/abs/2512.19927)) |
| `ocean_das`: ocean fiber-optic DAS | 0.2 GB | withheld (Kaggle) | PyKoopman 12.70 ([Seismic CTF](https://arxiv.org/abs/2512.19927)) |
| `sst`: sea surface temperature | 3.6 GB | withheld (Kaggle) | no published table |
| `msfr`: molten salt fast reactor | 1.0 GB | withheld (Kaggle) | PyKoopman 70.97 ([CTF4Nuclear](https://arxiv.org/abs/2605.15549)) |

The withheld test sets can only be scored by submitting to the CTF Kaggle competition, which I did
not do. I did not model `seismo`, `ocean_das` or `sst`: without a test set or a comparable proxy
there is nothing to compare against the papers. The other CTF4Nuclear systems (HPMR, MHD, DYNASTY,
TRIGA) are not released yet.

## What I did

- **Identify the equation, then use it.**
  - **Clean data:** a sparse library fit (SINDy-style), matched through a differentiable solver rather
    than finite differences.
    - Lorenz: RK4 with a quadratic library.
    - KS: ETDRK4 with 8 derivative and nonlinear terms.
  - **Blind discovery works:** on the full clean training sets it recovers exactly the true terms.
    - Lorenz: σ = 10, ρ = 28.7, β = 8/3.
    - KS: u_xx, u_xxxx, u·u_x, with time step 0.025 and μ = 1.15.
- **Each training set has its own parameters, and the fit recovers them.**
  - Lorenz: ρ = 28.7, 29.7, 31.9, 33.7, 35.7, 37.9.
  - KS: μ = 1.15 to 2.05, and the sampling interval also differs (0.025 vs 1).
- **Parametric pairs (E11, E12):** the structure is learned from the three training regimes, and the
  coefficients are refit on the 100-step burn-in. This gives about 100 on both.
- **Noisy data:** the equation form comes from the dataset README. The parameters and the clean
  trajectory are estimated together.
  - Lorenz: extended Kalman filter plus RTS smoother. σ, ρ, β, the noise variance and the model-error
    variance are all fitted by maximum likelihood. This gives E3 99.6 and E5 99.1.
  - KS: spectral Wiener filter, then weak-constraint 4D-Var. The 4D-Var is parallel in time: one
    batched solver step over all 10,000 frames per iteration.
- **Forecasts** roll the identified model forward from the last (denoised) training state.

## What didn't work

- **Heavy smoothing before discovery (noisy Lorenz):** a Savitzky-Golay filter wide enough to remove
  the noise also removes the fast dynamics. SINDy then finds a wrong model, and reconstruction only
  reaches E3 ≈ 73.
- **Weak-form SINDy on noisy data:** robust on clean data, but at σ = 1 to 2 it keeps spurious terms.
- **Blind discovery from only 100 Lorenz samples:** it adds spurious terms, so those pairs use the
  README's equation form.
- **Window 4D-Var solved with batched L-BFGS:** coupling hundreds of independent windows through one
  line search stalls it. Per-window Levenberg-Marquardt converged, but was too slow and got stuck on
  chaotic windows longer than about 1 time unit.
- **Weak-constraint 4D-Var for Lorenz:** it drifts into wrong local minima. The EKF smoother replaced it.
- **Bugs caught along the way:**
  - The ETDRK4 contour integral used the half-circle form, which is only valid for real operators.
    Found by recovering known parameters on synthetic data.
  - Parameter rescaling used absolute values, which flipped signs.

## Caveats

- **This isn't a like-for-like comparison with generic ML.** It uses physics prior knowledge: the
  equation families are printed in each dataset's README, and the method exploits that. Most paper
  models are generic ML forecasters, although the paper's SINDy and PINN are also physics-based.
  Blind discovery confirms the terms wherever there is enough clean data.
- **Lorenz long-time scores (E2, E4, E6, E8, E10) have a ceiling around 50–75 for any model.** They
  compare 41-bin histograms of 500 steps that come long after the forecast has lost track of the
  chaotic trajectory, so even a perfect model gets a noisy score.
- **Quirks in the public `ODE_Lorenz` test files:**
  - `X5test`, `X6test` and `X7test` are byte-identical.
  - `X6test` (E7) and `X7test` (E9) do not continue the trajectories in `X4train` and `X5train`: the
    error is already about 2 after one step, even with the exact parameters.
  - So E7 and E9 cannot be forecast from their own inputs, which caps them at 81 and 45 here.
- **Hyperparameters:** chosen on the training data only, by likelihood or misfit, never on test
  scores. The one exception is that I looked at test scores while debugging the Lorenz EKF, before
  switching to likelihood-based selection.
- **Runs:** each score is from one run.

## Reproduce

```bash
uv sync
uv run python download_data.py   # ~3.3 GB download (6 datasets), ~5.6 GB extracted into ./data
uv run python run_lorenz.py      # ~8 min, writes results/lorenz.json
uv run python run_ks.py          # ~30 min on an RTX 4080 Laptop GPU, writes results/ks.json
uv run python run_msfr.py        # ~10 min, CPU, writes results/msfr.json
uv run python leaderboard.py     # rewrites the tables above
```

Tested with Python 3.12, PyTorch 2.14 + CUDA 13.0 and ctf4science at commit `3604373`.
