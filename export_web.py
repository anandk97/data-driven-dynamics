"""Export compact data samples + model forecasts for the interactive results page (web/ctf-data.json)."""

import base64
import json
import sys

import numpy as np
import scipy.io as sio
import torch
from scipy.spatial import Delaunay

sys.path.insert(0, ".")
from ddd import ks  # noqa: E402
from ddd.data import ROOT, load  # noqa: E402
from run_lorenz import noisy  # noqa: E402

OUT = ROOT / "web" / "ctf-data.json"


def mat(ds, name, split="train"):
    d = sio.loadmat(ROOT / "data" / ds / split / f"{name}.mat")
    return next(v for k, v in d.items() if not k.startswith("__")).astype(np.float64)


def q8(a, lo=None, hi=None):
    """Quantize to uint8, base64. Returns (b64, lo, hi)."""
    lo = float(np.min(a)) if lo is None else lo
    hi = float(np.max(a)) if hi is None else hi
    b = np.clip(np.round((a - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    return {"b64": base64.b64encode(b.tobytes()).decode(), "lo": round(lo, 4), "hi": round(hi, 4),
            "shape": list(a.shape)}


def rk4_lorenz(x0, n, dt=0.05, sub=10, s=10.0, r=28.7, b=8 / 3):
    f = lambda x: np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])  # noqa: E731
    h, x, out = dt / sub, x0.copy(), []
    for _ in range(n):
        for _ in range(sub):
            k1 = f(x); k2 = f(x + h / 2 * k1); k3 = f(x + h / 2 * k2); k4 = f(x + h * k3)  # noqa: E702
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    return np.array(out)


def tris(nodes, max_edge=0.1):
    """Delaunay triangles of the mesh nodes, dropping the long edges that bridge the concave boundary."""
    t = Delaunay(nodes).simplices
    e = np.stack([np.linalg.norm(nodes[t[:, i]] - nodes[t[:, (i + 1) % 3]], axis=1) for i in range(3)], 1)
    return t[e.max(1) < max_edge].ravel().tolist()


def r2(a, d=2):
    return np.round(a, d).tolist()


def main():
    out = {}

    # ---------------- Lorenz ----------------
    X1 = mat("ODE_Lorenz", "X1train")
    T1 = mat("ODE_Lorenz", "X1test", "test")
    fc = rk4_lorenz(X1[-1], 400)
    # noisy reconstruction demo: pair 2 (noisy X2train -> clean X2test)
    Y = load("ODE_Lorenz", 2)[0][0]
    truth2 = mat("ODE_Lorenz", "X2test", "test")
    Xr, _, _ = noisy(Y, 2000)
    sl = slice(0, 300)
    out["lorenz"] = {
        "attractor": r2(X1[:4000]),
        "forecast": {"truth": r2(T1[:400]), "ours": r2(fc), "dt": 0.05},
        "denoise": {"noisy": r2(Y[sl]), "truth": r2(truth2[sl]), "ours": r2(Xr[sl]), "dt": 0.05},
    }

    # ---------------- KS ----------------
    K1 = mat("PDE_KS", "X1train")
    KT = mat("PDE_KS", "X1test", "test")
    m = ks.KS(c=np.array([0, 0, -0.025, 0, -0.02875, -0.025, 0, 0])).to(ks.DEV)
    kfc = ks.forecast(m, K1[-1], 1000)
    lo, hi = -3.2, 3.2
    out["ks"] = {
        "train": q8(K1[-2000::5, ::4], lo, hi),  # 400 frames (dt 0.125) x 256 points
        "truth": q8(KT[:1000:5, ::4], lo, hi),  # 200 x 256
        "ours": q8(kfc[:1000:5, ::4], lo, hi),
        "max_err": round(float(np.abs(kfc - KT[:1000]).max()), 4),
        "dt_train": 0.125, "dt_fc": 0.125, "L": float(32 * np.pi),
    }

    # ---------------- MSFR ----------------
    nodes = np.load(ROOT / "data" / "msfr" / "nodes.npy")
    M = np.load(ROOT / "data" / "msfr" / "train" / "X1train.npz")["X"]
    names = ["qPrompt", "qDecay", "T", "Ux", "Uz"]
    frames = np.arange(0, 2000, 80)  # 25 frames
    F = M[frames].reshape(len(frames), 5, 3880)
    out["msfr"] = {
        "nodes": r2(nodes, 3),
        "tris": tris(nodes),
        "dt": 0.05 * 80,
        "fields": {n: q8(F[:, i]) for i, n in enumerate(names)},
        "probe": {n: r2(M[:, i * 3880:(i + 1) * 3880].mean(1), 5) for i, n in enumerate(names)},
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
