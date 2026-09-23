"""Rebuild the README leaderboards from results/*.json and the published CTF numbers.

Paper numbers: Wyder et al., "Common Task Framework For a Critical Evaluation of Scientific Machine
Learning Algorithms", NeurIPS 2025 Datasets & Benchmarks, Table 1 (top 5 models + naive baselines).
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
E = [f"E{i}" for i in range(1, 13)]

PAPER = {
    "lorenz": {
        "LSTM": [99.34, 52.37, 97.42, 50.75, 96.44, 72.13, 66.61, 32.75, 36.39, 29.57, 80.58, 60.11],
        "DeepONet": [99.11, 45.28, 96.23, 62.32, 91.84, 14.35, 21.84, 40.75, 34.53, 25.15, 85.52, 76.68],
        "Reservoir": [99.91, 56.72, 97.41, -33.49, 95.07, 18.99, 65.21, 61.92, 45.41, -48.51, 99.80, 99.97],
        "KAN": [82.89, 20.53, 96.19, 55.20, 93.00, 66.69, 50.52, -2.45, 33.68, -31.47, 41.69, 60.85],
        "ODE-LSTM": [97.77, 17.07, 97.90, 69.92, 96.48, -82.40, 55.82, 15.20, 36.83, 14.56, 39.90, 40.95],
        "SINDy": [81.83, 36.00, 36.85, -96.80, -29.22, -18.27, 55.82, 15.20, 36.83, 14.56, 82.38, 15.07],
        "Baseline: average": [51.71, -91.20, 54.88, -91.87, 56.50, -91.33, 65.97, -91.07, 51.93, -90.27, 57.08,
                              60.88],
    },
    "ks": {
        "Reservoir": [99.97, 88.78, 88.61, 23.47, 80.73, -2.57, -12.38, -12.56, -100.00, -100.00, 32.39, 40.08],
        "LSTM": [95.22, -1.88, 90.11, -43.39, 79.83, -27.46, 7.28, 48.74, 4.45, 28.81, -54.07, -40.31],
        "ODE-LSTM": [80.09, 0.48, 88.65, -31.46, 52.18, -47.01, 1.71, 49.55, 6.37, 8.52, -54.07, -12.76],
        "DeepONet": [36.52, 17.41, -1.45, 6.52, 6.29, 24.50, -9.48, 1.49, -1.93, -0.15, -4.60, 8.77],
        "KAN": [-4.43, 4.89, 50.36, 5.29, 36.93, 24.69, -22.46, 26.47, -43.06, 1.75, 0.83, -2.75],
        "SINDy": [84.38, -13.81, -2.91, -100.00, -1.23, -88.53, -0.22, 45.42, -14.00, 34.37, 10.01, 10.51],
        "Baseline: zeros": [0.0] * 12,
    },
}
OURS = "**Ours: identified model + 4D-Var/EKF**"


def table(key):
    res = json.loads((ROOT / "results" / f"{key}.json").read_text())["scores"]
    rows = [(OURS, [res[e] for e in E], res["Avg"])]
    rows += [(name, v, sum(v) / 12) for name, v in PAPER[key].items()]
    rows.sort(key=lambda r: -r[2])
    best = {e: max(r[1][i] for r in rows[1:] if r[0] != OURS) for i, e in enumerate(E)}
    lines = ["| Rank | Model | **Avg** | " + " | ".join(E) + " |", "|---:|---|---:|" + "---:|" * 12]
    for rank, (name, v, avg) in enumerate(rows, 1):
        cells = []
        for i, e in enumerate(E):
            s = f"{v[i]:.1f}"
            cells.append(f"**{s}**" if name == OURS and v[i] > best[e] else s)
        a = f"**{avg:.2f}**" if name == OURS else f"{avg:.2f}"
        lines.append(f"| {rank} | {name} | {a} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for key in PAPER:
        if (ROOT / "results" / f"{key}.json").exists():
            text = re.sub(rf"(<!-- LEADERBOARD:{key}:START -->).*?(<!-- LEADERBOARD:{key}:END -->)",
                          lambda m: m.group(1) + "\n" + table(key) + "\n" + m.group(2), text, flags=re.S)
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
