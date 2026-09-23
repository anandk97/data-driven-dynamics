"""Load CTF datasets from ./data and score predictions with the official ctf4science metrics."""

from pathlib import Path

import numpy as np
import ctf4science.data_module as dm
from ctf4science.eval_module import evaluate

ROOT = Path(__file__).resolve().parent.parent
dm.top_dir = ROOT  # ctf4science looks for data/<name>/ next to its package; point it at this repo

# pair_id -> E-metric labels, in the paper's column order
E_LABELS = {1: ["E1", "E2"], 2: ["E3"], 3: ["E4"], 4: ["E5"], 5: ["E6"], 6: ["E7", "E8"], 7: ["E9", "E10"],
            8: ["E11"], 9: ["E12"]}
METRIC_ORDER = {1: ["short_time", "long_time"], 2: ["reconstruction"], 3: ["long_time"], 4: ["reconstruction"],
                5: ["long_time"], 6: ["short_time", "long_time"], 7: ["short_time", "long_time"],
                8: ["short_time"], 9: ["short_time"]}


def load(dataset, pair_id):
    """Return (list of training arrays [T, D], initialization array or None, number of steps to predict)."""
    train, init = dm.load_dataset(dataset, pair_id)
    n = len(dm.get_prediction_timesteps(dataset, pair_id))
    return train, init, n


def score(dataset, pair_id, prediction):
    """Official CTF scores for one pair, clipped to [-100, 100] like the paper."""
    res = evaluate(dataset, pair_id, prediction)
    return {e: float(np.clip(res[m], -100, 100)) for e, m in zip(E_LABELS[pair_id], METRIC_ORDER[pair_id])}
