"""The evaluation numbers docs/MODEL_DATA.md reports, shared by both models.

The models predict a high quantile of edge cost, not its mean, so they are
scored the way that doc's head comparison was: pinball loss at the trained
quantile, how often the real cost came in under the prediction (coverage,
which should sit near the quantile), and MAE for how far from typical the
prediction deliberately sits.

numpy only, on purpose. torch and LightGBM each bring their own OpenMP runtime,
and on macOS loading both into one process aborts (OMP Error #15) whichever
is imported first. The GBM side must not import torch through this module.
"""

from __future__ import annotations

import numpy as np


def evaluate(pred, target, quantile: float) -> dict[str, float]:
    """pinball / coverage / mae over rows where both values are present."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    keep = ~(np.isnan(pred) | np.isnan(target))
    pred, target = pred[keep], target[keep]
    if len(pred) == 0:
        raise ValueError("no rows with both a prediction and a target")
    err = target - pred
    return {
        "rows": int(len(pred)),
        "pinball": float(np.maximum(quantile * err, (quantile - 1) * err).mean()),
        "coverage": float((target <= pred).mean()),
        "mae": float(np.abs(err).mean()),
    }
