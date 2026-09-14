"""Gradient-boosted model of per-edge cost, a quantile by default.

One row in, one predicted cost out: a row is an edge at a moment, the same
shape as a training traversal row (scripts/training_data.py) or a snapshot row
(snapshot/build.py). Spatial and temporal context -- what happened upstream,
what this edge did in the last half hour -- has to arrive as feature columns;
the model itself sees each row alone. That keeps it the cheap baseline the
line-graph DCRNN has to beat.

Missing values are passed through as NaN rather than filled. LightGBM learns
which side of each split missing values belong on, which matters here because
absence of observations correlates with disruption (see snapshot/build.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml_model.features import FeatureSpec

DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


OBJECTIVES = ("quantile", "l1", "huber")


class QuantileGBM:
    """LightGBM with a pinball objective at `quantile`, or a point objective.

    objective "quantile" (default) predicts the `quantile`-th percentile.
    "l1" (MAE) predicts the median, and "huber" something between median and
    mean: squared loss within `huber_delta` seconds, absolute beyond it, so a
    stalled train pulls less than it would a mean. For those two `quantile`
    is stored as 0.5, the level they are scored at.

    params override DEFAULT_PARAMS; objective and alpha are always set from
    the arguments, so a params dict can't silently change what is predicted.
    """

    def __init__(self, spec: FeatureSpec, quantile: float = 0.9,
                 params: dict | None = None, objective: str = "quantile",
                 huber_delta: float = 30.0) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        if objective != "quantile":
            quantile = 0.5
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.spec = spec
        self.quantile = quantile
        self.objective = objective
        self.huber_delta = huber_delta
        # LightGBM reads alpha as the quantile for "quantile" and as delta for "huber".
        alpha = {"quantile": quantile, "huber": huber_delta}.get(objective)
        self.params = {**DEFAULT_PARAMS, **(params or {}), "objective": objective}
        self.params.pop("alpha", None)
        if alpha is not None:
            self.params["alpha"] = alpha
        self.booster: lgb.Booster | None = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Feature columns in spec order: numeric as float (nullable ints and
        bools included, pd.NA becoming NaN), categorical as pandas category.
        At predict time LightGBM maps categories back onto the codes it saw in
        training, so a label unseen then is treated as missing.
        """
        self.spec.check(frame)
        X = pd.DataFrame(index=frame.index)
        for column in self.spec.numeric:
            X[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        for column in self.spec.categorical:
            X[column] = frame[column].astype("string").astype("category")
        return X

    def _labelled(self, frame: pd.DataFrame,
                  reference: lgb.Dataset | None = None) -> lgb.Dataset:
        """A null target is a row with nothing to learn from, e.g. a residual
        target derived from an unmatched sched_edge_sec. Those are dropped here, and only those: has_schedule=0
        rows with a valid edge_sec are censored, not bad (docs/MODEL_DATA.md).
        """
        self.spec.check(frame, need_target=True)
        target = pd.to_numeric(frame[self.spec.target]).astype("float64")
        keep = target.notna()
        return lgb.Dataset(self._matrix(frame[keep]), label=target[keep],
                           categorical_feature=self.spec.categorical or "auto",
                           reference=reference, free_raw_data=False)

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            num_boost_round: int = 2000,
            early_stopping_rounds: int | None = 50) -> "QuantileGBM":
        """Train on `train`. With `valid`, stop once its loss hasn't
        improved for early_stopping_rounds and keep the best iteration.
        """
        train_set = self._labelled(train)
        valid_sets, callbacks = [], [lgb.log_evaluation(period=100)]
        if valid is not None:
            # reference makes validation reuse the training set's bin edges.
            valid_sets = [self._labelled(valid, reference=train_set)]
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
        self.booster = lgb.train(self.params, train_set,
                                 num_boost_round=num_boost_round,
                                 valid_sets=valid_sets, valid_names=["valid"],
                                 callbacks=callbacks)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """One predicted cost per row of `frame`, in row order, floored at 0
        since Dijkstra can't take a negative edge. Row order is what
        GraphSnapshot.weighted_graph expects, so a snapshot's edges frame can
        be passed straight through.
        """
        if self.booster is None:
            raise RuntimeError("model is not trained; call fit() or load()")
        best = self.booster.best_iteration or None
        pred = self.booster.predict(self._matrix(frame), num_iteration=best)
        return np.maximum(pred, 0.0)

    def save(self, directory: Path) -> None:
        if self.booster is None:
            raise RuntimeError("model is not trained; nothing to save")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / "model.txt")
        meta = {"spec": self.spec.to_dict(), "quantile": self.quantile,
                "objective": self.objective, "huber_delta": self.huber_delta,
                "params": self.params}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "QuantileGBM":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(FeatureSpec.from_dict(meta["spec"]), meta["quantile"], meta["params"],
                    meta.get("objective", "quantile"), meta.get("huber_delta", 30.0))
        model.booster = lgb.Booster(model_file=str(directory / "model.txt"))
        return model
