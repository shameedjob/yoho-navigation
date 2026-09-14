"""Turn per-edge tables into the windowed tensors the line-graph DCRNN reads.

Two inputs, both long-format DataFrames keyed by EDGE_KEY:

  features    one row per (edge, moment) -- e.g. a sequence of snapshot edge
              tables (snapshot/build.py) stacked with their `snapshot_ts`.
              Which columns count as features is a FeatureSpec's call.
  traversals  one row per observed traversal, e.g. scripts/training_data.py
              output. These are the labels, kept one per traversal.

Time is cut into steps of `step_sec`. Grid time `times[t]` is the *end* of
step t: feature rows with a timestamp in (times[t] - step, times[t]] land on
t, so a window ending at t only holds data known by times[t]. Horizon k of
origin t is every traversal *departing* in [times[t] + k*step, times[t] +
(k+1)*step). Departure, because that is the train a rider reaching from_node
then would board. Every label therefore lies strictly after everything its
window can see.

Labels stay sparse -- one entry per traversal per horizon -- rather than being
averaged into a dense [T, E, H] grid. The models predict a quantile, and
averaging traversals first would fit the quantile of bin means, a much tighter
distribution than the one a rider actually draws from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ml_model.features import EDGE_KEY, FeatureSpec
from ml_model.line_graph import LineGraph


class EdgeFeatureEncoder:
    """Maps spec columns to a fixed-width float vector per row.

    numeric   standardized, NaN -> 0, plus a 0/1 missing indicator per column,
              so the model can tell "no observation" from "average value"
    categorical  one-hot over the labels seen in fit(); unseen/null -> all 0
    present   one channel, 1 for a real row and 0 for a (t, edge) slot no row
              filled

    Also holds the target's mean/std. The model trains on the standardized
    target; pinball loss is scale-equivariant, so the quantile it learns
    survives un-standardizing.
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.vocab: dict[str, list[str]] = {}
        self.target_mean = 0.0
        self.target_std = 1.0

    @property
    def num_features(self) -> int:
        return (2 * len(self.spec.numeric)
                + sum(len(self.vocab[c]) for c in self.spec.categorical) + 1)

    def fit(self, features: pd.DataFrame, target) -> "EdgeFeatureEncoder":
        self.spec.check(features)
        for column in self.spec.numeric:
            values = pd.to_numeric(features[column]).astype("float64")
            std = values.std()
            self.mean[column] = float(values.mean()) if values.notna().any() else 0.0
            self.std[column] = float(std) if std and std > 0 else 1.0
        for column in self.spec.categorical:
            self.vocab[column] = sorted(features[column].dropna().astype(str).unique())
        target = pd.to_numeric(pd.Series(target)).astype("float64")
        self.target_mean = float(target.mean())
        std = target.std()
        self.target_std = float(std) if std and std > 0 else 1.0
        return self

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """[rows, num_features] for real rows."""
        self.spec.check(frame)
        blocks = []
        for column in self.spec.numeric:
            values = pd.to_numeric(frame[column]).astype("float64").to_numpy()
            missing = np.isnan(values)
            scaled = (values - self.mean[column]) / self.std[column]
            blocks.append(np.where(missing, 0.0, scaled)[:, None])
            blocks.append(missing[:, None].astype(np.float64))
        for column in self.spec.categorical:
            labels = frame[column].astype("string")
            codes = pd.Categorical(labels, categories=self.vocab[column]).codes
            one_hot = np.zeros((len(frame), len(self.vocab[column])))
            hit = codes >= 0
            one_hot[np.flatnonzero(hit), codes[hit]] = 1.0
            blocks.append(one_hot)
        blocks.append(np.ones((len(frame), 1)))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def absent(self) -> np.ndarray:
        """The vector for a slot with no row: every numeric missing, nothing
        else set, present = 0."""
        vector = np.zeros(self.num_features, dtype=np.float32)
        vector[1:2 * len(self.spec.numeric):2] = 1.0
        return vector

    def to_dict(self) -> dict:
        return {"spec": self.spec.to_dict(), "mean": self.mean, "std": self.std,
                "vocab": self.vocab, "target_mean": self.target_mean,
                "target_std": self.target_std}

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeFeatureEncoder":
        encoder = cls(FeatureSpec.from_dict(data["spec"]))
        encoder.mean, encoder.std = data["mean"], data["std"]
        encoder.vocab = data["vocab"]
        encoder.target_mean, encoder.target_std = data["target_mean"], data["target_std"]
        return encoder


def time_grid(start_ts: int, end_ts: int, step_sec: int) -> np.ndarray:
    """Grid step ends covering (start_ts, end_ts], aligned to multiples of
    step_sec so separately built grids line up."""
    first = -(-start_ts // step_sec) * step_sec
    last = -(-end_ts // step_sec) * step_sec
    return np.arange(first, last + 1, step_sec, dtype=np.int64)


def build_feature_tensor(
    features: pd.DataFrame,
    line_graph: LineGraph,
    encoder: EdgeFeatureEncoder,
    times: np.ndarray,
    time_col: str = "snapshot_ts",
) -> np.ndarray:
    """[T, E, F] over grid `times`. Rows for edges outside the line graph or
    times outside the grid are dropped; when several rows hit one slot the
    latest wins. Empty slots get encoder.absent().

    Memory is T * E * F * 4 bytes: a month at 5-minute steps over the subway
    line graph with ~25 channels is about 1.8 GB.
    """
    step_sec = int(times[1] - times[0]) if len(times) > 1 else 1
    X = np.broadcast_to(encoder.absent(), (len(times), len(line_graph), encoder.num_features)).copy()

    rows = features.sort_values(time_col, kind="stable")
    ts = rows[time_col].to_numpy(dtype=np.int64)
    t = -(-(ts - times[0]) // step_sec)  # ceil: a row belongs to the step it ends
    e = line_graph.indices(*(rows[c] for c in EDGE_KEY))
    keep = (t >= 0) & (t < len(times)) & (e >= 0)
    X[t[keep], e[keep]] = encoder.encode(rows[keep])
    return X


@dataclass
class SparseTargets:
    """One label per (origin step, edge, horizon), sorted by origin, with
    offsets[t]:offsets[t+1] slicing out origin t's labels."""

    origin: np.ndarray  # int64
    edge: np.ndarray  # int64
    horizon: np.ndarray  # int64
    value: np.ndarray  # float32, raw target units
    offsets: np.ndarray  # int64, len T + 1


def build_targets(
    traversals: pd.DataFrame,
    line_graph: LineGraph,
    times: np.ndarray,
    horizons: int,
    target: str = "edge_sec",
    time_col: str = "ts",
    duration_col: str | None = "edge_sec",
) -> SparseTargets:
    """Labels for every origin in `times`, `horizons` steps ahead.

    Departure time is `time_col - duration_col` (training rows stamp arrival
    at to_node); pass duration_col=None if `time_col` is already a departure.
    Rows with a null target are dropped.
    """
    step_sec = int(times[1] - times[0]) if len(times) > 1 else 1
    value = pd.to_numeric(traversals[target]).astype("float64").to_numpy()
    departs = traversals[time_col].to_numpy(dtype=np.int64)
    if duration_col is not None:
        departs = departs - pd.to_numeric(traversals[duration_col]).to_numpy(dtype=np.int64)
    edge = line_graph.indices(*(traversals[c] for c in EDGE_KEY))
    bin_ = (departs - times[0]) // step_sec  # floor: [times[b], times[b] + step)
    keep = (edge >= 0) & ~np.isnan(value)

    parts = []
    for k in range(horizons):
        origin = bin_ - k
        ok = keep & (origin >= 0) & (origin < len(times))
        parts.append((origin[ok], edge[ok], np.full(ok.sum(), k), value[ok]))
    origin, edge, horizon, value = (np.concatenate(p) for p in zip(*parts))

    order = np.argsort(origin, kind="stable")
    origin = origin[order]
    offsets = np.searchsorted(origin, np.arange(len(times) + 1))
    return SparseTargets(origin, edge[order].astype(np.int64),
                         horizon[order].astype(np.int64),
                         value[order].astype(np.float32), offsets.astype(np.int64))


class EdgeSequenceDataset(torch.utils.data.Dataset):
    """Sample i is the window of `window` steps ending at origins[i], and that
    origin's labels. Windows are views into X, never copied up front."""

    def __init__(self, X: torch.Tensor, targets: SparseTargets, window: int,
                 origins: np.ndarray) -> None:
        if (origins < window - 1).any():
            raise ValueError(f"every origin needs {window - 1} steps of history")
        self.X, self.targets, self.window, self.origins = X, targets, window, origins

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, i: int):
        t = int(self.origins[i])
        lo, hi = self.targets.offsets[t], self.targets.offsets[t + 1]
        return (self.X[t - self.window + 1: t + 1],
                torch.from_numpy(self.targets.edge[lo:hi]),
                torch.from_numpy(self.targets.horizon[lo:hi]),
                torch.from_numpy(self.targets.value[lo:hi]))

    @staticmethod
    def collate(batch):
        """X [batch, window, E, F]; labels flattened with their sample index."""
        X = torch.stack([item[0] for item in batch])
        sample = torch.cat([torch.full((len(item[1]),), i, dtype=torch.long)
                            for i, item in enumerate(batch)])
        edge, horizon, value = (torch.cat([item[j] for item in batch]) for j in (1, 2, 3))
        return X, sample, edge, horizon, value


def split_origins(num_steps: int, window: int, horizons: int,
                  valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/valid origins. `horizons` origins are skipped
    between the two, so no traversal labels both a training and a validation
    sample."""
    origins = np.arange(window - 1, num_steps)
    n_valid = int(round(len(origins) * valid_fraction))
    n_train = len(origins) - n_valid - horizons
    if n_train <= 0 or n_valid <= 0:
        raise ValueError(f"{num_steps} steps is too few for window {window}, "
                         f"{horizons} horizons and valid_fraction {valid_fraction}")
    return origins[:n_train], origins[-n_valid:]
