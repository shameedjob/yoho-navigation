"""A trained line-graph DCRNN with everything needed to use it live.

The network alone isn't enough to predict: it needs the exact line-graph node
order and feature encoding it was trained with, and the window/step it
expects. EdgeForecaster keeps those together in one checkpoint and turns a
window of per-edge rows into predicted edge costs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml_model.features import EDGE_KEY
from ml_model.line_graph import LineGraph
from ml_model.model import DCRNNForecaster
from ml_model.sequence import EdgeFeatureEncoder, build_feature_tensor


@dataclass
class ForecastConfig:
    window: int = 12
    step_sec: int = 300
    horizons: int = 6
    quantile: float = 0.9
    hidden_channels: int = 32
    K: int = 2
    num_layers: int = 1
    time_col: str = "snapshot_ts"


class EdgeForecaster:
    def __init__(self, model: DCRNNForecaster, encoder: EdgeFeatureEncoder,
                 line_graph: LineGraph, config: ForecastConfig) -> None:
        self.model, self.encoder = model, encoder
        self.line_graph, self.config = line_graph, config
        self.edge_index = torch.from_numpy(line_graph.edge_index)
        self.edge_weight = torch.from_numpy(line_graph.edge_weight)

    @classmethod
    def create(cls, encoder: EdgeFeatureEncoder, line_graph: LineGraph,
               config: ForecastConfig) -> "EdgeForecaster":
        model = DCRNNForecaster(encoder.num_features, config.hidden_channels,
                                config.horizons, config.K, config.num_layers)
        return cls(model, encoder, line_graph, config)

    def to_cost_units(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.encoder.target_std + self.encoder.target_mean

    @torch.no_grad()
    def predict(self, features: pd.DataFrame, at: int) -> pd.DataFrame:
        """Forecast from rows known by unix time `at`.

        `features` needs the rows for the `window` steps ending at `at`;
        older rows are ignored. Returns one row per (line-graph edge, horizon):
        EDGE_KEY columns, `horizon` (0-based), `horizon_start_sec` (seconds
        after `at` the horizon begins) and `pred`, floored at 0 for Dijkstra.
        """
        step = self.config.step_sec
        end = -(-at // step) * step
        times = np.arange(end - (self.config.window - 1) * step, end + 1, step)
        X = build_feature_tensor(features, self.line_graph, self.encoder, times,
                                 self.config.time_col)
        self.model.eval()
        out = self.model(torch.from_numpy(X)[None], self.edge_index, self.edge_weight)[0]
        pred = self.to_cost_units(out).clamp(min=0).numpy()  # [E, H]

        E, H = pred.shape
        keys = self.line_graph.edge_keys
        return pd.DataFrame({
            EDGE_KEY[0]: np.repeat([k[0] for k in keys], H),
            EDGE_KEY[1]: np.repeat([k[1] for k in keys], H),
            EDGE_KEY[2]: np.repeat([k[2] for k in keys], H),
            "horizon": np.tile(np.arange(H), E),
            "horizon_start_sec": np.tile(np.arange(H) * step + (end - at), E),
            "pred": pred.reshape(-1),
        })

    @staticmethod
    def align(predictions: pd.DataFrame, edges: pd.DataFrame,
              horizon: int = 0) -> pd.Series:
        """One horizon's predictions in the row order of `edges` (e.g. a
        GraphSnapshot's edges), NaN for edges the model doesn't cover --
        transfers and bus -- so GraphSnapshot.weighted_graph keeps their
        schedule cost."""
        chosen = predictions[predictions["horizon"] == horizon]
        lookup = dict(zip(zip(*(chosen[c] for c in EDGE_KEY)), chosen["pred"]))
        return pd.Series([lookup.get((f, t, bool(x)), np.nan)
                          for f, t, x in zip(*(edges[c] for c in EDGE_KEY))],
                         index=edges.index, dtype="float64")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "encoder": json.dumps(self.encoder.to_dict()),
            "config": asdict(self.config),
            "edge_keys": [list(k) for k in self.line_graph.edge_keys],
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "EdgeForecaster":
        data = torch.load(path, weights_only=True)
        line_graph = LineGraph([(f, t, bool(x)) for f, t, x in data["edge_keys"]],
                               data["edge_index"].numpy(), data["edge_weight"].numpy())
        forecaster = cls.create(EdgeFeatureEncoder.from_dict(json.loads(data["encoder"])),
                                line_graph, ForecastConfig(**data["config"]))
        forecaster.model.load_state_dict(data["state_dict"])
        return forecaster
