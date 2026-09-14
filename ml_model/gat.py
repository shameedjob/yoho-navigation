"""Graph attention edge model with temporal encoding, over the line graph.

The attention counterpart to the line-graph DCRNN (ml_model/model.py). Same
inputs -- a window of per-edge snapshot rows -- and the same output, a quantile
of each ride edge's cost, but built from attention instead of recurrence and
fixed diffusion:

  1. Temporal encoding. Every step's wall-clock time becomes sin/cos of time
     of day (daily and half-day periods) and of day of week, plus a weekend
     flag. It is added to each edge's projected features, together with a
     learned embedding of the step's position in the window. Unlike the
     standardized hour/dow columns in the feature rows, it is present even on
     steps where an edge has no row (padding, no train seen), and it wraps:
     23:50 sits next to 00:10.
  2. Temporal attention. Each edge attends over its own window (a transformer
     encoder layer, shared by all edges) and keeps the last step's output.
  3. Spatial attention. GATv2 layers over the line graph: each edge weighs its
     neighbours' summaries instead of averaging them with fixed diffusion
     weights. Links carry their type (ride or transfer, forward or reverse,
     self) as an edge attribute, so attention can treat the next stop on the
     same line differently from a transfer. Reverse links are added so
     congestion downstream can inform the edges feeding it.
  4. Readout from the spatial output plus the encoding of the time being
     predicted.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from ml_model.line_graph import LineGraph
from ml_model.replay import TZ
from ml_model.sequence import EdgeFeatureEncoder

TIME_CHANNELS = 9
LINK_TYPES = ("self", "ride", "ride_reverse", "transfer", "transfer_reverse")


def time_encoding(times: np.ndarray) -> np.ndarray:
    """[len(times), TIME_CHANNELS] cyclical encoding of unix seconds in NYC time."""
    local = pd.to_datetime(np.asarray(times, dtype=np.int64), unit="s", utc=True).tz_convert(TZ)
    day = (local.hour + local.minute / 60 + local.second / 3600).to_numpy() / 24
    week = (local.dayofweek.to_numpy() + day) / 7
    angles = 2 * math.pi * np.stack([day, 2 * day, week, 2 * week], axis=1)
    weekend = (local.dayofweek.to_numpy() >= 5).astype(np.float64)[:, None]
    return np.concatenate([np.sin(angles), np.cos(angles), weekend], axis=1).astype(np.float32)


def typed_links(line_graph: LineGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Line-graph links plus their reverses, with one-hot link types."""
    src, dst = line_graph.edge_index
    keys = line_graph.edge_keys
    ride = np.array([keys[s][1] == keys[d][0] for s, d in zip(src, dst)])
    forward = np.where(src == dst, 0, np.where(ride, 1, 3))
    real = src != dst
    index = np.concatenate([np.stack([src, dst]), np.stack([dst[real], src[real]])], axis=1)
    kind = np.concatenate([forward, forward[real] + 1])
    return (torch.from_numpy(index.astype(np.int64)),
            F.one_hot(torch.from_numpy(kind), len(LINK_TYPES)).float())


class TemporalGAT(torch.nn.Module):
    def __init__(self, in_channels: int, window: int, hidden: int = 32, heads: int = 4,
                 gat_layers: int = 2, horizons: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.inputs = torch.nn.Linear(in_channels, hidden)
        self.time = torch.nn.Linear(TIME_CHANNELS, hidden)
        self.lag = torch.nn.Parameter(torch.zeros(window, hidden))
        self.temporal = torch.nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=dropout, batch_first=True)
        self.spatial = torch.nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=len(LINK_TYPES),
                      add_self_loops=False, dropout=dropout)
            for _ in range(gat_layers))
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(hidden) for _ in range(gat_layers))
        self.readout = torch.nn.Sequential(torch.nn.Linear(2 * hidden, hidden), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden, horizons))

    def forward(self, x: torch.Tensor, step_time: torch.Tensor, target_time: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """x [B, W, N, F]; step_time [B, W, TIME_CHANNELS]; target_time
        [B, TIME_CHANNELS]. Returns [B, N, horizons] in standardized units."""
        B, W, N, _ = x.shape
        h = self.inputs(x) + (self.time(step_time) + self.lag)[:, :, None, :]
        h = self.temporal(h.transpose(1, 2).reshape(B * N, W, -1))[:, -1].reshape(B * N, -1)

        offsets = (torch.arange(B) * N).repeat_interleave(edge_index.shape[1])
        batch_index = edge_index.repeat(1, B) + offsets
        batch_attr = edge_attr.repeat(B, 1)
        for conv, norm in zip(self.spatial, self.norms):
            h = norm(h + F.elu(conv(h, batch_index, batch_attr)))

        when = self.time(target_time)[:, None, :].expand(B, N, -1)
        return self.readout(torch.cat([h.reshape(B, N, -1), when], dim=-1))


@dataclass
class GATConfig:
    window: int = 6
    step_sec: int = 600
    horizons: int = 1
    quantile: float = 0.9
    hidden: int = 32
    heads: int = 4
    gat_layers: int = 2
    dropout: float = 0.1


class GATForecaster:
    """TemporalGAT with the encoder, line graph and config it was trained with."""

    config_class = GATConfig

    def __init__(self, model: TemporalGAT, encoder: EdgeFeatureEncoder,
                 line_graph: LineGraph, config: GATConfig) -> None:
        self.model, self.encoder, self.line_graph, self.config = model, encoder, line_graph, config
        self.edge_index, self.edge_attr = typed_links(line_graph)

    @classmethod
    def create(cls, encoder: EdgeFeatureEncoder, line_graph: LineGraph,
               config: GATConfig) -> "GATForecaster":
        model = TemporalGAT(encoder.num_features, config.window, config.hidden, config.heads,
                            config.gat_layers, config.horizons, config.dropout)
        return cls(model, encoder, line_graph, config)

    def standardize(self, value: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """Labels into the model's output units; `edge` is each label's node."""
        return (value - self.encoder.target_mean) / self.encoder.target_std

    def to_cost_units(self, standardized: torch.Tensor,
                      edge: torch.Tensor | None = None) -> torch.Tensor:
        return standardized * self.encoder.target_std + self.encoder.target_mean

    def __call__(self, x: torch.Tensor, window_end: torch.Tensor) -> torch.Tensor:
        """x [B, W, N, F] for windows whose last step ends at unix seconds
        `window_end` [B]; predicts the step that follows."""
        step = self.config.step_sec
        ends = window_end.numpy().astype(np.int64)
        steps = ends[:, None] + step * np.arange(-self.config.window + 1, 1)[None, :]
        step_time = torch.from_numpy(time_encoding(steps.reshape(-1))).reshape(
            len(ends), self.config.window, TIME_CHANNELS)
        target_time = torch.from_numpy(time_encoding(ends + step // 2))
        return self.model(x, step_time, target_time, self.edge_index, self.edge_attr)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "encoder": json.dumps(self.encoder.to_dict()),
            "config": asdict(self.config),
            "edge_keys": [list(k) for k in self.line_graph.edge_keys],
            "edge_index": torch.from_numpy(self.line_graph.edge_index),
            "edge_weight": torch.from_numpy(self.line_graph.edge_weight),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "GATForecaster":
        data = torch.load(path, weights_only=True)
        line_graph = LineGraph([(f, t, bool(x)) for f, t, x in data["edge_keys"]],
                               data["edge_index"].numpy(), data["edge_weight"].numpy())
        forecaster = cls.create(EdgeFeatureEncoder.from_dict(json.loads(data["encoder"])),
                                line_graph, cls.config_class(**data["config"]))
        forecaster.model.load_state_dict(data["state_dict"])
        return forecaster
