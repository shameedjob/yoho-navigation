"""Graph WaveNet (Wu et al., 2019) edge model over the line graph.

The third graph model next to the line-graph DCRNN (ml_model/model.py) and the
temporal GAT (ml_model/gat.py), with the same inputs, outputs and training
script (ml_model/train_gat.py --architecture gwnet). What it adds:

  * Gated dilated causal convolutions over time instead of a GRU: a stack of
    layers with dilations 1, 2, 4, ... sees a long window cheaply, so the
    window can grow (--window 12 or 18) without recurrent training cost.
  * Diffusion graph convolution on three supports: the line graph's forward
    and backward random walks (what DCRNN uses), plus an *adaptive* adjacency
    softmax(relu(E1 E2^T)) learned from per-edge embeddings. That third
    support can link edges the line graph doesn't -- the local and express
    tracks of one corridor, lines sharing a merge upstream -- if their costs
    move together in training.

Time of day and day of week enter as input channels (ml_model.gat.time_encoding)
on every step, as in the original paper's time-of-day channel.

The adaptive support is a dense edges x edges matrix (~4.3M entries on the
subway line graph), the costliest part per step; --no-adaptive drops it to
measure what it buys.

With transfers (a line graph from build_line_graph(include_transfers=True),
config.transfers), the model also prices every transfer edge: the walk plus
the wait for the next train, as the edge cost Dijkstra adds when a rider
changes lines. The ~12k transfer nodes would make the backbone 7x bigger and
its adaptive matrix 45x, so they don't run through it. A transfer head reads
the backbone's final per-edge state instead: the mean over the rides arriving
at the transfer (how the rider gets there) and over the rides leaving its
to_node (the line being waited for -- its bunching and gaps), plus a learned
per-transfer embedding, the walk, and the time of day. Rides and transfers are
standardized separately (a transfer runs ~4x a ride), so neither dominates the
loss. One model, trained end to end; the origin wait stays a request-time model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ml_model.gat import TIME_CHANNELS, GATForecaster, time_encoding
from ml_model.line_graph import LineGraph
from ml_model.sequence import EdgeFeatureEncoder


def num_rides(line_graph: LineGraph) -> int:
    """Ride nodes come first in a line graph; transfer nodes, if any, follow."""
    return sum(1 for key in line_graph.edge_keys if not key[2])


def random_walks(line_graph: LineGraph) -> list[torch.Tensor]:
    """Sparse forward and backward transition matrices over the ride nodes,
    rows summing to 1: row i spreads from edge i to the edges after it
    (forward) or before it."""
    n = num_rides(line_graph)
    keep = (line_graph.edge_index < n).all(axis=0)
    src, dst = (torch.from_numpy(a[keep]) for a in line_graph.edge_index)
    weight = torch.from_numpy(line_graph.edge_weight[keep])
    supports = []
    for rows, cols in ((src, dst), (dst, src)):
        degree = torch.zeros(n).index_add_(0, rows, weight)
        values = weight / degree[rows]
        supports.append(torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce())
    return supports


class GraphConv(torch.nn.Module):
    """Diffusion over each support for `hops` steps, concatenated, then 1x1 conv."""

    def __init__(self, channels: int, supports: int, hops: int, dropout: float) -> None:
        super().__init__()
        self.hops, self.dropout = hops, dropout
        self.mix = torch.nn.Conv2d(channels * (1 + supports * hops), channels, 1)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        """x [B, C, N, T]."""
        B, C, N, T = x.shape
        out = [x]
        for A in supports:
            h = x
            for _ in range(self.hops):
                flat = h.permute(2, 0, 1, 3).reshape(N, -1)  # [N, B*C*T]
                flat = torch.sparse.mm(A, flat) if A.is_sparse else A @ flat
                h = flat.reshape(N, B, C, T).permute(1, 2, 0, 3)
                out.append(h)
        return F.dropout(self.mix(torch.cat(out, dim=1)), self.dropout, self.training)


class GraphWaveNet(torch.nn.Module):
    def __init__(self, in_channels: int, num_nodes: int, horizons: int = 1,
                 channels: int = 32, skip_channels: int = 64, dilations=(1, 2, 1, 2),
                 hops: int = 2, adaptive: bool = True, embed_dim: int = 10,
                 dropout: float = 0.3, num_transfers: int = 0, transfer_embed: int = 8) -> None:
        super().__init__()
        self.num_transfers = num_transfers
        if num_transfers:
            # Filled by the trainer / restored from the checkpoint.
            self.register_buffer("target_mean", torch.zeros(num_nodes + num_transfers))
            self.register_buffer("target_std", torch.ones(num_nodes + num_transfers))
            self.register_buffer("transfer_walk", torch.zeros(num_transfers))
            self.transfer_embedding = torch.nn.Embedding(num_transfers, transfer_embed)
            self.transfer_head = torch.nn.Sequential(
                torch.nn.Linear(2 * skip_channels + transfer_embed + 1 + TIME_CHANNELS, 64),
                torch.nn.ReLU(), torch.nn.Linear(64, horizons))
        self.dilations = tuple(dilations)
        self.receptive_field = 1 + sum(self.dilations)
        self.adaptive = adaptive
        if adaptive:
            self.source_embed = torch.nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
            self.target_embed = torch.nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        supports = 2 + int(adaptive)
        self.start = torch.nn.Conv2d(in_channels + TIME_CHANNELS, channels, 1)
        self.filters = torch.nn.ModuleList(
            torch.nn.Conv2d(channels, channels, (1, 2), dilation=(1, d)) for d in self.dilations)
        self.gates = torch.nn.ModuleList(
            torch.nn.Conv2d(channels, channels, (1, 2), dilation=(1, d)) for d in self.dilations)
        self.skips = torch.nn.ModuleList(
            torch.nn.Conv2d(channels, skip_channels, 1) for _ in self.dilations)
        self.graph_convs = torch.nn.ModuleList(
            GraphConv(channels, supports, hops, dropout) for _ in self.dilations)
        self.norms = torch.nn.ModuleList(torch.nn.BatchNorm2d(channels) for _ in self.dilations)
        self.end = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(skip_channels, 128, 1),
                                       torch.nn.ReLU(), torch.nn.Conv2d(128, horizons, 1))

    def forward(self, x: torch.Tensor, step_time: torch.Tensor,
                walks: list[torch.Tensor], transfer_in: torch.Tensor | None = None,
                transfer_out: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, W, N, F] over ride nodes; step_time [B, W, TIME_CHANNELS].
        Returns [B, N, horizons], or [B, N + transfers, horizons] with the
        transfer head, whose sparse [transfers, N] row-mean matrices
        transfer_in / transfer_out pick each transfer's arriving and onward rides."""
        B, W, N, _ = x.shape
        when = step_time[:, :, None, :].expand(B, W, N, TIME_CHANNELS)
        h = torch.cat([x, when], dim=-1).permute(0, 3, 2, 1)  # [B, C, N, W]
        if W < self.receptive_field:
            h = F.pad(h, (self.receptive_field - W, 0))
        h = self.start(h)
        supports = list(walks)
        if self.adaptive:
            supports.append(F.softmax(F.relu(self.source_embed @ self.target_embed.T), dim=1))
        skip = 0
        for filt, gate, to_skip, gconv, norm in zip(self.filters, self.gates, self.skips,
                                                    self.graph_convs, self.norms):
            residual = h
            h = torch.tanh(filt(h)) * torch.sigmoid(gate(h))
            s = to_skip(h)
            skip = s + (skip[..., -s.shape[-1]:] if torch.is_tensor(skip) else 0)
            h = gconv(h, supports) + residual[..., -h.shape[-1]:]
            h = norm(h)
        rides = self.end(skip[..., -1:]).squeeze(-1).transpose(1, 2)  # [B, N, horizons]
        if not self.num_transfers:
            return rides
        state = F.relu(skip[..., -1])  # [B, skip, N]
        flat = state.permute(2, 0, 1).reshape(N, -1)  # [N, B*skip]

        def gather(M):  # [transfers, B*skip] -> [B, transfers, skip]
            return torch.sparse.mm(M, flat).reshape(self.num_transfers, B, -1).transpose(0, 1)

        K = self.num_transfers
        head_in = torch.cat([
            gather(transfer_in), gather(transfer_out),
            self.transfer_embedding.weight[None].expand(B, K, -1),
            (self.transfer_walk / 300.0)[None, :, None].expand(B, K, 1),
            step_time[:, -1][:, None, :].expand(B, K, TIME_CHANNELS),
        ], dim=-1)
        return torch.cat([rides, self.transfer_head(head_in)], dim=1)


@dataclass
class GWNetConfig:
    window: int = 6
    step_sec: int = 600
    horizons: int = 1
    quantile: float = 0.9
    channels: int = 32
    skip_channels: int = 64
    dilations: tuple = (1, 2, 1, 2)
    hops: int = 2
    adaptive: bool = True
    embed_dim: int = 10
    dropout: float = 0.3
    transfers: bool = False
    transfer_embed: int = 8


class GWNetForecaster(GATForecaster):
    """GraphWaveNet with its encoder, line graph and config; same interface as GATForecaster."""

    config_class = GWNetConfig

    def __init__(self, model: GraphWaveNet, encoder: EdgeFeatureEncoder,
                 line_graph: LineGraph, config: GWNetConfig) -> None:
        super().__init__(model, encoder, line_graph, config)
        self.walks = random_walks(line_graph)
        self.num_rides = num_rides(line_graph)
        self.transfer_in = self.transfer_out = None
        if config.transfers:
            self.transfer_in, self.transfer_out = transfer_incidence(line_graph, self.num_rides)

    @classmethod
    def create(cls, encoder: EdgeFeatureEncoder, line_graph: LineGraph,
               config: GWNetConfig) -> "GWNetForecaster":
        rides = num_rides(line_graph)
        model = GraphWaveNet(encoder.num_features, rides, config.horizons,
                             config.channels, config.skip_channels, config.dilations,
                             config.hops, config.adaptive, config.embed_dim, config.dropout,
                             len(line_graph) - rides if config.transfers else 0,
                             config.transfer_embed)
        return cls(model, encoder, line_graph, config)

    def standardize(self, value: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        if not self.config.transfers:
            return super().standardize(value, edge)
        return (value - self.model.target_mean[edge]) / self.model.target_std[edge]

    def to_cost_units(self, standardized: torch.Tensor,
                      edge: torch.Tensor | None = None) -> torch.Tensor:
        """edge: the node of each value; None when the last dim runs over all nodes."""
        if not self.config.transfers:
            return super().to_cost_units(standardized, edge)
        if edge is None:
            return standardized * self.model.target_std + self.model.target_mean
        return standardized * self.model.target_std[edge] + self.model.target_mean[edge]

    def __call__(self, x: torch.Tensor, window_end: torch.Tensor) -> torch.Tensor:
        step = self.config.step_sec
        ends = window_end.numpy().astype(np.int64)
        steps = ends[:, None] + step * np.arange(-self.config.window + 1, 1)[None, :]
        step_time = torch.from_numpy(time_encoding(steps.reshape(-1))).reshape(
            len(ends), self.config.window, TIME_CHANNELS)
        return self.model(x[:, :, :self.num_rides], step_time, self.walks,
                          self.transfer_in, self.transfer_out)


def transfer_incidence(line_graph: LineGraph, rides: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse [transfers, rides] row-mean matrices: for each transfer node, the
    ride nodes linking into it (arriving) and out of it (onward)."""
    src, dst = line_graph.edge_index
    K = len(line_graph) - rides

    def matrix(rows, cols):
        rows, cols = torch.from_numpy(rows - rides), torch.from_numpy(cols)
        count = torch.zeros(K).index_add_(0, rows, torch.ones(len(rows)))
        values = 1.0 / count[rows]
        return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (K, rides)).coalesce()

    arriving = (src < rides) & (dst >= rides)
    onward = (src >= rides) & (dst < rides)
    return matrix(dst[arriving], src[arriving]), matrix(src[onward], dst[onward])
