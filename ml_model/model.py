"""DCRNN (Diffusion Convolutional Recurrent Neural Network) forecasting a
quantile of per-edge cost over the line graph (ml_model/line_graph.py).

Each "node" the network sees is a ride edge of the transit graph, so an output
is directly an edge cost for Dijkstra. Outputs are one per horizon step: a
long trip reaches its later edges well after the window was taken, and a
router can pick each edge's cost at the horizon the rider gets there.

Wraps torch_geometric_temporal's BatchedDCRNN cell -- a GRU whose gates use
diffusion (K-hop random-walk) graph convolutions instead of dense matmuls
-- with a linear readout to the forecast horizons.

Uses BatchedDCRNN rather than the library's plain DCRNN: DCRNN's DConv
layer materializes a dense num_nodes x num_nodes adjacency matrix on every
call, which benchmarked at >1s per single-step forward on this project's
~20k-node combined graph. BatchedDCRNN computes degree via scatter_add
instead, and processes a whole input window in one call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import BatchedDCRNN


class DCRNNForecaster(torch.nn.Module):
    """Stacked DCRNN layers over an input window, followed by a linear
    readout from the final step's hidden state.

    in_channels: input features per line-graph node per step
        (EdgeFeatureEncoder.num_features).
    hidden_channels: size of each DCRNN layer's hidden state.
    horizons: forecast steps predicted per node, one output each.
    K: diffusion step count each DCRNN layer convolves over. Higher K sees
        farther along and across lines per layer, at more compute.
    num_layers: number of stacked DCRNN cells, each processing the whole
        window before handing its per-step outputs to the next layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        horizons: int = 1,
        K: int = 2,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.recurrent_layers = torch.nn.ModuleList(
            [
                BatchedDCRNN(in_channels if i == 0 else hidden_channels, hidden_channels, K)
                for i in range(num_layers)
            ]
        )
        self.readout = torch.nn.Linear(hidden_channels, horizons)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """x: [batch, window_size, num_nodes, in_channels]. Returns
        [batch, num_nodes, horizons] in the encoder's standardized target
        units, read out of the last step's hidden state.
        """
        h = x
        for layer in self.recurrent_layers:
            h = layer(h, edge_index, edge_weight)
            h = F.relu(h)

        last_step = h[:, -1, :, :]
        return self.readout(last_step)


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantile: float
) -> torch.Tensor:
    """Mean pinball loss. At quantile=0.9 an under-prediction costs nine
    times an over-prediction, so the minimizer is the 90th percentile.
    """
    err = target - pred
    return torch.maximum(quantile * err, (quantile - 1) * err).mean()
