"""The line graph the edge-level DCRNN convolves over.

The model predicts a cost per graph *edge*, because that is what Dijkstra adds
up (per-stop lateness double-counts along a path, see docs/MODEL_DATA.md). So
its "nodes" are the transit graph's ride edges, and two ride edges are linked
when a train or rider can go from one straight onto the other:

  ride link      (a -> b) then (b -> c)             same train continuing
  transfer link  (a -> b), transfer b -> t, (t -> d)  rider changing lines

Transfer links are what let a delay on one line inform the lines it shares
stations with. Transfer edges themselves are not line-graph nodes: nothing
observes them live, and their costs come from headways (graph/subway_loader.py),
so they keep their schedule cost when the graph is reweighted.

Only subway by default, the one mode with live data. Bus edges would add
thousands of nodes that never carry an observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph import Graph

EdgeKey = tuple[str, str, bool]  # (from_node, to_node, is_transfer)


@dataclass
class LineGraph:
    """edge_keys: the transit-graph edge at each line-graph node index.
    edge_index / edge_weight: directed links between those indices, with a
    self-loop on every node (see build_line_graph).
    """

    edge_keys: list[EdgeKey]
    edge_index: np.ndarray  # [2, num_links], int64
    edge_weight: np.ndarray  # [num_links], float32

    def __post_init__(self) -> None:
        self.index_of = {key: i for i, key in enumerate(self.edge_keys)}

    def __len__(self) -> int:
        return len(self.edge_keys)

    def indices(self, from_nodes, to_nodes, is_transfer) -> np.ndarray:
        """Line-graph index per row, -1 where the edge isn't in this graph."""
        return np.array([
            self.index_of.get((f, t, bool(x)), -1)
            for f, t, x in zip(from_nodes, to_nodes, is_transfer)
        ], dtype=np.int64)


def build_line_graph(
    graph: Graph,
    modes: frozenset[str] = frozenset({"subway"}),
    ride_link_weight: float = 1.0,
    transfer_link_weight: float = 1.0,
    include_transfers: bool = False,
) -> LineGraph:
    """Line graph over `graph`'s ride edges whose both ends are in `modes`.

    include_transfers also makes every transfer edge a node, for models that
    price the walk and the wait for the next train (ml_model/graph_wavenet.py
    with transfers). Its links: the rides arriving at its from_node feed it,
    and it feeds the rides leaving its to_node. The ride-to-ride transfer links
    stay, so the ride nodes see the same neighbours either way. Transfer nodes
    come after all ride nodes, so ride indices match the ride-only graph.

    The weights are link strengths for diffusion, not travel times. They
    default to equal so the model starts with no built-in assumption about how
    strongly a transfer couples two lines compared with the next stop on the
    same line.

    A self-loop (weight 1.0) goes on every node. The diffusion convolution
    normalizes by raw in/out-degree with no epsilon, so a node with no links on
    one side -- the first or last segment of a line -- would divide by zero
    and spread inf to everything it touches.
    """
    def in_modes(node_id: str) -> bool:
        node = graph.get_node(node_id)
        return node is not None and node.mode in modes

    rides_from: dict[str, list[EdgeKey]] = {}
    transfers_from: dict[str, list[str]] = {}
    for from_id in graph:
        if not in_modes(from_id):
            continue
        for to_id, _time, is_transfer in graph.get_node(from_id).paths:
            if not in_modes(to_id):
                continue
            if is_transfer:
                transfers_from.setdefault(from_id, []).append(to_id)
            else:
                rides_from.setdefault(from_id, []).append((from_id, to_id, False))

    edge_keys = sorted({key for keys in rides_from.values() for key in keys})
    transfer_keys = sorted({(f, t, True) for f, tos in transfers_from.items() for t in tos}) \
        if include_transfers else []
    edge_keys += transfer_keys
    index_of = {key: i for i, key in enumerate(edge_keys)}

    links: dict[tuple[int, int], float] = {(i, i): 1.0 for i in range(len(edge_keys))}
    for key in edge_keys:
        if key[2]:
            continue
        source = index_of[key]
        arrive = key[1]
        for onward in rides_from.get(arrive, ()):
            pair = (source, index_of[onward])
            links[pair] = max(links.get(pair, 0.0), ride_link_weight)
        for via in transfers_from.get(arrive, ()):
            for onward in rides_from.get(via, ()):
                pair = (source, index_of[onward])
                links.setdefault(pair, transfer_link_weight)

    for key in transfer_keys:
        source = index_of[key]
        # a ride arriving at the transfer's from_node feeds it ...
        for incoming in (k for k in edge_keys if not k[2] and k[1] == key[0]):
            links[(index_of[incoming], source)] = transfer_link_weight
        # ... and it feeds the rides leaving its to_node
        for onward in rides_from.get(key[1], ()):
            links[(source, index_of[onward])] = transfer_link_weight
    pairs = sorted(links)
    edge_index = np.array(pairs, dtype=np.int64).T.reshape(2, -1)
    edge_weight = np.array([links[p] for p in pairs], dtype=np.float32)
    return LineGraph(edge_keys, edge_index, edge_weight)
