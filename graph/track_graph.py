"""Physical track segments, for modeling how trouble on one route reaches others.

The routing graph (graph/subway_loader.py) keys nodes by platform *and route*,
so a 4 and a 5 at the same platform are separate nodes joined only by transfer
edges. That is right for routing, since a rider boards a specific line, but it
cannot say that the 4 and 5 run on the same track between Grand Central and
14 St, which is exactly how a problem on one reaches the other.

A segment here is a directed pair of consecutive platforms, route-agnostic:
`631S>635S` is one segment carrying both the 4 and 5, while the 6 local covers
the same stretch through different segments. Every ride edge in the routing
graph maps onto one segment, many-to-one.

Why this structure, measured with an event study over 30 days of January 2025
observed traversals (48,970 issues, an issue being a single hop that ran 3+
minutes over its typical time). Shift is the change in hop time over the 30
minutes after an issue versus before; the interval resamples issues:

  responder                               mean shift   95% interval
  same route, same hop                        +17s      [+15, +18]
  other route on the SAME segment             +19s      [+17, +21]
  other route, same stations, other track      -3s      [ -7,  +1]
  routes sharing no platform (control)         +0s      [ -0,  +0]
  approaching, 1 segment upstream,
    same route / other routes                  +5s / +11s
  approaching, 2 segments upstream,
    same route / other routes                  +1s / +3s

So coupling runs through shared track, not shared stations: an issue on the 4
slows the 5 by about as much as it slows the next 4 (4 to 5 +35s, 5 to 4 +43s),
and does nothing measurable to the 6 on the parallel local track (4 to 6 +0s
over 497 issues). The slowdown also backs up onto trains approaching the
segment and fades within two segments, which is why `upstream` is the natural
direction for a message-passing model on this graph.

What does *not* spread by segment adjacency is lateness further down the line.
Trains that passed the issue ran their next five hops no slower, a mean of
-10s. Their lateness is still carried forward, but as a time shift on each
train (a train 10 minutes late stays about 9.7 minutes late 20 minutes later),
which belongs to a per-train model rather than to diffusion over this graph.

Caveat on the proxy: a consecutive platform pair usually means one physical
track, but GTFS stop ids don't distinguish local and express platforms at the
same station, so two routes can share a pair while running on different
tracks. Coupling strength per segment should be estimated from data rather
than assumed from `routes`; a mislabeled segment then simply shows none.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from graph.graph import Graph
from graph.subway_loader import load_routes


def segment_key(from_platform: str, to_platform: str) -> str:
    return f"{from_platform}>{to_platform}"


@dataclass
class Segment:
    key: str
    from_platform: str
    to_platform: str
    routes: set[str] = field(default_factory=set)
    # Scheduled seconds per route over this segment. Routes sharing a segment
    # usually agree to within seconds; a large spread is a hint the pair covers
    # two different tracks.
    seconds_by_route: dict[str, int] = field(default_factory=dict)
    # Scheduled trips per route over the whole feed. A route with a handful of
    # trips here is a part-time pattern, e.g. the late-night 4 running local on
    # Lexington, and shares the track only then.
    trips_by_route: dict[str, int] = field(default_factory=dict)
    successors: set[str] = field(default_factory=set)
    predecessors: set[str] = field(default_factory=set)

    @property
    def shared(self) -> bool:
        return len(self.routes) > 1


class TrackGraph:
    """Route-agnostic track segments derived from a routing `Graph`.

    `data_dir` is the static GTFS feed the graph was built from. It is needed
    because track continuity has to come from real trips, not from the routing
    graph's nodes: the late-night 4 runs local on Lexington, so the 4's Grand
    Central node has both local and express track entering and leaving it, and
    chaining segments through that node would invent a local-to-express
    connection no train makes -- putting the 6 upstream of the 4/5 express,
    the opposite of what was measured.
    """

    def __init__(self, graph: Graph, data_dir: Path = Path("data/gtfs_subway")):
        self.segments: dict[str, Segment] = {}
        # segment key -> the routing ride edges (from_node, to_node) running it
        self.routing_edges: dict[str, list[tuple[str, str]]] = {}
        self._segment_of_edge: dict[tuple[str, str], str] = {}

        for node_id in graph:
            node = graph.get_node(node_id)
            if node.mode != "subway":
                continue
            for to_id, seconds, is_transfer in node.paths:
                if is_transfer:
                    continue
                to_node = graph.get_node(to_id)
                if to_node is None:
                    continue
                key = segment_key(node.stop_id, to_node.stop_id)
                segment = self.segments.get(key)
                if segment is None:
                    segment = Segment(key, node.stop_id, to_node.stop_id)
                    self.segments[key] = segment
                    self.routing_edges[key] = []
                segment.routes.add(node.vehicle)
                segment.seconds_by_route[node.vehicle] = seconds
                self.routing_edges[key].append((node_id, to_id))
                self._segment_of_edge[(node_id, to_id)] = key

        self._link_from_trips(Path(data_dir))

    def _link_from_trips(self, data_dir: Path) -> None:
        """Connect segment a>b to b>c only where one trip actually runs a, b, c
        in that order, and count trips per route per segment.

        Relies on stop_times.txt listing each trip's stops consecutively in
        sequence order, the same assumption graph/subway_loader.py makes.
        """
        route_names = load_routes(data_dir)
        with open(data_dir / "trips.txt", newline="", encoding="utf-8") as f:
            route_of_trip = {row["trip_id"]: route_names.get(row["route_id"], row["route_id"])
                             for row in csv.DictReader(f)}

        prev_trip, prev_stop, prev_key = None, None, None
        with open(data_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trip, stop = row["trip_id"], row["stop_id"]
                if trip != prev_trip:
                    prev_trip, prev_stop, prev_key = trip, stop, None
                    continue
                key = segment_key(prev_stop, stop)
                segment = self.segments.get(key)
                if segment is not None:
                    route = route_of_trip.get(trip)
                    if route is not None:
                        segment.trips_by_route[route] = segment.trips_by_route.get(route, 0) + 1
                    if prev_key is not None and prev_key != key:
                        segment.predecessors.add(prev_key)
                        self.segments[prev_key].successors.add(key)
                prev_stop, prev_key = stop, key if segment is not None else None

    def __len__(self) -> int:
        return len(self.segments)

    def segment_for(self, from_node: str, to_node: str) -> Segment | None:
        """The track segment a routing ride edge runs on."""
        key = self._segment_of_edge.get((from_node, to_node))
        return self.segments.get(key) if key else None

    def shared_segments(self) -> list[Segment]:
        return [s for s in self.segments.values() if s.shared]

    def _walk(self, key: str, max_hops: int, forward: bool) -> dict[str, int]:
        seen = {key: 0}
        queue = deque([key])
        while queue:
            current = queue.popleft()
            if seen[current] == max_hops:
                continue
            segment = self.segments[current]
            for nxt in (segment.successors if forward else segment.predecessors):
                if nxt not in seen:
                    seen[nxt] = seen[current] + 1
                    queue.append(nxt)
        return seen

    def upstream(self, key: str, max_hops: int = 2) -> dict[str, int]:
        """Segments feeding into `key` within `max_hops`, with their distance.

        The default of 2 matches the measurement: a slowdown backs up onto the
        segment before it and fades by the one before that.
        """
        return self._walk(key, max_hops, forward=False)

    def downstream(self, key: str, max_hops: int = 2) -> dict[str, int]:
        return self._walk(key, max_hops, forward=True)

    def affected_routing_edges(self, key: str, max_hops: int = 2) -> dict[tuple[str, str], int]:
        """Every routing ride edge an issue on segment `key` is expected to slow:
        all routes on the segment itself, plus all routes on segments within
        `max_hops` upstream. Values are hop distance, 0 for the segment itself."""
        out: dict[tuple[str, str], int] = {}
        for seg, hops in self.upstream(key, max_hops).items():
            for edge in self.routing_edges[seg]:
                out[edge] = min(hops, out.get(edge, hops))
        return out

    def edge_index(self, direction: str = "upstream") -> tuple[np.ndarray, list[str]]:
        """(2 x E edge_index, segment keys in index order) for message passing.

        With direction="upstream" a message travels from a segment to the
        segments feeding it, the direction a blockage was measured to spread.
        "downstream" gives the reverse. No reverse copies are added: a model
        that wants both should ask for both and weight them separately.
        """
        if direction not in ("upstream", "downstream"):
            raise ValueError("direction must be 'upstream' or 'downstream'")
        keys = list(self.segments)
        index = {k: i for i, k in enumerate(keys)}
        src, dst = [], []
        for key, segment in self.segments.items():
            targets = segment.predecessors if direction == "upstream" else segment.successors
            for other in targets:
                src.append(index[key])
                dst.append(index[other])
        return np.array([src, dst], dtype=np.int64), keys
