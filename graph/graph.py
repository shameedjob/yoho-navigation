from __future__ import annotations

import heapq
import math
from dataclasses import replace

from graph.stop_node import StopNode

EARTH_RADIUS_M = 6_371_000

# Distinguishes "caller said nothing" from "caller explicitly said no period",
# since None is a meaningful value for service_period.
_UNSET = object()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class Graph:
    def __init__(self, default_period: str | None = None) -> None:
        self._nodes: dict[str, StopNode] = {}
        # Per-service-state edge times, keyed by (from, to, is_transfer). A
        # "period" here is a state name such as "Weekday:16-22"; see
        # graph/service_states.py, whose state_at maps a timestamp to one.
        # Kept beside `node.paths` rather than inside it so the 3-tuple shape
        # every existing reader unpacks stays as it is.
        self._period_times: dict[tuple[str, str, bool], dict[str, float]] = {}
        # Which period shortest_path assumes when the caller names none.
        self.default_period = default_period

    def add_node(self, node: StopNode) -> None:
        self._nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str, time: int, is_transfer: bool = False,
                 times_by_period: dict[str, float] | None = None) -> None:
        """Add an edge from `from_id` to `to_id` costing `time` seconds.

        `times_by_period` gives that cost per service state, e.g.
        {"Weekday:16-22": 90, "Saturday:22-04": 105}. A period missing from the mapping is
        one the edge does not run in, and routing in that period treats it as
        unreachable. `time` remains the period-agnostic cost used when no
        period is in play, so an edge carrying no mapping is always available.
        """
        node = self._nodes.get(from_id)
        if node is None:
            raise KeyError(f"no such node: {from_id!r}")
        node.paths.append((to_id, time, is_transfer))
        if times_by_period:
            self._period_times[(from_id, to_id, is_transfer)] = dict(times_by_period)

    def edge_time(self, from_id: str, to_id: str, is_transfer: bool,
                  time: float, period: str | None) -> float:
        """Cost of one edge in `period`, or infinity if it doesn't run then.

        An edge with no period mapping falls back to `time` and is treated as
        running in every period. That is what keeps a graph built without
        period data working unchanged, and what lets a combined subway and bus
        graph route when only the subway half carries periods.
        """
        if period is None:
            return time
        by_period = self._period_times.get((from_id, to_id, is_transfer))
        if by_period is None:
            return time
        return by_period.get(period, math.inf)

    def reweighted(self, period: str,
                   costs: dict[tuple[str, str, bool], float]) -> "Graph":
        """A copy of this graph with `period`'s cost replaced per edge.

        `costs` is keyed like the per-period table, (from, to, is_transfer).
        An edge that doesn't run in `period` stays unrunnable even if a cost
        is given for it: a prediction is a weight, not evidence of service.
        An edge with no period mapping has its period-agnostic time replaced.
        Other periods keep their costs, and the copy's default_period is
        `period`, so routing it without naming a period uses the new costs.
        Nodes are copied, so the original graph is never modified.
        """
        copy = Graph(default_period=period)
        for node_id, node in self._nodes.items():
            paths = []
            for to_id, time, is_transfer in node.paths:
                key = (node_id, to_id, is_transfer)
                if key in costs and key not in self._period_times:
                    time = round(costs[key])
                paths.append((to_id, time, is_transfer))
            copy.add_node(replace(node, paths=paths))
        for key, by_period in self._period_times.items():
            by_period = dict(by_period)
            if key in costs and period in by_period:
                by_period[period] = costs[key]
            copy._period_times[key] = by_period
        return copy

    def service_periods(self) -> set[str]:
        """Every period any edge in this graph declares."""
        return {p for times in self._period_times.values() for p in times}

    def get_node(self, node_id: str) -> StopNode | None:
        return self._nodes.get(node_id)

    def nodes_near(
        self, lat: float, lon: float, radius_m: float | None = None, limit: int | None = None
    ) -> list[tuple[StopNode, float]]:
        """Nodes near (lat, lon), sorted by distance ascending.

        radius_m restricts results to that many meters or closer (None =
        no restriction). limit caps the number of results returned (None =
        all matches).
        """
        results = [
            (node, haversine_m(lat, lon, node.lat, node.lon)) for node in self._nodes.values()
        ]
        if radius_m is not None:
            results = [pair for pair in results if pair[1] <= radius_m]
        results.sort(key=lambda pair: pair[1])
        if limit is not None:
            results = results[:limit]
        return results

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __iter__(self):
        return iter(self._nodes)

    def edge_count(self) -> int:
        return sum(len(node.paths) for node in self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def shortest_path(
        self,
        start_id: str,
        end_id: str,
        transfer_weight: float = 1.0,
        ignore_modes: set[str] | None = None,
        ignore_stops: set[str] | None = None,
        service_period: str | None = _UNSET,
        start_costs: dict[str, float] | None = None,
        end_ids: set[str] | None = None,
        end_costs: dict[str, float] | None = None,
    ) -> tuple[list[str], float] | None:
        """Dijkstra's algorithm over average edge times.

        service_period picks which service state's costs to use, e.g.
        "Weekday:16-22" (service_states.state_at gives the state for a
        departure time; a whole trip is priced in its departure state). Edges
        that don't run in that state cost infinity and are never relaxed, so a
        weekday-only edge simply cannot be taken on a Saturday. Left unset it falls back to the graph's `default_period`;
        pass None explicitly to ignore periods entirely and route on the
        period-agnostic `time` of every edge.

        transfer_weight multiplies the time of edges marked as transfers
        (see add_edge's is_transfer) -- above 1.0 to penalize transferring,
        below 1.0 to favor it, relative to riding. Ride edges are always
        taken at face value.

        ignore_modes excludes any node whose `mode` is in the set (e.g.
        {"bus"} to route subway-only). ignore_stops excludes any node
        whose `stop_id` is in the set, regardless of mode or route --
        useful for avoiding a specific stop/station. Note that for the
        subway, stop_id is the platform-level id (e.g. "101S"), not the
        parent station, since that's what ride/transfer edges key off of;
        blocking a whole station complex means listing all its platforms.

        start_costs lets the trip begin at other nodes too, each at a cost
        paid before its first edge -- e.g. the predicted wait for the next
        train at each platform near the rider. start_id starts at 0 unless it
        is given a cost there. The path begins at whichever start is cheapest
        overall, and the returned time includes that start's cost. Unknown or
        excluded ids in it are ignored.

        end_ids likewise lets the trip end at other nodes: arriving at any
        platform of the destination station doesn't pay a transfer to one
        particular platform. end_costs does the same with a cost paid after
        arriving, e.g. the walk from that stop to the destination; the trip
        ends where arrival plus that cost is lowest, and the returned time
        includes it. end_id and end_ids end at no extra cost unless end_costs
        gives them one.

        Returns (path of node ids from start to end, total time in seconds),
        or None if either id is missing/excluded, or end is unreachable
        from start under these constraints.
        """

        if service_period is _UNSET:
            service_period = self.default_period

        def allowed(node: StopNode) -> bool:
            if ignore_modes is not None and node.mode in ignore_modes:
                return False
            if ignore_stops is not None and node.stop_id in ignore_stops:
                return False
            return True

        if start_id not in self._nodes or end_id not in self._nodes:
            return None
        if not allowed(self._nodes[start_id]) or not allowed(self._nodes[end_id]):
            return None

        best_time: dict[str, float] = {start_id: 0}
        for node_id, cost in (start_costs or {}).items():
            if node_id in self._nodes and allowed(self._nodes[node_id]):
                best_time[node_id] = cost
        previous: dict[str, str] = {}
        visited: set[str] = set()
        queue: list[tuple[float, str]] = [(cost, node_id) for node_id, cost in best_time.items()]
        heapq.heapify(queue)

        targets = {n: 0.0 for n in {end_id, *(end_ids or ())}}
        targets.update(end_costs or {})
        targets = {n: c for n, c in targets.items() if n in self._nodes and allowed(self._nodes[n])}
        reached, best_end = None, math.inf
        while queue:
            time_so_far, node_id = heapq.heappop(queue)
            # Every later arrival costs at least this much before its end cost.
            if time_so_far >= best_end:
                break
            if node_id in visited:
                continue
            visited.add(node_id)

            if node_id in targets and time_so_far + targets[node_id] < best_end:
                reached, best_end = node_id, time_so_far + targets[node_id]

            node = self._nodes[node_id]
            for neighbor_id, edge_time, is_transfer in node.paths:
                if neighbor_id in visited:
                    continue
                neighbor = self._nodes.get(neighbor_id)
                if neighbor is None or not allowed(neighbor):
                    continue
                cost = self.edge_time(node_id, neighbor_id, is_transfer,
                                      edge_time, service_period)
                if cost == math.inf:
                    continue
                weighted_time = cost * transfer_weight if is_transfer else cost
                new_time = time_so_far + weighted_time
                if new_time < best_time.get(neighbor_id, float("inf")):
                    best_time[neighbor_id] = new_time
                    previous[neighbor_id] = node_id
                    heapq.heappush(queue, (new_time, neighbor_id))

        if reached is None:
            return None

        path = [reached]
        # A start that nothing improved on has no predecessor.
        while path[-1] in previous:
            path.append(previous[path[-1]])
        path.reverse()

        return path, best_end
