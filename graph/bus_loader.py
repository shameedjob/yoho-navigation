"""Build a Graph of the NYC bus network from the static GTFS feeds in
data/gtfs_b, data/gtfs_bx, data/gtfs_m, data/gtfs_q, data/gtfs_si.

Nodes are (stop, route) pairs, e.g. "100646::Bx9" -- a specific bus stop
while riding a specific route. Two kinds of edges are added:

- ride edges: from one stop to the next stop a trip visits on the same
  route, weighted by the average scheduled travel time between them.
- transfer edges: between different (stop, route) pairs at the exact same
  physical stop_id. Bus stops have no station/platform hierarchy or
  transfers.txt the way the subway does, so any two routes sharing a
  stop_id are treated as a walk-free transfer, weighted with the
  headway/2 rule -- half the average time between arrivals of the
  destination route at that stop.

stop_id is consistent across MTA's borough-partitioned bus feeds for the
same physical stop (a handful of border stops, like 100646 at the
Bronx/Queens line, appear in two feeds with identical id/name/lat-lon), so
all five feeds are merged together by stop_id.

Costs are kept per service state (graph/service_states.py), with the trips
running on each day taken from each feed's calendar: bus feeds carry
alternative calendars for the same trips (school days on and off), so
matching service ids by name would count those trips twice.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

from graph import Graph, StopNode
from graph.schedule_costs import DEFAULT_STATE, ScheduleCosts, base_cost
from graph.service_states import load_trip_days
from scripts.stop_routes import (
    BUS_FEEDS,
    load_routes,
    load_stop_coords,
    load_trip_routes,
    parse_gtfs_time,
)


def node_id(stop_id: str, route: str) -> str:
    return f"{stop_id}::{route}"


def build_bus_graph(
    data_dir: Path = Path("data"),
    feeds: list[str] = BUS_FEEDS,
    graph: Graph | None = None,
    headways: dict[str, dict[str, float]] | None = None,
    bucketed: bool = True,
) -> Graph:
    """Build the bus graph. Pass an existing `graph` to add bus nodes and
    edges into it (e.g. to combine with a subway graph) instead of
    creating a new one.

    Costs are per service state, as in the subway graph: every edge records
    its cost in each state it runs in (graph/service_states.py).

    Pass a dict as `headways` to have it filled with node id -> state ->
    average headway in seconds (mean gap between the route's arrivals at the
    stop) -- the same figure the headway/2 transfer rule uses, so other
    loaders can price a wait for these routes.
    """
    schedule = ScheduleCosts(bucketed)
    coords: dict[str, tuple[float, float]] = {}

    for feed in feeds:
        feed_dir = data_dir / feed
        routes = load_routes(feed_dir)
        trip_routes = load_trip_routes(feed_dir)
        trip_days = load_trip_days(feed_dir)

        for stop_id, coord in load_stop_coords(feed_dir).items():
            coords.setdefault(stop_id, coord)

        prev_trip_id = None
        prev_stop_id = None
        prev_departure = None

        with open(feed_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trip_id = row["trip_id"]
                route_id = trip_routes.get(trip_id)
                days = trip_days.get(trip_id)
                arrival = parse_gtfs_time(row["arrival_time"])
                departure = parse_gtfs_time(row["departure_time"])
                stop_id = row["stop_id"]

                if route_id is not None and days is not None and arrival is not None:
                    same_trip = trip_id == prev_trip_id
                    schedule.add_stop_time(days, routes.get(route_id, route_id), stop_id, arrival,
                                           prev_stop_id if same_trip else None,
                                           prev_departure if same_trip else None)

                prev_trip_id = trip_id
                prev_stop_id = stop_id
                prev_departure = departure

    if graph is None:
        graph = Graph(default_period=DEFAULT_STATE)

    def ensure_node(stop_id: str, route: str) -> str:
        nid = node_id(stop_id, route)
        if nid not in graph:
            lat, lon = coords[stop_id]
            graph.add_node(
                StopNode(id=nid, stop_id=stop_id, vehicle=route, mode="bus", lat=lat, lon=lon, paths=[])
            )
        return nid

    for key in schedule.legs.keys() | schedule.day_legs.keys():
        route, from_stop, to_stop = key
        per_state = {name: round(t) for name, t in schedule.ride_times(key).items()}
        if not per_state:
            continue
        graph.add_edge(ensure_node(from_stop, route), ensure_node(to_stop, route),
                       base_cost(per_state), is_transfer=False, times_by_period=per_state)

    routes_at: dict[str, list[str]] = defaultdict(list)
    headway_of: dict[str, dict[str, float]] = {}
    for stop_id, route in schedule.served():
        ensure_node(stop_id, route)
        routes_at[stop_id].append(route)
        by_state = schedule.headways(stop_id, route)
        if by_state:
            headway_of[node_id(stop_id, route)] = by_state
    if headways is not None:
        headways.update(headway_of)

    for stop_id, routes_here in routes_at.items():
        for from_route, to_route in itertools.permutations(routes_here, 2):
            per_state = {name: round(headway / 2)
                         for name, headway in headway_of.get(node_id(stop_id, to_route), {}).items()}
            if not per_state:
                continue
            graph.add_edge(
                node_id(stop_id, from_route),
                node_id(stop_id, to_route),
                base_cost(per_state),
                is_transfer=True,
                times_by_period=per_state,
            )

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feeds",
        type=lambda s: s.split(","),
        default=BUS_FEEDS,
        help=f"comma-separated feed dir names to include (default: all of {BUS_FEEDS})",
    )
    args = parser.parse_args()

    graph = build_bus_graph(args.data_dir, args.feeds)
    print(f"nodes: {len(graph)}")
    print(f"edges: {graph.edge_count()}")


if __name__ == "__main__":
    main()
