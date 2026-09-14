"""Build a Graph of the NYC subway from the static GTFS feed in data/gtfs_subway.

Nodes are (platform, route) pairs, e.g. "101S::1" -- a specific subway
platform while riding a specific line. Two kinds of edges are added:

- ride edges: from one platform to the next platform a trip visits on the
  same route, weighted by the average scheduled travel time between them.
- transfer edges: between different (platform, route) pairs reachable by
  walking within a station complex -- either platforms sharing the same
  parent_station, or two different parent stations connected by a row in
  transfers.txt (e.g. 168 St, where the 1 train and the A/C are modeled as
  separate parent stations, 112 and A09, joined by an in-system transfer).
  Weighted as transfers.txt's min_transfer_time (walking time, 0 if the
  pair has no explicit row) plus the headway/2 rule -- half the average
  time between arrivals of the destination route at the destination
  platform.

Every edge records its cost in each of 12 service states -- Weekday,
Saturday and Sunday, each in four six-hour buckets (graph/service_states.py).
Routing picks a state (`shortest_path(..., service_period="Saturday:22-04")`);
an edge that doesn't run then costs infinity and can't be taken. Costs are
per state because transfer time is driven by headway, which varies both by
day (Saturday runs about 1.25x weekday) and by hour (rush against overnight).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from graph import Graph, StopNode
from graph.schedule_costs import DEFAULT_STATE, ScheduleCosts, base_cost
from graph.service_states import load_trip_days
from scripts.stop_routes import load_stop_coords, parse_gtfs_time, route_display_name


def node_id(stop_id: str, route: str) -> str:
    return f"{stop_id}::{route}"


def load_routes(feed_dir: Path) -> dict[str, str]:
    with open(feed_dir / "routes.txt", newline="", encoding="utf-8") as f:
        return {row["route_id"]: route_display_name(row) for row in csv.DictReader(f)}


def load_trip_routes(feed_dir: Path, service_contains: str) -> dict[str, str]:
    with open(feed_dir / "trips.txt", newline="", encoding="utf-8") as f:
        return {
            row["trip_id"]: row["route_id"]
            for row in csv.DictReader(f)
            if service_contains in row["service_id"]
        }


def load_parent_stations(feed_dir: Path) -> dict[str, str]:
    """Map platform stop_id -> parent_station id."""
    parents: dict[str, str] = {}
    with open(feed_dir / "stops.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["parent_station"]:
                parents[row["stop_id"]] = row["parent_station"]
    return parents


def load_transfer_walk_times(feed_dir: Path) -> dict[tuple[str, str], int]:
    """Map (from_parent_station, to_parent_station) -> min_transfer_time
    seconds, from transfers.txt. Includes same-station rows (e.g. crossing
    between platforms) as well as connections between different parent
    stations in the same complex (e.g. 168 St's 112 <-> A09).
    """
    walk_times: dict[tuple[str, str], int] = {}
    with open(feed_dir / "transfers.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            walk_times[(row["from_stop_id"], row["to_stop_id"])] = int(row["min_transfer_time"])
    return walk_times


def build_subway_graph(
    data_dir: Path = Path("data/gtfs_subway"),
    graph: Graph | None = None,
    headways: dict[str, dict[str, float]] | None = None,
    bucketed: bool = True,
) -> Graph:
    """Build the subway graph. Pass an existing `graph` to add subway nodes
    and edges into it (e.g. to combine with a bus graph) instead of
    creating a new one.

    Every edge records its cost for each of the 12 service states in
    graph/service_states.py (Weekday/Saturday/Sunday x six-hour buckets), and
    routing picks one (`shortest_path(..., service_period="Saturday:22-04")`).
    An edge with no trips in a state doesn't run then and can't be taken.

    Costs are per state because they genuinely differ. Ride times move little,
    but a transfer is walk + headway/2, and headways range from a few minutes
    at rush hour to 20 overnight; a whole-day average priced both the same.

    Which trips run on each day comes from the feed's calendar
    (service_states.active_services), not from service id names.

    Pass a dict as `headways` to have it filled with node id -> state ->
    average headway in seconds, the figure the headway/2 transfer rule uses.

    bucketed=False keys costs by service day alone ("Weekday"), as before
    time-of-day buckets; only for benchmarking against the old costs.
    """
    routes = load_routes(data_dir)
    trip_days = load_trip_days(data_dir)
    trip_routes = load_trip_routes(data_dir, "")
    parent_of = load_parent_stations(data_dir)
    walk_times = load_transfer_walk_times(data_dir)
    coords = load_stop_coords(data_dir)

    schedule = ScheduleCosts(bucketed)
    prev_trip_id = None
    prev_stop_id = None
    prev_departure = None

    with open(data_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
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
        graph = Graph()
    graph.default_period = DEFAULT_STATE if bucketed else DEFAULT_STATE.split(":")[0]

    def ensure_node(stop_id: str, route: str) -> str:
        nid = node_id(stop_id, route)
        if nid not in graph:
            lat, lon = coords[stop_id]
            graph.add_node(
                StopNode(id=nid, stop_id=stop_id, vehicle=route, mode="subway", lat=lat, lon=lon, paths=[])
            )
        return nid

    for key in schedule.legs.keys() | schedule.day_legs.keys():
        route, from_stop, to_stop = key
        per_state = {name: round(t) for name, t in schedule.ride_times(key).items()}
        if not per_state:
            continue
        graph.add_edge(ensure_node(from_stop, route), ensure_node(to_stop, route),
                       base_cost(per_state), is_transfer=False, times_by_period=per_state)

    headway_of = {key: schedule.headways(*key) for key in schedule.served()}
    if headways is not None:
        for (stop_id, route), by_state in headway_of.items():
            if by_state:
                headways[node_id(stop_id, route)] = by_state

    platforms_by_parent: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for stop_id, route in schedule.served():
        parent = parent_of.get(stop_id)
        if parent is None:
            continue
        platforms_by_parent[parent].add((stop_id, route))
        ensure_node(stop_id, route)

    # Every station complex is connected to itself (same-parent platforms
    # can reach each other), plus whatever transfers.txt links to a
    # different parent station (e.g. 168 St's 112 <-> A09).
    connected_parent_pairs: set[tuple[str, str]] = {(p, p) for p in platforms_by_parent}
    for from_parent, to_parent in walk_times:
        if from_parent in platforms_by_parent and to_parent in platforms_by_parent:
            connected_parent_pairs.add((from_parent, to_parent))

    for from_parent, to_parent in connected_parent_pairs:
        walk_time = walk_times.get((from_parent, to_parent), 0)
        for from_stop, from_route in platforms_by_parent[from_parent]:
            for to_stop, to_route in platforms_by_parent[to_parent]:
                if (from_stop, from_route) == (to_stop, to_route):
                    continue
                # Arriving at a platform the destination route doesn't serve
                # in a state is not a long wait, it is no service at all.
                per_state = {name: round(walk_time + headway / 2)
                             for name, headway in headway_of[(to_stop, to_route)].items()}
                if not per_state:
                    continue
                graph.add_edge(
                    node_id(from_stop, from_route),
                    node_id(to_stop, to_route),
                    base_cost(per_state),
                    is_transfer=True,
                    times_by_period=per_state,
                )

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtfs_subway"))
    args = parser.parse_args()

    graph = build_subway_graph(args.data_dir)
    print(f"nodes: {len(graph)}")
    print(f"edges: {graph.edge_count()}")


if __name__ == "__main__":
    main()
