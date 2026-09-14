"""Build a single Graph combining the subway and bus networks, connected by
walkable subway<->bus transfers.

Requires data/processed/subway_bus_transfers.csv (see
scripts/subway_bus_transfers.py) for the cross-system links, since MTA's
GTFS data has no built-in connection between the two systems -- no shared
stop_id namespace, no transfers.txt entry bridging them.

A cross-system transfer is priced like the same-mode ones, walk plus
headway/2 for the route being boarded, so switching systems is never cheaper
than transferring within one:

  subway -> bus   walk + bus headway/2 at that stop + STATION_EXIT_SEC
  bus -> subway   walk + subway headway/2 at that platform

with the headway taken per service state, like every other edge.

STATION_EXIT_SEC covers climbing out of the station to street level, which
the CSV's walk (measured between station and stop coordinates) doesn't.
Until 2026-09 these edges cost the walk alone, which made hopping onto a bus
look cheaper than any subway transfer; once subway transfers were priced with
predicted waits (agent/tools.py), routes started zigzagging between the two.
Bus waits are schedule headways only: there is no live bus data in the
pipeline yet (see README.md).

To route subway-only or bus-only, or avoid specific stops, use
Graph.shortest_path's ignore_modes / ignore_stops / transfer_weight
parameters -- this loader always builds the full combined graph, and
those constraints are applied per-query, not at load time.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from graph import Graph
from graph.schedule_costs import DEFAULT_STATE, base_cost
from graph.bus_loader import BUS_FEEDS, build_bus_graph
from graph.bus_loader import node_id as bus_node_id
from graph.subway_loader import build_subway_graph, load_parent_stations


# Time to get from a subway platform up to the street, on top of the walk.
STATION_EXIT_SEC = 90


def build_combined_graph(
    data_dir: Path = Path("data"),
    bus_feeds: list[str] = BUS_FEEDS,
    transfers_csv: Path = Path("data/processed/subway_bus_transfers.csv"),
    access_sec: dict[tuple[str, str], float] | None = None,
    boarding_headway_sec: dict[tuple[str, str], dict[str, float]] | None = None,
    headways: dict[str, dict[str, float]] | None = None,
) -> Graph:
    """All 12 service states are built, and cross-system transfers are priced
    per state from the boarded route's headway in that state. A transfer onto
    a route that doesn't run in a state is closed then.

    Pass dicts to have them filled for every cross-system transfer, so a
    caller with better wait estimates can re-price the wait part:
    `access_sec` with its cost *without* the wait -- (from, to) -> walk, plus
    STATION_EXIT_SEC toward a bus -- and `boarding_headway_sec` with
    state -> average headway of the route boarded at `to`. `headways` is
    filled with node id -> state -> average headway, subway and bus alike."""
    graph = Graph(default_period=DEFAULT_STATE)
    subway_headways: dict[str, dict[str, float]] = {}
    bus_headways: dict[str, dict[str, float]] = {}
    build_subway_graph(data_dir / "gtfs_subway", graph=graph, headways=subway_headways)
    build_bus_graph(data_dir, bus_feeds, graph=graph, headways=bus_headways)
    if headways is not None:
        headways.update(subway_headways)
        headways.update(bus_headways)

    # subway_bus_transfers.csv keys transfers by parent station id, but
    # graph nodes are per-platform -- expand each station to every
    # (platform, route) node under it.
    parent_of = load_parent_stations(data_dir / "gtfs_subway")
    platforms_by_parent: dict[str, list[str]] = defaultdict(list)
    for nid in graph:
        node = graph.get_node(nid)
        if node.mode == "subway":
            platforms_by_parent[parent_of.get(node.stop_id, node.stop_id)].append(nid)

    with open(transfers_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bid = bus_node_id(row["bus_stop_id"], row["bus_route"])
            if bid not in graph:
                continue
            walk_time = int(row["walk_time_sec"])
            to_bus = {name: round(walk_time + STATION_EXIT_SEC + headway / 2)
                      for name, headway in bus_headways.get(bid, {}).items()}
            for sid in platforms_by_parent.get(row["station_id"], ()):
                to_subway = {name: round(walk_time + headway / 2)
                             for name, headway in subway_headways.get(sid, {}).items()}
                if to_subway:
                    graph.add_edge(bid, sid, base_cost(to_subway), is_transfer=True,
                                   times_by_period=to_subway)
                if to_bus:
                    graph.add_edge(sid, bid, base_cost(to_bus), is_transfer=True,
                                   times_by_period=to_bus)
                if access_sec is not None:
                    access_sec[(bid, sid)] = walk_time
                    access_sec[(sid, bid)] = walk_time + STATION_EXIT_SEC
                if boarding_headway_sec is not None:
                    if sid in subway_headways:
                        boarding_headway_sec[(bid, sid)] = subway_headways[sid]
                    if bid in bus_headways:
                        boarding_headway_sec[(sid, bid)] = bus_headways[bid]

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feeds",
        type=lambda s: s.split(","),
        default=BUS_FEEDS,
        help=f"comma-separated bus feed dir names to include (default: all of {BUS_FEEDS})",
    )
    parser.add_argument(
        "--transfers-csv", type=Path, default=Path("data/processed/subway_bus_transfers.csv")
    )
    args = parser.parse_args()

    graph = build_combined_graph(args.data_dir, args.feeds, args.transfers_csv)
    print(f"nodes: {len(graph)}")
    print(f"edges: {graph.edge_count()}")


if __name__ == "__main__":
    main()
