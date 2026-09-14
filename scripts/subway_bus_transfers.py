"""Find walkable subway<->bus transfers using stop coordinates, and write
them to a CSV.

MTA's static GTFS has no built-in link between the subway and bus systems
(no transfers.txt entry connects them, and the two use separate stop_id
namespaces), so this estimates walkable connections from haversine
distance between stop coordinates instead.

For each subway station and each bus route, only the single closest stop
that route serves to that station is kept -- not every nearby stop on the
route -- so a route running past a station for several consecutive stops
gets one transfer point, not several redundant ones.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from stop_routes import BUS_FEEDS, collect_stop_routes

EARTH_RADIUS_M = 6_371_000


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def load_subway_stations(data_dir: Path) -> list[tuple[str, str, float, float]]:
    """Station-level (location_type=1) subway stops: id, name, lat, lon."""
    stations = []
    with open(data_dir / "gtfs_subway" / "stops.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["location_type"] == "1":
                stations.append(
                    (row["stop_id"], row["stop_name"], float(row["stop_lat"]), float(row["stop_lon"]))
                )
    return stations


def load_bus_stop_coords(data_dir: Path, feeds: list[str]) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    for feed in feeds:
        with open(data_dir / feed / "stops.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                coords.setdefault(row["stop_id"], (float(row["stop_lat"]), float(row["stop_lon"])))
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/subway_bus_transfers.csv"))
    parser.add_argument(
        "--feeds",
        type=lambda s: s.split(","),
        default=BUS_FEEDS,
        help=f"comma-separated bus feed dir names to include (default: all of {BUS_FEEDS})",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=420.0,
        help="max walkable distance in meters to count as a transfer (default: 420, ~5 min)",
    )
    parser.add_argument(
        "--walk-speed-mps",
        type=float,
        default=1.4,
        help="assumed walking speed in meters/second (default: 1.4)",
    )
    args = parser.parse_args()

    stations = load_subway_stations(args.data_dir)
    bus_coords = load_bus_stop_coords(args.data_dir, args.feeds)
    _, bus_stop_routes = collect_stop_routes(args.data_dir, args.feeds)

    rows = []
    for station_id, station_name, s_lat, s_lon in stations:
        best_per_route: dict[str, tuple[str, float]] = {}

        for stop_id, (b_lat, b_lon) in bus_coords.items():
            dist = haversine_m(s_lat, s_lon, b_lat, b_lon)
            if dist > args.max_distance_m:
                continue
            for route in bus_stop_routes.get(stop_id, ()):
                current = best_per_route.get(route)
                if current is None or dist < current[1]:
                    best_per_route[route] = (stop_id, dist)

        for route, (stop_id, dist) in best_per_route.items():
            rows.append(
                (station_id, station_name, route, stop_id, round(dist), round(dist / args.walk_speed_mps))
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["station_id", "station_name", "bus_route", "bus_stop_id", "distance_m", "walk_time_sec"]
        )
        writer.writerows(rows)

    print(f"wrote {len(rows)} transfers to {args.output}")


if __name__ == "__main__":
    main()
