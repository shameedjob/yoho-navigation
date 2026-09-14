"""List each bus stop and the routes ("buses") that visit it.

Joins the borough-partitioned static GTFS bus feeds under data/
(trips.txt -> stop_times.txt -> stops.txt -> routes.txt) to answer,
for every stop in the system, which routes stop there.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

BUS_FEEDS = ["gtfs_b", "gtfs_bx", "gtfs_m", "gtfs_q", "gtfs_si"]


def route_display_name(route: dict[str, str]) -> str:
    return route["route_short_name"] or route["route_long_name"] or route["route_id"]


def load_routes(feed_dir: Path) -> dict[str, str]:
    with open(feed_dir / "routes.txt", newline="", encoding="utf-8") as f:
        return {row["route_id"]: route_display_name(row) for row in csv.DictReader(f)}


def load_stop_coords(feed_dir: Path) -> dict[str, tuple[float, float]]:
    with open(feed_dir / "stops.txt", newline="", encoding="utf-8") as f:
        return {
            row["stop_id"]: (float(row["stop_lat"]), float(row["stop_lon"]))
            for row in csv.DictReader(f)
        }


def load_trip_routes(feed_dir: Path, service_contains: str | None = None) -> dict[str, str]:
    """Map trip_id -> route_id. If service_contains is given, only trips
    whose service_id contains that substring are included (e.g. "Weekday"
    to exclude Saturday/Sunday/holiday service patterns).
    """
    with open(feed_dir / "trips.txt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if service_contains is None:
            return {row["trip_id"]: row["route_id"] for row in reader}
        return {
            row["trip_id"]: row["route_id"]
            for row in reader
            if service_contains in row["service_id"]
        }


def collect_stop_routes(
    data_dir: Path, feeds: list[str] = BUS_FEEDS
) -> tuple[dict[str, str], dict[str, set[str]]]:
    stop_names: dict[str, str] = {}
    stop_routes: dict[str, set[str]] = defaultdict(set)

    for feed in feeds:
        feed_dir = data_dir / feed
        routes = load_routes(feed_dir)
        trip_routes = load_trip_routes(feed_dir)

        with open(feed_dir / "stops.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stop_names.setdefault(row["stop_id"], row["stop_name"])

        with open(feed_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                route_id = trip_routes.get(row["trip_id"])
                if route_id is None:
                    continue
                stop_routes[row["stop_id"]].add(routes.get(route_id, route_id))

    return stop_names, stop_routes


def parse_gtfs_time(value: str) -> int | None:
    """Convert a GTFS "HH:MM:SS" time (hours may exceed 24) to seconds past midnight."""
    if not value:
        return None
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def collect_stop_route_headways(
    data_dir: Path, feeds: list[str] = BUS_FEEDS, service_contains: str = "Weekday"
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Average headway (seconds) per stop/route: the mean gap between a
    route's consecutive scheduled arrivals at that stop.

    Restricted to service_ids containing `service_contains` (default
    "Weekday") so weekday/Saturday/Sunday schedules aren't blended into
    one timeline, which would produce a meaningless headway.
    """
    stop_names: dict[str, str] = {}
    stop_route_times: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for feed in feeds:
        feed_dir = data_dir / feed
        routes = load_routes(feed_dir)
        trip_routes = load_trip_routes(feed_dir, service_contains=service_contains)

        with open(feed_dir / "stops.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stop_names.setdefault(row["stop_id"], row["stop_name"])

        with open(feed_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                route_id = trip_routes.get(row["trip_id"])
                if route_id is None:
                    continue
                arrival = parse_gtfs_time(row["arrival_time"])
                if arrival is None:
                    continue
                route_name = routes.get(route_id, route_id)
                stop_route_times[row["stop_id"]][route_name].append(arrival)

    headways: dict[str, dict[str, float]] = defaultdict(dict)
    for stop_id, route_times in stop_route_times.items():
        for route, times in route_times.items():
            times.sort()
            if len(times) < 2:
                continue
            gaps = [b - a for a, b in zip(times, times[1:])]
            headways[stop_id][route] = sum(gaps) / len(gaps)

    return stop_names, headways


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=None, help="write to a file instead of stdout")
    parser.add_argument(
        "--feeds",
        type=lambda s: s.split(","),
        default=BUS_FEEDS,
        help=f"comma-separated feed dir names to include (default: all of {BUS_FEEDS})",
    )
    args = parser.parse_args()

    stop_names, stop_routes = collect_stop_routes(args.data_dir, args.feeds)

    lines = []
    for stop_id, name in sorted(stop_names.items(), key=lambda kv: kv[1]):
        routes = sorted(stop_routes.get(stop_id, ()))
        if not routes:
            continue
        lines.append(f"{name} ({stop_id}): {', '.join(routes)}")

    output = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
