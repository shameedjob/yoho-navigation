"""Write the station data that viewer.html draws on its canvas.

One node per subway parent station (location_type=1 in stops.txt), at its
stop_lat/stop_lon, with the routes whose trips stop at any of its platforms
(trips.txt -> stop_times.txt). Route bullet styling comes straight from
routes.txt: route_short_name, route_color, route_text_color.

Output is a JS file (`window.SUBWAY = {...}`) rather than JSON so viewer.html
works opened straight from disk, where fetch() of a local file is blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from graph.subway_loader import load_parent_stations

EXPRESS_SUFFIX = "X"  # 6X, 7X, FX: drawn as a diamond bullet of the base route


def load_route_styles(feed_dir: Path) -> dict[str, dict]:
    with open(feed_dir / "routes.txt", newline="", encoding="utf-8") as f:
        return {
            row["route_id"]: {
                "id": row["route_id"],
                "label": row["route_short_name"],
                "name": row["route_long_name"],
                "color": "#" + row["route_color"],
                "text": "#" + row["route_text_color"],
                "express": len(row["route_id"]) > 1 and row["route_id"].endswith(EXPRESS_SUFFIX)
                and row["route_id"] != "SI",
                "sort": int(row["route_sort_order"]),
            }
            for row in csv.DictReader(f)
        }


def collect_station_routes(feed_dir: Path) -> dict[str, set[str]]:
    """Map parent station id -> route_ids of any trip stopping at one of its platforms."""
    parents = load_parent_stations(feed_dir)
    with open(feed_dir / "trips.txt", newline="", encoding="utf-8") as f:
        trip_routes = {row["trip_id"]: row["route_id"] for row in csv.DictReader(f)}

    station_routes: dict[str, set[str]] = defaultdict(set)
    with open(feed_dir / "stop_times.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            route_id = trip_routes.get(row["trip_id"])
            if route_id is None:
                continue
            station = parents.get(row["stop_id"], row["stop_id"])
            station_routes[station].add(route_id)
    return station_routes


def build(feed_dir: Path) -> dict:
    routes = load_route_styles(feed_dir)
    station_routes = collect_station_routes(feed_dir)

    stations = []
    with open(feed_dir / "stops.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["location_type"] != "1":
                continue
            served = station_routes.get(row["stop_id"], set())
            # A diamond variant only earns its own bullet where the base route doesn't also stop.
            shown = {
                r for r in served
                if not (routes[r]["express"] and r[:-1] in served)
            }
            stations.append({
                "id": row["stop_id"],
                "name": row["stop_name"],
                "lat": float(row["stop_lat"]),
                "lon": float(row["stop_lon"]),
                "routes": sorted(shown, key=lambda r: routes[r]["sort"]),
            })

    return {"routes": routes, "stations": stations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed-dir", type=Path, default=Path("data/gtfs_subway"))
    parser.add_argument("--output", type=Path, default=Path("viewer_data.js"))
    args = parser.parse_args()

    data = build(args.feed_dir)
    args.output.write_text("window.SUBWAY = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    unserved = sum(1 for s in data["stations"] if not s["routes"])
    print(f"wrote {len(data['stations'])} stations ({unserved} with no scheduled routes) to {args.output}")


if __name__ == "__main__":
    main()
