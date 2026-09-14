"""Build a bus-to-bus transfer table using the headway/2 rule of thumb:
for a route with roughly even service, a rider arriving at random waits on
average half its headway for the next one.

For each stop served by two or more routes, and each ordered pair
(from_bus, to_bus), avg_transfer_time_sec is to_bus's average headway at
that stop, divided by 2. This only depends on to_bus (not from_bus), so
rows sharing the same to_bus at a stop will share the same value -- that's
expected under this model, not a bug.

Restricted to weekday service by default (see --service-contains) so
weekend schedules don't distort the headway.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from stop_routes import BUS_FEEDS, collect_stop_route_headways


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/bus_transfers.csv"))
    parser.add_argument(
        "--feeds",
        type=lambda s: s.split(","),
        default=BUS_FEEDS,
        help=f"comma-separated feed dir names to include (default: all of {BUS_FEEDS})",
    )
    parser.add_argument(
        "--service-contains",
        default="Weekday",
        help="only include service_ids containing this substring (default: Weekday)",
    )
    args = parser.parse_args()

    _, headways = collect_stop_route_headways(args.data_dir, args.feeds, args.service_contains)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from_bus", "to_bus", "stop_id", "avg_transfer_time_sec"])
        for stop_id, route_headways in headways.items():
            if len(route_headways) < 2:
                continue
            for from_bus, to_bus in itertools.permutations(sorted(route_headways), 2):
                writer.writerow([from_bus, to_bus, stop_id, round(route_headways[to_bus] / 2)])


if __name__ == "__main__":
    main()
