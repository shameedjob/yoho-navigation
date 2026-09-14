"""Poll the live feeds for a while, then write a graph snapshot.

    python3 -m snapshot --polls 6 --interval 30 --out snapshot.csv

Run from the project root. Observed edge times only exist as differences
between polls, so a single poll yields alerts and coverage but no
observations; a few minutes of polling gives the first ones.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from graph.subway_loader import build_subway_graph, load_routes
from mta_api import MTAClient, feeds
from mta_api.alert_stations import DEFAULT_GTFS_DIR, StationMatcher
from snapshot.build import build_snapshot
from snapshot.schedule import DEFAULT_SUPPLEMENTED_DIR, StaticSchedule, ensure_supplemented
from snapshot.state import LiveState


def poll_once(client: MTAClient, state: LiveState) -> None:
    for feed_key in feeds.SUBWAY_FEEDS:
        try:
            polled_at, updates = client.get_feed_trip_updates(feed_key)
        except requests.RequestException as exc:
            # Skip rather than ingest an empty list, which would read as every
            # trip on the feed vanishing at once.
            print(f"  {feed_key}: {exc}")
            continue
        state.ingest_trip_updates(feed_key, updates,
                                  polled_at or datetime.now(timezone.utc))
    try:
        state.ingest_alerts(client.get_alerts(), datetime.now(timezone.utc))
    except requests.RequestException as exc:
        print(f"  alerts: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--polls", type=int, default=6)
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between polls; the feeds refresh about every 30s")
    ap.add_argument("--gtfs-dir", type=Path, default=DEFAULT_GTFS_DIR,
                    help="base static feed the graph is built from")
    ap.add_argument("--schedule-dir", type=Path,
                    help="static feed for per-trip schedules. Default: MTA's supplemented "
                         f"feed, cached in {DEFAULT_SUPPLEMENTED_DIR} and refreshed daily")
    ap.add_argument("--out", type=Path, default=Path("snapshot.csv"))
    args = ap.parse_args()

    graph = build_subway_graph(args.gtfs_dir)
    schedule_dir = args.schedule_dir or ensure_supplemented()
    state = LiveState(StationMatcher(gtfs_dir=args.gtfs_dir), load_routes(args.gtfs_dir),
                      StaticSchedule(schedule_dir))

    with MTAClient() as client:
        for i in range(args.polls):
            if i:
                time.sleep(args.interval)
            poll_once(client, state)
            print(f"poll {i + 1}/{args.polls}: {len(state.observations):,} observed "
                  f"traversals, {len(state.alerts)} subway alerts tracked")

    snap = build_snapshot(graph, state, datetime.now(timezone.utc))
    snap.to_csv(args.out)
    edges = snap.edges
    rides = edges[~edges["is_transfer"]]
    print(f"\nwrote {args.out}: {len(edges):,} edges, {snap.service_state} service")
    print(f"  ride edges running now     : {rides['runs_in_period'].sum():,} / {len(rides):,}")
    print(f"  ride edges observed        : {(rides['obs_count'] > 0).sum():,}")
    upcoming = rides[rides["next_train_eta_sec"].notna()]
    print(f"  ride edges with a next train: {len(upcoming):,}  "
          f"({upcoming['has_schedule'].mean():.0%} schedule-matched, "
          f"{upcoming['prior_delay_sec'].notna().sum():,} with a known delay)")
    print(f"  edges at an alerted station: {(edges['station_alert_count'] > 0).sum():,}")


if __name__ == "__main__":
    main()
