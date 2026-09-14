#!/usr/bin/env python3
"""Generate model training rows over a span of service dates.

One row per observed edge traversal: a train leaving one platform and arriving
at the next on the same route. That is the unit the router costs, so it is the
unit the model predicts.

    python3 -m scripts.training_data --start 2025-01-02 --end 2025-01-31
    python3 -m scripts.training_data --start 2026-01-01 --end 2026-05-31 --sample-days 30

Run it from the project root so `mta_api` and `graph` resolve. Each day needs
two network fetches, the subwaydata.nyc archive and the schedule dataset, and
archives are cached under --cache-dir so a re-run over an overlapping span is
cheap. See docs/MODEL_DATA.md for the column contract.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("This script needs pandas:  pip install pandas")

from mta_api.alert_stations import DEFAULT_GTFS_DIR, StationMatcher
from mta_api.schedule_history import (ALERT_TYPE_SEP, add_station_alerts,
                                      compute_delay,
                                      fetch_alerts_by_route_station,
                                      fetch_schedule, load_observed)

ARCHIVE_URL = "https://subwaydata.nyc/data/subwaydatanyc_{date}_csv.tar.xz"
TZ = "America/New_York"

# Edge traversals outside this band are not rides. Below the floor is the same
# train double-reported at one platform; above the ceiling is a terminal
# layover or a trip record that ran across a gap in the feed. Both would teach
# the model nonsense about how long a hop takes.
MIN_EDGE_SEC, MAX_EDGE_SEC = 20, 1200

COLUMNS = [
    # identity
    "service_date", "ts", "from_node", "to_node", "route_id", "direction",
    # targets
    "edge_sec", "has_schedule",
    # features
    "sched_edge_sec", "prior_delay_sec", "hour", "minute_of_day", "dow",
    "is_weekend", "station_alert_count", "station_alert_age_sec",
    "station_alert_types", "obs_last_edge_sec", "obs_last_age_sec",
]
# Graph-derived columns (route, is_transfer, service_period, runs_in_period,
# graph_edge_sec) are not stored here; snapshot.build.add_graph_columns joins
# them from the graph, the same code a live snapshot uses.

# How long an earlier train's edge time stays a usable observation. Same
# window a live snapshot uses (snapshot/build.py DEFAULT_OBS_WINDOW_SEC).
OBS_WINDOW_SEC = 1800

# Socrata "MTA Subway Schedules" datasets, one per year. A mismatched year
# returns zero rows.
SCHEDULE_DATASETS = {
    2021: "y63v-kht3", 2022: "rq86-r8pt", 2023: "7pnn-mafy",
    2024: "ebrw-j62c", 2025: "q9nv-uegs", 2026: "g8es-h7gb",
}
# No route-level alert count. The old `route_alert_count` counted alerts
# overlapping the whole trip, which needs the alert's end and the trip's end --
# neither is known mid-trip, so a live snapshot could never fill it and a model
# trained on it would see a column that is always null in production.
#
# No `delay_delta_sec` either. It is exactly edge_sec - sched_edge_sec, both of
# which are kept, and it was null whenever the schedule join failed. Derive it
# at training time if a residual target is ever wanted.


def daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def ensure_archive(service_date: str, cache_dir: Path) -> Path | None:
    """Download the day's archive unless it is already cached. Returns None if
    subwaydata.nyc has no archive for that date, which happens for very recent
    days and for scattered gaps in their history."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"subwaydatanyc_{service_date}_csv.tar.xz"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = ARCHIVE_URL.format(date=service_date)
    try:
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"  no archive for {service_date}: {exc}")
        return None
    path.write_bytes(data)
    return path


def route_display_names(gtfs_dir: Path) -> dict[str, str]:
    """route_id -> the name the graph builds node ids from, so the from_node /
    to_node emitted here are the same strings graph/subway_loader.py produces."""
    with open(gtfs_dir / "routes.txt", newline="", encoding="utf-8-sig") as handle:
        return {row["route_id"]: (row["route_short_name"] or row["route_long_name"]
                                  or row["route_id"])
                for row in csv.DictReader(handle)}


def add_last_observed(df: pd.DataFrame) -> pd.DataFrame:
    """The edge time of the most recent *earlier* train on the same edge.

    Mirrors a live snapshot's obs_last_edge_sec: only a train that had already
    reached to_node by the time this one reached from_node counts, which is
    the most a live snapshot could know before this train started the edge.
    A train finishing the edge while this one is on it would be fresher than
    anything live can see. Rows here have already passed the edge-time band,
    as live observations do. Null when no such train in OBS_WINDOW_SEC.

    obs_last_age_sec is measured at this train's arrival at from_node.
    """
    rows = df.reset_index(drop=True)
    rows["_row"] = rows.index
    starts = rows[["_row", "from_node", "to_node", "prev_arr"]] \
        .rename(columns={"prev_arr": "_at"}).sort_values("_at")
    starts["_at"] = starts["_at"].astype("int64")
    earlier = rows[["from_node", "to_node", "obs_arr", "edge_sec"]] \
        .rename(columns={"obs_arr": "_at", "edge_sec": "obs_last_edge_sec"}) \
        .assign(_seen=lambda f: f["_at"]).sort_values("_at")
    earlier["_at"] = earlier["_at"].astype("int64")
    matched = pd.merge_asof(starts, earlier, on="_at", by=["from_node", "to_node"],
                            direction="backward", tolerance=OBS_WINDOW_SEC)
    matched["obs_last_age_sec"] = matched["_at"] - matched["_seen"]
    matched = matched.set_index("_row")
    rows["obs_last_edge_sec"] = matched["obs_last_edge_sec"]
    rows["obs_last_age_sec"] = matched["obs_last_age_sec"]
    return rows.drop(columns="_row")


def build_day(service_date: str, archive: Path, matcher: StationMatcher,
              display: dict[str, str], dataset: str, app_token: str | None):
    """Delay-joined, alert-annotated edge rows for one service date."""
    observed = load_observed(file=str(archive))
    schedule = fetch_schedule(service_date, dataset, app_token=app_token)
    merged = compute_delay(observed, schedule)

    merged = add_station_alerts(
        merged, fetch_alerts_by_route_station(service_date, matcher,
                                              app_token=app_token))

    df = merged.sort_values(["trip_uid", "obs_arr"]).copy()
    grouped = df.groupby("trip_uid", sort=False)

    # Every field describing where the train came from is the previous row of
    # the same trip; everything else describes arrival at this row's stop.
    df["prev_station"] = grouped["station_id"].shift(1)
    df["prev_arr"] = grouped["obs_arr"].shift(1)
    df["prev_sched"] = grouped["sched_epoch"].shift(1)
    df["prev_delay"] = grouped["delay_sec"].shift(1)
    df = df[df["prev_arr"].notna()]

    df["edge_sec"] = df["obs_arr"] - df["prev_arr"]
    df = df[(df["edge_sec"] >= MIN_EDGE_SEC) & (df["edge_sec"] <= MAX_EDGE_SEC)]
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    df["sched_edge_sec"] = df["sched_epoch"] - df["prev_sched"]

    # A row with no schedule match is censored, not absent: the join fails more
    # often exactly when service is disrupted (81% match on express-to-local
    # trips against 88% on undisrupted ones), so dropping these rows would bias
    # the training set toward normal operation. They keep a valid edge_sec.
    df["has_schedule"] = df["sched_epoch"].notna() & df["prev_sched"].notna()

    route = df["route_id"].astype(str)
    df["route_name"] = route.map(lambda r: display.get(r, r))
    df["from_node"] = df["prev_station"].astype(str) + df["direction"].astype(str) \
        + "::" + df["route_name"]
    df["to_node"] = df["station_id"].astype(str) + df["direction"].astype(str) \
        + "::" + df["route_name"]

    local = pd.to_datetime(df["obs_arr"], unit="s", utc=True).dt.tz_convert(TZ)
    df["hour"] = local.dt.hour
    df["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    df["dow"] = local.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["service_date"] = service_date
    df["ts"] = df["obs_arr"].astype("int64")
    df["route_id"] = route
    df["prior_delay_sec"] = df["prev_delay"]
    df["station_alert_types"] = df["station_alert_types"].apply(
        lambda v: f" {ALERT_TYPE_SEP} ".join(v) if isinstance(v, list) else "")
    df = add_last_observed(df)

    for column in ("edge_sec", "sched_edge_sec", "prior_delay_sec",
                   "obs_last_edge_sec", "obs_last_age_sec"):
        df[column] = df[column].astype("float64").round().astype("Int64")
    df["has_schedule"] = df["has_schedule"].astype(int)

    return df[COLUMNS]


def main():
    ap = argparse.ArgumentParser(
        description="Generate per-edge training rows over a span of service dates.")
    ap.add_argument("--start", required=True, help="first service date, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last service date, inclusive")
    ap.add_argument("--out", default="training_data.csv")
    ap.add_argument("--cache-dir", default="sdn_cache",
                    help="where subwaydata.nyc archives are kept between runs")
    ap.add_argument("--gtfs-dir", default=str(DEFAULT_GTFS_DIR))
    ap.add_argument("--dataset",
                    help="Socrata schedules dataset. Default: the one for each "
                         f"date's year, {SCHEDULE_DATASETS}")
    ap.add_argument("--app-token", help="Socrata app token, avoids throttling")
    ap.add_argument("--sample-days", type=int, default=None, metavar="N",
                    help="generate N distinct days drawn at random from --start..--end "
                         "instead of every day in it")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --sample-days")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        sys.exit("--end is before --start")
    days = list(daterange(start, end))
    if args.sample_days is not None:
        if not 0 < args.sample_days <= len(days):
            sys.exit(f"--sample-days must be between 1 and {len(days)} for this span")
        days = sorted(random.Random(args.seed).sample(days, args.sample_days))
        print(f"sampled {len(days)} days (seed {args.seed}): "
              f"{', '.join(d.isoformat() for d in days)}")

    gtfs_dir = Path(args.gtfs_dir)
    matcher = StationMatcher(gtfs_dir=gtfs_dir)
    display = route_display_names(gtfs_dir)
    out = Path(args.out)

    # Appended per day rather than concatenated at the end, so a span that
    # fails partway through still leaves usable output.
    written, skipped = 0, []
    header_done = False
    for day in days:
        service_date = day.isoformat()
        print(f"[{service_date}]")
        archive = ensure_archive(service_date, Path(args.cache_dir))
        if archive is None:
            skipped.append(service_date)
            continue
        dataset = args.dataset or SCHEDULE_DATASETS.get(day.year)
        if dataset is None:
            print(f"  skipped: no schedules dataset known for {day.year}; pass --dataset")
            skipped.append(service_date)
            continue
        try:
            rows = build_day(service_date, archive, matcher, display,
                             dataset, args.app_token)
        except SystemExit as exc:
            print(f"  skipped: {exc}")
            skipped.append(service_date)
            continue
        rows.to_csv(out, mode="a" if header_done else "w",
                    header=not header_done, index=False)
        header_done = True
        written += len(rows)
        alerted = (rows["station_alert_count"] > 0).sum()
        print(f"  {len(rows):,} edge rows  ({alerted:,} at an alerted station, "
              f"{rows['has_schedule'].mean():.0%} schedule-matched)")

    print(f"\nwrote {out}  ({written:,} rows)")
    if skipped:
        print(f"no data for {len(skipped)} date(s): {', '.join(skipped)}")


if __name__ == "__main__":
    sys.exit(main())
