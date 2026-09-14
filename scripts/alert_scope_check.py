#!/usr/bin/env python3
"""Measure whether alerts whose header names no station carry delay signal.

    python3 -m scripts.alert_scope_check --start 2025-01-02 --end 2025-01-31

Run from the project root. Two phases:

1. Stop-level delays per service date, written to --training-dir as
   delays_<date>.csv. A date whose file already exists is skipped, so a run
   that stops partway resumes where it left off, and delays CSVs made earlier
   by mta_api/schedule_history.py can be dropped in to skip their dates.
2. Every stop row is put in the first group that applies, then compared with
   no-alert rows:

     named      an alert names this station on this route
     fallback   an alert on this route resolved no station at all
     elsewhere  an alert on this route named some other station
     none       no alert on this route

   An alert applies for two hours from its first update, as in
   add_station_alerts, so every group is computable live.

Why this exists: `route_alert_count` is the only training feature that sees
alerts naming no station, and it counts alerts by the trip's end time, which
is not known live. Over four days a route-wide fallback looked strong (2.30x)
until the baseline was matched on day as well as line and hour (1.10x), and
ten events drove all of it. This repeats the measurement over enough days
for events to stop deciding the answer.

Each group's late rate is divided by the no-alert late rate in matching
cells, under three cell definitions:

  line-hour      report_station_alerts' method, pooled over days
  day-line       matches the day, so a group that clusters on bad days isn't
                 credited with those days' delays. Covers nearly every row.
  day-line-hour  the strictest match, but a cell needs 30 no-alert rows and
                 disrupted cells rarely have them, so it drops most alert
                 rows (the share lost is printed). Read it as a check.

The interval is on the day-line ratio and bootstraps alert *events*, not
rows, because rows from one incident move together.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from mta_api.alert_stations import DEFAULT_GTFS_DIR, StationMatcher
from mta_api.schedule_history import (alert_route, compute_delay, fetch_schedule,
                                      load_observed)
from mta_api.service_alerts_history import fetch_subway_service_alerts
from scripts.training_data import daterange, ensure_archive

TZ = "America/New_York"
WINDOW_SEC = 7200
LATE_SEC = 300
# Same trimmed band as report_station_alerts: outside it is key-collision junk.
TRIM = (-1800, 7200)
MIN_CELL_ROWS = 30
GROUPS = ["named", "fallback", "elsewhere", "none"]
DELAY_COLUMNS = ["trip_uid", "route_id", "station_id", "direction", "line",
                 "obs_arr", "sched_epoch", "delay_sec"]


class Tee:
    """Print to the console and keep a copy for the report file."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)


# ----------------------------------------------------------------------
# phase 1: stop-level delays
# ----------------------------------------------------------------------
def ensure_delays(service_date: str, training_dir: Path, cache_dir: Path,
                  dataset: str, app_token: str | None) -> Path | None:
    out = training_dir / f"delays_{service_date}.csv"
    if out.exists() and out.stat().st_size > 0:
        print(f"  delays already in {out}")
        return out
    archive = ensure_archive(service_date, cache_dir)
    if archive is None:
        return None
    print("  loading observed stop events ...", flush=True)
    observed = load_observed(file=str(archive))
    print(f"  {len(observed):,} observed; querying schedule {dataset} ...", flush=True)
    try:
        schedule = fetch_schedule(service_date, dataset, app_token=app_token)
    except SystemExit as exc:
        print(f"  skipped: {exc}")
        return None
    merged = compute_delay(observed, schedule)
    matched = merged["sched_epoch"].notna().mean()
    # Written to a temp name and renamed, so an interrupted write never leaves
    # a partial file that the next run would mistake for a finished day.
    partial = out.with_suffix(".csv.partial")
    merged[DELAY_COLUMNS].to_csv(partial, index=False)
    partial.rename(out)
    print(f"  wrote {out.name}: {len(merged):,} stop rows, {matched:.0%} schedule-matched")
    return out


def load_delays(path: Path, service_date: str, cache_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False,
                     dtype={"station_id": str, "route_id": str, "line": str})
    if "route_id" not in df:
        # Delays files from schedule_history.py don't carry route_id; recover it
        # from the archive's trips table.
        archive = cache_dir / f"subwaydatanyc_{service_date}_csv.tar.xz"
        with tarfile.open(archive, "r:xz") as tf:
            for member in tf.getmembers():
                name = member.name.lower()
                if name.endswith(".csv") and "trips" in name:
                    trips = pd.read_csv(tf.extractfile(member), low_memory=False)
        df["route_id"] = df["trip_uid"].map(dict(zip(trips["trip_uid"],
                                                     trips["route_id"].astype(str))))
    return df


# ----------------------------------------------------------------------
# phase 2: alert scope groups
# ----------------------------------------------------------------------
def to_epoch(dt) -> float:
    local = pd.Timestamp(dt).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    return (local - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)


def index_alerts(service_date: str, matcher: StationMatcher, app_token: str | None):
    """Alert onsets indexed three ways, each entry (start_epoch, event key)."""
    day = datetime.fromisoformat(service_date)
    records = fetch_subway_service_alerts(day - timedelta(days=1), day + timedelta(days=2),
                                          app_token=app_token)
    named = defaultdict(list)     # (route, station) -> entries
    fallback = defaultdict(list)  # route -> entries
    anyroute = defaultdict(list)  # route -> entries
    headers = {}
    for record in records:
        start = to_epoch(record.time)
        if pd.isna(start):
            continue
        key = f"{record.event_id or record.time.isoformat()}"
        headers[key] = (record.status_label, record.headers[0] if record.headers else "")
        routes = [alert_route(r) for r in record.affected_trains]
        stations = defaultdict(set)
        for header in record.headers:
            for route, stops in matcher.affected_stations(header, routes).items():
                stations[route] |= set(stops)
        for route in routes:
            entry = (start, f"{key}|{route}")
            anyroute[route].append(entry)
            if stations.get(route):
                for stop in stations[route]:
                    named[(route, stop)].append(entry)
            else:
                fallback[route].append(entry)
    return named, fallback, anyroute, headers, len(records)


def nearest_live(entries, when):
    """Event key of the most recently started alert in its window, or None."""
    live = [(start, key) for start, key in entries if 0 <= when - start <= WINDOW_SEC]
    return max(live)[1] if live else None


def assign_groups(df: pd.DataFrame, named, fallback, anyroute) -> pd.DataFrame:
    groups, events = [], []
    for route, station, when in zip(df["aroute"], df["station_id"], df["obs_arr"]):
        for group, entries in (("named", named.get((route, station), ())),
                               ("fallback", fallback.get(route, ())),
                               ("elsewhere", anyroute.get(route, ()))):
            event = nearest_live(entries, when)
            if event is not None:
                groups.append(group)
                events.append(event)
                break
        else:
            groups.append("none")
            events.append(None)
    df["group"] = groups
    df["event"] = events
    return df


def prepare(df: pd.DataFrame, service_date: str) -> pd.DataFrame:
    df = df[df["delay_sec"].notna() & df["line"].notna() & df["route_id"].notna()].copy()
    df = df[(df["delay_sec"] >= TRIM[0]) & (df["delay_sec"] <= TRIM[1])]
    df["station_id"] = df["station_id"].astype(str)
    df["aroute"] = df["route_id"].map(alert_route)
    df["day"] = service_date
    df["hour"] = (pd.to_datetime(df["obs_arr"], unit="s", utc=True)
                  .dt.tz_convert(TZ).dt.hour)
    df["late"] = df["delay_sec"] > LATE_SEC
    return df


BASELINES = [("line-hour", ["line", "hour"]),
             ("day-line", ["day", "line"]),
             ("day-line-hour", ["day", "line", "hour"])]
INTERVAL_BASELINE = "day-line"


def event_interval(sub: pd.DataFrame, column: str, rng: np.random.Generator,
                   draws: int = 1000):
    """90% bootstrap interval for late/expected, resampling whole events."""
    sub = sub[sub[column].notna()]
    per_event = sub.groupby("event").agg(late=("late", "sum"), exp=(column, "sum"))
    if len(per_event) < 2:
        return None, None
    late, exp = per_event["late"].to_numpy(), per_event["exp"].to_numpy()
    picks = rng.integers(0, len(per_event), size=(draws, len(per_event)))
    ratios = late[picks].sum(axis=1) / np.maximum(exp[picks].sum(axis=1), 1e-9)
    return np.percentile(ratios, 5), np.percentile(ratios, 95)


def ratio(sub: pd.DataFrame, column: str) -> float:
    covered = sub[sub[column].notna()]
    if covered.empty:
        return float("nan")
    return covered["late"].mean() / covered[column].mean()


def report(rows: pd.DataFrame, headers: dict, out: Tee, top_events: int) -> None:
    rng = np.random.default_rng(0)
    clear = rows[rows["group"] == "none"]
    for name, keys in BASELINES:
        rate = clear.groupby(keys)["late"].agg(["mean", "size"])
        rate = rate[rate["size"] >= MIN_CELL_ROWS]["mean"].rename(f"exp_{name}")
        rows = rows.join(rate, on=keys)

    strict = f"exp_{BASELINES[-1][0]}"
    interval_col = f"exp_{INTERVAL_BASELINE}"
    out(f"\n{'':<10}{'':>10}{'':>8}{'':>12}{'--- late rate vs no-alert, by baseline ---':>48}")
    out(f"{'group':<10}{'rows':>10}{'events':>8}{'>300s late':>12}"
        + "".join(f"{name:>15}" for name, _ in BASELINES)
        + f"{'90% CI (events)':>18}{'lost to day-line-hour':>23}")
    for group in GROUPS:
        sub = rows[rows["group"] == group]
        if sub.empty:
            continue
        ci = "--"
        if group != "none":
            lo, hi = event_interval(sub, interval_col, rng)
            if lo is not None:
                ci = f"{lo:.2f}-{hi:.2f}x"
        events = sub["event"].nunique() if group != "none" else 0
        lost = sub[strict].isna().mean()
        out(f"{group:<10}{len(sub):>10,}{events:>8,}{sub['late'].mean():>12.1%}"
            + "".join(f"{ratio(sub, f'exp_{name}'):>14.2f}x" for name, _ in BASELINES)
            + f"{ci:>18}{lost:>23.0%}")

    fb = rows[(rows["group"] == "fallback") & rows[interval_col].notna()]
    if fb.empty:
        return
    per_event = fb.groupby(["day", "event"]).agg(rows=("late", "size"), late=("late", "mean"),
                                                exp=(interval_col, "mean"))
    per_event["ratio"] = per_event["late"] / per_event["exp"]
    above = (per_event["ratio"] > 1).mean()
    out(f"\nfallback events: {len(per_event)}, {above:.0%} of them above 1.0x "
        f"({INTERVAL_BASELINE} baseline)")
    out(f"largest {top_events} by rows:")
    for (day, event), row in per_event.sort_values("rows", ascending=False).head(top_events).iterrows():
        label, header = headers.get(event.rsplit("|", 1)[0], ("", ""))
        route = event.rsplit("|", 1)[1]
        out(f"  {day}  {route:<3} {int(row['rows']):>6,} rows  {row['ratio']:>5.2f}x  "
            f"[{label}] {header[:90]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first service date, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last service date, inclusive")
    ap.add_argument("--training-dir", type=Path, default=Path("data/training"))
    ap.add_argument("--cache-dir", type=Path, default=Path("sdn_cache"))
    ap.add_argument("--gtfs-dir", type=Path, default=DEFAULT_GTFS_DIR)
    ap.add_argument("--dataset", default="q9nv-uegs",
                    help="Socrata schedules dataset for the MATCHING YEAR (default 2025)")
    ap.add_argument("--app-token", help="Socrata app token, avoids throttling")
    ap.add_argument("--top-events", type=int, default=15)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        sys.exit("--end is before --start")
    args.training_dir.mkdir(parents=True, exist_ok=True)
    days = list(daterange(start, end))

    print(f"== phase 1: stop-level delays for {len(days)} day(s) -> {args.training_dir}")
    available = []
    for i, day in enumerate(days, 1):
        service_date = day.isoformat()
        print(f"[{i}/{len(days)} {service_date}]", flush=True)
        path = ensure_delays(service_date, args.training_dir, args.cache_dir,
                             args.dataset, args.app_token)
        if path is not None:
            available.append((service_date, path))

    if not available:
        sys.exit("no days with delay data")

    print(f"\n== phase 2: alert scope groups over {len(available)} day(s)")
    matcher = StationMatcher(gtfs_dir=args.gtfs_dir)
    frames, headers = [], {}
    for i, (service_date, path) in enumerate(available, 1):
        print(f"[{i}/{len(available)} {service_date}] fetching alerts ...", flush=True)
        named, fallback, anyroute, day_headers, n_records = index_alerts(
            service_date, matcher, args.app_token)
        headers.update(day_headers)
        rows = prepare(load_delays(path, service_date, args.cache_dir), service_date)
        rows = assign_groups(rows, named, fallback, anyroute)
        counts = rows["group"].value_counts()
        print(f"  {n_records} alert events nearby, {len(rows):,} rows in band: "
              + ", ".join(f"{g} {counts.get(g, 0):,}" for g in GROUPS))
        frames.append(rows[["day", "line", "hour", "late", "group", "event"]])

    out = Tee()
    first, last = available[0][0], available[-1][0]
    out(f"alert scope check, {len(available)} days from {first} to {last}")
    out(f"late = delay over {LATE_SEC}s; alert window {WINDOW_SEC // 3600}h from first update; "
        f"baseline cells need {MIN_CELL_ROWS} no-alert rows")
    report(pd.concat(frames, ignore_index=True), headers, out, args.top_events)

    report_path = args.training_dir / f"alert_scope_report_{first}_{last}.txt"
    report_path.write_text("\n".join(out.lines) + "\n")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
