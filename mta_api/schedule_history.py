#!/usr/bin/env python3
"""
add_scheduled_delay.py
--------------------------------------------------------------------
Fill delay_sec by joining observed subwaydata.nyc stop events to the
scheduled times in the NY State open-data "MTA Subway Schedules" set.

    delay_sec = observed_arrival - scheduled_arrival   (per trip, per stop)

Schedule source (confirmed schema):
    dataset q9nv-uegs = "MTA Subway Schedules: 2025"
    https://data.ny.gov/resource/q9nv-uegs.json
    fields used: train_id, gtfs_stop_id, direction, arrival_time,
                 departure_time, service_date, stop_order, line
  * train_id  == the vehicle_id in your subwaydata trips file
                 (e.g. "06 0401+ 125/PEL")            <- trip-level join key
  * gtfs_stop_id has NO N/S suffix  ("101") -> join to your stripped
    station_id + direction                            <- stop-level join key
  * arrival_time/departure_time are LOCAL (naive) timestamps.

!! YEAR MATTERS: q9nv-uegs holds 2025 only. For a 2024 (or 2023/2026)
   observed day, pass the matching year's dataset id via --dataset.
   The service_date you query must be the year of your subwaydata day.

The join is UNTESTED against the live feeds from here (data.ny.gov is
unreachable in this sandbox). The script PRINTS the match rate so you
can validate it on your machine -- that number is your go/no-go on
schedule-based delay vs. the headway fallback.

Usage
    python3 add_scheduled_delay.py --file subwaydatanyc_2025-03-04_csv.tar.xz \
        --service-date 2025-03-04 --features features_2025-03-04.csv
    python3 add_scheduled_delay.py --stop-times st.csv --trips tr.csv \
        --service-date 2025-03-04 --app-token $SODA_TOKEN
    # offline test / cached schedule:
    python3 add_scheduled_delay.py --stop-times st.csv --trips tr.csv \
        --schedule-csv schedule_2025-03-04.csv --service-date 2025-03-04
--------------------------------------------------------------------
"""

import argparse
import json
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("This script needs pandas:  pip install pandas")

try:
    from .alert_stations import DEFAULT_GTFS_DIR, StationMatcher
    from .service_alerts_history import fetch_subway_service_alerts
except ImportError:
    # Invoked as a plain script ("python3 mta_api/schedule_history.py"), which
    # puts mta_api/ on sys.path instead of the project root, so there's no
    # package to be relative to. Put the root on the path and import by package
    # name -- the sibling module uses relative imports of its own, so importing
    # it as a loose module wouldn't work either.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mta_api.alert_stations import DEFAULT_GTFS_DIR, StationMatcher
    from mta_api.service_alerts_history import fetch_subway_service_alerts

TZ = "America/New_York"
DIR_RE = re.compile(r"([NS])$")
SCHED_COLS = ["train_id", "line", "direction", "gtfs_stop_id",
              "stop_order", "arrival_time", "departure_time", "service_date"]


def norm_id(s):
    """Normalize the train/vehicle label: upper, single-spaced, trimmed, and
    the leading character dropped.

    That leading character (e.g. the "1" in "1A 0503 LEF/EUC" vs "0" in the
    schedule's "0A 0503 LEF/EUC" for the same physical train) is not applied
    consistently between subwaydata.nyc's observed feed and the MTA Subway
    Schedules dataset -- confirmed by comparing same line/time/path pairs
    across both sources. Numbered IRT lines happen to use "0" in both, which
    is why they joined fine; every lettered line used a different leading
    digit per source and matched 0% of the time before this fix.
    """
    normalized = str(s).strip().upper()
    # subwaydata.nyc's L-line ids sometimes omit the space that should follow
    # the "+" flag (e.g. "0500+RPY/8AV" instead of "0500+ RPY/8AV") -- true for
    # 339/550 of one day's L trips vs 0/several-hundred on every other line
    # checked (A, 6, D, 7, N), so it's an L-specific feed quirk, not a general
    # format. Insert it back before collapsing whitespace so the token count
    # lines up with the schedule's consistently-spaced format.
    normalized = re.sub(r"(?<=\d)\+(?=\S)", "+ ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized[1:]


# The schedule dataset uses a different station code than subwaydata.nyc for
# these termini -- confirmed by matching same route+time pairs across both
# sources (e.g. observed "07 0011+ MST/34H" == schedule "07 0011+ MST/34Y").
# Only the dominant/high-volume code per line is covered here; each line also
# has a handful of rarer codes (yard moves, alternate terminals) on one side
# only that this doesn't chase.
STATION_CODE_ALIASES = {"34H": "34Y", "962": "96S", "BCR": "NOT"}


def apply_station_alias(join_id):
    """Swap observed station codes in the path segment of a join_id for the
    schedule dataset's codes, per STATION_CODE_ALIASES."""
    prefix, sep, path = join_id.rpartition(" ")
    if not sep:
        return join_id
    aliased_path = "/".join(STATION_CODE_ALIASES.get(tok, tok) for tok in path.split("/"))
    return prefix + sep + aliased_path


# The schedule dataset embeds a different route letter than subwaydata.nyc's
# vehicle_id for these routes -- confirmed by matching same time+path pairs
# across both sources (e.g. observed "1Z 0726+ P-A/BRD" == schedule
# "0J 0726+ P-A/BRD"). Express/skip-stop variants get folded into their
# parent line (Z->J, FX->F, W->N); the Franklin Ave Shuttle's train_id uses
# "S" even though its own `line` column says "FS"; Staten Island Railway is
# labeled "SS" by subwaydata.nyc but "SI" in the schedule.
ROUTE_LETTER_OVERRIDES = {"Z": "J", "FX": "F", "W": "N", "FS": "S", "SS": "SI"}


def apply_route_override(join_id, route_id):
    """Swap the route-letter token of an observed join_id for the letter the
    schedule dataset actually uses, per ROUTE_LETTER_OVERRIDES."""
    alias = ROUTE_LETTER_OVERRIDES.get(route_id)
    if not alias:
        return join_id
    _, sep, rest = join_id.partition(" ")
    return alias + sep + rest


# --------------------------------------------------------------------
# fetch the schedule for one service_date via the Socrata (SODA) API
# --------------------------------------------------------------------
def fetch_schedule(service_date, dataset="q9nv-uegs", host="data.ny.gov",
                   app_token=None, page=50000):
    base = f"https://{host}/resource/{dataset}.json"
    rows, offset = [], 0
    headers = {"User-Agent": "delay-join/1.0"}
    if app_token:
        headers["X-App-Token"] = app_token
    while True:
        params = {
            "$select": ",".join(SCHED_COLS),
            "$where": f"service_date = '{service_date}T00:00:00.000'",
            "$limit": page,
            "$offset": offset,
            "$order": ":id",           # stable paging
        }
        url = base + "?" + urllib.parse.urlencode(params, safe="=':,")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            chunk = json.load(r)
        rows.extend(chunk)
        print(f"  fetched {len(rows):,} schedule rows...", end="\r")
        if len(chunk) < page:
            break
        offset += page
        time.sleep(0.2)
    print()
    if not rows:
        sys.exit(f"No schedule rows for {service_date} in dataset {dataset}. "
                 f"Wrong year? q9nv-uegs is 2025 only.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# observed stop events, with join keys attached
# --------------------------------------------------------------------
def load_observed(file=None, stop_times=None, trips=None):
    if file:
        st = tr = None
        with tarfile.open(file, "r:xz") as tf:
            for m in tf.getmembers():
                low = m.name.lower()
                if low.endswith(".csv") and "stop_times" in low:
                    st = pd.read_csv(tf.extractfile(m), low_memory=False)
                elif low.endswith(".csv") and "trips" in low:
                    tr = pd.read_csv(tf.extractfile(m), low_memory=False)
    else:
        st = pd.read_csv(stop_times, low_memory=False)
        tr = pd.read_csv(trips, low_memory=False)
    if st is None or tr is None:
        sys.exit("could not load both stop_times and trips")

    st = st.copy()
    st["obs_arr"] = st["arrival_time"].fillna(st["departure_time"])
    st = st.dropna(subset=["obs_arr"])
    st["obs_arr"] = st["obs_arr"].astype("int64")
    st["direction"] = st["stop_id"].astype(str).str.extract(DIR_RE)[0].fillna("X")
    st["station_id"] = st["stop_id"].astype(str).str.replace(DIR_RE, "", regex=True)
    st = st.merge(tr[["trip_uid", "vehicle_id", "route_id"]], on="trip_uid", how="left")
    st["join_id"] = [
        apply_station_alias(apply_route_override(norm_id(vehicle_id), route_id))
        for vehicle_id, route_id in zip(st["vehicle_id"], st["route_id"])
    ]
    return st


# --------------------------------------------------------------------
# the join + delay computation
# --------------------------------------------------------------------
def compute_delay(observed, schedule):
    sch = schedule.copy()
    # scheduled event time: arrival, fall back to departure
    sch["sched_local"] = sch.get("arrival_time")
    if "departure_time" in sch:
        sch["sched_local"] = sch["sched_local"].fillna(sch["departure_time"])
    sch["sched_local"] = pd.to_datetime(sch["sched_local"], errors="coerce")
    # local naive -> tz-aware -> unix epoch seconds.
    # Use Timedelta division (not astype int64) so it's correct regardless of
    # whether pandas parsed the strings at ns/us/s resolution. DST-safe.
    sched_aware = sch["sched_local"].dt.tz_localize(
        TZ, ambiguous="NaT", nonexistent="shift_forward")
    epoch0 = pd.Timestamp("1970-01-01", tz="UTC")
    sch["sched_epoch"] = (sched_aware - epoch0) / pd.Timedelta(seconds=1)
    sch["join_id"] = sch["train_id"].map(norm_id)
    sch["station_id"] = sch["gtfs_stop_id"].astype(str)
    sch["direction"] = sch["direction"].astype(str).str.strip().str[:1]

    # a train visits a stop once per service day (dedupe defensively)
    sch = (sch.dropna(subset=["sched_epoch"])
              .sort_values("sched_epoch")
              .drop_duplicates(["join_id", "station_id", "direction"], keep="first"))

    merged = observed.merge(
        sch[["join_id", "station_id", "direction", "sched_epoch", "line"]],
        on=["join_id", "station_id", "direction"], how="left")

    merged["delay_sec"] = (merged["obs_arr"] - merged["sched_epoch"]).round().astype("Int64")
    return merged


# --------------------------------------------------------------------
# service alerts active along each trip
# --------------------------------------------------------------------
# The alerts archive labels routes with a slightly different vocabulary than
# subwaydata.nyc's route_id: it has no express-variant codes, folding them into
# the parent line, and it calls Staten Island Railway "SI" where the observed
# feed uses both "SS" and "SI". Confirmed by diffing the two token sets over a
# sample period -- with these four aliases applied, every observed route_id has
# a counterpart in the alert vocabulary. Note this is a *different* mapping from
# ROUTE_LETTER_OVERRIDES above, which targets the schedule dataset's train_id
# conventions: alerts do carry Z, W and FS as distinct routes, so those stay.
ALERT_ROUTE_ALIASES = {"6X": "6", "7X": "7", "FX": "F", "SS": "SI"}

# The archive's status_label packs multiple concurrent conditions into one
# field, pipe-separated (e.g. "reroute | delays"), the same convention its
# `affected` column uses for routes. Split so a type array holds one condition
# per element rather than a compound string.
ALERT_TYPE_SEP = "|"


def alert_route(route_id):
    """Map an observed route_id onto the route vocabulary the alerts archive
    uses, per ALERT_ROUTE_ALIASES."""
    route = str(route_id).strip().upper()
    return ALERT_ROUTE_ALIASES.get(route, route)


def alert_types(status_label):
    """Split a status_label into its individual condition tokens."""
    return [t.strip() for t in str(status_label).split(ALERT_TYPE_SEP) if t.strip()]


def fetch_alerts_by_route(service_date, agency="NYCT Subway", app_token=None,
                          pad_days=1):
    """Fetch the service day's alerts, indexed by affected route.

    Each value is a list of (start_epoch, end_epoch, types) tuples in unix
    seconds, so they compare directly against the observed feed's epoch
    timestamps. The archive's timestamps are naive local time, so they're
    localized to TZ before conversion.

    The query window is padded by `pad_days` on each side because
    `fetch_subway_service_alerts` keys on when an alert *started*: an alert
    raised at 23:00 the night before is still active at 00:30 on the service
    date, and an unpadded query would miss it entirely.
    """
    day = datetime.fromisoformat(service_date)
    records = fetch_subway_service_alerts(
        day - timedelta(days=pad_days),
        day + timedelta(days=pad_days + 1),
        agency=agency,
        app_token=app_token,
    )

    epoch0 = pd.Timestamp("1970-01-01", tz="UTC")

    def to_epoch(dt):
        local = pd.Timestamp(dt).tz_localize(
            TZ, ambiguous="NaT", nonexistent="shift_forward")
        return (local - epoch0) / pd.Timedelta(seconds=1)

    by_route = defaultdict(list)
    for record in records:
        start = to_epoch(record.time)
        end = to_epoch(record.end_time)
        if pd.isna(start) or pd.isna(end):
            continue
        types = alert_types(record.status_label)
        for route in record.affected_trains:
            by_route[alert_route(route)].append((start, end, types))
    return by_route


def add_trip_alerts(merged, alerts_by_route):
    """Attach alert_count and alert_types to every stop-level row.

    An alert counts for a trip when it affects the trip's route and its span
    overlaps the trip's own span -- hence "along the trip": a delay that began
    two stops back is attached to every row of that trip, not only to the stop
    it was announced at. Both fields are therefore trip-level values repeated
    across the trip's rows.

    A fifth of archived alerts have a single update and so a zero-length span
    (see ServiceAlertRecord); those still register, because overlap is tested
    against the trip's whole span rather than an instant.
    """
    merged = merged.copy()
    spans = merged.groupby("trip_uid").agg(
        trip_start=("obs_arr", "min"),
        trip_end=("obs_arr", "max"),
        route=("route_id", "first"),
    )

    counts, types = {}, {}
    for trip_uid, row in spans.iterrows():
        candidates = alerts_by_route.get(alert_route(row["route"]), ())
        hits = [
            entry for entry in candidates
            if entry[0] <= row["trip_end"] and entry[1] >= row["trip_start"]
        ]
        counts[trip_uid] = len(hits)
        # Deduplicated because concurrent alerts routinely share a condition
        # (two separate "delays" events on one line), and sorted so the column
        # is stable across runs rather than ordered by fetch sequence.
        types[trip_uid] = sorted({t for entry in hits for t in entry[2]})

    merged["alert_count"] = merged["trip_uid"].map(counts).fillna(0).astype("int64")
    merged["alert_types"] = merged["trip_uid"].map(types)
    merged["alert_types"] = merged["alert_types"].apply(
        lambda v: v if isinstance(v, list) else [])
    return merged


def fetch_alerts_by_route_station(service_date, matcher, agency="NYCT Subway",
                                  app_token=None, pad_days=1):
    """Index the day's alerts by (route, station) rather than route alone.

    Keyed this way because the per-row lookup is then a dict hit plus a scan of
    the handful of alerts that ever touched that exact station, instead of a
    scan of every alert on the line.

    Station sets come from the alert headers, with ranges expanded along the
    route, so "no 6 train service between 3 Av-138 St and Pelham Bay Park"
    marks all 18 stops rather than the 2 named. Headers from every update of an
    event are unioned, since later updates sometimes name extra stations.
    """
    day = datetime.fromisoformat(service_date)
    records = fetch_subway_service_alerts(
        day - timedelta(days=pad_days),
        day + timedelta(days=pad_days + 1),
        agency=agency,
        app_token=app_token,
    )

    epoch0 = pd.Timestamp("1970-01-01", tz="UTC")

    def to_epoch(dt):
        local = pd.Timestamp(dt).tz_localize(
            TZ, ambiguous="NaT", nonexistent="shift_forward")
        return (local - epoch0) / pd.Timedelta(seconds=1)

    index = defaultdict(list)
    for record in records:
        start, end = to_epoch(record.time), to_epoch(record.end_time)
        if pd.isna(start) or pd.isna(end):
            continue
        types = alert_types(record.status_label)
        routes = [alert_route(r) for r in record.affected_trains]
        stations = defaultdict(set)
        for header in record.headers:
            for route, stops in matcher.affected_stations(header, routes).items():
                stations[route] |= set(stops)
        for route, stops in stations.items():
            for stop in stops:
                index[(route, stop)].append((start, end, types))
    return index


def add_station_alerts(merged, index, window_sec=7200):
    """Attach station-scoped alert columns to every stop-level row.

    An alert counts for a row when it names that row's station on that row's
    route and the stop falls inside the alert's effect window. Unlike the
    trip-level columns, this does not spread across the whole line.

    The window runs forward from the alert's FIRST update, not from its last.
    That is what the data supports: bucketing stop events by offset from the
    first update, delay lift is flat or negative beforehand and elevated for
    roughly two hours after, which is where `window_sec` comes from. Measuring
    from the last update instead lets a long-running event flag rows five or
    more hours old, and those rows are *below* the no-alert baseline for late
    arrivals -- they dilute the signal rather than extend it.
    """
    merged = merged.copy()
    counts, types, ages = [], [], []
    for route_id, station, when in zip(merged["route_id"], merged["station_id"],
                                       merged["obs_arr"]):
        live = [entry for entry in index.get((alert_route(route_id), str(station)), ())
                if 0 <= when - entry[0] <= window_sec]
        counts.append(len(live))
        types.append(sorted({t for entry in live for t in entry[2]}))
        # Age off the *nearest* alert start, so a row inside two overlapping
        # alerts reports the fresher one rather than an average of the two.
        ages.append(min((when - entry[0] for entry in live), default=None))

    merged["station_alert_count"] = counts
    merged["station_alert_types"] = types
    merged["station_alert_age_sec"] = pd.array(
        [None if a is None else int(a) for a in ages], dtype="Int64")
    return merged


def report_station_alerts(merged, trim=(-1800, 7200)):
    """Coverage and effect size for the station-scoped columns.

    Restricted to a trimmed delay band, because the untrimmed tail carries
    residual key-collision junk in the tens of thousands of seconds.

    The late-rate ratio is standardized: each alerted row's expected late rate
    is the no-alert rate in its own line, direction and hour cell, and the
    ratio is observed late rows over the sum of those expectations. An earlier
    version compared raw rates and only used the cells to filter rows, which
    reported 1.28x-1.63x on four days where the standardized figure was about
    1.0x. Alerts land on lines and hours that are already running late, so a
    raw ratio measures where alerts happen rather than what they do.

    One day is far too little to estimate this: rows from one alert move
    together. Treat the printed ratio as a smoke test, and use
    scripts/alert_scope_check.py for an estimate with an interval.
    """
    rows = len(merged)
    flagged = merged["station_alert_count"] > 0
    print(f"rows at an alerted station  : {flagged.sum():,}  "
          f"({flagged.sum()/rows:.1%})")

    band = merged[merged["delay_sec"].notna()]
    band = band[(band["delay_sec"] >= trim[0]) & (band["delay_sec"] <= trim[1])].copy()
    if len(band):
        band["hour"] = (pd.to_datetime(band["obs_arr"], unit="s", utc=True)
                        .dt.tz_convert(TZ).dt.hour)
        band["late"] = band["delay_sec"] > 300
        keys = ["line", "direction", "hour"]
        clear = band[band["station_alert_count"] == 0]
        cells = clear.groupby(keys).agg(base=("delay_sec", "median"),
                                        p_late=("late", "mean"),
                                        size=("late", "size"))
        cells = cells[cells["size"] >= 30]
        hit = band[band["station_alert_count"] > 0].join(cells, on=keys, how="inner")
        if len(hit):
            lift = hit["delay_sec"] - hit["base"]
            expected = hit["p_late"].sum()
            print(f"  median delay lift         : {lift.median():+.0f}s "
                  f"vs the same line, direction and hour with no alert")
            if expected:
                print(f"  over 300s late            : {hit['late'].sum():,} observed vs "
                      f"{expected:,.0f} expected  ({hit['late'].sum()/expected:.2f}x, "
                      f"standardized; one day, no interval)")

    counter = defaultdict(int)
    for values in merged.loc[flagged, "station_alert_types"]:
        for value in values:
            counter[value] += 1
    if counter:
        top = sorted(counter.items(), key=lambda kv: -kv[1])
        print("  types at those stations   : "
              + ", ".join(f"{k} {v:,}" for k, v in top[:6]))


def report_alerts(merged):
    rows = len(merged)
    with_alert = (merged["alert_count"] > 0).sum()
    trips = merged.drop_duplicates("trip_uid")
    trips_with = (trips["alert_count"] > 0).sum()
    print(f"rows on an alerted trip     : {with_alert:,}  ({with_alert/rows:.1%})")
    print(f"trips with >=1 active alert : {trips_with:,} / {len(trips):,}"
          f"  ({trips_with/len(trips):.1%})")
    counter = defaultdict(int)
    for values in trips.loc[trips["alert_count"] > 0, "alert_types"]:
        for value in values:
            counter[value] += 1
    if counter:
        top = sorted(counter.items(), key=lambda kv: -kv[1])
        print("  alert types on trips      : "
              + ", ".join(f"{k} {v:,}" for k, v in top[:6]))


def report(merged):
    n = len(merged)
    matched = merged["sched_epoch"].notna().sum()
    print("\n" + "=" * 60)
    print(f"observed stop events        : {n:,}")
    print(f"matched to a schedule row   : {matched:,}  ({matched/n:.1%})")
    if matched:
        d = merged.loc[merged["delay_sec"].notna(), "delay_sec"].astype(float)
        print(f"delay_sec  median {d.median():.0f}s   "
              f"p10 {d.quantile(.10):.0f}s   p90 {d.quantile(.90):.0f}s   "
              f"max {d.max():.0f}s")
        print(f"  (mostly small +/- values = healthy join; wild values = key mismatch)")
    print("=" * 60)
    if matched / n < 0.5:
        print("LOW MATCH RATE -> trust the headway columns as your target instead,\n"
              "  and check: right dataset year? train_id vs vehicle_id formatting?")


# --------------------------------------------------------------------
# fold mean delay back into the features grid (optional)
# --------------------------------------------------------------------
def update_features(merged, features_path, bin_sec=300):
    feats = pd.read_csv(features_path)
    # station_id can read back as int when all-numeric; force str to match
    feats["station_id"] = feats["station_id"].astype(str)
    feats["direction"] = feats["direction"].astype(str)
    m = merged.dropna(subset=["delay_sec"]).copy()
    m["station_id"] = m["station_id"].astype(str)
    m["direction"] = m["direction"].astype(str)
    m["bin_epoch"] = (m["obs_arr"] // bin_sec) * bin_sec
    m["ts"] = (pd.to_datetime(m["bin_epoch"], unit="s", utc=True)
                 .dt.tz_convert(_tz_of(feats)).map(lambda t: t.isoformat()))
    agg = (m.groupby(["ts", "station_id", "direction"])["delay_sec"]
             .mean().round(1).reset_index().rename(columns={"delay_sec": "delay_fill"}))
    out = feats.merge(agg, on=["ts", "station_id", "direction"], how="left")
    out["delay_sec"] = out["delay_fill"].where(out["delay_fill"].notna(), out.get("delay_sec"))
    out = out.drop(columns=["delay_fill"])
    dest = Path(features_path).with_name(Path(features_path).stem + "_with_delay.csv")
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  (delay_sec filled for "
          f"{out['delay_sec'].notna().sum():,}/{len(out):,} rows)")


def _tz_of(feats):
    # infer the tz used by the features ts column, default UTC
    s = str(feats["ts"].iloc[0]) if len(feats) else ""
    return "America/New_York" if s.endswith(("-04:00", "-05:00")) else "UTC"


def main():
    ap = argparse.ArgumentParser(description="Join observed stops to scheduled times -> delay_sec.")
    ap.add_argument("--file", help="subwaydatanyc archive")
    ap.add_argument("--stop-times")
    ap.add_argument("--trips")
    ap.add_argument("--service-date", required=True, help="YYYY-MM-DD of the observed day")
    ap.add_argument("--dataset", default="q9nv-uegs",
                    help="Socrata id for the schedules dataset OF THE MATCHING YEAR (default 2025)")
    ap.add_argument("--app-token", help="Socrata app token (avoids throttling)")
    ap.add_argument("--schedule-csv", help="use a cached/local schedule csv instead of the API")
    ap.add_argument("--features", help="features_<date>.csv to fill delay_sec into")
    ap.add_argument("--out", help="stop-level delays output (default delays_<date>.csv)")
    ap.add_argument("--no-alerts", action="store_true",
                    help="skip the service-alert join (one less network round trip)")
    ap.add_argument("--gtfs-dir", default=DEFAULT_GTFS_DIR,
                    help="local GTFS feed used to resolve alert headers to stations")
    args = ap.parse_args()

    observed = load_observed(args.file, args.stop_times, args.trips)
    if args.schedule_csv:
        schedule = pd.read_csv(args.schedule_csv, low_memory=False)
    else:
        print(f"[schedule] querying {args.dataset} for {args.service_date} ...")
        schedule = fetch_schedule(args.service_date, args.dataset, app_token=args.app_token)

    merged = compute_delay(observed, schedule)

    if not args.no_alerts:
        print(f"[alerts] querying the service-alerts archive for "
              f"{args.service_date} ...")
        merged = add_trip_alerts(
            merged, fetch_alerts_by_route(args.service_date, app_token=args.app_token))
        matcher = StationMatcher(gtfs_dir=args.gtfs_dir)
        merged = add_station_alerts(
            merged,
            fetch_alerts_by_route_station(args.service_date, matcher,
                                          app_token=args.app_token))

    report(merged)
    if not args.no_alerts:
        report_alerts(merged)
        report_station_alerts(merged)

    out = Path(args.out or f"delays_{args.service_date}.csv")
    cols = ["trip_uid", "station_id", "direction", "line",
            "obs_arr", "sched_epoch", "delay_sec",
            "alert_count", "alert_types",
            "station_alert_count", "station_alert_types", "station_alert_age_sec"]
    written = merged[[c for c in cols if c in merged]].copy()
    for column in ("alert_types", "station_alert_types"):
        if column in written:
            # Serialize the array with the same pipe convention the alerts
            # archive uses for its multi-valued columns, not a Python repr.
            written[column] = written[column].apply(
                lambda v: f" {ALERT_TYPE_SEP} ".join(v) if isinstance(v, list) else "")
    written.to_csv(out, index=False)
    print(f"wrote {out}  ({len(merged):,} stop-level rows)")

    if args.features:
        update_features(merged, args.features)


if __name__ == "__main__":
    main()