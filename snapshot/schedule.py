"""Scheduled stop times for live trips, from the static GTFS feed.

Training joined observed stops to the Socrata "MTA Subway Schedules" dataset
on a normalized vehicle label (see norm_id in mta_api/schedule_history.py).
Live trip updates carry no such label, but their trip_id is the tail of a
static GTFS trip_id:

    live    "080950_A..S58R"
    static  "ASP26GEN-A084-Saturday-00_080950_A..S58R"

The tail is origin time in hundredths of a minute, route, direction and path.
The path code is the unreliable part: live 6 trains run as "6..S04X001" while
the static feed has "6..S01R". Joining on (origin time, route, direction)
alone matched 378 of 536 live trips on a Saturday sample, against 157 on the
exact tail, with 2 keys ambiguous across the whole day's service.

Most of the rest were whole routes running a changed timetable -- SIR, 6, G
and J all 0% on that sample, with origin times minutes off every base trip.
MTA's *supplemented* static feed, the base schedule plus the coming week's
service changes, carries those trips: against the same live poll it matched
490 of 522 (93.9%) where the base feed in data/gtfs_subway matched 364
(69.7%). It is republished as changes are posted, so `ensure_supplemented`
refetches it once a day.

What still misses stays unmatched. Snapping a trip to the nearest-time static
trip would hand a disrupted train a schedule it isn't running, so those get
null schedule columns and has_schedule 0, the same censoring training uses.

Unverified: that this static schedule and the Socrata dataset agree for the
same trip. Both derive from MTA's timetable, but they are different sources.
"""

from __future__ import annotations

import csv
import io
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.stop_routes import parse_gtfs_time

TZ = ZoneInfo("America/New_York")

# origin time, route, direction
TRIP_KEY = re.compile(r"(?:^|_)(\d{6})_([^._]+)\.+([NS])")

SUPPLEMENTED_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_supplemented.zip"
DEFAULT_SUPPLEMENTED_DIR = Path("data/gtfs_supplemented")

WEEKDAY_COLUMNS = ("monday", "tuesday", "wednesday", "thursday", "friday",
                   "saturday", "sunday")


def trip_key(trip_id: str) -> tuple[str, str, str] | None:
    """(origin, route, direction) from a live id or a static id's tail."""
    match = TRIP_KEY.search(trip_id)
    return match.groups() if match else None


def ensure_supplemented(dest_dir=DEFAULT_SUPPLEMENTED_DIR,
                        max_age_sec: float = 24 * 3600, timeout: float = 120) -> Path:
    """Download and unpack the supplemented feed unless a copy younger than
    `max_age_sec` is already there. About 20MB. Returns the directory."""
    dest_dir = Path(dest_dir)
    marker = dest_dir / "stop_times.txt"
    if marker.exists() and time.time() - marker.stat().st_mtime < max_age_sec:
        return dest_dir
    with urllib.request.urlopen(SUPPLEMENTED_URL, timeout=timeout) as response:
        payload = response.read()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest_dir)
    # extractall keeps the archive's timestamps; the age check wants ours.
    marker.touch()
    return dest_dir


def service_day_epoch(service_date: date) -> int:
    """Unix seconds that GTFS times on `service_date` count from.

    GTFS defines times from "noon minus 12h", which is midnight except on the
    two DST change days, where it is an hour off midnight."""
    noon = datetime(service_date.year, service_date.month, service_date.day, 12, tzinfo=TZ)
    return int(noon.timestamp()) - 12 * 3600


class StaticSchedule:
    def __init__(self, gtfs_dir=Path("data/gtfs_subway")):
        gtfs_dir = Path(gtfs_dir)

        def read(name):
            with open(gtfs_dir / name, newline="", encoding="utf-8-sig") as handle:
                yield from csv.DictReader(handle)

        self._calendar = list(read("calendar.txt"))
        self._exceptions: dict[str, dict[str, str]] = defaultdict(dict)
        for row in read("calendar_dates.txt"):
            self._exceptions[row["date"]][row["service_id"]] = row["exception_type"]

        # service_id -> trip key -> static trip ids
        self._by_key: dict[str, dict[tuple, list[str]]] = defaultdict(lambda: defaultdict(list))
        for row in read("trips.txt"):
            key = trip_key(row["trip_id"])
            if key is not None:
                self._by_key[row["service_id"]][key].append(row["trip_id"])

        # static trip id -> stop_id -> seconds after the service day's origin
        self._stop_times: dict[str, dict[str, int]] = defaultdict(dict)
        for row in read("stop_times.txt"):
            seconds = parse_gtfs_time(row["arrival_time"] or row["departure_time"])
            if seconds is not None:
                self._stop_times[row["trip_id"]][row["stop_id"]] = seconds

        self._matches: dict[tuple[str, str], str | None] = {}

    def services_on(self, service_date: date) -> set[str]:
        stamp = service_date.strftime("%Y%m%d")
        column = WEEKDAY_COLUMNS[service_date.weekday()]
        active = {row["service_id"] for row in self._calendar
                  if row[column] == "1" and row["start_date"] <= stamp <= row["end_date"]}
        for service_id, kind in self._exceptions.get(stamp, {}).items():
            if kind == "1":
                active.add(service_id)
            else:
                active.discard(service_id)
        return active

    def match(self, trip_id: str, start_date: str) -> str | None:
        """Static trip id for a live (trip_id, start_date), or None when there
        is no match or more than one."""
        cache_key = (trip_id, start_date)
        if cache_key not in self._matches:
            self._matches[cache_key] = self._match(trip_id, start_date)
        return self._matches[cache_key]

    def _match(self, trip_id: str, start_date: str) -> str | None:
        key = trip_key(trip_id)
        try:
            day = datetime.strptime(start_date, "%Y%m%d").date()
        except ValueError:
            return None
        if key is None:
            return None
        candidates = [static_id for service_id in self.services_on(day)
                      for static_id in self._by_key.get(service_id, {}).get(key, ())]
        return candidates[0] if len(candidates) == 1 else None

    def scheduled_at(self, trip_id: str, start_date: str, stop_id: str) -> int | None:
        """Scheduled unix seconds for a live trip at a platform, or None."""
        static_id = self.match(trip_id, start_date)
        if static_id is None:
            return None
        seconds = self._stop_times[static_id].get(stop_id)
        if seconds is None:
            return None
        day = datetime.strptime(start_date, "%Y%m%d").date()
        return service_day_epoch(day) + seconds
