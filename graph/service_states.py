"""Service states: the (service day, time-of-day bucket) pairs schedule costs
are kept for.

There are 12: Weekday, Saturday and Sunday, each split into four six-hour
buckets aligned to the rush hours rather than to midnight:

  04-10  AM rush
  10-16  midday
  16-22  PM rush
  22-04  overnight, running past midnight

A service day starts at 04:00, so the overnight bucket belongs to the day it
began on: 01:00 early Sunday is "Saturday:22-04". `state_at` applies that rule
to a timestamp, and every place that needs a state from a clock -- routing,
live snapshots, training rows -- must go through it so they agree.

The GTFS side (`stop_time_states`) has to fit the same rule. MTA feeds put
the small hours on both sides of midnight: each service day has stop times
from 00:00 to 04:00 as well as past 24:00. A day's own times from 04:00 up
belong to it. Its times before 04:00 run on the calendar morning of that day,
so they belong to the *previous* night's overnight bucket:

  Weekday 00:00-04:00   Weekday nights (Mon-Thu) and Sunday night
  Saturday 00:00-04:00  Friday night, a Weekday state. Dropped: a Weekday
                        night is already described by Weekday trips, and
                        adding Saturday's would count Friday once in five
                        nights as if it were every night
  Sunday 00:00-04:00    Saturday night

Such times are shifted by 24h so they sort after the night's late trips and
headways are measured across one continuous timeline.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

DAYS = ("Weekday", "Saturday", "Sunday")
BUCKETS = ("04-10", "10-16", "16-22", "22-04")
DAY_START_HOUR = 4
BUCKET_HOURS = 6
_TZ = ZoneInfo("America/New_York")
STATES = tuple(f"{day}:{bucket}" for day in DAYS for bucket in BUCKETS)

# A bucket with fewer trips than this takes the day's average instead: a
# handful of trips says little about ride time, and a mean gap over them
# ignores the empty stretch around them. With none at all the edge is closed.
MIN_BUCKET_TRIPS = 6

# Whose overnight bucket a service day's pre-04:00 stop times belong to; see
# the module docstring.
_NIGHT_BEFORE = {"Weekday": ("Weekday", "Sunday"), "Saturday": (), "Sunday": ("Saturday",)}


def state(day: str, bucket: str) -> str:
    return f"{day}:{bucket}"


def day_of(state_name: str) -> str:
    return state_name.split(":", 1)[0]


def day_of_weekday(weekday: int) -> str:
    """Service day for a Python weekday number (Monday is 0)."""
    return {5: "Saturday", 6: "Sunday"}.get(weekday, "Weekday")


def service_day_at(local: datetime) -> str:
    """Weekday / Saturday / Sunday for a local time, with the day starting at
    04:00. Holidays, which run weekend service on a weekday, aren't handled."""
    return day_of_weekday((local - timedelta(hours=DAY_START_HOUR)).weekday())


def state_at(local: datetime) -> str:
    """The service state for a local (America/New_York) time."""
    shifted = local - timedelta(hours=DAY_START_HOUR)
    return state(day_of_weekday(shifted.weekday()), BUCKETS[shifted.hour // BUCKET_HOURS])


def states_of(epoch_seconds):
    """Vectorized `service_day_at` and `state_at` over a pandas Series of Unix
    seconds: (day Series, state Series), same index. Training rows and replay
    grids use this; it must agree with `state_at` exactly."""
    import pandas as pd

    # Wall-clock arithmetic, as datetime subtraction does in state_at: 04:00
    # local stays the boundary on DST change days.
    shifted = (pd.to_datetime(epoch_seconds, unit="s", utc=True).dt.tz_convert(_TZ)
               .dt.tz_localize(None) - pd.Timedelta(hours=DAY_START_HOUR))
    days = shifted.dt.weekday.map(day_of_weekday)
    buckets = (shifted.dt.hour // BUCKET_HOURS).map(dict(enumerate(BUCKETS)))
    return days, days + ":" + buckets


def stop_time_states(day: str, seconds: int) -> list[tuple[str, int]]:
    """(state, timeline seconds) pairs a stop time counts toward, for a trip on
    service `day` at GTFS `seconds` past its service day's midnight.

    Timeline seconds are comparable within a state: pre-04:00 times are moved
    a day later to follow the previous evening's trips.
    """
    if seconds >= DAY_START_HOUR * 3600:
        index = min((seconds // 3600 - DAY_START_HOUR) // BUCKET_HOURS, len(BUCKETS) - 1)
        return [(state(day, BUCKETS[index]), seconds)]
    return [(state(night, BUCKETS[-1]), seconds + 86400) for night in _NIGHT_BEFORE[day]]


def mean_gap(times: list[int]) -> float | None:
    """Average headway over sorted arrival times, or None with fewer than two."""
    if len(times) < 2:
        return None
    return (times[-1] - times[0]) / (len(times) - 1)


def per_state(by_state: dict[str, float], counts: dict[str, int],
              by_day: dict[str, float]) -> dict[str, float]:
    """Apply the sparse-bucket rule to one edge's figures.

    `by_state` is the figure computed from a state's own trips, `counts` how
    many trips that was, `by_day` the whole-day figure. A state with trips but
    fewer than MIN_BUCKET_TRIPS takes its day's figure when there is one. A
    state with no trips, or no computable figure, is left out: the edge doesn't
    run then.
    """
    out = {}
    for name, count in counts.items():
        if count <= 0:
            continue
        day_value = by_day.get(day_of(name))
        if count < MIN_BUCKET_TRIPS and day_value is not None:
            out[name] = day_value
        elif name in by_state:
            out[name] = by_state[name]
    return out


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def active_services(feed_dir: Path) -> dict[str, set[str]]:
    """Service ids for a typical Weekday, Saturday and Sunday of a feed.

    Matching service ids by name double-counts: bus feeds carry alternative
    calendars for the same trips ("EN_D6-Weekday" and "EN_D6-Weekday-SDon",
    school days on or off), and the subway feed has two date-range copies of
    each holiday service. So for each day type this looks at every date the
    feed covers, takes the set of services calendar.txt and calendar_dates.txt
    make active on it, and keeps the most common set -- the regular pattern,
    not a holiday.
    """
    weekly: dict[str, tuple[set[int], date, date]] = {}
    calendar = feed_dir / "calendar.txt"
    if calendar.exists():
        names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        with open(calendar, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                days = {i for i, name in enumerate(names) if row[name] == "1"}
                weekly[row["service_id"]] = (days, _parse_date(row["start_date"]),
                                             _parse_date(row["end_date"]))
    added: dict[date, set[str]] = {}
    removed: dict[date, set[str]] = {}
    exceptions = feed_dir / "calendar_dates.txt"
    if exceptions.exists():
        with open(exceptions, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                target = added if row["exception_type"] == "1" else removed
                target.setdefault(_parse_date(row["date"]), set()).add(row["service_id"])

    spans = [(start, end) for _, start, end in weekly.values()] + [(d, d) for d in added]
    if not spans:
        return {day: set() for day in DAYS}
    first = min(s for s, _ in spans)
    last = max(e for _, e in spans)

    patterns: dict[str, Counter] = {day: Counter() for day in DAYS}
    current = first
    while current <= last:
        on = {sid for sid, (days, start, end) in weekly.items()
              if current.weekday() in days and start <= current <= end}
        on = (on | added.get(current, set())) - removed.get(current, set())
        if on:
            patterns[day_of_weekday(current.weekday())][frozenset(on)] += 1
        current += timedelta(days=1)
    return {day: (set(counts.most_common(1)[0][0]) if counts else set())
            for day, counts in patterns.items()}


def load_trip_days(feed_dir: Path) -> dict[str, tuple[str, ...]]:
    """trip_id -> the service days it runs on, for trips in `active_services`.

    Usually one day, but not always: bus feeds' "-BM" services (e.g.
    "JG_D6-Weekday-SDon-BM") run Sunday through Thursday, whatever the name.
    """
    services = active_services(feed_dir)
    days_of_service: dict[str, tuple[str, ...]] = {}
    for day in DAYS:
        for sid in services[day]:
            days_of_service[sid] = days_of_service.get(sid, ()) + (day,)
    with open(feed_dir / "trips.txt", newline="", encoding="utf-8") as f:
        return {row["trip_id"]: days_of_service[row["service_id"]]
                for row in csv.DictReader(f) if row["service_id"] in days_of_service}
