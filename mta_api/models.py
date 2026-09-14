"""Plain-Python representations of the protobuf messages MTA returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class StopTimeUpdate:
    stop_id: str
    arrival: datetime | None
    departure: datetime | None


@dataclass
class TripUpdate:
    trip_id: str
    route_id: str
    start_date: str
    direction: str | None
    stop_time_updates: list[StopTimeUpdate] = field(default_factory=list)


@dataclass
class VehiclePosition:
    trip_id: str
    route_id: str
    current_stop_id: str | None
    status: str | None
    timestamp: datetime | None


@dataclass
class Alert:
    alert_id: str
    header_text: str
    description_text: str
    affected_route_ids: list[str] = field(default_factory=list)
    # (start, end) per GTFS-rt active_period; either side is None when unset.
    # Planned work is published ahead of time, so an alert present in the feed
    # is not necessarily in effect yet -- check these before counting it.
    active_periods: list[tuple[datetime | None, datetime | None]] = field(default_factory=list)
    # Agencies named by informed_entity. The all-alerts feed mixes subway
    # ("MTASBWY") with bus and rail, so filter on this rather than route id.
    agency_ids: list[str] = field(default_factory=list)
    # From MTA's proprietary alert extension, which the standard bindings
    # don't decode (see client._mercury_fields). `alert_type` is the live
    # condition, e.g. "Stops Skipped" -- the counterpart of the archive's
    # status_label. Field meanings are inferred from values, not documented:
    # created_at matched active_period start on every live incident checked.
    alert_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class ServiceAlertRecord:
    """One *event* from the NY Open Data historical service-alerts archive
    (https://data.ny.gov/Transportation/MTA-Service-Alerts-Beginning-April-2020/7kct-peq7),
    as opposed to `Alert`, which comes from MTA's live GTFS-realtime feed.

    The archive stores one row per *alert update*, not per incident: a single
    disruption appears as several rows sharing an `event_id`, numbered by
    `update_number`, each with a revised `header` as the situation develops.
    This record collapses one such group into a time span, so `time` is when
    the event was first announced and `end_time` is its last update.

    `end_time` is the best available proxy for "when it ended" -- the archive
    has no explicit resolution column, but MTA's final update is consistently
    the wind-down message ("...after we addressed a signal problem at X"),
    verified as 450 of 471 events over a sample week. Two caveats:

    - Roughly a fifth of events (95 of 471 in that week) have only one update,
      so `end_time == time` and the span is zero-length. Most are already
      phrased as closing messages, but some describe an in-progress problem
      that simply never got a follow-up. `update_count` lets callers filter
      these out when a real duration matters.
    - The end time is when MTA stopped *posting*, which trails actual recovery
      by an unknown amount, so treat spans as a lower bound.

    `status_label` is the event as first announced. It can change mid-event
    (54 of 471 in that week), typically escalating, so it describes the onset
    rather than the whole span. `affected_trains` is the opposite: the union
    across every update, since the set of affected lines also shifts.
    """

    status_label: str
    time: datetime
    end_time: datetime
    affected_trains: list[str] = field(default_factory=list)
    event_id: str | None = None
    update_count: int = 1
    # Every update's header text, oldest first. Kept as a list rather than
    # collapsed to one string because later updates sometimes name stations the
    # first one didn't, so resolving locations needs all of them.
    headers: list[str] = field(default_factory=list)

    @property
    def duration(self) -> timedelta:
        """How long the event was being actively updated. Zero for
        single-update events, where the true duration is unknown."""
        return self.end_time - self.time


def _to_datetime(unix_ts: int) -> datetime | None:
    if not unix_ts:
        return None
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)
