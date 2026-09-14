"""State carried between polls of the live MTA feeds.

A single poll is not enough to describe the network. Trip updates are
*predictions*, so an observed edge time only exists as the difference between
polls: a stop drops off the front of a trip's update list once the train has
left it, and the last time predicted for it is then effectively the observed
arrival. That is the same reconstruction subwaydata.nyc performs to produce
the archives the model was trained on. Checked against the live A feed over
two polls 45s apart: every dropped stop was a prefix of the trip's list and
none had a prediction still in the future.

Alerts need history too. The station alert feature is age since an alert was
*first* posted, with a two-hour effect window, and an alert keeps counting
inside that window even after it leaves the feed -- which is how training
computed it from the archive.

Nothing here reads the clock. Every method takes the time it applies at, so a
recorded sequence of polls replays to the same state.

`checkpoint` and `restore` carry that state across a process restart, so the
observation and alert history isn't lost with it. The previous poll's
predictions, which the next poll is diffed against, are only reused after a
short gap: diffed across a long one, every trip that finished during the
outage would log each of its stops as arrivals at minutes-old predicted times.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from mta_api.alert_stations import StationMatcher
from mta_api.models import Alert, TripUpdate
from mta_api.schedule_history import alert_route
from snapshot.schedule import StaticSchedule

# Same band scripts/training_data.py keeps. Below the floor is one train
# double-reported at a platform; above the ceiling is a layover or a gap in
# the feed, not a ride.
MIN_EDGE_SEC, MAX_EDGE_SEC = 20, 1200

# How far past the poll a dropped stop's last prediction may sit and still
# count as an arrival. A stop dropped while its prediction is well in the
# future was skipped or cut from the trip, not reached.
ARRIVAL_TOLERANCE_SEC = 60

# A restored checkpoint older than this starts a fresh diff baseline instead of
# diffing the next poll against its predictions. A few missed 30s polls are the
# same gap a failed feed fetch already leaves; more than that is an outage.
RESUME_DIFF_SEC = 120

CHECKPOINT_VERSION = 1

# Matches add_station_alerts' window_sec: delay lift is elevated for about two
# hours after an alert's first update and gone after.
ALERT_WINDOW_SEC = 7200

SUBWAY_AGENCY = "MTASBWY"

TripKey = tuple[str, str]  # (live trip_id, start_date)


@dataclass
class Observation:
    """One reconstructed edge traversal."""

    from_node: str
    to_node: str
    arrived_at: int  # unix seconds at to_node
    edge_sec: int


@dataclass
class AlertState:
    """One alert *event*, the unit the training archive counts.

    The live feed splits one incident into one entity per condition:
    "lmm:alert:267162:26" (Delays) and "lmm:alert:267162:29" (Stops Skipped)
    were published together for the same incident, where the archive stores a
    single event labelled "stops-skipped | delays". Entities are therefore
    grouped by `alert_event_key`, or one incident would count twice.
    """

    event_key: str
    kind: str  # "alert" for incidents, "planned_work", or whatever the id says
    first_seen: int
    last_seen: int
    alert_ids: set[str] = field(default_factory=set)
    # Every active_period start the feed has given. Planned work can carry
    # several (a nightly closure), incidents so far carry one.
    active_starts: set[int] = field(default_factory=set)
    route_ids: set[str] = field(default_factory=set)
    headers: set[str] = field(default_factory=set)
    # (alert route, station id without N/S), unioned over every header seen,
    # since later updates sometimes name stations earlier ones didn't. Same
    # union training takes over an event's updates.
    stations: set[tuple[str, str]] = field(default_factory=set)
    # Archive-style condition tokens keyed by the entity's created time, None
    # when the extension carried none. See `onset_types`.
    types_by_created: dict[int | None, set[str]] = field(default_factory=dict)

    @property
    def onset_types(self) -> set[str]:
        """Conditions as of the event's first posting.

        Training takes an event's label from its *first* update
        (ServiceAlertRecord.status_label), so a condition added later in the
        incident is not part of the feature. Live, that is the entities
        created at the event's earliest created time.
        """
        created = [c for c in self.types_by_created if c is not None]
        if created:
            return set(self.types_by_created[min(created)])
        return set(self.types_by_created.get(None, set()))

    def onset_at(self, at: int) -> int | None:
        """Live stand-in for the archive's first-update time, as of `at`.

        Taken from the feed's active_period starts, not from when this
        process first polled, so a snapshot started mid-incident still gets
        the true onset. Only an alert carrying no start at all falls back to
        `first_seen`, which runs young if polling began after it was posted.
        None when nothing has started by `at`.

        Incidents use the *earliest* start. An event groups one entity per
        condition, and a condition added mid-incident arrives with its own
        later start; taking the latest would move the incident's onset forward
        each time, where the archive keeps its first update. Planned work uses
        the latest start at or before `at`, so a recurring closure ages from
        tonight's start rather than its first night.

        For incidents active_period.start equalled the extension's created
        time to the second on every live subway incident checked (6 of 6),
        so this is the time the incident was first posted. Still unverified
        against the archive directly: it lags the live feed by weeks, so no
        event appears in both.
        """
        started = [s for s in self.active_starts if s <= at]
        if started:
            return min(started) if self.kind == "alert" else max(started)
        return self.first_seen if self.first_seen <= at else None


def _epoch(dt: datetime | None) -> int | None:
    return None if dt is None else int(dt.timestamp())


def alert_kind(alert_id: str) -> str:
    """"lmm:alert:267140:26" -> "alert", "lmm:planned_work:34644" -> "planned_work"."""
    parts = alert_id.split(":")
    return parts[1] if len(parts) >= 2 else alert_id


def alert_event_key(alert_id: str) -> str:
    """"lmm:alert:267162:29" -> "lmm:alert:267162".

    The trailing segment is not a version: it varies across unrelated alerts
    (24, 26, 29, 31, 34 seen in one poll) and one incident carried both :26
    (Delays) and :29 (Stops Skipped) at once, so it reads as a condition code.
    The number before it identifies the incident. Planned-work ids have no
    trailing segment and pass through unchanged.
    """
    parts = alert_id.split(":")
    if len(parts) >= 4 and parts[1] == "alert":
        return ":".join(parts[:3])
    return alert_id


def archive_alert_type(live_type: str | None) -> set[str]:
    """Live condition string -> the archive's status_label tokens.

    The archive writes conditions lowercase and hyphenated ("stops-skipped"),
    and the live extension title-cased with spaces ("Stops Skipped"). Split on
    "|" too, as training does, in case a live value ever carries several.
    """
    if not live_type:
        return set()
    return {re.sub(r"[\s\-]+", "-", token.strip().lower())
            for token in live_type.split("|") if token.strip()}


class LiveState:
    """Accumulates live feed polls into what a snapshot needs.

    `route_names` maps a GTFS route_id to the name graph node ids are built
    from (routes.txt short name), so observations join graph edges directly.
    `schedule` enables the schedule-derived columns; without it they stay null.
    """

    def __init__(self, matcher: StationMatcher, route_names: dict[str, str] | None = None,
                 schedule: StaticSchedule | None = None, retention_sec: int = 3 * 3600):
        self.matcher = matcher
        self.route_names = route_names or {}
        self.schedule = schedule
        self.retention_sec = retention_sec

        self.observations: list[Observation] = []
        # node -> arrival times, for "how long since any train reached here"
        self.arrivals: dict[str, list[int]] = defaultdict(list)
        self.alerts: dict[str, AlertState] = {}
        # feed key -> last successful poll, and route_id -> feed key as seen
        self.feed_polled_at: dict[str, int] = {}
        self.route_feed: dict[str, str] = {}

        # feed key -> trip key -> (route_id, [(stop_id, predicted ts)]) from
        # that feed's previous poll
        self._pending: dict[str, dict[tuple[str, str], tuple[str, list[tuple[str, int]]]]] = {}
        # trip key -> (node, arrival ts) of the trip's latest observed stop
        self._last_stop: dict[TripKey, tuple[str, int]] = {}
        # trip key -> (arrival ts, delay_sec or None) at that same stop. Kept
        # apart from _last_stop because it must outlive a stalled train: a
        # train stuck for 20 minutes is exactly the one whose delay matters.
        self.trip_delay: dict[TripKey, tuple[int, int | None]] = {}

    def node_id(self, stop_id: str, route_id: str) -> str:
        return f"{stop_id}::{self.route_names.get(route_id, route_id)}"

    # ------------------------------------------------------------------
    # trip updates -> observed arrivals and edge traversals
    # ------------------------------------------------------------------
    def ingest_trip_updates(self, feed_key: str, updates: list[TripUpdate],
                            polled_at: datetime) -> None:
        """Fold one *successful* poll of a feed into state.

        Only call this with a poll that actually returned. A failed fetch
        passed as an empty list would read as every trip having vanished at
        once, and their past-due stops as simultaneous arrivals.
        """
        now = int(polled_at.timestamp())
        current: dict[tuple[str, str], tuple[str, list[tuple[str, int]]]] = {}
        for tu in updates:
            stops = []
            for stu in tu.stop_time_updates:
                when = stu.arrival or stu.departure
                if when is not None:
                    stops.append((stu.stop_id, int(when.timestamp())))
            current[(tu.trip_id, tu.start_date)] = (tu.route_id, stops)
            if tu.route_id:
                self.route_feed[tu.route_id] = feed_key

        for trip_key, (route_id, previous) in self._pending.get(feed_key, {}).items():
            if trip_key in current:
                still_listed = {stop for stop, _ in current[trip_key][1]}
                dropped = [(s, t) for s, t in previous if s not in still_listed]
            else:
                # Trip finished or left the feed: whatever was already due
                # was reached, the rest never will be.
                dropped = previous
            for stop_id, predicted in dropped:
                if predicted <= now + ARRIVAL_TOLERANCE_SEC:
                    self._record_arrival(trip_key, stop_id, route_id, predicted)

        self._pending[feed_key] = current
        self.feed_polled_at[feed_key] = now
        self._prune(now)

    def _record_arrival(self, trip_key: TripKey, stop_id: str, route_id: str, at: int) -> None:
        node = self.node_id(stop_id, route_id)
        previous = self._last_stop.get(trip_key)
        # A stop can drop off a trip's list, reappear, and drop again. Seen
        # live as F21N -> F21N on one G trip; that is one arrival, not a hop.
        if previous is not None and previous[0] == node:
            return
        self.arrivals[node].append(at)
        self._last_stop[trip_key] = (node, at)
        # The delay at the latest observed stop, None when that stop has no
        # schedule match. Deliberately not carried over from an earlier
        # matched stop: that would report a stale lateness as current.
        scheduled = (self.schedule.scheduled_at(trip_key[0], trip_key[1], stop_id)
                     if self.schedule else None)
        self.trip_delay[trip_key] = (at, None if scheduled is None else at - scheduled)
        if previous is None:
            return
        from_node, left_at = previous
        edge_sec = at - left_at
        if MIN_EDGE_SEC <= edge_sec <= MAX_EDGE_SEC:
            self.observations.append(Observation(from_node, node, at, edge_sec))

    # ------------------------------------------------------------------
    # alerts -> station-scoped alert state with first-seen times
    # ------------------------------------------------------------------
    def ingest_alerts(self, alerts: list[Alert], polled_at: datetime) -> None:
        now = int(polled_at.timestamp())
        for alert in alerts:
            if SUBWAY_AGENCY not in alert.agency_ids:
                continue
            starts = [_epoch(start) for start, _ in alert.active_periods if start is not None]
            key = alert_event_key(alert.alert_id)
            state = self.alerts.get(key)
            if state is None:
                state = self.alerts[key] = AlertState(
                    event_key=key,
                    kind=alert_kind(alert.alert_id),
                    first_seen=now,
                    last_seen=now,
                )
            state.last_seen = now
            state.alert_ids.add(alert.alert_id)
            state.active_starts |= set(starts)
            created = _epoch(alert.created_at)
            state.types_by_created.setdefault(created, set()).update(
                archive_alert_type(alert.alert_type))
            state.route_ids |= set(alert.affected_route_ids)
            if alert.header_text and alert.header_text not in state.headers:
                state.headers.add(alert.header_text)
                routes = sorted({alert_route(r) for r in state.route_ids})
                for route, stops in self.matcher.affected_stations(alert.header_text,
                                                                   routes).items():
                    state.stations |= {(route, stop) for stop in stops}
        self._prune(now)

    # ------------------------------------------------------------------
    def upcoming_traversals(self, at: int) -> dict[tuple[str, str], tuple[int, TripKey, str, str, int]]:
        """For each edge, the next train due to traverse it, as of `at`.

        Returns {(from_node, to_node): (predicted arrival at from_node, trip
        key, from stop_id, to stop_id, predicted arrival at to_node)}, from
        consecutive pairs in each trip's remaining stop list. A train already
        between the two stops has left from_node and is not a train a rider
        there can board, so it is skipped.

        Uses each feed's latest poll only, and drops feeds polled after `at`,
        so a snapshot never sees predictions made after its own time.
        """
        best: dict[tuple[str, str], tuple[int, TripKey, str, str, int]] = {}
        for feed_key, trips in self._pending.items():
            if self.feed_polled_at.get(feed_key, at + 1) > at:
                continue
            for trip_key, (route_id, stops) in trips.items():
                for (from_stop, eta), (to_stop, eta_to) in zip(stops, stops[1:]):
                    edge = (self.node_id(from_stop, route_id), self.node_id(to_stop, route_id))
                    if edge not in best or eta < best[edge][0]:
                        best[edge] = (eta, trip_key, from_stop, to_stop, eta_to)
        return best

    def trips_as_of(self, at: int):
        """(trip key, route_id, remaining [(stop_id, predicted ts)]) for every
        trip in each feed's latest poll, skipping feeds polled after `at`."""
        for feed_key, trips in self._pending.items():
            if self.feed_polled_at.get(feed_key, at + 1) > at:
                continue
            for trip_key, (route_id, stops) in trips.items():
                yield trip_key, route_id, stops

    def last_stop(self, trip_key: TripKey) -> tuple[str, int] | None:
        """(node, arrival ts) of the trip's latest observed stop, if any."""
        return self._last_stop.get(trip_key)

    # ------------------------------------------------------------------
    # persistence across restarts
    # ------------------------------------------------------------------
    def checkpoint(self, at: int) -> dict:
        """Everything polls have accumulated, as JSON-safe data, as of `at`.
        The matcher, route names and schedule are rebuilt at startup instead."""
        return {
            "version": CHECKPOINT_VERSION,
            "at": at,
            "observations": [[o.from_node, o.to_node, o.arrived_at, o.edge_sec]
                             for o in self.observations],
            "arrivals": self.arrivals,
            "alerts": [{
                "event_key": a.event_key, "kind": a.kind,
                "first_seen": a.first_seen, "last_seen": a.last_seen,
                "alert_ids": sorted(a.alert_ids), "active_starts": sorted(a.active_starts),
                "route_ids": sorted(a.route_ids), "headers": sorted(a.headers),
                "stations": sorted(a.stations),
                "types_by_created": [[c, sorted(t)] for c, t in a.types_by_created.items()],
            } for a in self.alerts.values()],
            "feed_polled_at": self.feed_polled_at,
            "route_feed": self.route_feed,
            "pending": {feed: [[list(trip), route_id, stops]
                               for trip, (route_id, stops) in trips.items()]
                        for feed, trips in self._pending.items()},
            "last_stop": [[list(trip), node, ts] for trip, (node, ts) in self._last_stop.items()],
            "trip_delay": [[list(trip), ts, delay] for trip, (ts, delay) in self.trip_delay.items()],
        }

    def restore(self, data: dict, now: int, resume_diff_sec: int = RESUME_DIFF_SEC) -> bool:
        """Load a `checkpoint` into this (fresh) state as of `now`.

        History (observations, arrivals, alerts, trip delays) is always kept,
        then pruned to its usual windows at `now`. The diff baseline -- the
        previous poll's predictions, per-feed poll times, each trip's last stop
        -- is kept only if the checkpoint is at most `resume_diff_sec` old.
        Returns whether it was, i.e. whether the next poll continues diffing.
        """
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version {data.get('version')!r}")
        self.observations = [Observation(*row) for row in data["observations"]]
        self.arrivals = defaultdict(list, {n: list(ts) for n, ts in data["arrivals"].items()})
        self.alerts = {}
        for a in data["alerts"]:
            self.alerts[a["event_key"]] = AlertState(
                event_key=a["event_key"], kind=a["kind"],
                first_seen=a["first_seen"], last_seen=a["last_seen"],
                alert_ids=set(a["alert_ids"]), active_starts=set(a["active_starts"]),
                route_ids=set(a["route_ids"]), headers=set(a["headers"]),
                stations={tuple(st) for st in a["stations"]},
                types_by_created={c: set(t) for c, t in a["types_by_created"]},
            )
        self.route_feed = dict(data["route_feed"])
        self.trip_delay = {tuple(trip): (ts, delay) for trip, ts, delay in data["trip_delay"]}

        resumed = 0 <= now - data["at"] <= resume_diff_sec
        if resumed:
            self.feed_polled_at = dict(data["feed_polled_at"])
            self._pending = {feed: {tuple(trip): (route_id, [tuple(s) for s in stops])
                                    for trip, route_id, stops in trips}
                             for feed, trips in data["pending"].items()}
            self._last_stop = {tuple(trip): (node, ts) for trip, node, ts in data["last_stop"]}
        self._prune(now)
        return resumed

    def _prune(self, now: int) -> None:
        cutoff = now - self.retention_sec
        self.observations = [o for o in self.observations if o.arrived_at >= cutoff]
        for node in list(self.arrivals):
            kept = [t for t in self.arrivals[node] if t >= cutoff]
            if kept:
                self.arrivals[node] = kept
            else:
                del self.arrivals[node]
        self._last_stop = {k: v for k, v in self._last_stop.items()
                           if v[1] >= now - MAX_EDGE_SEC}
        self.trip_delay = {k: v for k, v in self.trip_delay.items() if v[0] >= cutoff}
        # An alert stays relevant for its window after onset, in the feed or not.
        self.alerts = {k: a for k, a in self.alerts.items()
                       if a.last_seen >= now - ALERT_WINDOW_SEC}
