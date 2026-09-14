"""Freeze live state into a per-edge snapshot of the graph at one instant.

One row per graph edge, ride and transfer alike, keyed by
(from_node, to_node, is_transfer) -- the same key Graph uses for its
per-period costs. Edges rather than nodes because the model predicts
`edge_sec` and the router adds up edge costs; node-level state (alerts at a
station, time since a train last arrived) is projected onto the edge that
arrives at that node, which is also how the training rows attach
`station_alert_*` to `to_node`.

Missing data stays missing. An edge nothing has traversed recently gets null
observation columns and `obs_count` 0, never a filled-in normal time, because
absence of trains correlates with disruption. `feed_age_sec` separates "no
trains seen" from "we haven't heard from that feed".

Columns named after docs/MODEL_DATA.md features mean the same thing they do
there, or are null. A training row describes one train's traversal; a snapshot
row describes the *next* train due to traverse the edge (the earliest
predicted arrival at from_node in the latest poll), since that is the one a
rider boarding there would take. The schedule-derived columns describe that
train:

  sched_edge_sec     its scheduled to_node time minus from_node time, from the
                     static GTFS feed (see snapshot/schedule.py)
  has_schedule       1 when both ends of that have a scheduled time
  prior_delay_sec    its delay at the latest stop it was observed arriving at.
                     Training measures delay exactly at from_node; live, the
                     train usually hasn't reached from_node yet, so this is
                     its current lateness, which MODEL_DATA.md names as the
                     live meaning. prior_delay_age_sec says how old it is.

Null when there's no upcoming train, no schedule match, or no observed stop
yet.

There is no `route_alert_count`: it needed the alert's and trip's end times,
unknown mid-trip, so it was dropped from training rows as well.

`station_alert_types` uses the condition decoded from MTA's alert extension,
mapped to the archive's tokens ("Stops Skipped" -> "stops-skipped"). Empty
string when no alert, as training writes it.

`graph_edge_sec` is the graph's average scheduled cost for the service period
and is *not* the same quantity as `sched_edge_sec`.

Every per-train column is measured on the clock training uses for that
column, not at the snapshot instant: the calendar and station alert columns
at the next train's predicted arrival at to_node, obs_last_age_sec at its
predicted arrival at from_node. Without a train due they fall back to the
snapshot time. docs/FEATURE_PARITY.md maps each feature's training and live
derivation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from graph import Graph
from graph.service_states import service_day_at, state_at, states_of
from mta_api.alert_stations import DIRECTION_SUFFIX
from mta_api.schedule_history import alert_route
from snapshot.state import ALERT_WINDOW_SEC, LiveState

TZ = ZoneInfo("America/New_York")

# Observations older than this don't describe current conditions for an edge.
DEFAULT_OBS_WINDOW_SEC = 1800

COLUMNS = [
    # identity
    "snapshot_ts", "from_node", "to_node", "route", "direction", "is_transfer",
    "service_period", "service_state", "runs_in_period", "graph_edge_sec",
    # MODEL_DATA.md features
    "has_schedule", "sched_edge_sec", "prior_delay_sec",
    "hour", "minute_of_day", "dow", "is_weekend",
    "station_alert_count", "station_alert_age_sec", "station_alert_types",
    # live observation state
    "obs_count", "obs_last_edge_sec", "obs_last_age_sec", "obs_median_edge_sec",
    "to_node_last_arrival_age_sec", "feed_age_sec",
    # the next train, which the schedule columns above describe
    "next_train_eta_sec", "prior_delay_age_sec",
]


@dataclass
class GraphSnapshot:
    at: datetime
    service_period: str  # service day, e.g. "Weekday"
    service_state: str   # schedule state costs are read in, e.g. "Weekday:16-22"
    edges: pd.DataFrame

    def to_csv(self, path) -> None:
        self.edges.to_csv(path, index=False)

    def weighted_graph(self, graph: Graph, predicted_edge_sec) -> Graph:
        """A copy of `graph` costed by model predictions for this snapshot.

        `predicted_edge_sec` is one value per row of `edges`, in row order
        (a list, array or Series). Missing values (None/NaN) keep the
        schedule cost. Costs change only for this snapshot's service state,
        and the copy routes in that state by default.
        """
        values = list(predicted_edge_sec)
        if len(values) != len(self.edges):
            raise ValueError(f"expected {len(self.edges)} predictions, got {len(values)}")
        costs = {}
        for from_id, to_id, is_transfer, value in zip(
                self.edges["from_node"], self.edges["to_node"],
                self.edges["is_transfer"], values):
            if value is None or pd.isna(value):
                continue
            costs[(from_id, to_id, bool(is_transfer))] = float(value)
        return graph.reweighted(self.service_state, costs)


def service_period_at(local: datetime) -> str:
    """Weekday / Saturday / Sunday for a local time, the service day starting
    at 04:00 (graph/service_states.py). Holidays aren't accounted for."""
    return service_day_at(local)


def add_graph_columns(rows: pd.DataFrame, graph: Graph) -> pd.DataFrame:
    """Join the graph-derived snapshot columns onto per-edge training rows.

    Training rows store only what was observed; route, is_transfer,
    service_period, service_state, runs_in_period and graph_edge_sec come from
    the graph so rebuilding the graph never means regenerating training data.

    The state is taken at the train's departure from from_node (`ts -
    edge_sec`), the training side of a snapshot's knowledge cutoff, with the
    same rule a snapshot applies to its own time (service_states.states_of).
    Rows whose edge the graph lacks get a null graph_edge_sec, runs_in_period
    False and route from the node id suffix.
    """
    periods, states = states_of(rows["ts"] - rows["edge_sec"])
    route, runs, cost = [], [], []
    for from_id, to_id, period in zip(rows["from_node"], rows["to_node"], states):
        node = graph.get_node(from_id)
        base = next((t for dest, t, transfer in (node.paths if node else ())
                     if dest == to_id and not transfer), None)
        value = (graph.edge_time(from_id, to_id, False, base, period)
                 if base is not None else math.inf)
        target = graph.get_node(to_id)
        route.append(target.vehicle if target else to_id.split("::")[-1])
        runs.append(value != math.inf)
        cost.append(value if value != math.inf else None)
    out = rows.copy()
    out["route"] = route
    out["is_transfer"] = False
    out["service_period"] = periods.to_numpy()
    out["service_state"] = states.to_numpy()
    out["runs_in_period"] = runs
    out["graph_edge_sec"] = pd.Series(cost, index=out.index, dtype="Float64").round().astype("Int64")
    return out


def alerts_by_station(state: LiveState, now: int,
                      alert_kinds: frozenset[str]) -> dict[tuple[str, str], list]:
    """(alert route, station) -> alert events naming it, known by `now`."""
    alerts_at: dict[tuple[str, str], list] = {}
    for alert in state.alerts.values():
        if alert.kind not in alert_kinds or alert.first_seen > now:
            continue
        for key in alert.stations:
            alerts_at.setdefault(key, []).append(alert)
    return alerts_at


def station_alert_summary(alerts_at: dict[tuple[str, str], list], keys, when: int):
    """(count, age of the freshest, "a | b" types, events) for alert events
    naming any of `keys` whose onset is 0..ALERT_WINDOW_SEC before `when`.
    Counted per event, as add_station_alerts counts archive events."""
    ages, types, events = [], set(), {}
    for key in keys:
        for alert in alerts_at.get(key, ()):
            if alert.event_key in events:
                continue
            onset = alert.onset_at(when)
            if onset is None or not 0 <= when - onset <= ALERT_WINDOW_SEC:
                continue
            events[alert.event_key] = (alert, onset)
            ages.append(when - onset)
            types |= alert.onset_types
    return (len(ages), (min(ages) if ages else None), " | ".join(sorted(types)),
            list(events.values()))


def route_ids_by_name(route_names: dict[str, str]) -> dict[str, set[str]]:
    """Invert route_id -> display name. Several ids share a name: GS, FS and H
    are all "S" in the graph."""
    by_name: dict[str, set[str]] = {}
    for route_id, name in route_names.items():
        by_name.setdefault(name, set()).add(route_id)
    return by_name


def build_snapshot(graph: Graph, state: LiveState, at: datetime,
                   obs_window_sec: int = DEFAULT_OBS_WINDOW_SEC,
                   alert_kinds: frozenset[str] = frozenset({"alert"})) -> GraphSnapshot:
    """Snapshot `graph` under `state` as of `at`.

    Only data timestamped at or before `at` is used, so a snapshot built from
    replayed polls doesn't see the future.

    `alert_kinds` defaults to incidents only, matching training: none of the
    12,315 archive updates in Q1 2025 carried a planned label, while planned
    work is most of the live feed on a given day.

    The station alert columns follow add_station_alerts exactly. They count
    alert *events* (see AlertState), take each event's conditions from its
    first posting, and are measured at the moment the train reaches to_node --
    training's clock is the traversal's arrival there, so a snapshot row uses
    the next train's predicted arrival at to_node, falling back to `at` when
    no train is due. Only alerts already ingested by `at` are used; an alert
    posted between `at` and the train's arrival can't be known live, which is
    the one unavoidable difference from training.
    """
    now = int(at.timestamp())
    local = at.astimezone(TZ)
    period = service_period_at(local)
    # One state for the whole snapshot, taken at its own time -- the clock
    # training's add_graph_columns and replay.snapshot_features match.
    service_state = state_at(local)
    ids_for_name = route_ids_by_name(state.route_names)
    upcoming = state.upcoming_traversals(now)

    alerts_at = alerts_by_station(state, now, alert_kinds)

    def station_alerts(keys, when: int):
        count, age, types, _ = station_alert_summary(alerts_at, keys, when)
        return count, age, types

    # edge -> observed traversals inside the window, oldest first
    recent: dict[tuple[str, str], list] = {}
    for obs in state.observations:
        if now - obs_window_sec <= obs.arrived_at <= now:
            recent.setdefault((obs.from_node, obs.to_node), []).append(obs)
    for group in recent.values():
        group.sort(key=lambda o: o.arrived_at)

    last_arrival = {}
    for node, times in state.arrivals.items():
        past = [t for t in times if t <= now]
        if past:
            last_arrival[node] = max(past)

    def calendar_at(ts: int) -> dict:
        """Local-time columns for `ts`, as training derives them from the
        traversal's arrival at to_node."""
        when = datetime.fromtimestamp(ts, TZ)
        return {
            "hour": when.hour,
            "minute_of_day": when.hour * 60 + when.minute,
            "dow": when.weekday(),
            "is_weekend": int(when.weekday() >= 5),
        }

    rows = []
    for from_id in graph:
        for to_id, base_time, is_transfer in graph.get_node(from_id).paths:
            to_node = graph.get_node(to_id)
            if to_node is None:
                continue
            route = to_node.vehicle
            cost = graph.edge_time(from_id, to_id, is_transfer, base_time, service_state)
            runs = cost != math.inf

            station = DIRECTION_SUFFIX.sub("", to_node.stop_id)
            direction = to_node.stop_id[-1] if DIRECTION_SUFFIX.search(to_node.stop_id) else None
            route_ids = ids_for_name.get(route, {route})

            observed = recent.get((from_id, to_id), []) if not is_transfer else []
            feed_times = [state.feed_polled_at[state.route_feed[rid]]
                          for rid in route_ids if rid in state.route_feed]
            arrived = last_arrival.get(to_id)

            eta = eta_to = sched_edge = delay = delay_age = None
            has_schedule = 0
            next_train = upcoming.get((from_id, to_id)) if not is_transfer else None
            if next_train is not None:
                eta, trip, from_stop, to_stop, eta_to = next_train
                if state.schedule is not None:
                    leave = state.schedule.scheduled_at(trip[0], trip[1], from_stop)
                    reach = state.schedule.scheduled_at(trip[0], trip[1], to_stop)
                    if leave is not None and reach is not None:
                        has_schedule = 1
                        sched_edge = reach - leave
                observed_delay = state.trip_delay.get(trip)
                if observed_delay is not None and observed_delay[0] <= now:
                    delay_age = now - observed_delay[0]
                    delay = observed_delay[1]

            # A prediction already in the past means the train is late against
            # its own forecast; it still reaches to_node no earlier than now.
            alert_clock = max(eta_to, now) if eta_to is not None else now
            # Training measures obs_last_age_sec when its train reaches
            # from_node, so here it is measured at the next train's predicted
            # arrival there. Observations are still only those seen by `now`.
            obs_clock = max(eta, now) if eta is not None else now
            alert_count, alert_age, alert_types = station_alerts(
                [(alert_route(rid), station) for rid in route_ids], alert_clock)

            rows.append({
                "snapshot_ts": now,
                "from_node": from_id,
                "to_node": to_id,
                "route": route,
                "direction": direction,
                "is_transfer": is_transfer,
                "service_period": period,
                "service_state": service_state,
                "runs_in_period": runs,
                "graph_edge_sec": cost if runs else None,
                "has_schedule": has_schedule,
                "sched_edge_sec": sched_edge,
                "prior_delay_sec": delay,
                # Same clock as the alert columns: the next train's predicted
                # arrival at to_node. Snapshot time ran a median 6 minutes
                # early and could fall on the previous day near midnight.
                **calendar_at(alert_clock),
                "station_alert_count": alert_count,
                # Freshest event, as add_station_alerts takes the minimum age.
                "station_alert_age_sec": alert_age,
                "station_alert_types": alert_types,
                "obs_count": len(observed),
                "obs_last_edge_sec": observed[-1].edge_sec if observed else None,
                "obs_last_age_sec": obs_clock - observed[-1].arrived_at if observed else None,
                "obs_median_edge_sec": (statistics.median(o.edge_sec for o in observed)
                                        if observed else None),
                "to_node_last_arrival_age_sec": now - arrived if arrived is not None else None,
                "feed_age_sec": now - max(feed_times) if feed_times else None,
                "next_train_eta_sec": None if eta is None else eta - now,
                "prior_delay_age_sec": delay_age if delay is not None else None,
            })

    edges = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("graph_edge_sec", "sched_edge_sec", "prior_delay_sec",
                   "station_alert_age_sec", "obs_last_edge_sec",
                   "obs_last_age_sec", "obs_median_edge_sec",
                   "to_node_last_arrival_age_sec", "feed_age_sec",
                   "next_train_eta_sec", "prior_delay_age_sec"):
        edges[column] = edges[column].astype("Float64").round().astype("Int64")
    return GraphSnapshot(at=at, service_period=period, service_state=service_state, edges=edges)
