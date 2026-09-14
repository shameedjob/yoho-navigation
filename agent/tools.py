"""Strands tools for the navigation agent: geocoding, pathfinding on the
combined subway+bus graph, and (eventually) real-time-aware predicted
paths.

get_position and get_path route on the static schedule graph (agent/routing.py,
which has no torch, so the alert scheduler can route without it).
get_predicted_path prices the trip under live conditions with the joint Graph
WaveNet (GRAPH_MODEL_PATH), as ml_model/benchmark_paths.py's
<model>_joint_eta_live router does. The polling service (snapshot/service.py)
holds no model: it serves the model's input window (/features/window), and
this process runs the model on it for every subway ride and transfer (walk +
wait) edge, the 90th percentile of each. Transfers between subway and bus are
priced at the same quantile (_cross_system_costs), and the trip may start at
any platform near the rider, each charged the wait for its next train
(/waits: the raw ETA, else 0.9 x typical headway).

Without a usable prediction -- no checkpoint, an empty or stale window --
ride and transfer costs stay on the schedule and it reports
model_adjusted=False; without /waits, waits_adjusted=False. No LightGBM: it
can't share a process with torch on macOS (OMP Error #15).
"""

from __future__ import annotations

import io
import math
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from strands import tool

from agent.routing import (SNAPSHOT_SERVICE_URL, SNAPSHOT_TIMEOUT_SEC, TZ, BUS_WAIT_HEADWAY_FRACTION, Route, Waits,
                           _access_sec, _boarding_headway_sec, _describe_path, _fetch_waits, _first_wait,
                           _get_graph, _get_stop_names, _route, _state_at, geocode_address, schedule_route)
from graph import Graph
from ml_model import live_window
from ml_model.graph_wavenet import GWNetForecaster
from ml_model.sequence import build_feature_tensor
from snapshot.build import GraphSnapshot

# The joint Graph WaveNet retrained on time-of-day bucketed schedule costs
# (docs/benchmarks/edges_2026-sample30_buckets.md, paths_2026-sample30_buckets.md).
GRAPH_MODEL_PATH = Path(os.environ.get(
    "GRAPH_MODEL_PATH", "ml_model/checkpoints/gwnet_joint_2026-sample30_buckets/gwnet.pt"))
# A prediction from a window whose newest grid time is older than this is
# stale: the model forecasts the 10 minutes after it.
MODEL_MAX_AGE_SEC = 2 * live_window.STEP_SEC
# Loaded once; False once a load has failed, so a missing checkpoint isn't
# retried on every call.
_graph_model: GWNetForecaster | None | bool = None
# Why the last call had no model prediction, for the tool's output.
_model_error: str | None = None
# (window grid times, ModelCosts) for the last window predicted.
_model_cache: tuple[tuple[int, ...], "ModelCosts"] | None = None
# ((window end, state), weighted graph) for the last prediction laid over the
# graph. Reweighting copies the whole combined graph, so repeat calls in the
# same 10 minutes reuse it.
_weighted: tuple[tuple[int, str], Graph] | None = None


@tool
def get_current_time() -> dict:
    """Get the current time and NYC service state.

    Returns:
        {
          "now_unix": current Unix timestamp (seconds),
          "now_local": New York local time, e.g. "Sun 2026-09-13 15:49",
          "service_state": current NYC transit schedule state, e.g. "Weekday:10-16",
          "service_period": "Weekday" or "Saturday" or "Sunday",
        }
    """
    now = datetime.now(TZ)
    state = _state_at(int(now.timestamp()))
    return {
        "now_unix": int(now.timestamp()),
        "now_local": now.strftime("%a %Y-%m-%d %H:%M"),  # New York time
        "service_state": state,
        "service_period": state.split(":")[0] if ":" in state else state,
    }


@tool
def get_position(address: str) -> tuple[float, float]:
    """Geocode a street address or place name to (latitude, longitude).

    Args:
        address: A street address or place name, e.g. "365 5th Avenue, New York, NY".

    Returns:
        (latitude, longitude) for the given address.
    """
    return geocode_address(address)


@tool
def get_path(start: tuple[float, float], end: tuple[float, float],
             departure_time: int | None = None) -> dict:
    """Find the fastest route between two coordinates on the combined
    subway+bus graph, using the average-time-weighted schedule for the day
    and time of departure (no live updates -- see get_predicted_path for that).

    Args:
        start: (latitude, longitude) of the trip's starting point.
        end: (latitude, longitude) of the trip's destination.
        departure_time: Unix timestamp (seconds) the trip leaves at; defaults
            to now. Schedule costs depend on it: late-night and weekend
            service runs less often, so waits and transfers cost more.

    Returns:
        {"steps": [{"stop_id", "stop_name", "mode", "route", "lat", "lon"}, ...],
         "total_time_sec": float, door to door: the walk to the first stop,
             the ride, and the walk from the last stop (no wait for the
             first vehicle),
         "walk_in_sec", "walk_out_sec": those two walks,
         "service_state": the schedule state priced, e.g. "Weekday:16-22"}
    """
    return schedule_route(start, end, departure_time)


def _fetch_snapshot() -> tuple[GraphSnapshot, bool]:
    """The polling service's latest snapshot, and whether it is warm (has
    covered the observation window, so obs_last_* features are populated).
    Raises if the service is unreachable or hasn't published yet.
    """
    response = requests.get(f"{SNAPSHOT_SERVICE_URL}/snapshot", timeout=SNAPSHOT_TIMEOUT_SEC)
    response.raise_for_status()
    payload = response.json()
    edges = pd.DataFrame(payload["rows"])
    edges["is_transfer"] = edges["is_transfer"].astype(bool)
    at = datetime.fromtimestamp(payload["snapshot_ts"], timezone.utc)
    first = edges.iloc[0] if len(edges) else {}
    return GraphSnapshot(at=at, service_period=first.get("service_period"),
                         service_state=first.get("service_state"), edges=edges), payload["warm"]


class ModelCosts:
    """One graph-model prediction: per-edge costs for the 10 minutes after the
    window's last grid time."""

    def __init__(self, costs: dict[tuple[str, str, bool], float],
                 transfer_wait_into: dict[str, float], window_end: int, real_steps: int) -> None:
        self.costs = costs                            # rides and subway transfers (walk + wait)
        self.transfer_wait_into = transfer_wait_into  # node -> median predicted wait onto it
        self.window_end = window_end
        self.real_steps = real_steps


def _get_graph_model() -> GWNetForecaster | None:
    """The checkpoint at GRAPH_MODEL_PATH, loaded once. None, with the reason in
    _model_error, when it's missing or doesn't fit the graph: its learned
    adjacency and per-transfer embeddings belong to specific edges, so a GTFS
    change that adds, drops or renames subway edges needs a retrained model."""
    global _graph_model, _model_error
    if _graph_model is None:
        try:
            model = GWNetForecaster.load(GRAPH_MODEL_PATH)
            model.model.eval()
            graph = _get_graph()
            subway = {(f, t, x) for f in graph if graph.get_node(f).mode == "subway"
                      for t, _, x in graph.get_node(f).paths
                      if graph.get_node(t) is not None and graph.get_node(t).mode == "subway"}
            keys = set(model.line_graph.edge_keys)
            if keys != subway:
                raise ValueError(f"checkpoint edges don't match the graph: {len(keys - subway)} "
                                 f"unknown, {len(subway - keys)} unpriced; retrain the model")
            if not model.config.transfers:
                raise ValueError("checkpoint has no transfer head (--with-transfers)")
            _graph_model = model
        except (OSError, ValueError, RuntimeError) as exc:
            _model_error = f"{type(exc).__name__}: {exc}"
            _graph_model = False
    return _graph_model or None


def _predict_window(model: GWNetForecaster, rows: pd.DataFrame, end: int) -> tuple[np.ndarray, int]:
    """(prediction per line-graph edge in cost seconds, real snapshots used) for
    the window ending at grid time `end`, from feature rows carrying a grid_ts
    column. Slots training would have padded are dropped before encoding."""
    times = live_window.window_times(end, model.config.step_sec, model.config.window)
    real = set(times) - live_window.padded(times, end)
    rows = rows[rows["grid_ts"].isin(real)]
    rows = rows.assign(is_transfer=rows["is_transfer"].astype(bool))
    X = build_feature_tensor(rows, model.line_graph, model.encoder, np.array(times), "grid_ts")
    with torch.no_grad():
        out = model(torch.from_numpy(X)[None], torch.tensor([end]))[:, :, 0]
        pred = model.to_cost_units(out).clamp(min=0)[0].numpy()
    return pred, len(real & set(rows["grid_ts"].unique()))


def _model_costs(now: int) -> ModelCosts | None:
    """The graph model run on the service's feature window, or None (reason in
    _model_error) when there is no model, no window, or the window is stale.

    The input tensor is built as training built it (ml_model/live_window.py):
    the last `window` grid times ending at the newest the service has, each
    slot the snapshot at that grid time, slots before the local day's start
    or missing from the service padded as "no row". Cached per window.
    """
    global _model_cache, _model_error
    model = _get_graph_model()
    if model is None:
        return None
    try:
        response = requests.get(f"{SNAPSHOT_SERVICE_URL}/features/window",
                                timeout=SNAPSHOT_TIMEOUT_SEC)
        response.raise_for_status()
        meta = response.json()
    except (requests.RequestException, ValueError) as exc:
        _model_error = f"feature window unavailable: {exc}"
        return None
    grid = sorted(int(t) for t in meta.get("grid_ts") or [])
    if not grid:
        _model_error = "feature window is empty"
        return None
    if (meta["step_sec"], meta["window"]) != (model.config.step_sec, model.config.window):
        _model_error = (f"service window {meta['window']} x {meta['step_sec']}s doesn't match the "
                        f"model's {model.config.window} x {model.config.step_sec}s")
        return None
    end = grid[-1]
    if now - end > MODEL_MAX_AGE_SEC:
        _model_error = f"feature window is stale: newest grid time {now - end}s ago"
        return None
    if _model_cache is not None and _model_cache[0] == tuple(grid):
        _model_error = None
        return _model_cache[1]

    rows = pd.DataFrame(meta["rows"])
    pred, real_steps = _predict_window(model, rows, end)
    keys = model.line_graph.edge_keys
    costs = {k: float(v) for k, v in zip(keys, pred)}
    walk = model.model.transfer_walk.numpy()
    waits_into = pd.DataFrame({"node": [k[1] for k in keys[model.num_rides:]],
                               "wait": pred[model.num_rides:] - walk})
    result = ModelCosts(costs, waits_into.groupby("node")["wait"].median().clip(lower=0).to_dict(),
                        end, real_steps)
    _model_cache = (tuple(grid), result)
    _model_error = None
    return result


def _cross_system_costs(model: ModelCosts, state: str) -> dict[tuple[str, str, bool], float]:
    """Subway <-> bus transfers re-priced to match model-priced subway transfers.

    Otherwise they keep the schedule's average wait (headway/2) while subway
    transfers carry the model's 90th-percentile wait, and a route saves
    "waiting" by stepping out to a bus for a stop and back in.

      bus -> subway   walk + the median predicted wait for riders transferring
                      onto that platform, else BUS_WAIT_HEADWAY_FRACTION of
                      the platform's scheduled headway
      subway -> bus   walk + station exit + BUS_WAIT_HEADWAY_FRACTION of the
                      bus's scheduled headway (no live bus data yet)
    """
    graph = _get_graph()
    costs = {}
    for (f, t), access in _access_sec.items():
        headway = _boarding_headway_sec.get((f, t), {}).get(state)
        if graph.get_node(f).mode == "bus" and t in model.transfer_wait_into:
            costs[(f, t, True)] = access + model.transfer_wait_into[t]
        elif headway is not None:
            costs[(f, t, True)] = access + BUS_WAIT_HEADWAY_FRACTION * headway
    return costs


def _weighted_graph(model: ModelCosts | None, state: str) -> Graph:
    """The combined graph with `state`'s costs replaced by the model's, or the
    graph itself without a model. Live costs are laid over the departure
    state, so an edge that doesn't run then stays closed. Cached per
    prediction and state: reweighting copies the whole combined graph."""
    global _weighted
    graph = _get_graph()
    if model is None:
        return graph
    key = (model.window_end, state)
    if _weighted is not None and _weighted[0] == key:
        return _weighted[1]
    costs = dict(model.costs)
    costs.update(_cross_system_costs(model, state))
    weighted = graph.reweighted(state, costs)
    _weighted = (key, weighted)
    return weighted


def _path_time(graph: Graph, node_ids: list[str], state: str) -> float | None:
    """Total cost of following `node_ids` on `graph` in `state`, or None if
    some hop doesn't run then. Where two edges join the same pair, the
    cheaper counts."""
    total = 0.0
    for from_id, to_id in zip(node_ids, node_ids[1:]):
        costs = [graph.edge_time(from_id, dest, is_transfer, t, state)
                 for dest, t, is_transfer in graph.get_node(from_id).paths if dest == to_id]
        cost = min(costs, default=math.inf)
        if cost == math.inf:
            return None
        total += cost
    return total


def _alerts_on_path(snapshot: GraphSnapshot, node_ids: list[str]) -> list[dict]:
    """Stations along the path with a live incident alert, from the snapshot's
    station_alert_* columns on the edges the path arrives by."""
    edges = snapshot.edges
    hops = pd.DataFrame({"from_node": node_ids[:-1], "to_node": node_ids[1:]})
    on_path = edges.merge(hops, on=["from_node", "to_node"])
    on_path = on_path[on_path["station_alert_count"].fillna(0) > 0]
    names = _get_stop_names()
    alerts = []
    for row in on_path.drop_duplicates("to_node").itertuples():
        stop_id = row.to_node.split("::")[0]
        alerts.append({
            "node_id": row.to_node,
            "stop_name": names.get(stop_id, stop_id),
            "alert_count": int(row.station_alert_count),
            "alert_age_sec": None if pd.isna(row.station_alert_age_sec)
                             else int(row.station_alert_age_sec),
            "alert_types": row.station_alert_types or "",
        })
    return alerts


@tool
def get_predicted_path(
    base_time: int,
    event_time: int,
    start_location: tuple[float, float],
    end_location: tuple[float, float],
) -> dict:
    """Re-predict a trip under live MTA conditions to say when to leave.

    Use this when you need real-time subway delays or want to say when to leave for a deadline.
    Compares schedule-based route against live delays from the MTA.

    Args:
        base_time: the initially planned trip duration in seconds (get_path's total_time_sec).
        event_time: Unix timestamp (seconds) of the deadline to arrive by.
        start_location: (latitude, longitude) of the trip's starting point.
        end_location: (latitude, longitude) of the destination/event.

    Returns:
        {
          "delay_sec": predicted trip time minus base_time (positive = slower),
          "leave_by_ts": when to leave to make the deadline,
          "slack_sec": seconds from now until leave_by_ts (negative = deadline already missed),
          "route_changed": whether the best route shifted due to live delays,
          "predicted": the live route with steps, total time, and waits,
          "alerts_on_route": service alerts along the route,
          "model_adjusted": whether live model pricing was available,
          "waits_adjusted": whether live wait predictions were available,
        }
    """
    snapshot, warm = _fetch_snapshot()
    waits = _fetch_waits()
    graph = _get_graph()
    now = int(time.time())
    # The trip is priced in its departure state: when the rider would leave on
    # the original plan, or now if that has passed.
    state = _state_at(max(now, event_time - base_time))
    model = _model_costs(now)
    weighted = _weighted_graph(model, state)

    planned = _route(graph, start_location, end_location, state)
    predicted = _route(weighted, start_location, end_location, state, waits)
    predicted_time = predicted.total_sec
    snapshot_ts = int(snapshot.at.timestamp())

    # The planned route re-priced live: its rides and transfers on the weighted
    # graph, the same walks, and the first wait at its own first stop.
    planned_wait = _first_wait(planned.node_ids[0], state, waits)
    planned_time = _path_time(weighted, planned.node_ids, state)
    if planned_time is not None:
        planned_time += planned.walk_in_sec + planned_wait + planned.walk_out_sec

    def summary(route: Route, g: Graph, total: float | None, wait: float) -> dict:
        return {"steps": _describe_path(g, route.node_ids), "total_time_sec": total,
                "origin_wait_sec": wait if waits is not None else None,
                "walk_in_sec": round(route.walk_in_sec), "walk_out_sec": round(route.walk_out_sec)}

    return {
        "planned": summary(planned, graph, planned_time, planned_wait),
        "predicted": summary(predicted, weighted, predicted_time, predicted.first_wait_sec),
        "route_changed": predicted.node_ids != planned.node_ids,
        "initial_time_sec": base_time,
        "delay_sec": round(predicted_time - base_time),
        "initial_leave_by_ts": event_time - base_time,
        "leave_by_ts": round(event_time - predicted_time),
        "slack_sec": round(event_time - predicted_time - now),
        "alerts_on_route": _alerts_on_path(snapshot, predicted.node_ids),
        "snapshot_ts": snapshot_ts,
        "snapshot_age_sec": now - snapshot_ts,
        "snapshot_warm": warm,
        "service_period": snapshot.service_period,
        "service_state": state,
        "model_adjusted": model is not None,
        "model": {"checkpoint": str(GRAPH_MODEL_PATH),
                  "window_end_ts": model.window_end if model else None,
                  "window_steps": model.real_steps if model else None,
                  "error": None if model else _model_error},
        "waits_adjusted": waits is not None,
        "waits_ts": waits.ts if waits else None,
    }


@tool
def compare_schedule_vs_live(
    start: tuple[float, float],
    end: tuple[float, float],
    arrival_deadline: int | None = None,
) -> dict:
    """Compare schedule-based routing against live delays to show real impact.

    Shows both routes side-by-side: schedule-based route vs live-adjusted route,
    total time difference, delay impact, and whether the route changed.

    Args:
        start: (latitude, longitude) of trip start.
        end: (latitude, longitude) of trip end.
        arrival_deadline: Unix timestamp (seconds) to arrive by; if set, also
            returns when you must leave to make it under live conditions.

    Returns:
        {
          "schedule_route": the schedule-based best route,
          "schedule_time_sec": schedule total time,
          "live_route": the live-adjusted best route,
          "live_time_sec": live total time,
          "delay_sec": how much slower live is vs schedule,
          "route_changed": whether best route shifted,
          "leave_by_ts": when to leave if deadline set (None otherwise),
          "alerts": any service alerts on the live route,
        }
    """
    now = int(time.time())
    schedule_result = get_path(start, end, departure_time=now)
    schedule_time = schedule_result["total_time_sec"]

    if arrival_deadline is None:
        arrival_deadline = now + int(schedule_time) + 600  # 10 min buffer

    live_result = get_predicted_path(
        base_time=int(schedule_time),
        event_time=arrival_deadline,
        start_location=start,
        end_location=end,
    )

    return {
        "schedule_route": schedule_result["steps"],
        "schedule_time_sec": schedule_time,
        "live_route": live_result["predicted"]["steps"],
        "live_time_sec": live_result["predicted"]["total_time_sec"],
        "delay_sec": live_result["delay_sec"],
        "route_changed": live_result["route_changed"],
        "leave_by_ts": live_result["leave_by_ts"],
        "slack_sec": live_result["slack_sec"],
        "alerts": live_result["alerts_on_route"],
        "model_adjusted": live_result["model_adjusted"],
        "waits_adjusted": live_result["waits_adjusted"],
    }

