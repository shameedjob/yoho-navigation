"""Long-running live poller that serves graph snapshots over HTTP.

    python3 -m snapshot.service --port 8791

One process owns the LiveState: a background thread polls every subway trip
update feed and the alerts feed, rebuilds the snapshot after each poll, and
caches it with a per-node state table. HTTP handlers only read the cache, so a
slow client never delays polling. This is the process meant to run on AWS; the
local version uses only the standard library so it containerizes as is.

Endpoints, all GET, JSON unless `?format=csv`:

  /health           polling status, feed ages, whether state is warmed up
  /features         edge keys + ml_model.features.MODEL_FEATURES, one row per
                    edge. The rows a model scores. Filters: route, from_node,
                    to_node, due=1 (only edges with a train due)
  /snapshot         every snapshot column (same filters)
  /nodes            one state row per graph node. Filter: route
  /nodes/<node_id>  that node's state, its alert events, the feature rows
                    of the edges into and out of it, and its wait row
  /features/window  the graph model's input: ride-edge feature rows for each
                    of the last `window` grid times (grid_ts column), built at
                    those times (ml_model/live_window.py). Meta: step_sec,
                    window, grid_ts present
  /waits            one row per node: wait_sec, what a router charges for the
                    first train there -- the raw ETA, else 0.9 x typical
                    headway, else no service (wait_source says which) -- plus
                    the origin-wait features. Filters: route, node

The service holds no model. The agent (agent/tools.py) runs the graph model on
/features/window and prices transfers with it; this process only polls, keeps
the window, and serves. typical_headway_sec needs --typical-headways (a
typical_headway.csv from ml_model.train_waits); without it platforms with no
ETA get no wait. Headway columns need arrivals history, so they fill in as
polling warms up.

Observed edge times need history: `warm` in /health turns true once polling
has covered the 30-minute observation window. Before that, obs_last_* is
mostly null (docs/FEATURE_PARITY.md).

Every poll cycle, published or failed, appends the /health status as one JSON
line to <status-dir>/snapshot_status_<local date>.jsonl (default logs/status),
so polling gaps, feed staleness and errors can be reviewed after the fact.
`published` says whether that cycle produced a new snapshot. Pass
--status-dir "" to turn it off.

After every published snapshot, the latest one is kept in --state-dir (default
data/live), each file overwritten in place:

  live_state.json.gz      LiveState.checkpoint(): observations, arrivals, alert
                          history and the diff baseline. Loaded at startup so a
                          restart keeps its history; see LiveState.restore for
                          when polling resumes diffing against it.
  snapshot_latest.csv.gz  the snapshot table itself, the graph with its trains
  feature_window.csv.gz   the /features/window rows, so a restart keeps the
                          model's history instead of up to an hour short

/health reports what was restored under `checkpoint`. --state-dir "" disables
both saving and restoring.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import signal
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from graph.subway_loader import build_subway_graph, load_routes
from ml_model import live_window
from ml_model.features import EDGE_KEY, MODEL_FEATURES
from ml_model.waits import ORIGIN_FEATURES
from mta_api import MTAClient
from mta_api.alert_stations import DEFAULT_GTFS_DIR, DIRECTION_SUFFIX, StationMatcher
from mta_api.schedule_history import alert_route
from snapshot.__main__ import poll_once
from snapshot.build import (DEFAULT_OBS_WINDOW_SEC, TZ, alerts_by_station,
                            build_snapshot, route_ids_by_name,
                            station_alert_summary)
from snapshot.schedule import StaticSchedule, ensure_supplemented
from snapshot.state import LiveState
from snapshot.waits import WaitFeatures, load_typical, with_origin_wait

log = logging.getLogger("snapshot.service")

FEATURE_COLUMNS = ["snapshot_ts", *EDGE_KEY, "runs_in_period", *MODEL_FEATURES.columns]
WINDOW_COLUMNS = ["grid_ts", *EDGE_KEY, "runs_in_period", *MODEL_FEATURES.columns]

NODE_COLUMNS = [
    "node_id", "stop_id", "station", "direction", "route",
    "last_arrival_age_sec", "arrivals_in_window", "next_departure_eta_sec",
    "station_alert_count", "station_alert_age_sec", "station_alert_types",
    "in_edges", "out_edges", "observed_in_edges",
]


def node_states(graph, state: LiveState, edges: pd.DataFrame, now: int,
                window_sec: int = DEFAULT_OBS_WINDOW_SEC,
                alert_kinds: frozenset[str] = frozenset({"alert"})):
    """(table of node states, node_id -> alert events) at `now`.

    Node alerts are measured at `now`, not at a train's arrival: this is the
    station as it stands, where edge rows describe the next train.
    """
    ids_for_name = route_ids_by_name(state.route_names)
    alerts_at = alerts_by_station(state, now, alert_kinds)
    rides = edges[~edges["is_transfer"].astype(bool)]
    next_departure = rides.groupby("from_node")["next_train_eta_sec"].min()
    in_edges = edges.groupby("to_node").size()
    out_edges = edges.groupby("from_node").size()
    observed_in = rides[rides["obs_count"] > 0].groupby("to_node").size()

    rows, events_by_node = [], {}
    for node_id in graph:
        node = graph.get_node(node_id)
        stop_id = node.stop_id
        station = DIRECTION_SUFFIX.sub("", stop_id)
        direction = stop_id[-1] if DIRECTION_SUFFIX.search(stop_id) else None
        times = [t for t in state.arrivals.get(node_id, ()) if t <= now]
        route_ids = ids_for_name.get(node.vehicle, {node.vehicle})
        count, age, types, events = station_alert_summary(
            alerts_at, [(alert_route(rid), station) for rid in route_ids], now)
        events_by_node[node_id] = [{
            "event_key": alert.event_key,
            "types": sorted(alert.onset_types),
            "onset_ts": onset,
            "age_sec": now - onset,
            "headers": sorted(alert.headers),
        } for alert, onset in events]
        eta = next_departure.get(node_id)
        rows.append({
            "node_id": node_id,
            "stop_id": stop_id,
            "station": station,
            "direction": direction,
            "route": node.vehicle,
            "last_arrival_age_sec": now - max(times) if times else None,
            "arrivals_in_window": sum(1 for t in times if t >= now - window_sec),
            "next_departure_eta_sec": None if eta is None or pd.isna(eta) else int(eta),
            "station_alert_count": count,
            "station_alert_age_sec": age,
            "station_alert_types": types,
            "in_edges": int(in_edges.get(node_id, 0)),
            "out_edges": int(out_edges.get(node_id, 0)),
            "observed_in_edges": int(observed_in.get(node_id, 0)),
        })
    table = pd.DataFrame(rows, columns=NODE_COLUMNS)
    for column in ("last_arrival_age_sec", "next_departure_eta_sec", "station_alert_age_sec"):
        table[column] = table[column].astype("Float64").round().astype("Int64")
    return table, events_by_node


class Poller:
    """Owns the live state and the latest snapshot; polls on its own thread."""

    STATE_FILE = "live_state.json.gz"
    SNAPSHOT_FILE = "snapshot_latest.csv.gz"
    WINDOW_FILE = "feature_window.csv.gz"

    def __init__(self, gtfs_dir: Path, interval: float, status_dir: Path | None = None,
                 state_dir: Path | None = None, typical_headways: Path | None = None,
                 step_sec: int = live_window.STEP_SEC, window: int = live_window.WINDOW) -> None:
        self.gtfs_dir = gtfs_dir
        self.typical_headways = typical_headways
        self.typical: dict | None = None
        self.wait_features: WaitFeatures | None = None
        self.waits_error: str | None = None
        self.step_sec = step_sec
        self.window_len = window
        # grid time -> ride-edge feature rows built at it. Replaced whole, like latest.
        self.window: dict[int, pd.DataFrame] = {}
        self.window_error: str | None = None
        self.interval = interval
        self.status_dir = status_dir
        self.state_dir = state_dir
        self.checkpoint_info: dict = {"restored_from_ts": None, "resumed_diffing": False,
                                      "last_saved_ts": None, "last_save_sec": None,
                                      "last_save_error": None}
        self.status = "starting"
        self.started_at = time.time()
        self.first_poll_at: float | None = None
        self.polls = 0
        self.last_poll_at: float | None = None
        self.last_cycle_sec: float | None = None
        self.last_error: str | None = None
        self.graph = None
        self.state: LiveState | None = None
        # Replaced whole after each poll, never mutated, so readers need no lock.
        self.latest: dict | None = None
        self._schedule_day = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _load(self) -> None:
        log.info("building graph from %s", self.gtfs_dir)
        self.graph = build_subway_graph(self.gtfs_dir)
        self.wait_features = WaitFeatures(self.graph)
        if self.typical_headways is not None:
            self.typical = load_typical(self.typical_headways)
            log.info("loaded typical headways from %s", self.typical_headways)
        self.state = LiveState(StationMatcher(gtfs_dir=self.gtfs_dir),
                               load_routes(self.gtfs_dir))
        self._restore_checkpoint()
        self._restore_window()
        self._refresh_schedule()

    def _restore_checkpoint(self) -> None:
        """Load the saved state, if any. A missing or unreadable file starts clean."""
        if self.state_dir is None:
            return
        path = self.state_dir / self.STATE_FILE
        if not path.exists():
            return
        now = int(time.time())
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                saved = json.load(f)
            resumed = self.state.restore(saved["state"], now)
        except Exception as exc:
            log.warning("ignoring unreadable checkpoint %s: %s", path, exc)
            self.state = LiveState(self.state.matcher, self.state.route_names)
            return
        saved_at = saved["state"]["at"]
        self.checkpoint_info.update(restored_from_ts=saved_at, resumed_diffing=resumed)
        # Warm-up is continuous polling coverage, so it only carries over when the
        # gap was short enough to resume diffing.
        if resumed:
            self.first_poll_at = saved["service"]["first_poll_at"]
        log.info("restored checkpoint from %ds ago: %d observations, %d alert events, %s",
                 now - saved_at, len(self.state.observations), len(self.state.alerts),
                 "resuming diffs" if resumed else "fresh diff baseline (gap too long)")

    def _restore_window(self) -> None:
        """Load the saved feature window, keeping grid times still inside a
        window ending now. A missing or unreadable file starts empty."""
        if self.state_dir is None:
            return
        path = self.state_dir / self.WINDOW_FILE
        if not path.exists():
            return
        try:
            rows = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            log.warning("ignoring unreadable feature window %s: %s", path, exc)
            return
        oldest = live_window.grid_floor(int(time.time()), self.step_sec) \
            - self.step_sec * (self.window_len - 1)
        self.window = {int(t): part.reset_index(drop=True)
                       for t, part in rows.groupby("grid_ts") if t >= oldest}
        log.info("restored feature window: %d of %d grid times", len(self.window), self.window_len)

    def _update_window(self, now: int) -> None:
        """On the first poll at or past a grid time, add the snapshot *at* that
        grid time to the window and drop times that fell out of it. Never
        raises: the window is the model's input, not the service's."""
        grid = live_window.grid_floor(now, self.step_sec)
        if grid in self.window or (self.window and grid < max(self.window)):
            return
        try:
            snap = build_snapshot(self.graph, self.state, datetime.fromtimestamp(grid, timezone.utc))
            rides = snap.edges[~snap.edges["is_transfer"].astype(bool)]
            rows = rides.assign(grid_ts=grid)[WINDOW_COLUMNS].reset_index(drop=True)
            oldest = grid - self.step_sec * (self.window_len - 1)
            self.window = {t: f for t, f in self.window.items() if t >= oldest} | {grid: rows}
            self.window_error = None
            log.info("feature window: added %s, %d of %d grid times",
                     datetime.fromtimestamp(grid, TZ).strftime("%H:%M"), len(self.window),
                     self.window_len)
        except Exception as exc:
            self.window_error = f"{type(exc).__name__}: {exc}"
            log.exception("feature window update failed")
            return
        if self.state_dir is not None:
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True)
                _write_atomic(self.state_dir / self.WINDOW_FILE, gzip.compress(
                    pd.concat(self.window.values()).to_csv(index=False).encode()))
            except Exception as exc:
                log.warning("could not save feature window: %s", exc)

    def window_rows(self) -> pd.DataFrame:
        window = self.window
        if not window:
            return pd.DataFrame(columns=WINDOW_COLUMNS)
        return pd.concat([window[t] for t in sorted(window)], ignore_index=True)

    def _refresh_schedule(self) -> None:
        """Reload the supplemented schedule once per local day; MTA republishes it."""
        today = datetime.now(TZ).date()
        if self._schedule_day == today:
            return
        try:
            self.state.schedule = StaticSchedule(ensure_supplemented())
            self._schedule_day = today
            log.info("loaded supplemented schedule for %s", today)
        except Exception as exc:  # keep polling on yesterday's schedule
            log.warning("schedule refresh failed: %s", exc)

    def run(self) -> None:
        try:
            self._load()
        except Exception as exc:
            self.status, self.last_error = "failed", f"startup: {exc}"
            log.exception("startup failed")
            return
        self.status = "warming"
        with MTAClient() as client:
            while not self._stop.is_set():
                began = time.time()
                published = False
                try:
                    self._refresh_schedule()
                    poll_once(client, self.state)
                    self._publish()
                    published = True
                    self.last_error = None
                except Exception as exc:  # one bad poll must not end the service
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    log.exception("poll failed")
                self.last_cycle_sec = time.time() - began
                self._record_status(published)
                self._stop.wait(max(0.0, self.interval - self.last_cycle_sec))

    def _publish(self) -> None:
        at = datetime.now(timezone.utc)
        now = int(at.timestamp())
        self.polls += 1
        self.last_poll_at = now
        if self.first_poll_at is None:
            self.first_poll_at = now
        snap = build_snapshot(self.graph, self.state, at)
        nodes, events = node_states(self.graph, self.state, snap.edges, now)
        origin_waits = self._waits(now)
        self._update_window(now)
        self.latest = {"snapshot": snap, "nodes": nodes, "events": events,
                       "origin_waits": origin_waits,
                       "at": now, "polled_feeds": dict(self.state.feed_polled_at),
                       "observations": len(self.state.observations),
                       # Features count incidents only; planned work is tracked
                       # but ignored (build_snapshot's alert_kinds).
                       "alerts": dict(Counter(a.kind for a in self.state.alerts.values()))}
        if self.warm:
            self.status = "ready"
        self._save_checkpoint(snap, now)
        edges = snap.edges
        log.info("poll %d: %d observations, alert events %s, %d edges with a train due, "
                 "cycle %.1fs", self.polls, self.latest["observations"],
                 self.latest["alerts"], int(edges["next_train_eta_sec"].notna().sum()),
                 time.time() - at.timestamp())

    def _waits(self, now: int) -> pd.DataFrame | None:
        """The origin-wait table for this poll. A failure here is logged and
        reported in /health but never stops the snapshot from publishing."""
        try:
            origin = with_origin_wait(self.wait_features.origin_waits(self.state, now, self.typical))
        except Exception as exc:
            self.waits_error = f"{type(exc).__name__}: {exc}"
            log.exception("wait features failed")
            return None
        self.waits_error = None
        return origin

    def _save_checkpoint(self, snap, now: int) -> None:
        """Overwrite the saved state and latest snapshot. Never raises."""
        if self.state_dir is None:
            return
        began = time.time()
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            payload = {"service": {"first_poll_at": self.first_poll_at},
                       "state": self.state.checkpoint(now)}
            _write_atomic(self.state_dir / self.STATE_FILE,
                          gzip.compress(json.dumps(payload, separators=(",", ":")).encode()))
            _write_atomic(self.state_dir / self.SNAPSHOT_FILE,
                          gzip.compress(snap.edges.to_csv(index=False).encode()))
        except Exception as exc:
            self.checkpoint_info["last_save_error"] = f"{type(exc).__name__}: {exc}"
            log.warning("could not save checkpoint to %s: %s", self.state_dir, exc)
            return
        self.checkpoint_info.update(last_saved_ts=now, last_save_sec=round(time.time() - began, 2),
                                    last_save_error=None)

    def _record_status(self, published: bool) -> None:
        """Append this cycle's health to the day's status file. Never raises:
        a full disk must not stop polling."""
        if self.status_dir is None:
            return
        at = datetime.now(TZ)
        record = {"recorded_at": at.isoformat(timespec="seconds"), "published": published,
                  **self.health()}
        path = self.status_dir / f"snapshot_status_{at.date().isoformat()}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            log.warning("could not record status to %s: %s", path, exc)

    @property
    def warm(self) -> bool:
        return (self.first_poll_at is not None and self.last_poll_at is not None
                and self.last_poll_at - self.first_poll_at >= DEFAULT_OBS_WINDOW_SEC)

    def health(self) -> dict:
        latest = self.latest or {}
        now = time.time()
        feeds = latest.get("polled_feeds", {})
        return {
            "status": self.status,
            "warm": self.warm,
            "uptime_sec": round(now - self.started_at),
            "polls": self.polls,
            "poll_interval_sec": self.interval,
            "last_poll_age_sec": None if self.last_poll_at is None else round(now - self.last_poll_at),
            "last_cycle_sec": None if self.last_cycle_sec is None else round(self.last_cycle_sec, 1),
            # first_poll_at can be restored from a checkpoint before this process polls.
            "polling_covered_sec": (None if self.first_poll_at is None or self.last_poll_at is None
                                    else self.last_poll_at - self.first_poll_at),
            "warm_after_sec": DEFAULT_OBS_WINDOW_SEC,
            "snapshot_ts": latest.get("at"),
            "service_period": latest["snapshot"].service_period if latest else None,
            "service_state": latest["snapshot"].service_state if latest else None,
            "observations": latest.get("observations"),
            "alert_events_by_kind": latest.get("alerts"),
            "feed_age_sec": {k: round(now - v) for k, v in sorted(feeds.items())},
            "last_error": self.last_error,
            "typical_headways": str(self.typical_headways) if self.typical_headways else None,
            "waits_error": self.waits_error,
            "feature_window": {
                "grid_ts": sorted(self.window), "steps": len(self.window),
                "window": self.window_len, "step_sec": self.step_sec,
                "newest_age_sec": round(now - max(self.window)) if self.window else None,
                "error": self.window_error,
            },
            "checkpoint": dict(self.checkpoint_info),
        }


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a crash mid-write never leaves a
    truncated file where the last good one was."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _filter(frame: pd.DataFrame, query: dict, allowed: tuple[str, ...]) -> pd.DataFrame:
    for column in allowed:
        if column in query:
            frame = frame[frame[column].astype(str) == query[column]]
    if query.get("due") in ("1", "true") and "next_train_eta_sec" in frame:
        frame = frame[frame["next_train_eta_sec"].notna()]
    return frame


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def node_waits(latest: dict, node_id: str) -> dict:
    """A node's origin-wait row."""
    origin = latest.get("origin_waits")
    if origin is None:
        return {"wait": None}
    match = origin[origin["node"] == node_id]
    return {"wait": _records(match)[0] if len(match) else None}


def make_handler(poller: Poller):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Read-only data; lets viewer.html (opened from disk or another port) fetch it.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code: int = 200) -> None:
            self._send(code, json.dumps(payload, allow_nan=False).encode(), "application/json")

        def _table(self, frame: pd.DataFrame, query: dict, **extra) -> None:
            if query.get("format") == "csv":
                self._send(200, frame.to_csv(index=False).encode(), "text/csv")
            else:
                self._json({**extra, "count": len(frame), "rows": _records(frame)})

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            path = url.path.rstrip("/") or "/"
            if path in ("/", "/health"):
                return self._json(poller.health())
            latest = poller.latest
            if latest is None:
                return self._json({"error": "no snapshot yet", **poller.health()}, 503)
            edges = latest["snapshot"].edges
            meta = {"snapshot_ts": latest["at"], "warm": poller.warm}
            if path == "/features":
                frame = _filter(edges, query, ("route", "from_node", "to_node"))
                return self._table(frame[FEATURE_COLUMNS], query, **meta,
                                   features=MODEL_FEATURES.to_dict())
            if path == "/snapshot":
                return self._table(_filter(edges, query, ("route", "from_node", "to_node")),
                                   query, **meta)
            if path == "/nodes":
                return self._table(_filter(latest["nodes"], query, ("route",)), query, **meta)
            if path.startswith("/nodes/"):
                node_id = unquote(path[len("/nodes/"):])
                nodes = latest["nodes"]
                match = nodes[nodes["node_id"] == node_id]
                if match.empty:
                    return self._json({"error": f"unknown node {node_id!r}"}, 404)
                return self._json({
                    **meta,
                    "node": _records(match)[0],
                    "alert_events": latest["events"].get(node_id, []),
                    "in_edges": _records(edges.loc[edges["to_node"] == node_id, FEATURE_COLUMNS]),
                    "out_edges": _records(edges.loc[edges["from_node"] == node_id, FEATURE_COLUMNS]),
                    **node_waits(latest, node_id),
                })
            if path == "/features/window":
                frame = _filter(poller.window_rows(), query, ("from_node", "to_node"))
                return self._table(frame, query, **meta, step_sec=poller.step_sec,
                                   window=poller.window_len, grid_ts=sorted(poller.window),
                                   features=MODEL_FEATURES.to_dict())
            if path == "/waits":
                frame = latest["origin_waits"]
                if frame is None:
                    return self._json({"error": "wait features unavailable",
                                       "waits_error": poller.waits_error}, 503)
                return self._table(_filter(frame, query, ("route", "node")), query, **meta,
                                   features=ORIGIN_FEATURES.to_dict(),
                                   typical_headways=poller.health()["typical_headways"])
            self._json({"error": "not found",
                        "endpoints": ["/health", "/features", "/features/window", "/snapshot",
                                      "/nodes", "/nodes/<node_id>", "/waits"]}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between poll starts; the feeds refresh about every 30s")
    ap.add_argument("--gtfs-dir", type=Path, default=DEFAULT_GTFS_DIR)
    ap.add_argument("--status-dir", default="logs/status",
                    help='directory for the per-poll status JSONL files; "" disables them')
    ap.add_argument("--state-dir", default="data/live",
                    help='directory for the latest state checkpoint and snapshot; "" disables them')
    ap.add_argument("--typical-headways", type=Path, default=None,
                    help="typical_headway.csv from ml_model.train_waits: the /waits fallback "
                         "where a platform has no ETA")
    ap.add_argument("--window-step-sec", type=int, default=live_window.STEP_SEC,
                    help="feature window grid step; must match the agent's graph model")
    ap.add_argument("--window", type=int, default=live_window.WINDOW,
                    help="feature window length in steps; must match the agent's graph model")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    poller = Poller(args.gtfs_dir, args.interval,
                    status_dir=Path(args.status_dir) if args.status_dir else None,
                    state_dir=Path(args.state_dir) if args.state_dir else None,
                    typical_headways=args.typical_headways,
                    step_sec=args.window_step_sec, window=args.window)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(poller))
    threading.Thread(target=poller.run, name="poller", daemon=True).start()

    def shutdown(signum, frame):
        log.info("signal %s, shutting down", signum)
        poller.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info("serving on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
