"""Schedule routing on the combined subway+bus graph, geocoding, and live
first-train waits from the snapshot service -- everything the router needs
except the graph model. No torch: the alert scheduler routes with this alone,
while the chat agent (agent/tools.py) adds the Graph WaveNet's live pricing.
"""

from __future__ import annotations

import csv
import io
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from geocoding import geocode
from graph import Graph
from graph.combined_loader import build_combined_graph
from graph.service_states import state_at

DATA_DIR = Path("data")
GEOCODER_USER_AGENT = "yoho-navigation-agent/0.1"
SNAPSHOT_SERVICE_URL = os.environ.get("SNAPSHOT_SERVICE_URL", "http://127.0.0.1:8791")
SNAPSHOT_TIMEOUT_SEC = 10
# Stops a trip may start or end at: every node, subway or bus, within
# ACCESS_RADIUS_M of the coordinates, or within START_RADIUS_EXTRA_M beyond the
# nearest node when even that is farther. Each is charged its walk.
ACCESS_RADIUS_M = 400
START_RADIUS_EXTRA_M = 150
# Straight-line walking speed, as scripts/subway_bus_transfers.py prices the
# walks between subway stations and bus stops.
WALK_SPEED_MPS = 1.4
TZ = ZoneInfo("America/New_York")
# Bus waits with waits on: this fraction of the scheduled headway, the 90th
# percentile of a uniformly random wait -- the quantile the wait models are
# trained at (ml_model/train_waits.py --quantile). Half the headway, the
# schedule's average wait, would make buses look cheap next to subway
# transfers priced at their 90th percentile.
BUS_WAIT_HEADWAY_FRACTION = 0.9


# Building the combined graph takes several seconds (parses the full GTFS
# feeds), so it's built once lazily and cached, not per tool call.
_graph: Graph | None = None
# (bus, subway) / (subway, bus) -> transfer cost without its wait, and
# state -> headway of the route boarded, from build_combined_graph, so
# cross-system transfers can be re-priced at the model's quantile.
_access_sec: dict[tuple[str, str], float] = {}
_boarding_headway_sec: dict[tuple[str, str], dict[str, float]] = {}
# node -> state -> scheduled headway, subway and bus, for first-train waits at
# bus stops (which have no live data).
_headways: dict[str, dict[str, float]] = {}
_stop_names: dict[str, str] | None = None

# Startup warm-up (agent.tools.warm_up) builds these on a background thread while
# requests may already be asking for them: the locks make a request wait for that
# build instead of starting a second one.
_graph_lock = threading.Lock()
_stop_names_lock = threading.Lock()


def _get_graph() -> Graph:
    global _graph
    with _graph_lock:
        if _graph is None:
            _graph = build_combined_graph(DATA_DIR, access_sec=_access_sec,
                                          boarding_headway_sec=_boarding_headway_sec,
                                          headways=_headways)
    return _graph


def _get_stop_names() -> dict[str, str]:
    with _stop_names_lock:
        return _load_stop_names()


def _load_stop_names() -> dict[str, str]:
    global _stop_names
    if _stop_names is None:
        names: dict[str, str] = {}
        feed_dirs = [DATA_DIR / "gtfs_subway"] + [
            DATA_DIR / feed for feed in ("gtfs_b", "gtfs_bx", "gtfs_m", "gtfs_q", "gtfs_si")
        ]
        for feed_dir in feed_dirs:
            with open(feed_dir / "stops.txt", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    names.setdefault(row["stop_id"], row["stop_name"])
        _stop_names = names
    return _stop_names


def _access_walks(graph: Graph, location: tuple[float, float]) -> dict[str, float]:
    """Node -> walking seconds from `location`, for every node a trip may start
    or end at there, whatever its mode.

    Snapping to the single nearest node used to decide the mode: a bus stop on
    the corner beat the subway entrance a few metres farther, and the route
    had to begin on that bus. Charging each candidate its walk lets the
    router weigh them instead.
    """
    lat, lon = location
    nearest = graph.nodes_near(lat, lon, limit=1)
    if not nearest:
        raise ValueError(f"no graph nodes found near {location!r}")
    radius = max(ACCESS_RADIUS_M, nearest[0][1] + START_RADIUS_EXTRA_M)
    return {node.id: distance / WALK_SPEED_MPS
            for node, distance in graph.nodes_near(lat, lon, radius_m=radius)}


def _state_at(ts: float) -> str:
    """The service state (graph/service_states.py) a trip leaving at Unix time
    `ts` is priced in."""
    return state_at(datetime.fromtimestamp(ts, TZ))


def _first_wait(node_id: str, state: str, waits: "Waits | None") -> float:
    """Wait for the first vehicle at `node_id`: /waits for a subway platform,
    BUS_WAIT_HEADWAY_FRACTION of the scheduled headway at a bus stop, so that
    neither mode looks free next to the other. 0 without waits (schedule
    routing has no first-train wait)."""
    if waits is None:
        return 0.0
    if node_id in waits.origin:
        return waits.origin[node_id]
    headway = _headways.get(node_id, {}).get(state)
    return BUS_WAIT_HEADWAY_FRACTION * headway if headway is not None else 0.0


class Route:
    def __init__(self, node_ids: list[str], total_sec: float, walk_in_sec: float,
                 first_wait_sec: float, walk_out_sec: float) -> None:
        self.node_ids = node_ids
        self.total_sec = total_sec  # walks and first wait included
        self.walk_in_sec = walk_in_sec
        self.first_wait_sec = first_wait_sec
        self.walk_out_sec = walk_out_sec


MODES = ("subway", "bus")
_MODE_WORDS = {"subway": "subway", "subways": "subway", "train": "subway", "trains": "subway", "metro": "subway",
               "bus": "bus", "buses": "bus", "busses": "bus"}
_LINE_NOISE = re.compile(r"\b(the|train|trains|line|lines|bus|buses|route)\b", re.I)


@dataclass(frozen=True)
class Avoid:
    """Modes and lines a trip must not use."""
    modes: frozenset = field(default_factory=frozenset)
    lines: frozenset = field(default_factory=frozenset)

    def __bool__(self) -> bool:
        return bool(self.modes or self.lines)

    def allows(self, node) -> bool:
        if node.mode in self.modes:
            return False
        vehicle = node.vehicle.upper()
        return not (vehicle in self.lines or
                    (node.mode == "subway" and vehicle.endswith("X") and vehicle[:-1] in self.lines))

    def describe(self) -> dict:
        return {"modes": sorted(self.modes), "lines": sorted(self.lines)}


def parse_avoid(modes: list[str] | None = None, lines: list[str] | None = None) -> Avoid:
    """Avoid from what the model passes: modes like "bus"/"subways", lines like
    "L", "the 6 train", "B38". Raises ValueError with a message it can act on."""
    mode_set = set()
    for m in modes or []:
        key = _MODE_WORDS.get(str(m).strip().lower())
        if key is None:
            raise ValueError(f"avoid_modes must be 'subway' or 'bus', got {m!r}")
        mode_set.add(key)
    if mode_set == set(MODES):
        raise ValueError("can't avoid both subway and bus: there'd be nothing left to ride")
    line_set = set()
    for raw in lines or []:
        line = _LINE_NOISE.sub(" ", str(raw)).strip().upper().replace(" ", "")
        if not line:
            raise ValueError(f"couldn't read a line name from {raw!r}")
        line_set.add(line)
    unknown = sorted(line_set - known_lines())
    if unknown:
        raise ValueError(f"unknown line(s) {', '.join(unknown)}; use names like 'L', '6', 'B38', 'M15'")
    return Avoid(frozenset(mode_set), frozenset(line_set))


_known_lines: set[str] | None = None


def known_lines() -> set[str]:
    """Every line/route name in the graph, upper-case (subway and bus)."""
    global _known_lines
    if _known_lines is None:
        graph = _get_graph()
        _known_lines = {graph.get_node(n).vehicle.upper() for n in graph}
    return _known_lines


def _route(graph: Graph, start: tuple[float, float], end: tuple[float, float], state: str,
           waits: "Waits | None" = None, avoid: Avoid | None = None) -> Route:
    """Cheapest trip from `start` to `end`, walks included.

    It may board at any stop near `start` (_access_walks), paying the walk
    there plus the first wait (_first_wait), and leave the network at any stop
    near `end`, paying the walk from it. Ending at the stop reached first
    rather than a particular one also matters once transfers carry real waits:
    otherwise a route could end with a transfer onto one particular platform
    -- even a line that isn't running -- after already arriving. The whole
    trip is priced in `state`, its departure state. `avoid` removes modes and
    lines entirely, including as the stop boarded or left at."""
    walk_in, walk_out = _access_walks(graph, start), _access_walks(graph, end)
    if avoid:
        walk_in = {n: w for n, w in walk_in.items() if avoid.allows(graph.get_node(n))}
        walk_out = {n: w for n, w in walk_out.items() if avoid.allows(graph.get_node(n))}
        if not walk_in or not walk_out:
            where = "the start" if not walk_in else "the destination"
            raise ValueError(f"no stop near {where} is left after avoiding {avoid.describe()}")
    start_costs = {n: w + _first_wait(n, state, waits) for n, w in walk_in.items()}
    result = graph.shortest_path(min(start_costs, key=start_costs.get),
                                 min(walk_out, key=walk_out.get), service_period=state,
                                 start_costs=start_costs, end_costs=walk_out,
                                 ignore_modes=set(avoid.modes) if avoid else None,
                                 ignore_routes=set(avoid.lines) if avoid else None)
    if result is None:
        suffix = f" while avoiding {avoid.describe()}" if avoid else ""
        raise ValueError(f"no path found between {start!r} and {end!r}{suffix}")
    node_ids, total = result
    first = node_ids[0]
    return Route(node_ids, total, walk_in[first], _first_wait(first, state, waits),
                 walk_out[node_ids[-1]])


def _describe_path(graph: Graph, node_ids: list[str]) -> list[dict]:
    names = _get_stop_names()
    steps = []
    for node_id in node_ids:
        node = graph.get_node(node_id)
        steps.append(
            {
                "stop_id": node.stop_id,
                "stop_name": names.get(node.stop_id, node.stop_id),
                "mode": node.mode,
                "route": node.vehicle,
                "lat": node.lat,
                "lon": node.lon,
            }
        )
    return steps


class Waits:
    """First-train waits from the service's latest poll (/waits)."""

    def __init__(self, origin: dict[str, float], ts: int) -> None:
        # node -> wait for its next train: raw ETA, else 0.9 x typical headway,
        # else no service (snapshot.waits.with_origin_wait)
        self.origin = origin
        self.ts = ts


def _fetch_waits() -> Waits | None:
    """Waits from /waits, or None when the service has none yet. Never raises:
    routing without waits is the fallback, not an error."""
    try:
        response = requests.get(f"{SNAPSHOT_SERVICE_URL}/waits", params={"format": "csv"},
                                timeout=SNAPSHOT_TIMEOUT_SEC)
        response.raise_for_status()
        origin = pd.read_csv(io.StringIO(response.text))
    except (requests.RequestException, ValueError):
        return None
    if "wait_sec" not in origin or not len(origin):
        return None
    return Waits(origin=dict(zip(origin["node"], origin["wait_sec"].astype(float))),
                 ts=int(origin["snapshot_ts"].iloc[0]))


def geocode_address(address: str) -> tuple[float, float]:
    """(lat, lon) of the best NYC match for an address or place name (geocoding/search.py);
    raises ValueError if nothing matches."""
    return geocode(address)


def schedule_route(start: tuple[float, float], end: tuple[float, float], departure_time: int | None = None,
                   waits: Waits | None = None, avoid: Avoid | None = None) -> dict:
    """Fastest route on the schedule for the departure's service state. With
    `waits`, the first-train wait is priced in and reported (first_wait_sec).

    Returns {"steps", "total_time_sec", "walk_in_sec", "walk_out_sec", "service_state"}
    (plus "first_wait_sec" with waits)."""
    graph = _get_graph()
    state = _state_at(time.time() if departure_time is None else departure_time)
    route = _route(graph, start, end, state, waits, avoid)
    out = {"steps": _describe_path(graph, route.node_ids), "total_time_sec": route.total_sec,
           "walk_in_sec": round(route.walk_in_sec), "walk_out_sec": round(route.walk_out_sec),
           "service_state": state}
    if waits is not None:
        out["first_wait_sec"] = round(route.first_wait_sec)
    return out
