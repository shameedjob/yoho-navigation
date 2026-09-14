"""Live alerts and delays per subway station, from the snapshot service, for
the agent's station_status tool and the map.

A station here is a GTFS parent station ("626", 86 St). The service reports per
platform node ("626N::4"): an incident alert count/types, and the next train's
delay versus the schedule (/waits next_train_delay_sec). Platforms are rolled up
to their station: the worst delay, every alert. No torch and no transit graph --
names and coordinates come from GTFS stops.txt.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache

import requests

from agent.routing import DATA_DIR, SNAPSHOT_SERVICE_URL, SNAPSHOT_TIMEOUT_SEC

# A next train this late or later counts as delayed; below it is normal jitter.
DELAYED_SEC = 120
MAX_STATIONS = 25


@lru_cache(maxsize=1)
def stations() -> dict[str, dict]:
    """Parent station id -> {"name", "lat", "lon"} from GTFS stops.txt."""
    out = {}
    with open(DATA_DIR / "gtfs_subway" / "stops.txt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("location_type") == "1":
                out[row["stop_id"]] = {"name": row["stop_name"], "lat": float(row["stop_lat"]),
                                       "lon": float(row["stop_lon"])}
    return out


def station_of(stop_id: str) -> str:
    """Platform stop id ("626N") -> parent station id ("626")."""
    return stop_id[:-1] if stop_id[-1:] in ("N", "S") else stop_id


# GTFS station names abbreviate ("14 St-Union Sq"); people don't ("Union Square").
_ABBREVIATIONS = {"street": "st", "streets": "sts", "avenue": "av", "ave": "av", "square": "sq", "place": "pl",
                  "parkway": "pkwy", "boulevard": "blvd", "road": "rd", "center": "ctr", "centre": "ctr",
                  "heights": "hts", "junction": "jct", "terminal": "term", "plaza": "plz", "park": "pk",
                  "saint": "st", "fort": "ft", "mount": "mt", "north": "n", "south": "s", "east": "e", "west": "w"}


def _norm(text: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return " ".join(_ABBREVIATIONS.get(w, w) for w in words)


def find_stations(name: str) -> list[str]:
    """Station ids whose name matches `name` (case/punctuation-insensitive), best
    (shortest name) first. Every word must be a whole word of the station name,
    or a prefix of one when 4+ letters ("metro" -> MetroTech) -- so "86" never
    matches "186 St"."""
    words = _norm(name).split()
    if not words:
        return []

    def matches(station_words: list[str]) -> bool:
        return all(w in station_words or (len(w) >= 4 and any(sw.startswith(w) for sw in station_words))
                   for w in words)

    hits = [(len(s["name"]), sid) for sid, s in stations().items() if matches(_norm(s["name"]).split())]
    return [sid for _, sid in sorted(hits)]


def _line_matches(route: str, line: str) -> bool:
    """"6" matches the 6 and the 6 express (6X); "S" only shuttles named S."""
    route, line = route.upper(), line.upper()
    return route == line or (route.endswith("X") and route[:-1] == line)


def fetch_live() -> tuple[list[dict], dict[str, dict]]:
    """(/nodes rows, node -> /waits row) from the snapshot service; raises if it's down."""
    nodes = requests.get(f"{SNAPSHOT_SERVICE_URL}/nodes", timeout=SNAPSHOT_TIMEOUT_SEC)
    nodes.raise_for_status()
    waits = requests.get(f"{SNAPSHOT_SERVICE_URL}/waits", timeout=SNAPSHOT_TIMEOUT_SEC)
    waits.raise_for_status()
    return nodes.json()["rows"], {w["node"]: w for w in waits.json()["rows"]}


def fetch_alert_headers(node_id: str) -> list[str]:
    """The alert texts at one platform node (/nodes/<node_id>), newest wording last."""
    resp = requests.get(f"{SNAPSHOT_SERVICE_URL}/nodes/{node_id}", timeout=SNAPSHOT_TIMEOUT_SEC)
    resp.raise_for_status()
    headers: list[str] = []
    for event in resp.json().get("alert_events", []):
        if event.get("headers"):
            headers.append(event["headers"][-1])
    return headers


def station_status(station_ids: list[str] | None = None, line: str | None = None, only_problems: bool = False,
                   *, live=fetch_live, alert_headers=fetch_alert_headers) -> list[dict]:
    """Status of the given stations and/or a line's stations:
    [{"station_id", "name", "lat", "lon", "lines", "alert_types", "alerts" (texts),
      "delay_sec" (worst next-train delay, None without live data), "delayed_lines", "status"}]
    status is "alert", "delayed" or "ok". only_problems keeps alert/delayed stations."""
    nodes, waits = live()
    wanted = set(station_ids or [])
    by_station: dict[str, dict] = {}
    for node in nodes:
        sid = station_of(node["stop_id"])
        if sid not in stations():
            continue
        if wanted and sid not in wanted:
            continue
        if line and not _line_matches(node["route"], line):
            continue
        s = by_station.setdefault(sid, {"station_id": sid, **stations()[sid], "lines": set(), "alert_types": set(),
                                        "alert_nodes": [], "delay_sec": None, "delayed_lines": set()})
        s["lines"].add(node["route"])
        if node.get("station_alert_count"):
            s["alert_types"].update(t for t in (node.get("station_alert_types") or "").split("|") if t)
            s["alert_nodes"].append(node["node_id"])
        delay = (waits.get(node["node_id"]) or {}).get("next_train_delay_sec")
        if delay is not None:
            s["delay_sec"] = max(s["delay_sec"] or 0, round(delay))
            if delay >= DELAYED_SEC:
                s["delayed_lines"].add(node["route"])

    out = []
    for s in by_station.values():
        status = "alert" if s["alert_nodes"] else "delayed" if (s["delay_sec"] or 0) >= DELAYED_SEC else "ok"
        if only_problems and status == "ok":
            continue
        out.append({**s, "status": status, "alerts": [], "lines": sorted(s["lines"]),
                    "alert_types": sorted(s["alert_types"]), "delayed_lines": sorted(s["delayed_lines"])})
    rank = {"alert": 0, "delayed": 1, "ok": 2}
    out.sort(key=lambda s: (rank[s["status"]], -(s["delay_sec"] or 0), s["name"]))
    out = out[:MAX_STATIONS]
    for s in out:  # alert texts only for the stations returned: one request per alerted platform
        texts: list[str] = []
        for node_id in s.pop("alert_nodes"):
            try:
                texts += [t for t in alert_headers(node_id) if t not in texts]
            except requests.RequestException:
                pass
        s["alerts"] = texts
    return out
