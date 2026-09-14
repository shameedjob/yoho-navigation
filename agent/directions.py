"""Route directions built in code from get_path-style steps: the model gets a
summary, the rider gets directions and legs. Pure Python, so the web process,
the scheduler and tests can use it without loading torch (agent/tools.py)."""

from __future__ import annotations

import math


def _rides(steps: list[dict]) -> list[list[dict]]:
    """The route's rides: runs of consecutive steps on the same mode and route,
    at least two stops long. A run of one stop is a stop walked through between
    rides (e.g. a bus stop on the corner before the subway), not a ride."""
    runs: list[list[dict]] = []
    for step in steps:
        if runs and (runs[-1][-1]["mode"], runs[-1][-1]["route"]) == (step["mode"], step["route"]):
            runs[-1].append(step)
        else:
            runs.append([step])
    return [run for run in runs if len(run) >= 2]


HOME = "home"  # a trip end at the user's home: never sent with coordinates


def endpoint(value) -> dict | None:
    """A trip end for the browser: {"kind": "place", "lat", "lon"}, {"kind": "home"}
    (no coordinates: home never leaves the server), or None when unknown."""
    if value == HOME:
        return {"kind": "home"}
    if value is None:
        return None
    lat, lon = value
    return {"kind": "place", "lat": float(lat), "lon": float(lon)}


def route_payload(steps: list[dict], total_time_sec: float, walk_in_sec: float | None = None,
                  walk_out_sec: float | None = None, start=None, end=None) -> dict:
    """The route as the browser draws it (web/static/route_map.js): the summary plus
    "total_time_sec", "walk_in_sec", "walk_out_sec", "start"/"end" (see endpoint),
    "legs" and "directions". `start` / `end`: (lat, lon), HOME, or None."""
    coords = lambda e: tuple(e) if e not in (None, HOME) else None
    return {**route_summary(steps, total_time_sec), "total_time_sec": total_time_sec,
            "walk_in_sec": walk_in_sec, "walk_out_sec": walk_out_sec,
            "start": endpoint(start), "end": endpoint(end),
            "legs": route_legs(steps, walk_in_sec, walk_out_sec, coords(start), coords(end)),
            "directions": format_directions(steps, total_time_sec, walk_in_sec, walk_out_sec)}


def _line_label(mode: str, route: str) -> str:
    return f"the {route} train" if mode == "subway" else f"the {route} {mode}"


def route_summary(steps: list[dict], total_time_sec: float) -> dict:
    """What the model needs about a route: how long, which lines, how many
    transfers. No station names -- the rider gets those from format_directions."""
    rides = _rides(steps)
    return {"minutes": math.ceil(total_time_sec / 60),
            "lines": [_line_label(r[0]["mode"], r[0]["route"]) for r in rides],
            "transfers": max(0, len(rides) - 1)}


def route_legs(steps: list[dict], walk_in_sec: float | None = None, walk_out_sec: float | None = None,
               start: tuple[float, float] | None = None, end: tuple[float, float] | None = None) -> list[dict]:
    """The route as legs for the browser to draw: walks and rides in order, each
    ride with its line and every stop (name, lat, lon) along it. With `start` /
    `end` coordinates, the first walk gets a `from` and the last a `to` point, so
    the walks to and from the network can be drawn too; without them (e.g. the
    user's private home) those ends stay "start" / "destination"."""
    rides = _rides(steps)
    point = lambda s: {"stop_id": s["stop_id"], "stop_name": s["stop_name"], "lat": s["lat"], "lon": s["lon"]}
    origin = {"stop_name": "Start", "lat": start[0], "lon": start[1]} if start else "start"
    destination = {"stop_name": "Destination", "lat": end[0], "lon": end[1]} if end else "destination"
    if not rides:
        return [{"type": "walk", "from": origin, "to": destination, "sec": walk_in_sec}]
    legs: list[dict] = [{"type": "walk", "from": origin, "to": point(rides[0][0]),
                         "sec": walk_in_sec if rides[0][0] is steps[0] else None}]
    for n, ride in enumerate(rides):
        if n:  # always: same-named stops (Fulton St on the C and the 4) are still different platforms
            legs.append({"type": "walk", "from": point(rides[n - 1][-1]), "to": point(ride[0]), "sec": None})
        legs.append({"type": "ride", "mode": ride[0]["mode"], "route": ride[0]["route"],
                     "stops": len(ride) - 1, "path": [point(s) for s in ride]})
    legs.append({"type": "walk", "from": point(rides[-1][-1]), "to": destination,
                 "sec": walk_out_sec if rides[-1][-1] is steps[-1] else None})
    return legs


def format_directions(steps: list[dict], total_time_sec: float,
                      walk_in_sec: float | None = None, walk_out_sec: float | None = None) -> str:
    """Step-by-step directions for the rider, built in code so the model can't
    garble them: board and alight stations of each ride with its stop count,
    the walks between, and the door-to-door time."""
    rides = _rides(steps)
    minutes = math.ceil(total_time_sec / 60)
    if not rides:
        return f"Walk to your destination (under {minutes} min)."

    def walk(sec: float | None) -> str:
        return f" ({max(1, round(sec / 60))} min)" if sec else ""

    lines = []
    first = rides[0]
    # The walk-in time is to steps[0]; it only fits if the first ride boards there.
    lines.append(f"Walk to {first[0]['stop_name']}{walk(walk_in_sec) if first[0] is steps[0] else ''}.")
    for n, ride in enumerate(rides):
        if n:
            lines.append(f"Transfer at {rides[n - 1][-1]['stop_name']}"
                         + ("." if ride[0]["stop_name"] == rides[n - 1][-1]["stop_name"]
                            else f": walk to {ride[0]['stop_name']}."))
        stops = len(ride) - 1
        lines.append(f"Take {_line_label(ride[0]['mode'], ride[0]['route'])} {stops} stop{'s' if stops != 1 else ''}, "
                     f"{ride[0]['stop_name']} to {ride[-1]['stop_name']}.")
    lines.append(f"Walk to your destination{walk(walk_out_sec) if rides[-1][-1] is steps[-1] else ''}.")
    lines.append(f"Under {minutes} min door to door.")
    return "\n".join(lines)
