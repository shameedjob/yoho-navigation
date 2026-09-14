"""Build the per-request navigation agent for one signed-in user.

User-specific tools are closures over the caller's uid, created per request.
The model never passes a user id, so it can't be talked into reading someone
else's data, and the home location is decrypted inside the tool call that
needs it and never handed back to the model: the tool returns the route, not
the coordinates. (The route's first stop does reveal the station nearest home;
that's inherent to giving directions from there.)

agent/tools.py loads torch and the transit graph at import time, so it's
imported lazily here -- the web process starts fast and tests don't need it.
"""
from __future__ import annotations

import functools
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands.models.openai import OpenAIModel
from accounts.home import load_home
from agent.directions import HOME, endpoint as _endpoint, route_payload as _route_payload
from accounts.trips import (LOCAL_TIME_FORMATS, local_label, no_home as _no_home, parse_event_time as _parse_start,
                            parse_local_time, plan_leave_by, transit_geocode, transit_router)
from storage import FieldCipher, UserStore

SYSTEM_PROMPT = """You are Yoho, a concise assistant for getting around New York City on public transit.
Route tools return a summary (minutes, lines, transfers, departure and arrival times). The route is drawn on the user's map
automatically: when a result has "highlighted_on_map": true, tell the user you've highlighted the
route on their map. Don't list stations or stops yourself; if they ask which stations or where to
transfer, call route_stations.
Whenever you give a trip, say when it departs and when it arrives (the tool's local times: departs/arrives,
or leave_by/arrive_by), not only how many minutes it takes.
Answer briefly unless the user asks for detail. Use tools to geocode and plan; don't invent information.
For trips starting or ending at the user's home, use route_from_home / route_to_home. The home
location is private: you can't see it, don't ask the user for it, and don't guess it.
If a tool says no home is set, tell the user they can set it on their Profile page.
For trips from where the user is right now ("from here", "near me", "take me to X" with no start),
use route_from_here; route_to_home with no start also leaves from here. Their location is private too.
If a tool says their location isn't available, ask them to allow location access or give a starting address.
Use get_current_time when you need to know today's schedule (weekday, Saturday, or Sunday).
Never ask the user for coordinates or to look something up on a map. For a place name you're not sure of
(a school, store, venue, "the Starbucks near Union Square"), call search_places. One clear match: use its lat/lon.
Several different places: list them briefly (name and address) and ask which one; they're also pinned on the map.
Nothing found: ask for a nearby street address or cross streets.
If the user wants to skip subways, buses or particular lines ("no buses", "avoid the L", "not the 4 or 5"),
pass avoid_modes / avoid_lines to the routing tool; say what was avoided.
For alerts or delays at stations or on a line ("any delays on the 4?", "is 86 St OK?", "delays on my route"),
call station_status; results are marked on the map. For trips with several stops or a detour, call
plan_multi_stop_trip with the stops in order ("home" is allowed).
Times you pass to tools are New York local time in words, e.g. "14:00", "2pm", "Monday 14:00", "tomorrow 9:30".
Never ask the user for a date you can express that way. "Arrive by"/"get there by"/"when should I leave" -> leave_by
(with start when the user names one); "leaving at"/"at <time>" -> get_path with depart_at.
Only help with transit, routing, and the user's schedule; politely decline anything else."""

MAX_STORED_MESSAGES = 20
# leave_by starts from the device's location only for trips this soon; later ones
# fall back to where the user will be then (calendar event or home).
CURRENT_LOCATION_WINDOW_SEC = 3 * 3600
HERE_WORDS = ("here", "my location", "current location", "where i am")


def make_user_tools(uid: str, store: UserStore, cipher: FieldCipher, log: RouteLog | None = None,
                    location: tuple[float, float] | None = None) -> list:
    """`location`: the browser's current (lat, lon) for this request, or None when
    the user hasn't shared it. Like home, it's used inside tools and never returned
    to the model (the browser gets it back as the route's start; it's its own)."""
    @tool
    def route_from_here(destination: tuple[float, float], depart_at: str | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Plan a transit route from where the user is right now (their device's location) to a destination.

        Args:
            destination: (latitude, longitude) of where the user is going; use get_position first for an address.
            depart_at: when they leave, New York local time ("Monday 14:00", "2pm"); defaults to now.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            The route summary with "departs"/"arrives" local times, as get_path returns it, or {"error": "no_location"} when the
            user's location isn't available. The location itself is not included.
        """
        if location is None:
            return _no_location()
        departure = _departure(depart_at)
        if isinstance(departure, dict):
            return departure
        from agent.tools import get_path
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        leave = departure or int(time.time())
        return _timed(_routed(lambda: _summarized(log, get_path(start=location, end=tuple(destination),
                                                                departure_time=leave, avoid=avoid),
                                                  start=location, end=tuple(destination)), avoid), leave)

    @tool
    def route_from_home(destination: tuple[float, float], depart_at: str | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Plan a transit route from the signed-in user's saved home to a destination.

        Args:
            destination: (latitude, longitude) of where the user is going; use get_position first for an address.
            depart_at: when they leave, New York local time ("Monday 14:00", "2pm"); defaults to now.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            The route summary with "departs"/"arrives" local times, as get_path returns it. The home location itself is not included.
        """
        departure = _departure(depart_at)
        if isinstance(departure, dict):
            return departure
        home = load_home(store, cipher, uid)
        if home is None:
            return _no_home()
        from agent.tools import get_path
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        leave = departure or int(time.time())
        return _timed(_routed(lambda: _summarized(log, get_path(start=(home.lat, home.lon), end=tuple(destination),
                                                                departure_time=leave, avoid=avoid),
                                                  start=HOME, end=tuple(destination)), avoid), leave)

    @tool
    def route_to_home(start: tuple[float, float] | None = None, depart_at: str | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Plan a transit route from a starting point to the signed-in user's saved home.

        Args:
            start: (latitude, longitude) the user is leaving from; use get_position first for an address.
                Omit to leave from where the user is now.
            depart_at: when they leave, New York local time ("Monday 14:00", "2pm"); defaults to now.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            The route summary with "departs"/"arrives" local times, as get_path returns it. The home location itself is not included.
        """
        departure = _departure(depart_at)
        if isinstance(departure, dict):
            return departure
        if start is None:
            if location is None:
                return _no_location()
            start = location
        home = load_home(store, cipher, uid)
        if home is None:
            return _no_home()
        from agent.tools import get_path
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        leave = departure or int(time.time())
        return _timed(_routed(lambda: _summarized(log, get_path(start=tuple(start), end=(home.lat, home.lon),
                                                                departure_time=leave, avoid=avoid),
                                                  start=tuple(start), end=HOME), avoid), leave)

    @tool
    def upcoming_events(hours_ahead: int = 24) -> list[dict]:
        """List the signed-in user's synced Google Calendar events starting within the next hours_ahead hours.

        Args:
            hours_ahead: How far ahead to look, in hours (max 336).

        Returns:
            [{"summary", "start", "end", "all_day", "location"}], location may be null.
        """
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=max(1, min(int(hours_ahead), 336)))
        out = []
        for e in store.list_calendar_events(uid):
            start = _parse_start(e.get("start"))
            if start is None or not (now - timedelta(hours=1) <= start <= horizon):
                continue
            location = (cipher.decrypt(e["location_enc"], user_id=uid, field=f"event_location:{e['id']}")
                        if e.get("location_enc") else None)
            out.append({"summary": e.get("summary"), "start": e.get("start"), "end": e.get("end"),
                        "all_day": e.get("all_day", False), "location": location})
        return out

    @tool
    def route_to_event(event_id: str, event_location: str) -> dict:
        """Plan a live-adjusted route from home to an upcoming calendar event.

        Combines geocoding, home-based routing, and live delay checking in one step.

        Args:
            event_id: Calendar event ID (from upcoming_events).
            event_location: The event's location (address or place name).

        Returns:
            {
              "event_location": (lat, lon) of the event,
              "minutes", "lines", "transfers": the live-adjusted route's summary,
              "total_time_sec": door-to-door travel time,
              "delay_vs_schedule_sec": how much slower than schedule,
              "leave_by": local time to leave to make the event on time,
              "arrive_by": the event's start, local time,
              "alerts": service alerts on the route,
            }
        """
        from agent.tools import get_position, get_predicted_path

        # Find the event in the user's calendar
        event_time = None
        for e in store.list_calendar_events(uid):
            if e.get("id") == event_id:
                start_str = e.get("start")
                if start_str:
                    start_dt = _parse_start(start_str)
                    if start_dt:
                        event_time = int(start_dt.timestamp())
                break

        if event_time is None:
            raise ValueError(f"event {event_id} not found or has no start time")

        # Get home location
        home = load_home(store, cipher, uid)
        if home is None:
            return {"error": "no_home_set", "message": "Set your home location on the Profile page."}

        # Geocode the event location
        dest_pos = get_position(event_location)

        # Get schedule-based route
        from agent.tools import get_path
        schedule_result = get_path(start=(home.lat, home.lon), end=dest_pos)
        schedule_time = schedule_result["total_time_sec"]

        # Get live-adjusted route with deadline
        live_result = get_predicted_path(
            base_time=int(schedule_time),
            event_time=event_time,
            start_location=(home.lat, home.lon),
            end_location=dest_pos,
        )

        predicted = live_result["predicted"]
        return {
            "event_location": dest_pos,
            **_record(log, predicted, start=HOME, end=dest_pos),
            "total_time_sec": predicted["total_time_sec"],
            "delay_vs_schedule_sec": live_result["delay_sec"],
            "leave_by": local_label(live_result["leave_by_ts"]),
            "arrive_by": local_label(event_time),
            "slack_sec": live_result["slack_sec"],
            "alerts": live_result["alerts_on_route"],
            "on_time": live_result["slack_sec"] > 0,
        }

    @tool
    def leave_by(destination: tuple[float, float], arrival_time: str, start: tuple[float, float] | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """When the user must leave to reach a destination by arrival_time.

        Starts at `start` if the user named a starting place; otherwise where they're
        assumed to be: their device's current location for arrivals in the next 3 hours,
        else their calendar event at that time (during it or up to an hour after), else home.

        Args:
            destination: (latitude, longitude); use get_position first for an address.
            arrival_time: when to arrive, New York local time: "14:00", "2pm", "Monday 14:00",
                "tomorrow 9:30" or "YYYY-MM-DD HH:MM".
            start: (latitude, longitude) of a starting place the user named; omit to use their
                current location / event / home.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            {"leave_by" (local time to tell the user), "arrive_by", "slack_min" (negative =
             already late), "arrives_if_leaving_now" (only when already late), "minutes", "lines", "transfers",
             "start_source": "given", "current_location", "event" or "home", "event_summary"/"event_location" when from
             an event, "live": whether live MTA predictions were used, "alerts"}.
        """
        now = int(time.time())
        try:
            arrive = parse_local_time(arrival_time, now)
        except ValueError as exc:
            return {"error": "bad_arrival_time", "message": str(exc)}
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        here = not start and location is not None and arrive - now <= CURRENT_LOCATION_WINDOW_SEC
        try:
            plan = plan_leave_by(store, cipher, uid, tuple(destination), arrive, now,
                                 geocode=transit_geocode, route=functools.partial(transit_router, avoid=avoid),
                                 start=location if here else tuple(start) if start else None)
        except ValueError as exc:
            return {"error": "no_route", "message": str(exc)}
        if here and "error" not in plan:
            plan["start_source"] = "current_location"
        if avoid and "error" not in plan:
            plan["avoided"] = avoid.describe()
        if "error" not in plan and plan["slack_min"] < 0:
            plan["arrives_if_leaving_now"] = local_label(round(now + plan["total_time_sec"]))
        plan.pop("leave_by_ts", None)
        start_point = plan.pop("start_point", None)  # never to the model; home isn't sent to the browser either
        start_end = HOME if plan.get("start_source") == "home" else start_point
        return _summarized(log, plan, start=start_end, end=tuple(destination))

    @tool
    def plan_multi_stop_trip(stops: list[str], depart_at: str | None = None,
                             stay_minutes: list[int] | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Plan one trip through several stops in order, e.g. a detour or errands on the way.

        Args:
            stops: at least 2 places in travel order: an address, a place name, "home", "here"
                (where the user is now), or "lat,lon".
            depart_at: when the trip starts, New York local time ("Monday 9:00", "2pm"); omit for now.
            stay_minutes: minutes spent at each intermediate stop, in order (defaults to 0).
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            {"legs": [{"from", "to", "minutes", "lines", "transfers", "departs", "arrives"}],
             "total_travel_minutes", "trip_arrives", "stops": [{"input", "matched"}], "highlighted_on_map"}
            or {"error", "message"} naming the stop that couldn't be found.
        """
        if len(stops) < 2:
            return {"error": "too_few_stops", "message": "give at least a start and a destination"}
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        departure = _departure(depart_at)
        if isinstance(departure, dict):
            return departure
        from agent.tools import get_path
        from geocoding import search_places as search

        resolved: list[tuple[Endpoint, tuple[float, float], str]] = []  # (endpoint, coords, name)
        for raw in stops:
            text = str(raw).strip()
            if text.lower() in ("home", "my home"):
                home = load_home(store, cipher, uid)
                if home is None:
                    return _no_home()
                resolved.append((HOME, (home.lat, home.lon), "home"))
                continue
            if text.lower() in HERE_WORDS:
                if location is None:
                    return _no_location()
                resolved.append((location, location, "your location"))
                continue
            match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", text)
            if match:
                point = (float(match.group(1)), float(match.group(2)))
                resolved.append((point, point, text))
                continue
            found = search(text, limit=1)
            if not found:
                return {"error": "stop_not_found", "message": f"couldn't find {text!r}; try search_places or an address"}
            place = found[0]
            label = f"{place['name']}, {place['address']}" if place.get("address") else place["name"]
            resolved.append(((place["lat"], place["lon"]), (place["lat"], place["lon"]), label))

        stays = list(stay_minutes or [])
        clock = departure or int(time.time())
        legs, segments, total = [], [], 0.0
        for i in range(len(resolved) - 1):
            (start_ep, start, start_name), (end_ep, end, end_name) = resolved[i], resolved[i + 1]
            try:
                route = get_path(start=start, end=end, departure_time=clock, avoid=avoid)
            except ValueError as exc:
                return {"error": "no_route", "message": f"leg {i + 1} ({start_name} -> {end_name}): {exc}"}
            arrives = round(clock + route["total_time_sec"])
            summary = _route_summary(route)
            legs.append({"from": start_name, "to": end_name, **summary,
                         "departs": local_label(clock), "arrives": local_label(arrives)})
            segments.append((route, start_ep, end_ep))
            total += route["total_time_sec"]
            stay = stays[i] if i < len(stays) and stays[i] else 0
            clock = arrives + int(stay) * 60
        if log is not None:
            log.record_trip(segments, [name for _, _, name in resolved])
        return {**({"avoided": avoid.describe()} if avoid else {}),
                "legs": legs, "total_travel_minutes": round(total / 60), "trip_arrives": legs[-1]["arrives"],
                "stops": [{"input": str(raw), "matched": name} for raw, (_, _, name) in zip(stops, resolved)],
                "highlighted_on_map": log is not None}

    return [route_from_here, route_from_home, route_to_home, upcoming_events, route_to_event, leave_by,
            plan_multi_stop_trip]


def _no_location() -> dict:
    return {"error": "no_location",
            "message": "The user's location isn't available. Ask them to allow location access in the browser, "
                       "or for a starting address."}


def _avoid(modes: list[str] | None, lines: list[str] | None):
    """The model's avoid_modes/avoid_lines as an Avoid, or an error dict it can read."""
    if not modes and not lines:
        return None
    from agent.routing import parse_avoid
    try:
        return parse_avoid(modes, lines)
    except ValueError as exc:
        return {"error": "bad_avoid", "message": str(exc)}


def _routed(plan, avoid) -> dict:
    """Run a routing call; a route that can't avoid what was asked becomes an error
    the model can explain, and a successful one says what it avoided."""
    try:
        result = plan()
    except ValueError as exc:
        return {"error": "no_route", "message": str(exc)}
    return {**result, "avoided": avoid.describe()} if avoid and "error" not in result else result


def _timed(result: dict, leave: int) -> dict:
    """A route result with the local times it departs and arrives, leaving at
    `leave`; errors (and results without a total time) unchanged."""
    if "error" in result or result.get("total_time_sec") is None:
        return result
    return {**result, "departs": local_label(leave), "arrives": local_label(round(leave + result["total_time_sec"]))}


def _route_summary(route: dict) -> dict:
    from agent.directions import route_summary
    return route_summary(route["steps"], route["total_time_sec"])


def _departure(value: str | None) -> int | dict | None:
    """A model-stated local time as Unix time; None when omitted; an error dict
    the model can read and correct when it can't be parsed."""
    if value is None or not str(value).strip():
        return None
    try:
        return parse_local_time(value, int(time.time()))
    except ValueError as exc:
        return {"error": "bad_time", "message": str(exc)}


class RouteLog:
    """The last route planned in this request. Its step-by-step directions go to
    the rider in code (directions_text), not through the model: route tools hand
    the model only a summary, and station names only via route_stations. A small
    model given the raw steps rewrote them -- every stop became a "transfer"."""

    def __init__(self) -> None:
        self.route: dict | None = None
        self.places: list[dict] | None = None  # the last search_places results, pinned on the map
        self.stations: list[dict] | None = None  # the last station_status results, marked on the map

    @property
    def directions(self) -> str | None:
        return self.route["directions"] if self.route else None

    def record(self, steps: list[dict], total_time_sec: float,
               walk_in_sec: float | None = None, walk_out_sec: float | None = None,
               start: Endpoint = None, end: Endpoint = None) -> dict:
        """`start` / `end`: the trip's (lat, lon), HOME for the user's home (sent to
        the browser without coordinates, like everywhere else), or None if unknown."""
        self.route = _route_payload(steps, total_time_sec, walk_in_sec, walk_out_sec, start, end)
        # The chat page draws the last recorded route (web/static/route_map.js).
        return {k: self.route[k] for k in ("minutes", "lines", "transfers")} | {"highlighted_on_map": True}

    def record_trip(self, segments: list[tuple[dict, Endpoint, Endpoint]], stop_names: list[str]) -> None:
        """A multi-stop trip as one drawable route: each segment is (get_path result,
        start, end) and stop_names has one name per stop (len(segments) + 1)."""
        import math
        from agent.directions import format_directions, route_legs, route_summary
        coords = lambda e: e if isinstance(e, tuple) else None
        legs, directions, lines, transfers, total = [], [], [], 0, 0.0
        for i, (route, start, end) in enumerate(segments):
            summary = route_summary(route["steps"], route["total_time_sec"])
            legs += route_legs(route["steps"], route.get("walk_in_sec"), route.get("walk_out_sec"),
                               coords(start), coords(end))
            directions.append(f"Leg {i + 1}, to {stop_names[i + 1]}:\n" + format_directions(
                route["steps"], route["total_time_sec"], route.get("walk_in_sec"), route.get("walk_out_sec")))
            lines += [line for line in summary["lines"] if line not in lines]
            transfers += summary["transfers"]
            total += route["total_time_sec"]
        waypoints = [{**_endpoint(end), "name": stop_names[i + 1], "order": i + 1}
                     for i, (_, _, end) in enumerate(segments[:-1])]
        self.route = {"minutes": math.ceil(total / 60), "lines": lines, "transfers": transfers,
                      "total_time_sec": total, "walk_in_sec": segments[0][0].get("walk_in_sec"),
                      "walk_out_sec": segments[-1][0].get("walk_out_sec"),
                      "start": _endpoint(segments[0][1]), "end": _endpoint(segments[-1][2]),
                      "waypoints": waypoints, "legs": legs, "directions": "\n\n".join(directions)}


Endpoint = "tuple[float, float] | str | None"


def _record(log: RouteLog | None, route: dict, start: Endpoint = None, end: Endpoint = None) -> dict:
    """Log a route dict with steps and return its summary (or the raw steps with no log)."""
    if log is None:
        return {"steps": route["steps"]}
    return log.record(route["steps"], route["total_time_sec"], route.get("walk_in_sec"), route.get("walk_out_sec"),
                      start=_coords_or_home(start), end=_coords_or_home(end))


def _coords_or_home(value: Endpoint):
    return value if value in (None, HOME) else (float(value[0]), float(value[1]))


def _summarized(log: RouteLog | None, route: dict, start: Endpoint = None, end: Endpoint = None) -> dict:
    """`route` with its steps replaced by the summary; unchanged with no log, or
    when it's an error rather than a route."""
    if log is None or "steps" not in route:
        return route
    return {**{k: v for k, v in route.items() if k != "steps"}, **_record(log, route, start, end)}


def make_route_tools(log: RouteLog) -> list:
    """The shared routing tools, wrapped so their routes go through `log`."""
    from agent import tools

    @tool
    def get_path(start: tuple[float, float], end: tuple[float, float], depart_at: str | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Fastest route between two coordinates on the subway+bus schedule for the departure time.

        Args:
            start: (latitude, longitude) of the trip's start.
            end: (latitude, longitude) of the destination.
            depart_at: when the user leaves, New York local time ("Monday 14:00", "2pm",
                "tomorrow 9:30"); omit for now. Late-night and weekend service is slower.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            {"minutes", "lines", "transfers", "departs", "arrives" (local times), "total_time_sec",
             "walk_in_sec", "walk_out_sec", "service_state"}
        """
        departure = _departure(depart_at)
        if isinstance(departure, dict):
            return departure
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        leave = departure or int(time.time())
        return _timed(_routed(lambda: _summarized(log, tools.get_path(start=start, end=end, departure_time=leave,
                                                                      avoid=avoid),
                                                  start=start, end=end), avoid), leave)

    @tool
    def get_predicted_path(base_time: int, event_time: str, start_location: tuple[float, float],
                           end_location: tuple[float, float], avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Re-price a trip under live MTA conditions and say when to leave to arrive by a deadline.

        Args:
            base_time: get_path's total_time_sec for this trip (a duration, not a clock time).
            event_time: the deadline, New York local time ("Monday 14:00", "2pm").
            start_location: (latitude, longitude) of the start.
            end_location: (latitude, longitude) of the destination.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            {"delay_sec", "leave_by", "arrive_by" (local times), "slack_sec", "route_changed", "predicted": {"minutes",
             "lines", "transfers", "total_time_sec"}, "alerts_on_route", "model_adjusted", "waits_adjusted"}
        """
        deadline = _departure(event_time)
        if not isinstance(deadline, int):
            return deadline or {"error": "bad_time", "message": f"event_time must be {LOCAL_TIME_FORMATS}"}
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        try:
            res = tools.get_predicted_path(base_time=base_time, event_time=deadline, start_location=start_location,
                                           end_location=end_location, avoid=avoid)
        except ValueError as exc:
            return {"error": "no_route", "message": str(exc)}
        planned = {k: v for k, v in res["planned"].items() if k != "steps"}
        return {**res, "planned": planned,
                "predicted": _summarized(log, res["predicted"], start=start_location, end=end_location),
                "leave_by": local_label(res["leave_by_ts"]), "arrive_by": local_label(deadline),
                **({"avoided": avoid.describe()} if avoid else {})}

    @tool
    def compare_schedule_vs_live(start: tuple[float, float], end: tuple[float, float],
                                 arrival_deadline: str | None = None, avoid_modes: list[str] | None = None, avoid_lines: list[str] | None = None) -> dict:
        """Compare the scheduled route with the live-adjusted one.

        Args:
            start: (latitude, longitude) of the start.
            end: (latitude, longitude) of the destination.
            arrival_deadline: when to arrive, New York local time ("Monday 14:00", "2pm"), if any.
            avoid_modes: leave a mode out entirely: ["bus"] for "no buses", ["subway"] for "buses only".
            avoid_lines: lines or bus routes to leave out, e.g. ["L"], ["4", "5"], ["B38"].

        Returns:
            {"departs" (local time; both routes leave now), "schedule_route", "live_route": summaries
             ({"minutes", "lines", "transfers", "arrives"}), "delay_sec", "route_changed", "leave_by" (local time),
             "slack_sec", "alerts"}
        """
        deadline = _departure(arrival_deadline)
        if isinstance(deadline, dict):
            return deadline
        avoid = _avoid(avoid_modes, avoid_lines)
        if isinstance(avoid, dict):
            return avoid
        try:
            res = tools.compare_schedule_vs_live(start=start, end=end, arrival_deadline=deadline, avoid=avoid)
        except ValueError as exc:
            return {"error": "no_route", "message": str(exc)}
        from agent.directions import route_summary
        now = int(time.time())
        arrives = lambda sec: local_label(round(now + sec))
        return {**res, "departs": local_label(now),
                "schedule_route": {**route_summary(res["schedule_route"], res["schedule_time_sec"]),
                                   "arrives": arrives(res["schedule_time_sec"])},
                "live_route": {**log.record(res["live_route"], res["live_time_sec"],
                                            start=_coords_or_home(start), end=_coords_or_home(end)),
                               "arrives": arrives(res["live_time_sec"])},
                "leave_by": local_label(res["leave_by_ts"])}

    @tool
    def search_places(query: str) -> dict:
        """Search New York City for places matching a name or partial address, to pick one to route to.
        Use it whenever get_position fails or a name could be several places.

        Args:
            query: what the user said, e.g. "PS 26 the Jesse Owens School", "MoMA", "Starbucks Union Square".

        Returns:
            {"places": [{"name", "address", "kind", "lat", "lon"}] best first (up to 5), "pinned_on_map": bool}
        """
        from geocoding import search_places as search
        places = search(query)
        log.places = places or None
        return {"places": places, "pinned_on_map": bool(places)}

    @tool
    def station_status(stations: list[str] | None = None, line: str | None = None,
                       along_route: bool = False, only_problems: bool = False) -> dict:
        """Live service alerts and delays at subway stations; they're marked on the user's map.

        Args:
            stations: station names, e.g. ["86 St", "Union Sq"].
            line: a subway line, e.g. "4", to check all its stations.
            along_route: check the stations of the route just planned in this conversation turn.
            only_problems: return only stations with an alert or a delay (use for "any delays on the 4?").

        Returns:
            {"stations": [{"name", "lines", "status" ("alert"|"delayed"|"ok"), "delay_min", "alerts"}],
             "shown_on_map"} or {"error", "message"}
        """
        from agent.station_status import find_stations, station_of
        from agent.station_status import station_status as live_status
        ids: list[str] = []
        for name in stations or []:
            found = find_stations(name)
            if not found:
                return {"error": "station_not_found", "message": f"no subway station matches {name!r}"}
            ids += [i for i in found if i not in ids]
        if along_route:
            if not log.route:
                return {"error": "no_route", "message": "plan the route first, then check it"}
            ids += [station_of(stop["stop_id"]) for leg in log.route["legs"] if leg["type"] == "ride"
                    and leg["mode"] == "subway" for stop in leg["path"] if station_of(stop["stop_id"]) not in ids]
        if not ids and not line:
            return {"error": "nothing_to_check", "message": "name stations, a line, or use along_route"}
        try:
            found = live_status(ids or None, line=line, only_problems=only_problems)
        except Exception as exc:  # snapshot service down
            return {"error": "live_data_unavailable", "message": str(exc)[:200]}
        log.stations = found or None
        return {"stations": [{"name": s["name"], "lines": s["lines"], "status": s["status"],
                              "delay_min": round((s["delay_sec"] or 0) / 60), "alerts": s["alerts"]} for s in found],
                "shown_on_map": bool(found)}

    @tool
    def route_stations() -> str:
        """Station-by-station directions for the last route planned. Call only when the user
        asks which stations or stops, or where to transfer."""
        return log.directions or "No route has been planned yet."

    return [get_path, get_predicted_path, compare_schedule_vs_live, search_places, station_status, route_stations]


def make_model(model_id: str | None = None, region: str | None = None):
    """Amazon Bedrock Mantle when a model id is configured (YOHO_AGENT_MODEL), else local Ollama.

    Mantle serves open-weight models such as google.gemma-4-* over an OpenAI-compatible
    API; Strands mints its bearer token from boto3's usual credential chain (env keys,
    AWS_PROFILE, or the instance role), which needs bedrock-mantle:CreateInference and
    bedrock-mantle:CallWithBearerToken.
    """
    if model_id:
        return OpenAIModel(
            model_id=model_id,
            bedrock_mantle_config={"region": region} if region else {},
            # Gemma 4 on Mantle rejects more than one tool call per turn.
            params={"temperature": 0.2, "parallel_tool_calls": False},
        )
    return OllamaModel(
        host="http://localhost:11434",
        model_id="gemma4:e2b",
        temperature=0.2,
        # Ollama loads models with a 4096-token context by default. The system
        # prompt and tool specs alone take ~1.8k, so a turn with thinking and a
        # route result overflows it and Strands raises MaxTokensReachedException.
        options={"num_ctx": 32768},
    )


def build_agent(uid: str, store: UserStore, cipher: FieldCipher, history: list[dict],
                location: tuple[float, float] | None = None, *,
                model_id: str | None = None, region: str | None = None) -> Agent:
    from agent.tools import get_current_time, get_position
    log = RouteLog()
    agent = Agent(
        model=make_model(model_id, region),
        system_prompt=SYSTEM_PROMPT,
        messages=history,
        tools=[get_current_time, get_position, *make_route_tools(log), *make_user_tools(uid, store, cipher, log, location)],
        callback_handler=None,
    )
    agent.route_log = log
    return agent


def reply_text(result: Any) -> str:
    return "".join(block.get("text", "") for block in result.message.get("content", [])).strip()


def stations_payload(agent: Any) -> list[dict] | None:
    """The last station_status results, for the browser to mark; None if none."""
    log = getattr(agent, "route_log", None)
    return getattr(log, "stations", None) if log is not None else None


def places_payload(agent: Any) -> list[dict] | None:
    """The last place search this agent ran, for the browser to pin; None if none."""
    log = getattr(agent, "route_log", None)
    return log.places if log is not None else None


def route_payload(agent: Any) -> dict | None:
    """The last route this agent planned, for the caller to send the browser with
    the reply: {"minutes", "lines", "transfers", "total_time_sec", "walk_in_sec",
    "walk_out_sec", "legs" (walks and rides, each ride's stops with lat/lon),
    "directions" (step-by-step text)}. None if no route was planned."""
    log = getattr(agent, "route_log", None)
    return log.route if log is not None else None


def token_usage(agent: Any) -> tuple[int, int]:
    """(input, output) tokens this agent has spent. A fresh agent per request
    makes this the request's cost; read it even after a failed call, since
    tokens spent before the failure still count."""
    usage = agent.event_loop_metrics.accumulated_usage
    return int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))


TOOL_BLOCKS = ("toolUse", "toolResult")


def _is_prompt(message: dict) -> bool:
    """A user message the user typed, as opposed to one carrying tool results."""
    return message.get("role") == "user" and not any("toolResult" in block for block in message.get("content", []))


def _without_tool_calls(messages: list[dict]) -> list[dict]:
    """Drop toolUse/toolResult blocks, then messages left empty, merging the
    same-role neighbours that leaves behind (e.g. "Let me check." + the answer)
    since some providers reject two assistant messages in a row."""
    out: list[dict] = []
    for message in messages:
        content = [block for block in message.get("content", []) if not any(k in block for k in TOOL_BLOCKS)]
        if not content:
            continue
        if out and out[-1]["role"] == message.get("role"):
            out[-1] = {**out[-1], "content": [*out[-1]["content"], *content]}
        else:
            out.append({**message, "content": content})
    return out


def trim_history(messages: list[dict], limit: int = MAX_STORED_MESSAGES) -> list[dict]:
    """The history to store: tool calls only from the latest request (they're
    most of the context, and older ones are rarely needed), then the last
    `limit` messages, starting at a plain user message so the kept history never
    opens on a dangling tool result. The latest request is kept whole even if
    it alone runs past `limit`."""
    last = next((i for i in range(len(messages) - 1, -1, -1) if _is_prompt(messages[i])), 0)
    earlier = _without_tool_calls(messages[:last])
    if earlier and messages[last:] and earlier[-1]["role"] == messages[last].get("role"):
        earlier.pop()  # an earlier turn that ended without a reply; don't open two user turns in a row
    history = [*earlier, *messages[last:]]
    kept = history[max(0, min(len(history) - limit, len(earlier))):]
    while kept and not _is_prompt(kept[0]):
        kept.pop(0)
    return kept
