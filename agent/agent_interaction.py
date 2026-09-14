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
print("hi")

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from accounts.home import load_home
from accounts.trips import (no_home as _no_home, parse_event_time as _parse_start, parse_local_time,
                            plan_leave_by, transit_geocode, transit_router)
from storage import FieldCipher, UserStore

SYSTEM_PROMPT = """You are Yoho, a concise assistant for getting around New York City on public transit.
Route tools return a summary (minutes, lines, transfers). The route is drawn on the user's map
automatically: when a result has "highlighted_on_map": true, tell the user you've highlighted the
route on their map. Don't list stations or stops yourself; if they ask which stations or where to
transfer, call route_stations.
Answer briefly unless the user asks for detail. Use tools to geocode and plan; don't invent information.
For trips starting or ending at the user's home, use route_from_home / route_to_home. The home
location is private: you can't see it, don't ask the user for it, and don't guess it.
If a tool says no home is set, tell the user they can set it on their Profile page.
Use get_current_time when you need to know today's schedule (weekday, Saturday, or Sunday).
For "when should I leave" questions, use leave_by; it picks the start (current event or home) itself.
Only help with transit, routing, and the user's schedule; politely decline anything else."""

MAX_STORED_MESSAGES = 20


def make_user_tools(uid: str, store: UserStore, cipher: FieldCipher, log: RouteLog | None = None) -> list:
    @tool
    def route_from_home(destination: tuple[float, float], departure_time: int | None = None) -> dict:
        """Plan a transit route from the signed-in user's saved home to a destination.

        Args:
            destination: (latitude, longitude) of where the user is going; use get_position first for an address.
            departure_time: Unix timestamp (seconds) of departure; defaults to now.

        Returns:
            The route summary, as get_path returns it. The home location itself is not included.
        """
        home = load_home(store, cipher, uid)
        if home is None:
            return _no_home()
        from agent.tools import get_path
        return _summarized(log, get_path(start=(home.lat, home.lon), end=tuple(destination), departure_time=departure_time))

    @tool
    def route_to_home(start: tuple[float, float], departure_time: int | None = None) -> dict:
        """Plan a transit route from a starting point to the signed-in user's saved home.

        Args:
            start: (latitude, longitude) the user is leaving from; use get_position first for an address.
            departure_time: Unix timestamp (seconds) of departure; defaults to now.

        Returns:
            The route summary, as get_path returns it. The home location itself is not included.
        """
        home = load_home(store, cipher, uid)
        if home is None:
            return _no_home()
        from agent.tools import get_path
        return _summarized(log, get_path(start=tuple(start), end=(home.lat, home.lon), departure_time=departure_time))

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
              "leave_by_ts": when to leave to make the event on time,
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
            **_record(log, predicted),
            "total_time_sec": predicted["total_time_sec"],
            "delay_vs_schedule_sec": live_result["delay_sec"],
            "leave_by_ts": live_result["leave_by_ts"],
            "slack_sec": live_result["slack_sec"],
            "alerts": live_result["alerts_on_route"],
            "on_time": live_result["slack_sec"] > 0,
        }

    @tool
    def leave_by(destination: tuple[float, float], arrival_time: str) -> dict:
        """When the user must leave to reach a destination by arrival_time.

        The trip starts where the user is assumed to be: at their current calendar
        event (during it or up to an hour after it ends), otherwise at home.

        Args:
            destination: (latitude, longitude); use get_position first for an address.
            arrival_time: New York local time to arrive by, "HH:MM" (24-hour; the next
                time it occurs) or "YYYY-MM-DD HH:MM". E.g. 9 PM tonight -> "21:00".

        Returns:
            {"leave_by" (local time to tell the user), "arrive_by", "slack_min" (negative =
             already late), "minutes", "lines", "transfers",
             "start_source": "event" or "home", "event_summary"/"event_location" when from
             an event, "live": whether live MTA predictions were used, "alerts"}.
        """
        now = int(time.time())
        try:
            arrive = parse_local_time(arrival_time, now)
        except ValueError as exc:
            return {"error": "bad_arrival_time", "message": str(exc)}
        plan = plan_leave_by(store, cipher, uid, tuple(destination), arrive, now,
                             geocode=transit_geocode, route=transit_router)
        plan.pop("leave_by_ts", None)
        return _summarized(log, plan)

    return [route_from_home, route_to_home, upcoming_events, route_to_event, leave_by]


class RouteLog:
    """The last route planned in this request. Its step-by-step directions go to
    the rider in code (directions_text), not through the model: route tools hand
    the model only a summary, and station names only via route_stations. A small
    model given the raw steps rewrote them -- every stop became a "transfer"."""

    def __init__(self) -> None:
        self.route: dict | None = None

    @property
    def directions(self) -> str | None:
        return self.route["directions"] if self.route else None

    def record(self, steps: list[dict], total_time_sec: float,
               walk_in_sec: float | None = None, walk_out_sec: float | None = None) -> dict:
        from agent.directions import format_directions, route_legs, route_summary
        summary = route_summary(steps, total_time_sec)
        self.route = {**summary, "total_time_sec": total_time_sec,
                      "walk_in_sec": walk_in_sec, "walk_out_sec": walk_out_sec,
                      "legs": route_legs(steps, walk_in_sec, walk_out_sec),
                      "directions": format_directions(steps, total_time_sec, walk_in_sec, walk_out_sec)}
        # The chat page draws the last recorded route (web/static/route_map.js).
        return {**summary, "highlighted_on_map": True}


def _record(log: RouteLog | None, route: dict) -> dict:
    """Log a route dict with steps and return its summary (or the raw steps with no log)."""
    if log is None:
        return {"steps": route["steps"]}
    return log.record(route["steps"], route["total_time_sec"], route.get("walk_in_sec"), route.get("walk_out_sec"))


def _summarized(log: RouteLog | None, route: dict) -> dict:
    """`route` with its steps replaced by the summary; unchanged with no log, or
    when it's an error rather than a route."""
    if log is None or "steps" not in route:
        return route
    return {**{k: v for k, v in route.items() if k != "steps"}, **_record(log, route)}


def make_route_tools(log: RouteLog) -> list:
    """The shared routing tools, wrapped so their routes go through `log`."""
    from agent import tools

    @tool
    def get_path(start: tuple[float, float], end: tuple[float, float], departure_time: int | None = None) -> dict:
        """Fastest route between two coordinates on the subway+bus schedule (no live data).

        Args:
            start: (latitude, longitude) of the trip's start.
            end: (latitude, longitude) of the destination.
            departure_time: Unix timestamp of departure; defaults to now.

        Returns:
            {"minutes", "lines", "transfers", "total_time_sec", "walk_in_sec", "walk_out_sec", "service_state"}
        """
        return _summarized(log, tools.get_path(start=start, end=end, departure_time=departure_time))

    @tool
    def get_predicted_path(base_time: int, event_time: int, start_location: tuple[float, float],
                           end_location: tuple[float, float]) -> dict:
        """Re-price a trip under live MTA conditions and say when to leave to arrive by a deadline.

        Args:
            base_time: get_path's total_time_sec for this trip (a duration, not a clock time).
            event_time: Unix timestamp of the deadline.
            start_location: (latitude, longitude) of the start.
            end_location: (latitude, longitude) of the destination.

        Returns:
            {"delay_sec", "leave_by_ts", "slack_sec", "route_changed", "predicted": {"minutes",
             "lines", "transfers", "total_time_sec"}, "alerts_on_route", "model_adjusted", "waits_adjusted"}
        """
        res = tools.get_predicted_path(base_time=base_time, event_time=event_time,
                                       start_location=start_location, end_location=end_location)
        planned = {k: v for k, v in res["planned"].items() if k != "steps"}
        return {**res, "planned": planned, "predicted": _summarized(log, res["predicted"])}

    @tool
    def compare_schedule_vs_live(start: tuple[float, float], end: tuple[float, float],
                                 arrival_deadline: int | None = None) -> dict:
        """Compare the scheduled route with the live-adjusted one.

        Args:
            start: (latitude, longitude) of the start.
            end: (latitude, longitude) of the destination.
            arrival_deadline: Unix timestamp to arrive by, if any.

        Returns:
            {"schedule_route", "live_route": summaries ({"minutes", "lines", "transfers"}),
             "delay_sec", "route_changed", "leave_by_ts", "slack_sec", "alerts"}
        """
        res = tools.compare_schedule_vs_live(start=start, end=end, arrival_deadline=arrival_deadline)
        from agent.directions import route_summary
        return {**res, "schedule_route": route_summary(res["schedule_route"], res["schedule_time_sec"]),
                "live_route": log.record(res["live_route"], res["live_time_sec"])}

    @tool
    def route_stations() -> str:
        """Station-by-station directions for the last route planned. Call only when the user
        asks which stations or stops, or where to transfer."""
        return log.directions or "No route has been planned yet."

    return [get_path, get_predicted_path, compare_schedule_vs_live, route_stations]


def build_agent(uid: str, store: UserStore, cipher: FieldCipher, history: list[dict]) -> Agent:
    from agent.tools import get_current_time, get_position
    log = RouteLog()
    agent = Agent(
        model=OllamaModel(
            host="http://localhost:11434",
            model_id="gemma4:e2b",
            temperature=0.2,
            # Ollama loads models with a 4096-token context by default. The system
            # prompt and tool specs alone take ~1.8k, so a turn with thinking and a
            # route result overflows it and Strands raises MaxTokensReachedException.
            options={"num_ctx": 32768},
        ),  # None -> Strands' default (Bedrock)
        system_prompt=SYSTEM_PROMPT,
        messages=history,
        tools=[get_current_time, get_position, *make_route_tools(log), *make_user_tools(uid, store, cipher, log)],
        callback_handler=None,
    )
    agent.route_log = log
    return agent


def reply_text(result: Any) -> str:
    return "".join(block.get("text", "") for block in result.message.get("content", [])).strip()


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


def trim_history(messages: list[dict], limit: int = MAX_STORED_MESSAGES) -> list[dict]:
    """Keep the last `limit` messages, starting at a plain user message so the
    kept history never opens on a dangling tool result."""
    kept = list(messages[-limit:])
    while kept and not (kept[0].get("role") == "user"
                        and not any("toolResult" in block for block in kept[0].get("content", []))):
        kept.pop(0)
    return kept
