"""Plan a trip to arrive somewhere on time, for the chat agent's leave_by tool
and the scheduler's event alerts alike.

The trip starts where the user is assumed to be: at the calendar event at the
top of their queue if now falls within it or the hour after it ends
(scheduler/calendar_hook.py), otherwise at home. Home coordinates never leave
this module; callers get the route and where it started from ("event"/"home").

Routing is passed in (`route`), so tests don't load the transit graph. Two
real routers: transit_router (chat agent) prices rides with the Graph WaveNet
and needs torch; schedule_router (alert scheduler) doesn't -- schedule costs
plus live first-train waits from the snapshot service.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from accounts.home import load_home
from scheduler.calendar_hook import CalendarHandler
from storage import FieldCipher, UserStore

NYC = ZoneInfo("America/New_York")

Coords = tuple[float, float]
# address -> (lat, lon); may return None or raise ValueError when it can't.
Geocoder = Callable[[str], "Coords | None"]
# (start, end, arrive_ts) -> {"steps", "total_time_sec", "walk_in_sec", "walk_out_sec",
#                             "live", "alerts", optionally "delay_vs_schedule_sec"}
Router = Callable[[Coords, Coords, int], dict]


def no_home() -> dict:
    return {"error": "no_home_set", "message": "The user has not set a home location. They can set it on the Profile page."}


def parse_event_time(value: str | None) -> datetime | None:
    """A stored event's start/end as an aware datetime (naive -> UTC), or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def local_label(ts: int) -> str:
    """e.g. "Sun Sep 13, 8:27 PM" in New York time."""
    return datetime.fromtimestamp(ts, NYC).strftime("%a %b %-d, %-I:%M %p")


LOCAL_TIME_FORMATS = ('"HH:MM", "2pm", "Monday 14:00", "tomorrow 9:30" or "YYYY-MM-DD HH:MM" '
                      '(New York time)')
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _parse_clock(text: str) -> tuple[int, int]:
    text = text.strip().lower().replace(" ", "").replace(".", "")
    for fmt in ("%H:%M", "%I:%M%p", "%I%p", "%H"):
        try:
            clock = datetime.strptime(text, fmt)
            return clock.hour, clock.minute
        except ValueError:
            pass
    raise ValueError(text)


def parse_local_time(value: str, now: int) -> int:
    """Unix time for a New York local time the model states in words, because it
    can't do epoch math: "14:00" / "2pm" (the next time it occurs), "Monday 14:00"
    (the next Monday; today if that time hasn't passed), "today 9:30",
    "tomorrow 9:30", or "YYYY-MM-DD HH:MM"."""
    raw = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return int(datetime.strptime(raw, fmt).replace(tzinfo=NYC).timestamp())
        except ValueError:
            pass
    words = raw.lower().replace(",", " ").replace(" at ", " ").split()
    current = datetime.fromtimestamp(now, NYC)
    day_word = next((w for w in words if w in ("today", "tonight", "tomorrow")
                     or any(d.startswith(w) and len(w) >= 3 for d in _WEEKDAYS)), None)
    clock_text = " ".join(w for w in words if w != day_word)
    try:
        hour, minute = _parse_clock(clock_text)
    except ValueError:
        raise ValueError(f"time must be {LOCAL_TIME_FORMATS}, got {raw!r}")
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if day_word == "tomorrow":
        target += timedelta(days=1)
    elif day_word in ("today", "tonight"):
        pass
    elif day_word is not None:
        weekday = next(i for i, d in enumerate(_WEEKDAYS) if d.startswith(day_word))
        target += timedelta(days=(weekday - current.weekday()) % 7)
        if target.timestamp() < now:
            target += timedelta(days=7)
    elif target.timestamp() < now:
        target += timedelta(days=1)
    return int(target.timestamp())


def calendar_handler(uid: str, store: UserStore) -> CalendarHandler:
    """A calendar queue of the user's timed events, rebuilt from the store.
    Locations stay encrypted in the store; only the chosen event's is decrypted.
    All-day events are skipped: they'd pin the start location for the whole day."""
    handler = CalendarHandler()
    for e in store.list_calendar_events(uid):
        start, end = parse_event_time(e.get("start")), parse_event_time(e.get("end"))
        if e.get("all_day") or start is None or end is None:
            continue
        handler.add_event(id=e["id"], time=int(start.timestamp()),
                          duration=max(0, int((end - start).total_seconds())),
                          lat=None, long=None, address=None)
    return handler


def event_location(store: UserStore, cipher: FieldCipher, uid: str, event: dict) -> str | None:
    return (cipher.decrypt(event["location_enc"], user_id=uid, field=f"event_location:{event['id']}")
            if event.get("location_enc") else None)


def _geocode(geocode: Geocoder, address: str) -> Coords | None:
    try:
        coords = geocode(address)
    except ValueError:
        return None
    return tuple(coords) if coords else None


def resolve_start(store: UserStore, cipher: FieldCipher, uid: str, now: int,
                  geocode: Geocoder) -> tuple[Coords | None, dict]:
    """Where a trip leaving at `now` starts, and what to report about it (never
    the home coordinates). Coordinates are None when neither is known."""
    event = calendar_handler(uid, store).start_event(now)
    note = None
    if event is not None:
        row = next((e for e in store.list_calendar_events(uid) if e.get("id") == event.id), {"id": event.id})
        address = event_location(store, cipher, uid, row)
        info = {"start_source": "event", "event_summary": row.get("summary"), "event_location": address}
        if address:
            coords = _geocode(geocode, address)
            if coords is not None:
                return coords, info
            note = f"couldn't geocode the current event's location ({address!r}); started from home"
        else:
            note = "the current event has no location; started from home"
    home = load_home(store, cipher, uid)
    if home is None:
        return None, no_home()
    return (home.lat, home.lon), {"start_source": "home", "start_note": note}


def plan_leave_by(store: UserStore, cipher: FieldCipher, uid: str, destination: Coords, arrive: int, now: int,
                  *, geocode: Geocoder, route: Router, start: Coords | None = None,
                  located_at: int | None = None) -> dict:
    """When to leave to reach `destination` by `arrive`: from `start` when the
    user gave one, else from resolve_start (current event or home).

    Returns {"leave_by_ts", "leave_by", "arrive_by", "slack_min", "steps",
    "total_time_sec", "walk_in_sec", "walk_out_sec", "live", "alerts",
    "start_source", ...}, or an {"error", "message"} dict."""
    if start is not None:
        info: dict = {"start_source": "given"}
    else:
        # Where the rider is at `located_at` (e.g. their expected departure), else now.
        start, info = resolve_start(store, cipher, uid, located_at or now, geocode)
        if start is None:
            return info
    trip = route(tuple(start), tuple(destination), arrive)
    leave_ts = round(arrive - trip["total_time_sec"])
    # start_point is for drawing the walk from the start; callers must not hand it
    # to the model or browser when start_source is "home".
    return {"leave_by_ts": leave_ts, "leave_by": local_label(leave_ts), "arrive_by": local_label(arrive),
            "slack_min": round((leave_ts - now) / 60), **trip, **info, "start_point": tuple(start)}


def transit_router(start: Coords, end: Coords, arrive: int, avoid=None) -> dict:
    """The agent's router: live MTA prediction (graph model over the snapshot
    service's window), falling back to the schedule when that's down. Loads torch."""
    from agent.tools import get_path, get_predicted_path
    schedule = get_path(start=start, end=end, avoid=avoid)
    try:
        live = get_predicted_path(base_time=int(schedule["total_time_sec"]), event_time=arrive,
                                  start_location=start, end_location=end, avoid=avoid)
    except Exception:  # snapshot service down or cold: the schedule is still an answer
        live = None
    if live is None:
        return {"steps": schedule["steps"], "total_time_sec": schedule["total_time_sec"],
                "walk_in_sec": schedule["walk_in_sec"], "walk_out_sec": schedule["walk_out_sec"],
                "live": False, "alerts": []}
    predicted = live["predicted"]
    return {"steps": predicted["steps"], "total_time_sec": predicted["total_time_sec"],
            "walk_in_sec": predicted["walk_in_sec"], "walk_out_sec": predicted["walk_out_sec"],
            "live": True, "delay_vs_schedule_sec": live["delay_sec"], "alerts": live["alerts_on_route"]}


def schedule_router(start: Coords, end: Coords, arrive: int, avoid=None) -> dict:
    """The scheduler's router, without torch: the schedule for the departure's
    service state, plus the live wait for the first train when the snapshot
    service is up (live_waits)."""
    from agent.routing import _fetch_waits, schedule_route
    waits = _fetch_waits()
    trip = schedule_route(start, end, waits=waits, avoid=avoid)
    return {"steps": trip["steps"], "total_time_sec": trip["total_time_sec"],
            "walk_in_sec": trip["walk_in_sec"], "walk_out_sec": trip["walk_out_sec"],
            "live": False, "live_waits": waits is not None, "alerts": []}


def transit_geocode(address: str) -> Coords | None:
    from agent.routing import geocode_address
    return geocode_address(address)
