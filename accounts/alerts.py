"""Proactive "time to leave" alerts for calendar events.

    calendar change -> webhook -> sync_calendar -> refresh_due_checks
    scheduler, every minute -> run_due_checks -> plan_leave_by -> email (SNS)
    scheduler, hourly -> renew_watches

A due check is one row per upcoming timed event with a location. The email goes
out YOHO_ALERT_LEAD_MINUTES before the rider has to *leave* -- which depends on
the trip, unknown when the calendar syncs. So each check runs in two phases:

  plan   PLAN_AHEAD_SEC before the event: route the trip to learn the leave
         time, then move the check to (leave time - lead). If that moment has
         already passed (a late-added event, a very long trip), send right away.
  send   at (leave time - lead): route again with fresh data, check the route's
         stations for live alerts and delays, and email it with a link to
         /trip/<trip_id>, where the signed-in rider sees the same trip on the map.

Checks live in the store, not in a process, so any number of web workers can
write them, one scheduler reads them, and a restart loses nothing.

Alerts are planned in code (accounts/trips.py), not by the chat agent: no
tokens spent, and the directions in the email can't be garbled by the model.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import uuid

from accounts.calendar_sync import user_credentials
from accounts.trips import Geocoder, Router, event_location, local_label, parse_event_time, plan_leave_by, resolve_start
from agent.directions import HOME, _line_label, format_directions, route_payload
from integrations.google import CalendarAccessRevoked
from integrations.google.calendar import start_watch, stop_watch
from storage import FieldCipher, UserStore

log = logging.getLogger(__name__)

ALERT_LEAD_SEC = int(os.environ.get("YOHO_ALERT_LEAD_MINUTES", "60")) * 60
# The plan phase runs this long before the event: longer than any trip plus the lead.
PLAN_AHEAD_SEC = 4 * 3600
MAX_ATTEMPTS = 3
RENEW_WITHIN_SEC = 24 * 3600
WEBHOOK_PATH = "/webhooks/google-calendar"


# --- due checks -------------------------------------------------------------

def due_check_id(uid: str, event_id: str) -> str:
    return f"{uid}__{event_id}"


def refresh_due_checks(store: UserStore, uid: str, now: int, lead_sec: int = ALERT_LEAD_SEC) -> int:
    """Rebuild the user's due checks from their stored events. A check for the
    same event start is kept as it is -- sent stays sent (a re-sync doesn't
    re-alert), a planned send time stays planned; a new or moved event gets a
    fresh check in the plan phase. Returns how many are pending."""
    existing = {c["id"]: c for c in store.list_due_checks(uid)}
    checks = []
    for event in store.list_calendar_events(uid):
        start = parse_event_time(event.get("start"))
        if event.get("all_day") or start is None or start.timestamp() <= now or not event.get("location_enc"):
            continue  # nothing to alert: all-day, started, or nowhere to route to
        cid = due_check_id(uid, event["id"])
        prev = existing.get(cid)
        if prev and prev.get("event_start") == event["start"]:
            checks.append(prev)
            continue
        checks.append({"id": cid, "uid": uid, "event_id": event["id"], "event_start": event["start"],
                       "check_at": int(start.timestamp()) - PLAN_AHEAD_SEC, "phase": "plan",
                       "status": "pending", "attempts": 0})
    store.replace_due_checks(uid, checks)
    return sum(c["status"] == "pending" for c in checks)


def _clock(ts: int) -> str:
    """e.g. "8:14 AM" (the email is about one day's trip)."""
    return local_label(ts).split(", ")[-1]


def _status_lines(problems: list[dict] | None) -> list[str]:
    """The email's live-status section: route_problems' findings, "all clear",
    or that live data couldn't be checked (None)."""
    if problems is None:
        return ["Live delays and service alerts couldn't be checked, so this route is based on the schedule."]
    if not problems:
        return ["No delays or service alerts are reported on your route right now."]
    lines = ["Delays and service alerts on your route:"]
    for p in problems:
        notes = [f"{_line_label(d['mode'], d['route'])} is running about {d['delay_min']} min late"
                 for d in p["delays"]]
        notes += p["alerts"] or ([f"service alert ({', '.join(p['alert_types'])})"] if p["alert_types"]
                                 else ["service alert"] if not p["delays"] else [])
        lines.append(f"- {p['name']}: " + "; ".join(notes))
    if any(p["delays"] for p in problems):
        lines.append("Allow extra time: the ETA above doesn't include these delays.")
    return lines


def compose_alert(plan: dict, summary: str | None, location: str | None, now: int,
                  problems: list[dict] | None = None) -> tuple[str, str]:
    """(subject, body) for one alert email. `problems` is route_problems' result
    for the route, or None when live status couldn't be checked:

        Subject: YoHo Alert: Dentist (delays on your route)
        Your trip to Barclays Center is on schedule. To make sure you get there on
        time, be sure to follow this route from home at 8:14 AM:
        <directions>
        ETA: 9:00 AM

        Delays and service alerts on your route:
        - Atlantic Av-Barclays Ctr: the 2 train is running about 6 min late
        Allow extra time: the ETA above doesn't include these delays.
    """
    what = summary or "your event"
    where = location or what
    start = (f"your current event ({plan['event_summary']})" if plan.get("start_source") == "event"
             and plan.get("event_summary") else "your current event" if plan.get("start_source") == "event"
             else "home")
    directions = format_directions(plan["steps"], plan["total_time_sec"], plan.get("walk_in_sec"),
                                   plan.get("walk_out_sec"))
    if plan["slack_min"] >= 0:
        intro = (f"Your trip to {where} is on schedule. To make sure you get there on time, "
                 f"be sure to follow this route from {start} at {_clock(plan['leave_by_ts'])}:")
        eta = plan["leave_by_ts"] + round(plan["total_time_sec"])
    else:
        intro = (f"Your trip to {where} is running behind: you'd have needed to leave at "
                 f"{_clock(plan['leave_by_ts'])}. Leave now and follow this route from {start}:")
        eta = now + round(plan["total_time_sec"])
    lines = [intro, "", directions, "", f"ETA: {_clock(eta)}", "", *_status_lines(problems)]
    if plan.get("start_note"):
        lines += ["", f"Note: {plan['start_note']}."]
    flag = (" (delays on your route)" if any(p["delays"] for p in problems or [])
            else " (service alert on your route)" if problems else "")
    return f"YoHo Alert: {what}{flag}", "\n".join(lines)


def _live_problems(steps: list[dict]) -> list[dict] | None:
    """route_problems for the email, or None when the snapshot service is down:
    the alert still goes out, saying live status wasn't checked."""
    from agent.station_status import route_problems
    try:
        return route_problems(steps)
    except Exception as exc:
        log.warning("live route status unavailable: %s", exc)
        return None


def _plan_from_departure(store, cipher, uid, destination, arrive: int, now: int, *, geocode, route) -> dict:
    """plan_leave_by with the trip starting where the rider will be when they
    *leave*, not where they are when the check runs: at 5:30 they may still be at
    a museum they'll have left by a 7:20 departure. Routed from now's location
    first, then from the location at that leave time; if the two disagree (a
    start near the hour-after-an-event cutoff) the earlier leave time wins."""
    first = plan_leave_by(store, cipher, uid, destination, arrive, now, geocode=geocode, route=route)
    if "error" in first:
        raise ValueError(first["message"])
    start_then, info = resolve_start(store, cipher, uid, max(now, first["leave_by_ts"]), geocode)
    if start_then is None or tuple(start_then) == tuple(first["start_point"]):
        return first  # same place either way: no second route
    second = plan_leave_by(store, cipher, uid, destination, arrive, now, geocode=geocode, route=route,
                           start=tuple(start_then))
    second.update(info)  # report "home"/"event" rather than "given"
    return min(first, second, key=lambda p: p["leave_by_ts"])


def trip_field(trip_id: str) -> str:
    """The cipher field an emailed trip (due check trip_enc) is bound to."""
    return f"trip:{trip_id}"


def emailed_trip(store: UserStore, cipher: FieldCipher, uid: str, trip_id: str) -> dict | None:
    """The trip emailed to `uid` under `trip_id` -- {"subject", "body", "sent_at",
    "route"} -- or None. Only the user's own checks are searched, so a link
    forwarded to someone else shows them nothing. It lasts as long as the
    check: until a calendar sync after the event starts, or the event moves."""
    for check in store.list_due_checks(uid):
        if check.get("trip_id") == trip_id and check.get("trip_enc"):
            return cipher.decrypt_json(check["trip_enc"], user_id=uid, field=trip_field(trip_id))
    return None


def run_due_checks(store: UserStore, cipher: FieldCipher, notifier, now: int,
                   *, geocode: Geocoder, route: Router, route_status=None,
                   link_base: str | None = None) -> list[dict]:
    """Run every pending check that's due, in its phase (see the module doc).
    `route_status(steps)` gives the live problems on a route to send, or None when
    unknown; defaults to the snapshot service's alerts and delays. `link_base` is
    the web app's public origin for the email's trip link; None leaves it out.
    Returns one {"id", "status", ...} per check handled: "planned" (moved to its
    send time, with "send_at"), "sent", "expired" or a failure. A failure stays
    pending and is retried next run, up to MAX_ATTEMPTS; a check whose event has
    already started expires unsent."""
    results = []
    for check in store.due_checks_before(now):
        if check.get("status") != "pending":
            continue
        uid, cid = check["uid"], check["id"]
        event = next((e for e in store.list_calendar_events(uid) if e.get("id") == check["event_id"]), None)
        start = parse_event_time(event.get("start")) if event else None
        if event is None or start is None or start.timestamp() <= now:
            store.update_due_check(cid, {"status": "expired"})
            results.append({"id": cid, "status": "expired"})
            continue
        try:
            address = event_location(store, cipher, uid, event)
            destination = geocode(address) if address else None
            if not destination:
                raise ValueError(f"couldn't geocode the event location {address!r}")
            plan = _plan_from_departure(store, cipher, uid, tuple(destination), int(start.timestamp()), now,
                                        geocode=geocode, route=route)
            send_at = plan["leave_by_ts"] - ALERT_LEAD_SEC
            if check.get("phase", "plan") == "plan" and send_at > now:
                store.update_due_check(cid, {"phase": "send", "check_at": send_at, "attempts": 0,
                                             "planned_leave_by": plan["leave_by"]})
                results.append({"id": cid, "status": "planned", "send_at": send_at, "leave_by": plan["leave_by"]})
                continue
            problems = (route_status or _live_problems)(plan["steps"])
            subject, body = compose_alert(plan, event.get("summary"), address, now, problems)
            trip_id = secrets.token_urlsafe(16)
            start_end = HOME if plan.get("start_source") == "home" else plan.get("start_point")
            trip = {"subject": subject, "body": body, "sent_at": local_label(now),
                    "route": route_payload(plan["steps"], plan["total_time_sec"], plan.get("walk_in_sec"),
                                           plan.get("walk_out_sec"), start_end, tuple(destination))}
            email = f"{body}\n\nSee this trip on your map: {link_base.rstrip('/')}/trip/{trip_id}" if link_base else body
            message_id = notifier.send(uid, subject, email)
        except Exception as exc:
            attempts = check.get("attempts", 0) + 1
            status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
            log.warning("due check %s attempt %d failed: %s", cid, attempts, exc)
            store.update_due_check(cid, {"attempts": attempts, "last_error": str(exc)[:500], "status": status})
            results.append({"id": cid, "status": status, "error": str(exc)})
            continue
        # Encrypted: the route holds the event's (and a start event's) coordinates.
        store.update_due_check(cid, {"status": "sent", "sent_at": now, "message_id": message_id,
                                     "leave_by": plan["leave_by"], "trip_id": trip_id,
                                     "trip_enc": cipher.encrypt_json(trip, user_id=uid, field=trip_field(trip_id))})
        results.append({"id": cid, "status": "sent", "subject": subject, "body": email, "trip_id": trip_id})
    return results


# --- watch channels ---------------------------------------------------------

def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def verify_channel(store: UserStore, channel_id: str | None, token: str | None) -> dict | None:
    """The stored channel if the id exists and the token matches, else None.
    Only a hash of the token is stored; compared in constant time."""
    if not channel_id or not token:
        return None
    channel = store.get_watch_channel(channel_id)
    if channel is None or not hmac.compare_digest(channel.get("token_hash", ""), token_hash(token)):
        return None
    return channel


def start_watch_for_user(store: UserStore, cipher: FieldCipher, oauth, uid: str, webhook_base_url: str) -> dict:
    """Open a new push channel for the user's primary calendar and store it."""
    channel_id, token = str(uuid.uuid4()), secrets.token_urlsafe(32)
    address = webhook_base_url.rstrip("/") + WEBHOOK_PATH
    watch = start_watch(user_credentials(store, cipher, oauth, uid), address, channel_id, token)
    store.save_watch_channel(channel_id, {"uid": uid, "resource_id": watch["resource_id"],
                                          "expiration": watch["expiration"], "token_hash": token_hash(token),
                                          "address": address})
    return {"id": channel_id, **watch}


def ensure_watch(store: UserStore, cipher: FieldCipher, oauth, uid: str, webhook_base_url: str, now: int) -> dict | None:
    """Start a channel unless the user already has one that isn't about to expire."""
    live = [c for c in store.list_watch_channels(uid) if c["expiration"] / 1000 > now + RENEW_WITHIN_SEC]
    return None if live else start_watch_for_user(store, cipher, oauth, uid, webhook_base_url)


def stop_channel(store: UserStore, cipher: FieldCipher, oauth, channel: dict) -> None:
    try:
        stop_watch(user_credentials(store, cipher, oauth, channel["uid"]), channel["id"], channel["resource_id"])
    except Exception as exc:  # already expired or revoked: dropping our row is all that's left
        log.info("stopping channel %s: %s", channel["id"], exc)
    store.delete_watch_channel(channel["id"])


def renew_watches(store: UserStore, cipher: FieldCipher, oauth, webhook_base_url: str, now: int) -> list[str]:
    """Replace channels expiring within RENEW_WITHIN_SEC: start the new one
    first, then stop the old, so no change falls in a gap. Users whose Google
    access was revoked are marked revoked and lose their channels."""
    renewed = []
    for channel in store.list_watch_channels():
        if channel["expiration"] / 1000 > now + RENEW_WITHIN_SEC:
            continue
        uid = channel["uid"]
        try:
            start_watch_for_user(store, cipher, oauth, uid, webhook_base_url)
        except CalendarAccessRevoked:
            store.update_user(uid, {"google": {"revoked": True}})
            store.delete_watch_channel(channel["id"])
            continue
        except Exception:
            log.exception("renewing channel %s failed; will retry", channel["id"])
            continue
        stop_channel(store, cipher, oauth, channel)
        renewed.append(uid)
    return renewed


def ensure_email_subscription(store: UserStore, notifier, uid: str) -> str | None:
    """Subscribe the user's sign-in email to their alerts once."""
    user = store.get_user(uid) or {}
    if not user.get("email") or (user.get("alerts") or {}).get("subscription"):
        return None
    arn = notifier.subscribe_email(uid, user["email"])
    store.update_user(uid, {"alerts": {"subscription": arn or "pending confirmation"}})
    return arn


__all__ = ["ALERT_LEAD_SEC", "WEBHOOK_PATH", "compose_alert", "due_check_id", "emailed_trip", "ensure_email_subscription",
           "ensure_watch", "local_label", "refresh_due_checks", "renew_watches", "run_due_checks",
           "start_watch_for_user", "stop_channel", "verify_channel"]
