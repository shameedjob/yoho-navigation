"""Proactive "time to leave" alerts for calendar events.

    calendar change -> webhook -> sync_calendar -> refresh_due_checks
    scheduler, every minute -> run_due_checks -> plan_leave_by -> email (SNS)
    scheduler, hourly -> renew_watches

A due check is one row per upcoming timed event with a location: check it at
event start minus YOHO_ALERT_LEAD_MINUTES. Checks live in the store, not in a
process, so any number of web workers can write them, one scheduler reads them,
and a restart loses nothing.

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
from accounts.trips import Geocoder, Router, event_location, local_label, parse_event_time, plan_leave_by
from agent.directions import format_directions
from integrations.google import CalendarAccessRevoked
from integrations.google.calendar import start_watch, stop_watch
from storage import FieldCipher, UserStore

log = logging.getLogger(__name__)

ALERT_LEAD_SEC = int(os.environ.get("YOHO_ALERT_LEAD_MINUTES", "60")) * 60
MAX_ATTEMPTS = 3
RENEW_WITHIN_SEC = 24 * 3600
WEBHOOK_PATH = "/webhooks/google-calendar"


# --- due checks -------------------------------------------------------------

def due_check_id(uid: str, event_id: str) -> str:
    return f"{uid}__{event_id}"


def refresh_due_checks(store: UserStore, uid: str, now: int, lead_sec: int = ALERT_LEAD_SEC) -> int:
    """Rebuild the user's due checks from their stored events. A check already
    sent for the same event start stays sent, so a re-sync doesn't re-alert; a
    moved event gets a fresh check. Returns how many are pending."""
    existing = {c["id"]: c for c in store.list_due_checks(uid)}
    checks = []
    for event in store.list_calendar_events(uid):
        start = parse_event_time(event.get("start"))
        if event.get("all_day") or start is None or start.timestamp() <= now or not event.get("location_enc"):
            continue  # nothing to alert: all-day, started, or nowhere to route to
        cid = due_check_id(uid, event["id"])
        prev = existing.get(cid)
        if prev and prev.get("event_start") == event["start"] and prev.get("status") != "pending":
            checks.append(prev)
            continue
        checks.append({"id": cid, "uid": uid, "event_id": event["id"], "event_start": event["start"],
                       "check_at": int(start.timestamp()) - lead_sec, "status": "pending", "attempts": 0})
    store.replace_due_checks(uid, checks)
    return sum(c["status"] == "pending" for c in checks)


def compose_alert(plan: dict, summary: str | None) -> tuple[str, str]:
    """(subject, body) for one alert email."""
    what = summary or "your event"
    if plan["slack_min"] < 0:
        subject = f"Running late for {what}: leave now"
    else:
        subject = f"Leave by {plan['leave_by'].split(', ')[-1]} for {what}"
    start = ("your current event" + (f" ({plan['event_summary']})" if plan.get("event_summary") else "")
             if plan.get("start_source") == "event" else "home")
    lines = [
        f"{what} starts {plan['arrive_by']}.",
        f"Leave by {plan['leave_by']} ({plan['slack_min']} min from now)" if plan["slack_min"] >= 0
        else f"You'd need to have left {-plan['slack_min']} min ago -- leave as soon as you can.",
        f"Starting from {start}.",
        "",
        format_directions(plan["steps"], plan["total_time_sec"], plan.get("walk_in_sec"), plan.get("walk_out_sec")),
        "",
        "Times use live MTA predictions." if plan.get("live")
        else "Times use the MTA schedule with the live wait for your first train." if plan.get("live_waits")
        else "Times use the MTA schedule (live data unavailable).",
    ]
    if plan.get("alerts"):
        lines.append("Service alerts on your route: " + ", ".join(a["stop_name"] for a in plan["alerts"]))
    if plan.get("start_note"):
        lines.append(f"Note: {plan['start_note']}.")
    return subject, "\n".join(lines)


def run_due_checks(store: UserStore, cipher: FieldCipher, notifier, now: int,
                   *, geocode: Geocoder, route: Router) -> list[dict]:
    """Send every pending check that's due. Returns one {"id", "status", ...}
    per check handled. A failure stays pending and is retried next run, up to
    MAX_ATTEMPTS; a check whose event has already started expires unsent."""
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
            plan = plan_leave_by(store, cipher, uid, tuple(destination), int(start.timestamp()), now,
                                 geocode=geocode, route=route)
            if "error" in plan:
                raise ValueError(plan["message"])
            subject, body = compose_alert(plan, event.get("summary"))
            message_id = notifier.send(uid, subject, body)
        except Exception as exc:
            attempts = check.get("attempts", 0) + 1
            status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
            log.warning("due check %s attempt %d failed: %s", cid, attempts, exc)
            store.update_due_check(cid, {"attempts": attempts, "last_error": str(exc)[:500], "status": status})
            results.append({"id": cid, "status": status, "error": str(exc)})
            continue
        store.update_due_check(cid, {"status": "sent", "sent_at": now, "message_id": message_id,
                                     "leave_by": plan["leave_by"]})
        results.append({"id": cid, "status": "sent", "subject": subject, "body": body})
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


__all__ = ["ALERT_LEAD_SEC", "WEBHOOK_PATH", "compose_alert", "due_check_id", "ensure_email_subscription",
           "ensure_watch", "local_label", "refresh_due_checks", "renew_watches", "run_due_checks",
           "start_watch_for_user", "stop_channel", "verify_channel"]
