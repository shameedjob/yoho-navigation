from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from accounts.alerts import (ALERT_LEAD_SEC, PLAN_AHEAD_SEC, due_check_id, ensure_email_subscription, refresh_due_checks,
                             renew_watches, run_due_checks, token_hash)
from accounts.home import Home, save_home
from integrations.aws import LogNotifier, SnsNotifier
from integrations.google import CalendarAccessRevoked
from tests.test_web import UID, WRITE, env, login  # noqa: F401  (env is a fixture)

NOW = int(datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc).timestamp())
DENTIST = (40.735, -73.99)
STEPS = [{"stop_id": "1", "stop_name": "Bedford Av", "mode": "subway", "route": "L", "lat": 40.71, "lon": -73.95},
         {"stop_id": "2", "stop_name": "1 Av", "mode": "subway", "route": "L", "lat": 40.73, "lon": -73.98},
         {"stop_id": "3", "stop_name": "Union Sq", "mode": "subway", "route": "L", "lat": 40.735, "lon": -73.99}]


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def event(cipher, eid, start, minutes=60, location="10 Union Sq E, New York", all_day=False):
    return {"id": eid, "summary": eid.title(), "start": iso(start), "end": iso(start + minutes * 60),
            "all_day": all_day,
            "location_enc": cipher.encrypt(location, user_id=UID, field=f"event_location:{eid}") if location else None}


def fake_route(calls):
    def route(start, end, arrive):
        calls.append((start, end, arrive))
        return {"steps": STEPS, "total_time_sec": 1500, "walk_in_sec": 120, "walk_out_sec": 60,
                "live": True, "alerts": []}
    return route


def geocode(address):
    return DENTIST if "Union Sq" in address else None


@pytest.fixture
def user(env):  # noqa: F811
    login(env)
    save_home(env.store, env.cipher, UID, Home("home", 40.71, -73.95))
    return env


# --- due checks ---------------------------------------------------------------

def test_refresh_due_checks_picks_timed_future_events_with_a_location(user):
    c = user.cipher
    user.store.replace_calendar_events(UID, [
        event(c, "dentist", NOW + 3 * 3600),
        event(c, "started", NOW - 600),
        event(c, "holiday", NOW + 3600, all_day=True),
        event(c, "call", NOW + 3600, location=None),
    ])
    assert refresh_due_checks(user.store, UID, NOW) == 1
    [check] = user.store.list_due_checks(UID)
    assert check["id"] == due_check_id(UID, "dentist")
    assert check["check_at"] == NOW + 3 * 3600 - PLAN_AHEAD_SEC and check["phase"] == "plan" and check["status"] == "pending"


def test_resync_keeps_sent_checks_but_reschedules_moved_events(user):
    c = user.cipher
    user.store.replace_calendar_events(UID, [event(c, "dentist", NOW + 3 * 3600), event(c, "gym", NOW + 5 * 3600)])
    refresh_due_checks(user.store, UID, NOW)
    for cid in (due_check_id(UID, "dentist"), due_check_id(UID, "gym")):
        user.store.update_due_check(cid, {"status": "sent"})
    user.store.replace_calendar_events(UID, [event(c, "dentist", NOW + 3 * 3600), event(c, "gym", NOW + 6 * 3600)])
    assert refresh_due_checks(user.store, UID, NOW) == 1
    status = {x["event_id"]: x["status"] for x in user.store.list_due_checks(UID)}
    assert status == {"dentist": "sent", "gym": "pending"}


# --- the alert job ------------------------------------------------------------------

def test_due_check_sends_one_email_with_directions_then_stops(user):
    user.store.replace_calendar_events(UID, [event(user.cipher, "dentist", NOW + 3600)])
    refresh_due_checks(user.store, UID, NOW - 3600)
    notifier, calls = LogNotifier(), []
    [result] = run_due_checks(user.store, user.cipher, notifier, NOW, geocode=geocode, route=fake_route(calls))
    assert result["status"] == "sent"
    assert calls == [((40.71, -73.95), DENTIST, NOW + 3600)]  # from home, arriving at the start
    [mail] = notifier.sent
    assert mail["uid"] == UID and mail["subject"] == "YoHo Alert: Dentist"
    assert "Take the L train 2 stops, Bedford Av to Union Sq." in mail["body"]
    assert "follow this route from home" in mail["body"] and "40.71" not in mail["body"]  # no home coordinates
    assert run_due_checks(user.store, user.cipher, notifier, NOW + 60, geocode=geocode, route=fake_route([])) == []
    assert len(notifier.sent) == 1


def test_trip_starts_at_the_current_event(user):
    c = user.cipher
    user.store.replace_calendar_events(UID, [
        event(c, "work", NOW - 3600, minutes=90, location="89 E 42nd St, New York"),
        event(c, "dentist", NOW + 3600),
    ])
    refresh_due_checks(user.store, UID, NOW - 7200)
    calls, notifier = [], LogNotifier()
    geo = lambda a: (40.7527, -73.9772) if "42nd" in a else geocode(a)
    run_due_checks(user.store, user.cipher, notifier, NOW, geocode=geo, route=fake_route(calls))
    assert calls[0][0] == (40.7527, -73.9772)
    assert "follow this route from your current event (Work)" in notifier.sent[0]["body"]


def test_failures_retry_then_give_up_and_started_events_expire(user):
    user.store.replace_calendar_events(UID, [event(user.cipher, "dentist", NOW + 3600, location="nowhere")])
    refresh_due_checks(user.store, UID, NOW - 3600)
    notifier = LogNotifier()
    statuses = [run_due_checks(user.store, user.cipher, notifier, NOW + i, geocode=geocode, route=fake_route([]))[0]["status"]
                for i in range(3)]
    assert statuses == ["pending", "pending", "failed"] and notifier.sent == []

    user.store.replace_calendar_events(UID, [event(user.cipher, "gym", NOW + 60)])
    refresh_due_checks(user.store, UID, NOW - 3600)
    [result] = run_due_checks(user.store, user.cipher, notifier, NOW + 120, geocode=geocode, route=fake_route([]))
    assert result["status"] == "expired"


# --- webhook ------------------------------------------------------------------

HEADERS = {"X-Goog-Channel-ID": "chan-1", "X-Goog-Channel-Token": "secret-token", "X-Goog-Resource-State": "exists"}


def add_channel(env):  # noqa: F811
    env.store.save_watch_channel("chan-1", {"uid": UID, "resource_id": "res-1", "token_hash": token_hash("secret-token"),
                                            "expiration": (NOW + 7 * 86400) * 1000})


def test_webhook_rejects_unknown_channels_and_bad_tokens(user):
    add_channel(user)
    assert user.client.post("/webhooks/google-calendar", headers={**HEADERS, "X-Goog-Channel-ID": "nope"}).status_code == 404
    assert user.client.post("/webhooks/google-calendar", headers={**HEADERS, "X-Goog-Channel-Token": "wrong"}).status_code == 404
    assert user.client.post("/webhooks/google-calendar", headers={k: v for k, v in HEADERS.items()
                                                                  if k != "X-Goog-Channel-Token"}).status_code == 404


def test_webhook_sync_handshake_does_nothing(user):
    add_channel(user)
    ran = []
    user.app.extensions["yoho"].run_background = ran.append
    assert user.client.post("/webhooks/google-calendar", headers={**HEADERS, "X-Goog-Resource-State": "sync"}).status_code == 200
    assert ran == []


def test_webhook_change_resyncs_calendar_and_due_checks(user):
    add_channel(user)
    user.app.extensions["yoho"].run_background = lambda fn: fn()
    start = datetime.now(timezone.utc) + timedelta(hours=4)
    user.fetch["items"] = [{"id": "ev9", "status": "confirmed", "summary": "Lunch", "location": "10 Union Sq E",
                            "start": {"dateTime": start.isoformat()}, "end": {"dateTime": (start + timedelta(hours=1)).isoformat()}}]
    assert user.client.post("/webhooks/google-calendar", headers=HEADERS).status_code == 200
    assert [e["id"] for e in user.store.list_calendar_events(UID)] == ["ev9"]
    assert [c["event_id"] for c in user.store.list_due_checks(UID)] == ["ev9"]


# --- channels and SNS -----------------------------------------------------------

def test_renew_replaces_expiring_channels_and_drops_revoked(user, monkeypatch):
    add_channel(user)
    started, stopped = [], []
    monkeypatch.setattr("accounts.alerts.start_watch",
                        lambda creds, address, cid, token, **_: started.append(address) or {"resource_id": "res-2", "expiration": (NOW + 8 * 86400) * 1000})
    monkeypatch.setattr("accounts.alerts.stop_watch", lambda creds, cid, rid: stopped.append(cid))
    soon = NOW + 7 * 86400 - 3600  # expires within the day
    assert renew_watches(user.store, user.cipher, user.oauth, "https://yoho.example", soon) == [UID]
    assert started == ["https://yoho.example/webhooks/google-calendar"] and stopped == ["chan-1"]
    assert [c["resource_id"] for c in user.store.list_watch_channels(UID)] == ["res-2"]

    def revoked(*a, **k):
        raise CalendarAccessRevoked("invalid_grant")
    monkeypatch.setattr("accounts.alerts.start_watch", revoked)
    assert renew_watches(user.store, user.cipher, user.oauth, "https://yoho.example", NOW + 30 * 86400) == []
    assert user.store.list_watch_channels(UID) == [] and user.store.get_user(UID)["google"]["revoked"]


def test_sns_subscribes_with_uid_filter_and_publishes_with_uid_attribute(user):
    calls = []
    client = SimpleNamespace(subscribe=lambda **kw: calls.append(("subscribe", kw)) or {"SubscriptionArn": "pending confirmation"},
                             publish=lambda **kw: calls.append(("publish", kw)) or {"MessageId": "m-1"})
    sns = SnsNotifier("arn:aws:sns:us-east-1:123:yoho-alerts", client=client)
    # login already subscribed through the app's dry-run notifier; subscribing again is a no-op
    assert ensure_email_subscription(user.store, sns, UID) is None
    user.store.update_user(UID, {"alerts": None})
    assert ensure_email_subscription(user.store, sns, UID) == "pending confirmation"
    assert sns.send(UID, "x" * 150, "body") == "m-1"
    (_, sub), (_, pub) = calls
    assert sub["Protocol"] == "email" and sub["Endpoint"] == "rider@example.com"
    assert json.loads(sub["Attributes"]["FilterPolicy"]) == {"uid": [UID]}
    assert pub["MessageAttributes"]["uid"]["StringValue"] == UID and len(pub["Subject"]) == 100


def test_parse_local_time_understands_worded_times():
    from zoneinfo import ZoneInfo
    from accounts.trips import parse_local_time
    ny = ZoneInfo("America/New_York")
    now = int(datetime(2026, 9, 13, 22, 16, tzinfo=ny).timestamp())  # a Sunday night
    at = lambda *a: int(datetime(*a, tzinfo=ny).timestamp())
    assert parse_local_time("Monday 14:00", now) == at(2026, 9, 14, 14, 0)
    assert parse_local_time("monday at 2pm", now) == at(2026, 9, 14, 14, 0)
    assert parse_local_time("2pm", now) == at(2026, 9, 14, 14, 0)        # already past today -> tomorrow
    assert parse_local_time("today 23:00", now) == at(2026, 9, 13, 23, 0)
    assert parse_local_time("Sunday 20:00", now) == at(2026, 9, 20, 20, 0)  # tonight's has passed -> next week
    assert parse_local_time("2026-09-15 08:00", now) == at(2026, 9, 15, 8, 0)
    with pytest.raises(ValueError):
        parse_local_time("whenever", now)


def test_leave_by_uses_a_start_the_user_named_even_without_home(env):  # noqa: F811
    from accounts.trips import plan_leave_by
    login(env)  # no home set
    calls = []
    plan = plan_leave_by(env.store, env.cipher, UID, DENTIST, NOW + 3600, NOW,
                         geocode=geocode, route=fake_route(calls), start=(40.6847, -73.9773))
    assert plan["start_source"] == "given" and calls[0][0] == (40.6847, -73.9773)


def test_leave_by_tool_starts_from_the_device_location_only_for_soon_trips(env, monkeypatch):  # noqa: F811
    from zoneinfo import ZoneInfo
    from agent import agent_interaction as ai
    login(env)
    save_home(env.store, env.cipher, UID, Home("home", 40.6782, -73.9442))
    calls = []
    router = fake_route(calls)
    monkeypatch.setattr(ai, "transit_router", lambda start, end, arrive, avoid=None: router(start, end, arrive))
    monkeypatch.setattr(ai.time, "time", lambda: NOW)
    here = (40.7359, -73.9911)
    leave_by = {t.tool_name: t for t in ai.make_user_tools(UID, env.store, env.cipher, location=here)}["leave_by"]
    local = lambda ts: datetime.fromtimestamp(ts, ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")

    assert leave_by(destination=DENTIST, arrival_time=local(NOW + 3600))["start_source"] == "current_location"
    assert leave_by(destination=DENTIST, arrival_time=local(NOW + 86400))["start_source"] == "home"  # tomorrow
    assert [c[0] for c in calls] == [here, (40.6782, -73.9442)]


def test_alert_is_sent_an_hour_before_leaving_not_before_the_event(user):
    """plan phase learns the leave time; the send phase runs lead time before it."""
    user.store.replace_calendar_events(UID, [event(user.cipher, "dentist", NOW + 4 * 3600)])
    refresh_due_checks(user.store, UID, NOW - 3600)
    [check] = user.store.list_due_checks(UID)
    notifier, calls = LogNotifier(), []
    # the plan phase runs PLAN_AHEAD_SEC before the event and sends nothing
    [planned] = run_due_checks(user.store, user.cipher, notifier, check["check_at"], geocode=geocode, route=fake_route(calls))
    leave = NOW + 4 * 3600 - 1500  # the fake trip takes 25 min
    assert planned["status"] == "planned" and planned["send_at"] == leave - ALERT_LEAD_SEC and notifier.sent == []
    assert run_due_checks(user.store, user.cipher, notifier, planned["send_at"] - 60, geocode=geocode, route=fake_route(calls)) == []
    [sent] = run_due_checks(user.store, user.cipher, notifier, planned["send_at"], geocode=geocode, route=fake_route(calls))
    assert sent["status"] == "sent" and len(calls) == 2  # routed again with fresh data at send time
    body = notifier.sent[0]["body"]
    assert body.startswith("Your trip to 10 Union Sq E, New York is on schedule.") and body.rstrip().endswith("ETA: 12:00 PM")


def test_late_added_event_is_sent_immediately_and_says_leave_now(user):
    user.store.replace_calendar_events(UID, [event(user.cipher, "dentist", NOW + 600)])  # 10 min away, 25 min trip
    refresh_due_checks(user.store, UID, NOW)
    notifier = LogNotifier()
    [sent] = run_due_checks(user.store, user.cipher, notifier, NOW, geocode=geocode, route=fake_route([]))
    assert sent["status"] == "sent" and "running behind" in notifier.sent[0]["body"]
    assert "Leave now" in notifier.sent[0]["body"] and "ETA: 8:25 AM" in notifier.sent[0]["body"]
