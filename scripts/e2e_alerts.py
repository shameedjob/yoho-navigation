"""End-to-end run of calendar change -> webhook -> due check -> route -> email.

Every step runs the real code path. Each outside service is real when its keys
are configured (.env, see web/config.py and .env.example) and faked otherwise,
so the same command works before and after you add keys:

  service           real when                                     otherwise
  Google Calendar   --uid USER (a user who signed in via           a demo user and event in memory
                    `python -m web`, YOHO_STORE=firestore)
  Calendar watch    --uid and YOHO_WEBHOOK_BASE_URL                a stored fake channel
  Routing           always (schedule graph + live waits from the   --fake-route
                    snapshot service; no torch)
  Geocoding         always (Nominatim)                             --fake-route
  Email             YOHO_SNS_TOPIC_ARN + AWS keys                  logged (dry run)

    python -m scripts.e2e_alerts                        # all local
    python -m scripts.e2e_alerts --email you@x.com      # SNS: subscribe; click the AWS link, run again
    python -m scripts.e2e_alerts --uid <google sub>     # your real calendar and a real watch channel

The webhook is delivered through Flask's test client, carrying the channel's
real id and token, so validation and re-sync run exactly as for Google's POST.
To see Google itself call it, run `python -m web` behind the public URL in
YOHO_WEBHOOK_BASE_URL and edit an event.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from accounts import alerts
from accounts.calendar_sync import REFRESH_TOKEN_FIELD
from accounts.home import Home, load_home, save_home
from accounts.trips import local_label, schedule_router, transit_geocode
from integrations.aws import LogNotifier, SnsNotifier
from integrations.google.calendar import start_watch
from integrations.google.oauth import SCOPES
from storage import FieldCipher, MemoryStore
from storage.crypto import generate_key_entry

DEMO_UID = "demo-user"
DEMO_HOME = Home("demo home (Crown Heights)", 40.6782, -73.9442)
DEMO_EVENT_LOCATION = "10 Union Sq E, New York, NY"


def step(n: int, text: str) -> None:
    print(f"\n[{n}] {text}")


def fake_route(start, end, arrive):
    steps = [{"stop_id": "250S", "stop_name": "Crown Hts-Utica Av", "mode": "subway", "route": "4", "lat": 40.6689, "lon": -73.9329},
             {"stop_id": "239S", "stop_name": "Franklin Av-Medgar Evers College", "mode": "subway", "route": "4", "lat": 40.6707, "lon": -73.9581},
             {"stop_id": "635S", "stop_name": "14 St-Union Sq", "mode": "subway", "route": "4", "lat": 40.7347, "lon": -73.9900}]
    return {"steps": steps, "total_time_sec": 1800, "walk_in_sec": 300, "walk_out_sec": 120, "live": False, "alerts": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uid", help="a real signed-in user (Google sub) in Firestore; default: in-memory demo user")
    parser.add_argument("--email", help="subscribe this address to the demo user's SNS alerts")
    parser.add_argument("--fake-route", action="store_true", help="skip geocoding and the transit graph")
    parser.add_argument("--dry-run-email", action="store_true", help="log the email even if SNS is configured")
    parser.add_argument("--keep-watch", action="store_true", help="leave the real watch channel open afterwards")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from web import create_app
    from web.__main__ import _load_dotenv
    from web.config import Settings
    _load_dotenv()

    real_google = bool(args.uid)
    sns_topic = None if args.dry_run_email else os.environ.get("YOHO_SNS_TOPIC_ARN")
    webhook_base = os.environ.get("YOHO_WEBHOOK_BASE_URL")

    step(1, "Services")
    if real_google:
        settings = Settings.from_env()
        if settings.store != "firestore":
            print("  --uid needs YOHO_STORE=firestore: the user signed in through another process")
            return 2
        from storage.firestore_store import FirestoreStore
        store = FirestoreStore(settings.firebase_credentials, settings.firebase_project_id)
        oauth = None  # create_app builds the real GoogleOAuth from settings
    else:
        settings = Settings(secret_key="e2e", google_client_id="e2e", google_client_secret="e2e",
                            google_redirect_uri="http://localhost/cb", data_keys=generate_key_entry("e2e"),
                            store="memory", dev=True, sns_topic_arn=sns_topic,
                            aws_region=os.environ.get("AWS_REGION"))
        store = MemoryStore()
        oauth = SimpleNamespace(credentials=lambda refresh_token, scopes: SimpleNamespace(refresh_token=refresh_token))
    notifier = SnsNotifier(sns_topic, os.environ.get("AWS_REGION")) if sns_topic else LogNotifier()
    app = create_app(settings, store=store, oauth=oauth, notifier=notifier, run_background=lambda fn: fn(),
                     agent_factory=lambda *a, **k: None)
    svc = app.extensions["yoho"]
    cipher: FieldCipher = svc.cipher
    route, geocode = (fake_route, lambda a: (40.7347, -73.9900)) if args.fake_route else (schedule_router, transit_geocode)
    print(f"  Google Calendar: {'real' if real_google else 'fake (demo user)'}")
    print(f"  watch channel:   {'real' if real_google and webhook_base else 'fake'}")
    print(f"  routing:         {'fake' if args.fake_route else 'real (schedule graph + live first-train waits; no torch)'}")
    print(f"  email:           {'SNS ' + sns_topic if sns_topic else 'dry run (logged)'}")

    step(2, "User")
    if real_google:
        uid = args.uid
        user = store.get_user(uid)
        if user is None:
            print(f"  no user {uid!r} in Firestore -- sign in once with `python -m web`")
            return 2
        print(f"  {user.get('email')} (home {'set' if load_home(store, cipher, uid) else 'NOT set -- set it on the Profile page'})")
    else:
        uid = DEMO_UID
        start = datetime.now(timezone.utc) + timedelta(seconds=alerts.ALERT_LEAD_SEC + 20 * 60)
        store.upsert_user(uid, {"email": args.email or "demo@example.com", "google": {
            "refresh_token_enc": cipher.encrypt("demo-refresh-token", user_id=uid, field=REFRESH_TOKEN_FIELD),
            "scopes": list(SCOPES)}})
        save_home(store, cipher, uid, DEMO_HOME)
        demo_items = [{"id": "e2edemo1", "status": "confirmed", "summary": "Demo dinner", "location": DEMO_EVENT_LOCATION,
                       "start": {"dateTime": start.isoformat()}, "end": {"dateTime": (start + timedelta(hours=1)).isoformat()}}]
        import accounts.calendar_sync
        accounts.calendar_sync.fetch_upcoming_events = lambda credentials, **_: demo_items
        print(f"  {uid}: home {DEMO_HOME.label}; Google will return 'Demo dinner' at {local_label(int(start.timestamp()))}")
    if sns_topic and (args.email or real_google):
        arn = alerts.ensure_email_subscription(store, notifier, uid)
        if arn:
            print(f"  subscribed {store.get_user(uid)['email']} to alerts: {arn}")
            print("  -> click the confirmation link AWS just emailed, then run this again to receive the alert")

    step(3, "Calendar change notification -> webhook")
    channel_id, token, real_channel = str(uuid.uuid4()), secrets.token_urlsafe(32), None
    if real_google and webhook_base:
        from accounts.calendar_sync import user_credentials
        address = webhook_base.rstrip("/") + alerts.WEBHOOK_PATH
        watch = start_watch(user_credentials(store, cipher, svc.oauth, uid), address, channel_id, token)
        real_channel = {"id": channel_id, "uid": uid, **watch}
        print(f"  opened a real watch channel -> {address} (expires {local_label(watch['expiration'] // 1000)})")
    else:
        watch = {"resource_id": "e2e-fake-resource", "expiration": int((time.time() + 7 * 86400) * 1000)}
        if real_google:
            print("  YOHO_WEBHOOK_BASE_URL unset: using a stored fake channel (Google won't call it)")
    store.save_watch_channel(channel_id, {"uid": uid, "resource_id": watch["resource_id"], "expiration": watch["expiration"],
                                          "token_hash": alerts.token_hash(token), "address": "e2e"})
    client = app.test_client()
    headers = {"X-Goog-Channel-ID": channel_id, "X-Goog-Channel-Token": token}
    for state in ("sync", "exists"):
        resp = client.post(alerts.WEBHOOK_PATH, headers={**headers, "X-Goog-Resource-State": state})
        print(f"  POST {alerts.WEBHOOK_PATH} state={state}: {resp.status_code}")
    bad = client.post(alerts.WEBHOOK_PATH, headers={**headers, "X-Goog-Channel-Token": "wrong", "X-Goog-Resource-State": "exists"})
    print(f"  POST with a wrong token: {bad.status_code} (expected 404)")
    events = store.list_calendar_events(uid)
    print(f"  re-synced {len(events)} events; synced at {store.get_user(uid).get('calendar_synced_at')}")

    step(4, "Due checks")
    pending = sorted((c for c in store.list_due_checks(uid) if c["status"] == "pending"), key=lambda c: c["check_at"])
    if not pending:
        print("  nothing to alert: no upcoming timed event with a location in the next two weeks")
        return 1
    for c in pending[:5]:
        summary = next((e.get("summary") for e in events if e["id"] == c["event_id"]), "?")
        print(f"  {summary!r}: check at {local_label(c['check_at'])}")
    check = pending[0]
    now = max(int(time.time()), check["check_at"])
    if now > time.time():
        print(f"  simulating the clock at {local_label(now)} (the first check's time)")

    step(5, "Scheduler: run due checks -> route -> email")
    results = alerts.run_due_checks(store, cipher, notifier, now, geocode=geocode, route=route)
    for r in results:
        print(f"  {r['id']}: {r['status']}" + (f" -- {r['error']}" if r.get("error") else ""))
        if r["status"] == "sent":
            print(f"\n  Subject: {r['subject']}\n  " + r["body"].replace("\n", "\n  "))
    sent = next((c for c in store.list_due_checks(uid) if c["id"] == check["id"]), {})
    if sent.get("message_id"):
        print(f"\n  message id: {sent['message_id']}")

    if real_channel and not args.keep_watch:
        alerts.stop_channel(store, cipher, svc.oauth, store.get_watch_channel(channel_id))
        print("\n  stopped the real watch channel (--keep-watch to leave it open)")
    elif not real_channel:
        store.delete_watch_channel(channel_id)
    return 0 if any(r["status"] == "sent" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
