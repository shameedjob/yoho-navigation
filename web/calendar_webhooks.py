"""Google Calendar push notifications: POST /webhooks/google-calendar.

Google sends an empty body; everything is in X-Goog-* headers. A channel we
don't know or a wrong token gets 404. The first notification on a new channel
("sync") is only a handshake. Anything else means the calendar changed: re-sync
it in the background (which also refreshes alert due checks) and answer 200
right away, since Google retries slow or failed deliveries.

The endpoint has no session: the per-channel secret token is the credential.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, request

from accounts.alerts import WEBHOOK_PATH, verify_channel
from accounts.calendar_sync import CalendarNotConnected, sync_calendar
from integrations.google import CalendarAccessRevoked

log = logging.getLogger(__name__)
bp = Blueprint("calendar_webhooks", __name__)


@bp.post(WEBHOOK_PATH)
def google_calendar():
    svc = current_app.extensions["yoho"]
    channel = verify_channel(svc.store, request.headers.get("X-Goog-Channel-ID"),
                             request.headers.get("X-Goog-Channel-Token"))
    if channel is None:
        return "", 404
    state = request.headers.get("X-Goog-Resource-State", "")
    if state == "sync":
        return "", 200

    uid = channel["uid"]
    store, cipher, oauth = svc.store, svc.cipher, svc.oauth

    def resync():
        try:
            sync_calendar(store, cipher, oauth, uid)
        except (CalendarNotConnected, CalendarAccessRevoked):
            log.info("calendar no longer connected for channel %s", channel["id"])
        except Exception:
            log.exception("webhook re-sync failed for channel %s", channel["id"])

    svc.run_background(resync)
    return "", 200
