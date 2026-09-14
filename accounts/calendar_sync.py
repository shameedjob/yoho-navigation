"""Sync a user's Google Calendar into the store (used after login and by
POST /api/calendar/sync)."""

from __future__ import annotations

from datetime import datetime, timezone

from integrations.google import CalendarAccessRevoked, GoogleOAuth
from integrations.google.calendar import fetch_upcoming_events, to_stored_events
from integrations.google.oauth import CALENDAR_SCOPE
from storage import FieldCipher, UserStore

REFRESH_TOKEN_FIELD = "google_refresh_token"


class CalendarNotConnected(Exception):
    """No usable Calendar grant: never granted, unticked at consent, or revoked."""


def calendar_connected(user: dict | None) -> bool:
    google = (user or {}).get("google") or {}
    return bool(google.get("refresh_token_enc")) and CALENDAR_SCOPE in google.get("scopes", []) and not google.get("revoked")


def user_credentials(store: UserStore, cipher: FieldCipher, oauth: GoogleOAuth, uid: str):
    """Google credentials from the user's stored refresh token."""
    user = store.get_user(uid)
    if not calendar_connected(user):
        raise CalendarNotConnected(uid)
    google = user["google"]
    refresh_token = cipher.decrypt(google["refresh_token_enc"], user_id=uid, field=REFRESH_TOKEN_FIELD)
    return oauth.credentials(refresh_token, google["scopes"])


def sync_calendar(store: UserStore, cipher: FieldCipher, oauth: GoogleOAuth, uid: str) -> int:
    """Replace the stored events with the next two weeks', and the user's alert
    due checks with them. Returns the event count."""
    from accounts.alerts import refresh_due_checks  # alerts imports this module

    credentials = user_credentials(store, cipher, oauth, uid)
    try:
        items = fetch_upcoming_events(credentials)
    except CalendarAccessRevoked:
        store.update_user(uid, {"google": {"revoked": True}})
        raise
    rows = to_stored_events(uid, items, cipher)
    store.replace_calendar_events(uid, rows)
    now = datetime.now(timezone.utc)
    store.update_user(uid, {"calendar_synced_at": now.isoformat()})
    refresh_due_checks(store, uid, int(now.timestamp()))
    return len(rows)
