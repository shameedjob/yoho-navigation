"""Pull a user's upcoming Google Calendar events.

This is the on-demand sync (POST /api/calendar/sync, and right after login):
a full re-read of the primary calendar over a fixed look-ahead window. Push
notifications plus incremental syncTokens (docs/PLAN_FLASK_GOOGLE.md, phases
4-5) can replace it later; Google doesn't allow syncToken together with the
timeMin/timeMax window used here, so that's a separate code path, not a flag.

Event locations are encrypted before they're stored -- like the home address,
a location tied to a time is exactly the data worth protecting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build

from storage import FieldCipher

LOOKAHEAD_DAYS = 14
MAX_EVENTS = 500


class CalendarAccessRevoked(Exception):
    """The refresh token no longer works; the user has to reconnect Google."""


def fetch_upcoming_events(credentials, now: datetime | None = None, days: int = LOOKAHEAD_DAYS) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
    items: list[dict] = []
    page_token = None
    try:
        while len(items) < MAX_EVENTS:
            resp = service.events().list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=(now + timedelta(days=days)).isoformat(),
                singleEvents=True,  # expand recurring events into instances
                orderBy="startTime",
                maxResults=250,
                pageToken=page_token,
                fields="items(id,status,summary,location,start,end,updated),nextPageToken",
            ).execute()
            items.extend(resp.get("items", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except RefreshError as e:
        raise CalendarAccessRevoked(str(e)) from e
    finally:
        service.close()
    return items[:MAX_EVENTS]


def to_stored_events(uid: str, items: list[dict], cipher: FieldCipher) -> list[dict]:
    """Google event resources -> storage rows. Cancelled events are dropped;
    all-day events keep their date (no time) in start/end."""
    rows = []
    for item in items:
        if item.get("status") == "cancelled":
            continue
        start, end = item.get("start", {}), item.get("end", {})
        location = item.get("location")
        rows.append({
            "id": item["id"],
            "summary": item.get("summary", ""),
            "start": start.get("dateTime") or start.get("date"),
            "end": end.get("dateTime") or end.get("date"),
            "all_day": "date" in start and "dateTime" not in start,
            "location_enc": cipher.encrypt(location, user_id=uid, field=f"event_location:{item['id']}") if location else None,
            "updated": item.get("updated"),
        })
    return rows


WATCH_TTL_SEC = 7 * 24 * 3600  # Google's cap for events.watch channels


def start_watch(credentials, address: str, channel_id: str, token: str, ttl_sec: int = WATCH_TTL_SEC) -> dict:
    """Ask Google to POST to `address` when the primary calendar changes.
    `address` must be HTTPS on a domain verified for the OAuth project. Returns
    {"resource_id", "expiration" (Unix ms)}."""
    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
    try:
        resp = service.events().watch(calendarId="primary", body={
            "id": channel_id, "type": "web_hook", "address": address, "token": token,
            "params": {"ttl": str(ttl_sec)},
        }).execute()
    except RefreshError as e:
        raise CalendarAccessRevoked(str(e)) from e
    finally:
        service.close()
    return {"resource_id": resp["resourceId"], "expiration": int(resp["expiration"])}


def stop_watch(credentials, channel_id: str, resource_id: str) -> None:
    service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
    try:
        service.channels().stop(body={"id": channel_id, "resourceId": resource_id}).execute()
    except RefreshError as e:
        raise CalendarAccessRevoked(str(e)) from e
    finally:
        service.close()
