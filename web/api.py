"""JSON API used by the frontend pages. Every route needs a signed-in session;
writes also need the X-Yoho-Request: 1 header (see auth.login_required_api)."""

from __future__ import annotations

import logging
import threading

from flask import Blueprint, jsonify, request, session

from accounts.calendar_sync import REFRESH_TOKEN_FIELD, CalendarNotConnected, calendar_connected, sync_calendar
from accounts.home import Home, HomeOutOfArea, clear_home, has_home, in_service_area, save_home
from accounts.quota import quota_status, record_usage
from integrations.google import CalendarAccessRevoked

from .auth import login_required_api, services

log = logging.getLogger(__name__)
bp = Blueprint("api", __name__, url_prefix="/api")

MAX_MESSAGE_CHARS = 2000
MAX_ADDRESS_CHARS = 200

# One agent call per user at a time, per process. With several gunicorn
# workers a user could run one per worker; see accounts/quota.py for why the
# overshoot this allows stays small.
_inflight: set[str] = set()
_inflight_lock = threading.Lock()


def _error(code: str, status: int, **extra):
    return jsonify(error=code, **extra), status


@bp.get("/me")
@login_required_api
def me(uid: str):
    svc = services()
    user = svc.store.get_user(uid)
    return jsonify(
        email=user.get("email"),
        name=user.get("name"),
        picture=user.get("picture"),
        home_set=has_home(user),
        calendar={"connected": calendar_connected(user), "synced_at": user.get("calendar_synced_at")},
        usage=quota_status(svc.store, uid, svc.settings.weekly_token_limit, user).as_dict(),
    )


@bp.put("/me/home")
@login_required_api
def set_home(uid: str):
    """Body: {"lat": .., "lon": ..} from browser geolocation, or {"address": ".."}.
    The response says only that home is set; the location isn't echoed back."""
    svc = services()
    body = request.get_json(silent=True) or {}
    if "lat" in body and "lon" in body:
        try:
            home = Home(label=None, lat=float(body["lat"]), lon=float(body["lon"]))
        except (TypeError, ValueError):
            return _error("invalid_coordinates", 400)
    elif isinstance(body.get("address"), str) and body["address"].strip():
        address = body["address"].strip()[:MAX_ADDRESS_CHARS]
        coords = svc.geocode(address)
        if coords is None:
            return _error("address_not_found", 422)
        home = Home(label=address, lat=coords[0], lon=coords[1])
    else:
        return _error("expected lat/lon or address", 400)
    try:
        save_home(svc.store, svc.cipher, uid, home)
    except HomeOutOfArea:
        return _error("outside_service_area", 422)
    return jsonify(home_set=True)


@bp.delete("/me/home")
@login_required_api
def delete_home(uid: str):
    clear_home(services().store, uid)
    return jsonify(home_set=False)


@bp.post("/calendar/sync")
@login_required_api
def calendar_sync(uid: str):
    svc = services()
    try:
        count = sync_calendar(svc.store, svc.cipher, svc.oauth, uid)
    except CalendarNotConnected:
        return _error("calendar_not_connected", 409, reconnect_url="/auth/google/login")
    except CalendarAccessRevoked:
        return _error("calendar_access_revoked", 409, reconnect_url="/auth/google/login")
    return jsonify(synced=count)


@bp.get("/calendar/events")
@login_required_api
def calendar_events(uid: str):
    events = services().store.list_calendar_events(uid)
    return jsonify(events=[{"id": e["id"], "summary": e.get("summary"), "start": e.get("start"), "end": e.get("end"),
                            "all_day": e.get("all_day", False), "has_location": bool(e.get("location_enc"))}
                           for e in events])


def _client_location(value) -> tuple[float, float] | None:
    """The browser's {"lat", "lon"} sent with a chat message, if usable. It's used
    for this request only and never stored; anything invalid or outside the
    service area is ignored, and the agent then says it doesn't know where the user is."""
    if not isinstance(value, dict):
        return None
    try:
        lat, lon = float(value["lat"]), float(value["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lon) if in_service_area(lat, lon) else None


@bp.get("/trips/<trip_id>")
@login_required_api
def emailed_trip(uid: str, trip_id: str):
    """A trip from an alert email, for /trip/<trip_id>: {"subject", "body" (the
    email text), "sent_at" (local time), "route" (as /api/chat sends it)}. 404 when
    it isn't this user's or is gone."""
    from accounts.alerts import emailed_trip as load_trip
    svc = services()
    trip = load_trip(svc.store, svc.cipher, uid, trip_id)
    if trip is None:
        return _error("trip_not_found", 404)
    return jsonify(subject=trip["subject"], body=trip["body"], sent_at=trip["sent_at"], route=trip["route"])


@bp.post("/chat")
@login_required_api
def chat(uid: str):
    from agent.agent_interaction import (places_payload, reply_text, route_payload, stations_payload, token_usage,
                                         trim_history)

    svc = services()
    body = request.get_json(silent=True) or {}
    message = str(body.get("message") or "").strip()
    if not message:
        return _error("empty_message", 400)
    if len(message) > MAX_MESSAGE_CHARS:
        return _error("message_too_long", 413, max_chars=MAX_MESSAGE_CHARS)

    status = quota_status(svc.store, uid, svc.settings.weekly_token_limit)
    if status.exceeded:
        return _error("token_limit_reached", 429, usage=status.as_dict())

    with _inflight_lock:
        if uid in _inflight:
            return _error("request_in_progress", 409)
        _inflight.add(uid)
    try:
        agent = svc.agent_factory(uid, svc.store, svc.cipher, svc.store.get_conversation(uid),
                                  location=_client_location(body.get("location")))
        try:
            result = agent(message)
        finally:
            input_tokens, output_tokens = token_usage(agent)
            if input_tokens or output_tokens:
                record_usage(svc.store, uid, input_tokens, output_tokens)
        svc.store.set_conversation(uid, trim_history(agent.messages))
    except Exception:
        log.exception("agent call failed")
        return _error("agent_failed", 502)
    finally:
        with _inflight_lock:
            _inflight.discard(uid)

    # The route is sent as data built in code, not by the model: the browser shows
    # its directions and can draw its legs. route is null when none was planned.
    reply, route = reply_text(result), route_payload(agent)
    places = None if route else places_payload(agent)  # a planned route supersedes the search that led to it
    if route and "map" not in reply.lower():
        reply = f"{reply} I've highlighted the route on your map.".strip()
    return jsonify(reply=reply, route=route, places=places, stations=stations_payload(agent),
                   usage=quota_status(svc.store, uid, svc.settings.weekly_token_limit).as_dict())


@bp.delete("/chat")
@login_required_api
def clear_chat(uid: str):
    services().store.set_conversation(uid, [])
    return jsonify(cleared=True)


@bp.delete("/me")
@login_required_api
def delete_account(uid: str):
    """Revoke the Google grant (best effort), then delete everything we hold."""
    svc = services()
    google = (svc.store.get_user(uid) or {}).get("google") or {}
    if google.get("refresh_token_enc"):
        try:
            svc.oauth.revoke(svc.cipher.decrypt(google["refresh_token_enc"], user_id=uid, field=REFRESH_TOKEN_FIELD))
        except Exception:
            log.warning("token revoke failed during account deletion", exc_info=True)
    svc.store.delete_user(uid)
    session.clear()
    return jsonify(deleted=True)
