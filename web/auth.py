"""Google sign-in: /auth/google/login -> Google -> /auth/google/callback.

The session cookie holds only the user's Google `sub`; tokens stay server-side
(the refresh token encrypted in Firestore, access tokens only in memory).
"""

from __future__ import annotations

import hmac
import logging
from functools import wraps

from flask import Blueprint, abort, current_app, jsonify, redirect, request, session, url_for

from accounts.calendar_sync import REFRESH_TOKEN_FIELD, CalendarNotConnected, sync_calendar
from integrations.google import CalendarAccessRevoked

log = logging.getLogger(__name__)
bp = Blueprint("auth", __name__, url_prefix="/auth")


def services():
    return current_app.extensions["yoho"]


def current_uid() -> str | None:
    return session.get("uid")


def login_required_api(view):
    """401 JSON for API calls without a session, and a CSRF check on writes.

    SameSite=Lax already keeps the cookie off cross-site POSTs; requiring a
    custom header on top means a write must come from same-origin JavaScript
    (a cross-site form can't set headers, and a cross-site fetch that does
    would need CORS approval this app never gives).
    """
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = current_uid()
        if uid is None or services().store.get_user(uid) is None:
            session.clear()
            return jsonify(error="unauthenticated"), 401
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("X-Yoho-Request") != "1":
            return jsonify(error="missing X-Yoho-Request header"), 403
        return view(uid, *args, **kwargs)
    return wrapper


@bp.get("/google/login")
def google_login():
    url, state, verifier = services().oauth.authorization_url()
    session["oauth_state"] = state
    session["oauth_verifier"] = verifier
    return redirect(url)


@bp.get("/google/callback")
def google_callback():
    svc = services()
    expected_state = session.pop("oauth_state", None)
    verifier = session.pop("oauth_verifier", None)
    if request.args.get("error"):  # user cancelled or denied consent
        return redirect(url_for("pages.login", error="denied"))
    state = request.args.get("state", "")
    if not expected_state or not verifier or not hmac.compare_digest(state, expected_state):
        abort(400, "OAuth state mismatch")
    code = request.args.get("code")
    if not code:
        abort(400, "missing authorization code")

    try:
        grant = svc.oauth.exchange_code(code, state, verifier)
    except Exception:
        log.exception("Google token exchange failed")
        return redirect(url_for("pages.login", error="google"))
    ident = grant.identity
    if not ident.email_verified:
        return redirect(url_for("pages.login", error="unverified_email"))

    uid = ident.sub
    fields: dict = {"email": ident.email, "name": ident.name, "picture": ident.picture}
    if grant.refresh_token:
        fields["google"] = {
            "refresh_token_enc": svc.cipher.encrypt(grant.refresh_token, user_id=uid, field=REFRESH_TOKEN_FIELD),
            "scopes": grant.scopes,
            "revoked": None,
        }
    svc.store.upsert_user(uid, fields)

    session.clear()  # drop anything from before login
    session.permanent = True
    session["uid"] = uid

    try:
        sync_calendar(svc.store, svc.cipher, svc.oauth, uid)
    except (CalendarNotConnected, CalendarAccessRevoked):
        pass  # the profile page shows the calendar as disconnected
    except Exception:
        log.exception("initial calendar sync failed for a user")  # don't block sign-in on it
    _start_alerts(svc, uid)
    return redirect(url_for("pages.chat"))


def _start_alerts(svc, uid: str) -> None:
    """Best effort after sign-in: push notifications for calendar changes (when a
    public webhook URL is configured) and the alert email subscription."""
    import time
    from accounts.alerts import ensure_email_subscription, ensure_watch
    if svc.settings.webhook_base_url:
        try:
            ensure_watch(svc.store, svc.cipher, svc.oauth, uid, svc.settings.webhook_base_url, int(time.time()))
        except (CalendarNotConnected, CalendarAccessRevoked):
            pass
        except Exception:
            log.exception("starting calendar watch failed for a user")
    try:
        ensure_email_subscription(svc.store, svc.notifier, uid)
    except Exception:
        log.exception("subscribing alert email failed for a user")


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("pages.login"))
