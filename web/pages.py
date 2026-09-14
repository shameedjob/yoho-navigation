"""Frontend pages, rendered from web/templates (recreated from
design_handoff_yoho_chat/). Pages hold no data; they call /api/*."""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, session, url_for

from .auth import current_uid, services

bp = Blueprint("pages", __name__)

_LOGIN_ERRORS = {
    "denied": "Google sign-in was cancelled.",
    "google": "Couldn't finish signing in with Google. Try again.",
    "unverified_email": "Your Google account's email isn't verified.",
}


def _signed_in() -> bool:
    uid = current_uid()
    return uid is not None and services().store.get_user(uid) is not None


def _chat_page(trip_id: str | None = None):
    # Each page load starts a fresh conversation: the page shows only the greeting,
    # so the agent mustn't keep answering from history the user can no longer see.
    services().store.set_conversation(current_uid(), [])
    return render_template("chat.html", trip_id=trip_id)


@bp.get("/")
def chat():
    return _chat_page() if _signed_in() else redirect(url_for("pages.login"))


@bp.get("/trip/<trip_id>")
def trip(trip_id: str):
    """The link in an alert email: the chat page, showing the emailed trip on the
    map (it loads /api/trips/<trip_id>). Signed out, it comes back here after login."""
    if not _signed_in():
        session["next"] = url_for("pages.trip", trip_id=trip_id)
        return redirect(url_for("pages.login"))
    return _chat_page(trip_id)


@bp.get("/profile")
def profile():
    return render_template("profile.html") if _signed_in() else redirect(url_for("pages.login"))


@bp.get("/login")
def login():
    if _signed_in():
        return redirect(url_for("pages.chat"))
    return render_template("login.html", error=_LOGIN_ERRORS.get(request.args.get("error", "")))
