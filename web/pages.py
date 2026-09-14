"""Frontend pages, rendered from web/templates (recreated from
design_handoff_yoho_chat/). Pages hold no data; they call /api/*."""

from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, url_for

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


@bp.get("/")
def chat():
    return render_template("chat.html") if _signed_in() else redirect(url_for("pages.login"))


@bp.get("/profile")
def profile():
    return render_template("profile.html") if _signed_in() else redirect(url_for("pages.login"))


@bp.get("/login")
def login():
    if _signed_in():
        return redirect(url_for("pages.chat"))
    return render_template("login.html", error=_LOGIN_ERRORS.get(request.args.get("error", "")))
